import os
import json
import time
import logging
import pickle
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding_benchmarker')


@dataclass
class BenchmarkConfig:
    """Configuration for embedding model benchmarking"""
    original_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    fine_tuned_model_path: str = None  # Path to fine-tuned model
    output_dir: str = "benchmark_results"  # Directory to save benchmark results
    test_dataset_path: str = None  # Path to test dataset
    chroma_dir: str = "./chroma_db"  # Directory for Chroma DB
    num_evaluation_queries: int = 100  # Number of queries to evaluate
    top_k_values: List[int] = None  # Top-k values to evaluate
    seed: int = 42  # Random seed for reproducibility
    
    # Dimension reduction parameters
    enable_dimension_reduction: bool = True  # Whether to use dimension reduction
    target_dimension: int = 512  # Target dimension for reduction
    reduction_method: str = "pca"  # Method for dimension reduction
    
    def __post_init__(self):
        if self.top_k_values is None:
            self.top_k_values = [1, 3, 5, 10, 20]
        os.makedirs(self.output_dir, exist_ok=True)


class EmbeddingBenchmarker:
    """
    Benchmarks embedding models on domain-specific data
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """Initialize the benchmarker with configuration"""
        self.config = config or BenchmarkConfig()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Maps to hold model information
        self.models = {}
        self.indices = {}
        self.query_engines = {}
        
        # Dimension reducers (if enabled)
        self.reducers = {}
        
        logger.info(f"Initializing EmbeddingBenchmarker")
        logger.info(f"Original model: {self.config.original_model_name}")
        logger.info(f"Fine-tuned model: {self.config.fine_tuned_model_path}")
        
        if self.config.enable_dimension_reduction:
            logger.info(f"Dimension reduction enabled: {self.config.reduction_method} to {self.config.target_dimension}d")
    
    def _setup_model(self, model_name_or_path: str, model_id: str, sample_texts: List[str] = None):
        """Set up a model for benchmarking, with optional dimension reduction"""
        try:
            # Import necessary modules
            from sentence_transformers import SentenceTransformer
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Try to load as a Sentence Transformer model
            model = SentenceTransformer(model_name_or_path)
            original_dim = model.get_sentence_embedding_dimension()
            
            # Initialize embedding model for LlamaIndex
            if self.config.enable_dimension_reduction and sample_texts:
                # Import dimension reducer
                try:
                    from embedding_dimension_reducer import (
                        ReducedDimensionHuggingFaceEmbedding,
                        EmbeddingDimensionReducer,
                        DimensionReductionConfig
                    )
                    
                    # Use reduced dimension embedding
                    reducer_config = DimensionReductionConfig(
                        reduction_method=self.config.reduction_method,
                        target_dim=self.config.target_dimension
                    )
                    
                    reducer = EmbeddingDimensionReducer(reducer_config)
                    self.reducers[model_id] = reducer
                    
                    # Create reduced embedding model
                    embed_model = ReducedDimensionHuggingFaceEmbedding(
                        model_name=model_name_or_path,
                        target_dim=self.config.target_dimension,
                        reduction_method=self.config.reduction_method
                    )
                    
                    # Prepare with sample texts
                    embed_model.prepare(sample_texts)
                    
                    # Record dimensions
                    target_dim = self.config.target_dimension
                    logger.info(f"Reduced {model_id} embeddings from {original_dim}d to {target_dim}d")
                except ImportError:
                    logger.warning("Dimension reducer not available, using standard embedding model without dimension reduction")
                    # Regular embedding model without dimension reduction
                    embed_model = HuggingFaceEmbedding(
                        model_name=model_name_or_path
                    )
                    target_dim = original_dim
            else:
                # Regular embedding model without dimension reduction
                embed_model = HuggingFaceEmbedding(
                    model_name=model_name_or_path
                )
                target_dim = original_dim
            
            self.models[model_id] = {
                "name": model_name_or_path,
                "model": model,
                "llama_index_embed": embed_model,
                "original_dim": original_dim,
                "target_dim": target_dim
            }
            
            logger.info(f"Successfully loaded model: {model_name_or_path} as {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {model_name_or_path}: {e}")
            return False
    
    def setup_models(self, sample_texts: List[str] = None):
        """Set up all models for benchmarking, with sample texts for dimension reduction"""
        # Set up original model
        original_success = self._setup_model(
            self.config.original_model_name, 
            "original",
            sample_texts
        )
        
        # Set up fine-tuned model if provided
        fine_tuned_success = False
        if self.config.fine_tuned_model_path:
            fine_tuned_success = self._setup_model(
                self.config.fine_tuned_model_path,
                "fine_tuned",
                sample_texts
            )
        
        return original_success and (not self.config.fine_tuned_model_path or fine_tuned_success)
    
    def _create_vector_index(self, 
                            documents: List, 
                            model_id: str, 
                            collection_name: str):
        """Create a vector index for a model using Chroma"""
        # Import necessary modules
        from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
        
        model_dir = os.path.join(self.config.chroma_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Set up Chroma client and collection
        chroma_client = chromadb.PersistentClient(path=model_dir)
        
        # Check if collection exists and delete it
        try:
            collections = chroma_client.list_collections()
            for collection in collections:
                if collection.name == collection_name:
                    chroma_client.delete_collection(collection_name)
                    logger.info(f"Deleted existing collection: {collection_name}")
                    break
        except Exception as e:
            logger.warning(f"Error checking/deleting collection: {e}")
        
        # Create new collection
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Set embedding model
        Settings.embed_model = self.models[model_id]["llama_index_embed"]
        
        # Create index
        try:
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True
            )
            logger.info(f"Created index for {model_id} with {len(documents)} documents")
            return index
            
        except Exception as e:
            logger.error(f"Error creating index for {model_id}: {e}")
            return None
    
    def prepare_data_from_test_dataset(self) -> Tuple[List, List[Dict], List[str]]:
        """
        Prepare documents and test queries from a test dataset
        
        Returns:
            Tuple of (documents, test_queries, sample_texts)
        """
        from llama_index.core import Document
        
        if not self.config.test_dataset_path:
            raise ValueError("Test dataset path not provided")
            
        # Load the dataset
        try:
            with open(self.config.test_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Check data format
            if isinstance(data, dict) and 'data' in data:
                qa_pairs = data['data']
            elif isinstance(data, list):
                qa_pairs = data
            else:
                raise ValueError(f"Unexpected data format in {self.config.test_dataset_path}")
                
            logger.info(f"Loaded {len(qa_pairs)} QA pairs from test dataset")
                
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            raise
        
        # Create documents from answers
        documents = []
        sample_texts = []  # For dimension reduction training
        for idx, item in enumerate(qa_pairs):
            answer = item.get('answer', '')
            if not answer:
                continue
                
            doc = Document(
                text=answer,
                metadata={
                    "source": f"test_dataset_{idx}",
                    "question": item.get('question', ''),
                    "subtopic": item.get('subtopic', 'general_finance'),
                    "qa_idx": idx
                }
            )
            documents.append(doc)
            
            # Add to sample texts for dimension reduction
            sample_texts.append(answer)
        
        logger.info(f"Created {len(documents)} documents from QA pairs")
        
        # Create test queries from questions
        test_queries = []
        for idx, item in enumerate(qa_pairs):
            question = item.get('question', '')
            if not question:
                continue
                
            query_item = {
                "query": question,
                "qa_idx": idx,
                "subtopic": item.get('subtopic', 'general_finance'),
                "expected_doc_idx": idx,  # The index of the document that should be retrieved
                "complexity": item.get('complexity', 'basic')
            }
            test_queries.append(query_item)
            
            # Add questions to sample texts as well
            sample_texts.append(question)
        
        logger.info(f"Created {len(test_queries)} test queries from QA pairs")
        
        # If we need to limit the number of evaluation queries
        if len(test_queries) > self.config.num_evaluation_queries:
            # Try to maintain a good distribution of subtopics and complexity
            subtopics = {}
            for query in test_queries:
                subtopic = query.get('subtopic', 'general_finance')
                if subtopic not in subtopics:
                    subtopics[subtopic] = []
                subtopics[subtopic].append(query)
            
            # Calculate how many queries to select from each subtopic
            total_queries = sum(len(queries) for queries in subtopics.values())
            selected_queries = []
            
            for subtopic, queries in subtopics.items():
                # Proportional selection
                n_select = max(1, int(len(queries) / total_queries * self.config.num_evaluation_queries))
                # Ensure we don't select more than we have
                n_select = min(n_select, len(queries))
                # Randomly select queries
                selected = random.sample(queries, n_select)
                selected_queries.extend(selected)
            
            # If we still need more queries, randomly select from remaining
            if len(selected_queries) < self.config.num_evaluation_queries:
                remaining = [q for q in test_queries if q not in selected_queries]
                n_remaining = self.config.num_evaluation_queries - len(selected_queries)
                if remaining and n_remaining > 0:
                    selected_queries.extend(random.sample(remaining, min(n_remaining, len(remaining))))
            
            # If we selected too many, trim
            if len(selected_queries) > self.config.num_evaluation_queries:
                selected_queries = random.sample(selected_queries, self.config.num_evaluation_queries)
                
            test_queries = selected_queries
            logger.info(f"Selected {len(test_queries)} test queries for evaluation")
        
        # Limit sample texts for dimension reduction training
        max_samples = 1000  # Limit sample size for dimension reduction
        if len(sample_texts) > max_samples:
            sample_texts = random.sample(sample_texts, max_samples)
            logger.info(f"Selected {len(sample_texts)} sample texts for dimension reduction")
        
        return documents, test_queries, sample_texts
    
    def prepare_vector_indices(self, documents: List):
        """Prepare vector indices for all models"""
        # Ensure LLM is set to None
        from llama_index.core import Settings
        Settings.llm = None
        
        for model_id in self.models:
            # Create collection name with timestamp to avoid conflicts
            collection_name = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create vector index
            index = self._create_vector_index(documents, model_id, collection_name)
            
            if index:
                self.indices[model_id] = index
                
                # Create query engine with response_mode="no_text" to avoid using LLM
                self.query_engines[model_id] = index.as_query_engine(
                    similarity_top_k=max(self.config.top_k_values),
                    response_mode="no_text"  # This mode doesn't need an LLM
                )
                
                logger.info(f"Created query engine for {model_id}")
            else:
                logger.error(f"Failed to create index for {model_id}")
    
    def evaluate_retrieval(self, test_queries: List[Dict]) -> Dict:
        """
        Evaluate retrieval performance of all models
        
        Args:
            test_queries: List of test query items
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        # Initialize metrics for each model and k value
        for model_id in self.query_engines:
            results[model_id] = {
                "precision": {k: [] for k in self.config.top_k_values},
                "recall": {k: [] for k in self.config.top_k_values},  # Added recall
                "f1": {k: [] for k in self.config.top_k_values},      # Added F1
                "reciprocal_rank": [],
                "retrieval_time": [],
                "subtopic_performance": {},
                "complexity_performance": {},
                "query_results": []
            }
        
        # Run queries and evaluate
        for query_item in tqdm(test_queries, desc="Evaluating queries"):
            query_text = query_item["query"]
            expected_doc_idx = query_item["expected_doc_idx"]
            subtopic = query_item.get("subtopic", "general_finance")
            complexity = query_item.get("complexity", "basic")
            
            # Initialize subtopic and complexity metrics if needed
            for model_id in results:
                if subtopic not in results[model_id]["subtopic_performance"]:
                    results[model_id]["subtopic_performance"][subtopic] = {
                        "precision": {k: [] for k in self.config.top_k_values},
                        "recall": {k: [] for k in self.config.top_k_values},  # Added recall
                        "f1": {k: [] for k in self.config.top_k_values},      # Added F1
                        "reciprocal_rank": []
                    }
                
                if complexity not in results[model_id]["complexity_performance"]:
                    results[model_id]["complexity_performance"][complexity] = {
                        "precision": {k: [] for k in self.config.top_k_values},
                        "recall": {k: [] for k in self.config.top_k_values},  # Added recall
                        "f1": {k: [] for k in self.config.top_k_values},      # Added F1
                        "reciprocal_rank": []
                    }
            
            # Run query for each model
            for model_id, query_engine in self.query_engines.items():
                start_time = time.time()
                
                try:
                    # Get response
                    response = query_engine.query(query_text)
                    
                    # Extract source nodes
                    if hasattr(response, 'source_nodes'):
                        source_nodes = response.source_nodes
                        retrieved_scores = []
                        retrieved_idxs = []
                        
                        # Get document indices and scores
                        for node in source_nodes:
                            if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                                metadata = node.node.metadata
                                if 'qa_idx' in metadata:
                                    retrieved_idxs.append(metadata['qa_idx'])
                                    # Add score if available
                                    if hasattr(node, 'score'):
                                        retrieved_scores.append(node.score)
                    else:
                        retrieved_idxs = []
                        retrieved_scores = []
                        
                    # Calculate retrieval time
                    retrieval_time = time.time() - start_time
                    results[model_id]["retrieval_time"].append(retrieval_time)
                    
                    # Calculate metrics for each k
                    for k in self.config.top_k_values:
                        # Get top-k results
                        top_k_idxs = retrieved_idxs[:k]
                        
                        # Calculate precision@k (1 if correct doc in top-k, 0 otherwise)
                        precision_k = 1.0 if expected_doc_idx in top_k_idxs else 0.0
                        results[model_id]["precision"][k].append(precision_k)
                        
                        # For simple case with one relevant doc, recall is same as precision
                        # In a more complex scenario with multiple relevant docs, this would be different
                        recall_k = precision_k
                        results[model_id]["recall"][k].append(recall_k)
                        
                        # Calculate F1 score (harmonic mean of precision and recall)
                        if precision_k > 0 and recall_k > 0:
                            f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                        else:
                            f1_k = 0.0
                        results[model_id]["f1"][k].append(f1_k)
                        
                        # Add to subtopic and complexity metrics
                        results[model_id]["subtopic_performance"][subtopic]["precision"][k].append(precision_k)
                        results[model_id]["subtopic_performance"][subtopic]["recall"][k].append(recall_k)
                        results[model_id]["subtopic_performance"][subtopic]["f1"][k].append(f1_k)
                        
                        results[model_id]["complexity_performance"][complexity]["precision"][k].append(precision_k)
                        results[model_id]["complexity_performance"][complexity]["recall"][k].append(recall_k)
                        results[model_id]["complexity_performance"][complexity]["f1"][k].append(f1_k)
                    
                    # Calculate reciprocal rank (1/position of first relevant result)
                    if expected_doc_idx in retrieved_idxs:
                        rank = retrieved_idxs.index(expected_doc_idx) + 1
                        rr = 1.0 / rank
                    else:
                        rr = 0.0
                        
                    results[model_id]["reciprocal_rank"].append(rr)
                    results[model_id]["subtopic_performance"][subtopic]["reciprocal_rank"].append(rr)
                    results[model_id]["complexity_performance"][complexity]["reciprocal_rank"].append(rr)
                    
                    # Store query result
                    results[model_id]["query_results"].append({
                        "query": query_text,
                        "expected_doc_idx": expected_doc_idx,
                        "retrieved_idxs": retrieved_idxs[:max(self.config.top_k_values)],
                        "retrieved_scores": retrieved_scores[:max(self.config.top_k_values)],
                        "subtopic": subtopic,
                        "complexity": complexity,
                        "reciprocal_rank": rr,
                        "retrieval_time": retrieval_time
                    })
                    
                except Exception as e:
                    logger.error(f"Error querying {model_id} with '{query_text[:50]}...': {e}")
                    # Add zeros for failed queries
                    for k in self.config.top_k_values:
                        results[model_id]["precision"][k].append(0.0)
                        results[model_id]["recall"][k].append(0.0)
                        results[model_id]["f1"][k].append(0.0)
                    results[model_id]["reciprocal_rank"].append(0.0)
                    results[model_id]["retrieval_time"].append(0.0)
        
        # Calculate mean metrics
        for model_id in results:
            # Overall metrics
            for k in self.config.top_k_values:
                results[model_id][f"precision@{k}"] = np.mean(results[model_id]["precision"][k]) if results[model_id]["precision"][k] else 0.0
                results[model_id][f"recall@{k}"] = np.mean(results[model_id]["recall"][k]) if results[model_id]["recall"][k] else 0.0
                results[model_id][f"f1@{k}"] = np.mean(results[model_id]["f1"][k]) if results[model_id]["f1"][k] else 0.0
            
            results[model_id]["mean_reciprocal_rank"] = np.mean(results[model_id]["reciprocal_rank"]) if results[model_id]["reciprocal_rank"] else 0.0
            results[model_id]["mean_retrieval_time"] = np.mean(results[model_id]["retrieval_time"]) if results[model_id]["retrieval_time"] else 0.0
            
            # Subtopic metrics
            for subtopic in results[model_id]["subtopic_performance"]:
                for k in self.config.top_k_values:
                    precision_k = results[model_id]["subtopic_performance"][subtopic]["precision"][k]
                    recall_k = results[model_id]["subtopic_performance"][subtopic]["recall"][k]
                    f1_k = results[model_id]["subtopic_performance"][subtopic]["f1"][k]
                    
                    if precision_k:
                        results[model_id]["subtopic_performance"][subtopic][f"precision@{k}"] = np.mean(precision_k)
                    if recall_k:
                        results[model_id]["subtopic_performance"][subtopic][f"recall@{k}"] = np.mean(recall_k)
                    if f1_k:
                        results[model_id]["subtopic_performance"][subtopic][f"f1@{k}"] = np.mean(f1_k)
                
                rr = results[model_id]["subtopic_performance"][subtopic]["reciprocal_rank"]
                if rr:
                    results[model_id]["subtopic_performance"][subtopic]["mean_reciprocal_rank"] = np.mean(rr)
            
            # Complexity metrics
            for complexity in results[model_id]["complexity_performance"]:
                for k in self.config.top_k_values:
                    precision_k = results[model_id]["complexity_performance"][complexity]["precision"][k]
                    recall_k = results[model_id]["complexity_performance"][complexity]["recall"][k]
                    f1_k = results[model_id]["complexity_performance"][complexity]["f1"][k]
                    
                    if precision_k:
                        results[model_id]["complexity_performance"][complexity][f"precision@{k}"] = np.mean(precision_k)
                    if recall_k:
                        results[model_id]["complexity_performance"][complexity][f"recall@{k}"] = np.mean(recall_k)
                    if f1_k:
                        results[model_id]["complexity_performance"][complexity][f"f1@{k}"] = np.mean(f1_k)
                
                rr = results[model_id]["complexity_performance"][complexity]["reciprocal_rank"]
                if rr:
                    results[model_id]["complexity_performance"][complexity]["mean_reciprocal_rank"] = np.mean(rr)
            
            # Add dimension information
            model_info = self.models.get(model_id, {})
            results[model_id]["original_dimension"] = model_info.get("original_dim", 0)
            results[model_id]["target_dimension"] = model_info.get("target_dim", 0)
        
        return results
    
    def generate_comparison_plots(self, results: Dict):
        """Generate comparison plots for benchmark results"""
        # Create results directory
        plots_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Extract model names
        model_names = list(results.keys())
        if len(model_names) < 2:
            logger.warning("Not enough models for comparison plots")
            return
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Precision@k comparison plot
        plt.figure(figsize=(10, 6))
        
        for model_id in model_names:
            precision_values = [results[model_id][f"precision@{k}"] for k in self.config.top_k_values]
            plt.plot(self.config.top_k_values, precision_values, marker='o', linewidth=2, label=model_id)
        
        plt.xlabel('k', fontsize=14)
        plt.ylabel('Precision@k', fontsize=14)
        plt.title('Precision@k Comparison', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        precision_plot_path = os.path.join(plots_dir, "precision_comparison.png")
        plt.savefig(precision_plot_path)
        plt.close()
        
        # 2. Recall@k comparison plot
        plt.figure(figsize=(10, 6))
        
        for model_id in model_names:
            recall_values = [results[model_id][f"recall@{k}"] for k in self.config.top_k_values]
            plt.plot(self.config.top_k_values, recall_values, marker='o', linewidth=2, label=model_id)
        
        plt.xlabel('k', fontsize=14)
        plt.ylabel('Recall@k', fontsize=14)
        plt.title('Recall@k Comparison', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        recall_plot_path = os.path.join(plots_dir, "recall_comparison.png")
        plt.savefig(recall_plot_path)
        plt.close()
        
        # 3. F1@k comparison plot
        plt.figure(figsize=(10, 6))
        
        for model_id in model_names:
            f1_values = [results[model_id][f"f1@{k}"] for k in self.config.top_k_values]
            plt.plot(self.config.top_k_values, f1_values, marker='o', linewidth=2, label=model_id)
        
        plt.xlabel('k', fontsize=14)
        plt.ylabel('F1@k', fontsize=14)
        plt.title('F1@k Comparison', fontsize=16)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        f1_plot_path = os.path.join(plots_dir, "f1_comparison.png")
        plt.savefig(f1_plot_path)
        plt.close()
        
        # 4. MRR and Retrieval Time Comparison (Bar chart)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(model_names))
        width = 0.35
        
        # MRR bars
        mrr_values = [results[model_id]["mean_reciprocal_rank"] for model_id in model_names]
        ax1.bar(x - width/2, mrr_values, width, label='Mean Reciprocal Rank', color='skyblue')
        ax1.set_ylabel('Mean Reciprocal Rank', fontsize=14)
        ax1.set_ylim(0, 1.0)
        
        # Retrieval time bars
        ax2 = ax1.twinx()
        time_values = [results[model_id]["mean_retrieval_time"] for model_id in model_names]
        ax2.bar(x + width/2, time_values, width, label='Mean Retrieval Time (s)', color='lightgreen')
        ax2.set_ylabel('Time (seconds)', fontsize=14)
        
        # Set up x-axis
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names)
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.title('MRR and Retrieval Time Comparison', fontsize=16)
        plt.tight_layout()
        
        mrr_time_plot_path = os.path.join(plots_dir, "mrr_time_comparison.png")
        plt.savefig(mrr_time_plot_path)
        plt.close()
        
        logger.info(f"Generated comparison plots in {plots_dir}")
        return plots_dir
    
    def run_benchmark(self) -> Dict:
        """Run a complete benchmark of all models"""
        # Import needed libraries
        from llama_index.core import Settings
        
        # Disable LLM for benchmarking
        Settings.llm = None
        logger.info("Set LLM to None for benchmarking")
    
        # Prepare data from test dataset
        documents, test_queries, sample_texts = self.prepare_data_from_test_dataset()
        
        # Set up models with sample texts for dimension reduction
        if not self.setup_models(sample_texts):
            logger.error("Failed to set up models. Exiting.")
            return None
        
        # Prepare vector indices
        self.prepare_vector_indices(documents)
        
        # Evaluate retrieval
        results = self.evaluate_retrieval(test_queries)
        
        # Generate comparison plots
        plots_dir = self.generate_comparison_plots(results)
        
        # Save results
        results_file = os.path.join(
            self.config.output_dir, 
            f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        # Clean up results for JSON serialization
        clean_results = {}
        for model_id, model_results in results.items():
            clean_results[model_id] = {}
            
            # Copy scalar metrics
            for k in self.config.top_k_values:
                clean_results[model_id][f"precision@{k}"] = float(model_results[f"precision@{k}"])
                clean_results[model_id][f"recall@{k}"] = float(model_results[f"recall@{k}"])
                clean_results[model_id][f"f1@{k}"] = float(model_results[f"f1@{k}"])
            
            clean_results[model_id]["mean_reciprocal_rank"] = float(model_results["mean_reciprocal_rank"])
            clean_results[model_id]["mean_retrieval_time"] = float(model_results["mean_retrieval_time"])
            
            # Add dimension information
            clean_results[model_id]["original_dimension"] = int(model_results.get("original_dimension", 0))
            clean_results[model_id]["target_dimension"] = int(model_results.get("target_dimension", 0))
            
            # Clean subtopic performance
            clean_results[model_id]["subtopic_performance"] = {}
            for subtopic, metrics in model_results["subtopic_performance"].items():
                clean_results[model_id]["subtopic_performance"][subtopic] = {}
                for metric, value in metrics.items():
                    if isinstance(value, list):
                        continue
                    # Skip dictionary values or convert numeric values to float
                    if isinstance(value, dict):
                        continue
                    if isinstance(value, (int, float)):
                        clean_results[model_id]["subtopic_performance"][subtopic][metric] = float(value)
                    else:
                        try:
                            clean_results[model_id]["subtopic_performance"][subtopic][metric] = float(value)
                        except (TypeError, ValueError):
                            # If conversion fails, just use the original value
                            clean_results[model_id]["subtopic_performance"][subtopic][metric] = value
            
            # Clean complexity performance
            clean_results[model_id]["complexity_performance"] = {}
            for complexity, metrics in model_results["complexity_performance"].items():
                clean_results[model_id]["complexity_performance"][complexity] = {}
                for metric, value in metrics.items():
                    if isinstance(value, list):
                        continue
                    # Skip dictionary values or convert numeric values to float
                    if isinstance(value, dict):
                        continue
                    if isinstance(value, (int, float)):
                        clean_results[model_id]["complexity_performance"][complexity][metric] = float(value)
                    else:
                        try:
                            clean_results[model_id]["complexity_performance"][complexity][metric] = float(value)
                        except (TypeError, ValueError):
                            # If conversion fails, just use the original value
                            clean_results[model_id]["complexity_performance"][complexity][metric] = value
            
            # Store a sample of query results
            clean_results[model_id]["query_sample"] = model_results["query_results"][:10]
        
        # Add configuration information
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "original_model": self.config.original_model_name,
            "fine_tuned_model": self.config.fine_tuned_model_path,
            "enable_dimension_reduction": self.config.enable_dimension_reduction,
            "target_dimension": self.config.target_dimension,
            "reduction_method": self.config.reduction_method,
            "k_values": self.config.top_k_values,
            "num_evaluation_queries": self.config.num_evaluation_queries
        }
        
        # Final output structure
        final_results = {
            "metadata": metadata,
            "results": clean_results
        }
        
        # Save to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
        
        # Calculate and report improvement
        if "original" in clean_results and "fine_tuned" in clean_results:
            improvements = {}
            
            # Precision@k improvement
            for k in self.config.top_k_values:
                orig_precision = clean_results["original"][f"precision@{k}"]
                ft_precision = clean_results["fine_tuned"][f"precision@{k}"]
                
                if orig_precision > 0:
                    rel_improvement = (ft_precision - orig_precision) / orig_precision * 100
                    abs_improvement = ft_precision - orig_precision
                    improvements[f"precision@{k}"] = {
                        "original": orig_precision,
                        "fine_tuned": ft_precision,
                        "absolute_improvement": abs_improvement,
                        "relative_improvement_percent": rel_improvement
                    }
            
            # Recall@k improvement
            for k in self.config.top_k_values:
                orig_recall = clean_results["original"][f"recall@{k}"]
                ft_recall = clean_results["fine_tuned"][f"recall@{k}"]
                
                if orig_recall > 0:
                    rel_improvement = (ft_recall - orig_recall) / orig_recall * 100
                    abs_improvement = ft_recall - orig_recall
                    improvements[f"recall@{k}"] = {
                        "original": orig_recall,
                        "fine_tuned": ft_recall,
                        "absolute_improvement": abs_improvement,
                        "relative_improvement_percent": rel_improvement
                    }
            
            # F1@k improvement
            for k in self.config.top_k_values:
                orig_f1 = clean_results["original"][f"f1@{k}"]
                ft_f1 = clean_results["fine_tuned"][f"f1@{k}"]
                
                if orig_f1 > 0:
                    rel_improvement = (ft_f1 - orig_f1) / orig_f1 * 100
                    abs_improvement = ft_f1 - orig_f1
                    improvements[f"f1@{k}"] = {
                        "original": orig_f1,
                        "fine_tuned": ft_f1,
                        "absolute_improvement": abs_improvement,
                        "relative_improvement_percent": rel_improvement
                    }
            
            # MRR improvement
            orig_mrr = clean_results["original"]["mean_reciprocal_rank"]
            ft_mrr = clean_results["fine_tuned"]["mean_reciprocal_rank"]
            
            if orig_mrr > 0:
                rel_improvement = (ft_mrr - orig_mrr) / orig_mrr * 100
                abs_improvement = ft_mrr - orig_mrr
                improvements["mean_reciprocal_rank"] = {
                    "original": orig_mrr,
                    "fine_tuned": ft_mrr,
                    "absolute_improvement": abs_improvement,
                    "relative_improvement_percent": rel_improvement
                }
            
            # Retrieval time improvement (negative means faster, which is better)
            orig_time = clean_results["original"]["mean_retrieval_time"]
            ft_time = clean_results["fine_tuned"]["mean_retrieval_time"]
            
            if orig_time > 0:
                time_diff = orig_time - ft_time
                time_pct = (time_diff / orig_time) * 100
                
                improvements["mean_retrieval_time"] = {
                    "original": orig_time,
                    "fine_tuned": ft_time,
                    "absolute_improvement": time_diff,
                    "relative_improvement_percent": time_pct
                }
            
            # Save improvements to file
            improvements_file = os.path.join(
                self.config.output_dir, 
                f"improvements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(improvements_file, 'w', encoding='utf-8') as f:
                json.dump(improvements, f, indent=2)
            
            logger.info(f"Improvement metrics saved to {improvements_file}")
            
            # Log key improvements
            logger.info("\nKey Improvements:")
            for metric, data in improvements.items():
                logger.info(f"{metric}: {data['relative_improvement_percent']:.2f}% improvement")
            
            # Add improvements to final results
            final_results["improvements"] = improvements
        
        return final_results
    
    def get_summary(self, results: Dict) -> str:
        """Generate a human-readable summary of benchmark results"""
        if not results:
            return "No benchmark results available."
        
        # Extract results data
        clean_results = results.get("results", {})
        metadata = results.get("metadata", {})
        improvements = results.get("improvements", {})
        
        summary = []
        summary.append("# Embedding Model Benchmark Summary\n")
        
        # Configuration summary
        summary.append("## Configuration\n")
        summary.append(f"- **Original Model:** {metadata.get('original_model', 'Unknown')}\n")
        if metadata.get('fine_tuned_model'):
            summary.append(f"- **Fine-tuned Model:** {metadata.get('fine_tuned_model', 'None')}\n")
        
        if metadata.get('enable_dimension_reduction'):
            summary.append(f"- **Dimension Reduction:** {metadata.get('reduction_method', 'Unknown')} to {metadata.get('target_dimension')} dimensions\n")
        
        summary.append(f"- **Evaluation Queries:** {metadata.get('num_evaluation_queries', 0)}\n")
        summary.append(f"- **Date:** {metadata.get('timestamp', datetime.now().isoformat())}\n")
        
        # Models compared
        summary.append("\n## Models Compared\n")
        for model_id in clean_results:
            model_name = metadata.get(f"{model_id}_model", model_id)
            orig_dim = clean_results[model_id].get("original_dimension", 0)
            target_dim = clean_results[model_id].get("target_dimension", 0)
            
            if orig_dim > 0 and orig_dim != target_dim:
                summary.append(f"- **{model_id}**: {model_name} ({target_dim}d reduced from {orig_dim}d)\n")
            else:
                summary.append(f"- **{model_id}**: {model_name} ({target_dim}d)\n")
        
        # Overall metrics
        summary.append("\n## Overall Performance\n")
        
        # Add table header for precision@k
        summary.append("### Precision@k\n")
        header = "| Model | " + " | ".join([f"P@{k}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        divider = "| --- | " + " | ".join(["---" for _ in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        
        summary.append(header)
        summary.append(divider)
        
        for model_id in clean_results:
            row = f"| {model_id} | "
            row += " | ".join([f"{clean_results[model_id].get(f'precision@{k}', 0):.3f}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])])
            row += " |\n"
            summary.append(row)
        
        # Add table header for recall@k
        summary.append("\n### Recall@k\n")
        header = "| Model | " + " | ".join([f"R@{k}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        divider = "| --- | " + " | ".join(["---" for _ in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        
        summary.append(header)
        summary.append(divider)
        
        for model_id in clean_results:
            row = f"| {model_id} | "
            row += " | ".join([f"{clean_results[model_id].get(f'recall@{k}', 0):.3f}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])])
            row += " |\n"
            summary.append(row)
        
        # Add table header for F1@k
        summary.append("\n### F1@k\n")
        header = "| Model | " + " | ".join([f"F1@{k}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        divider = "| --- | " + " | ".join(["---" for _ in metadata.get('k_values', [1, 3, 5, 10, 20])]) + " |\n"
        
        summary.append(header)
        summary.append(divider)
        
        for model_id in clean_results:
            row = f"| {model_id} | "
            row += " | ".join([f"{clean_results[model_id].get(f'f1@{k}', 0):.3f}" for k in metadata.get('k_values', [1, 3, 5, 10, 20])])
            row += " |\n"
            summary.append(row)
        
        # MRR and Retrieval Time
        summary.append("\n### Mean Reciprocal Rank and Retrieval Time\n")
        summary.append("| Model | MRR | Retrieval Time (s) |\n")
        summary.append("| --- | --- | --- |\n")
        
        for model_id in clean_results:
            mrr = clean_results[model_id].get("mean_reciprocal_rank", 0)
            time = clean_results[model_id].get("mean_retrieval_time", 0)
            summary.append(f"| {model_id} | {mrr:.3f} | {time:.3f} |\n")
        
        # Improvements
        if improvements:
            summary.append("\n## Improvements\n")
            
            # Precision@k improvements
            for k in [1, 3, 5, 10, 20]:
                key = f"precision@{k}"
                if key in improvements:
                    imp = improvements[key]
                    abs_imp = imp.get("absolute_improvement", 0)
                    rel_imp = imp.get("relative_improvement_percent", 0)
                    summary.append(f"- **Precision@{k}**: {abs_imp:.3f} absolute / {rel_imp:.2f}% relative improvement\n")
            
            # Recall@k improvements
            for k in [1, 3, 5, 10, 20]:
                key = f"recall@{k}"
                if key in improvements:
                    imp = improvements[key]
                    abs_imp = imp.get("absolute_improvement", 0)
                    rel_imp = imp.get("relative_improvement_percent", 0)
                    summary.append(f"- **Recall@{k}**: {abs_imp:.3f} absolute / {rel_imp:.2f}% relative improvement\n")
            
            # F1@k improvements
            for k in [1, 3, 5, 10, 20]:
                key = f"f1@{k}"
                if key in improvements:
                    imp = improvements[key]
                    abs_imp = imp.get("absolute_improvement", 0)
                    rel_imp = imp.get("relative_improvement_percent", 0)
                    summary.append(f"- **F1@{k}**: {abs_imp:.3f} absolute / {rel_imp:.2f}% relative improvement\n")
            
            # MRR improvement
            if "mean_reciprocal_rank" in improvements:
                mrr_imp = improvements["mean_reciprocal_rank"]
                abs_imp = mrr_imp.get("absolute_improvement", 0)
                rel_imp = mrr_imp.get("relative_improvement_percent", 0)
                summary.append(f"- **Mean Reciprocal Rank**: {abs_imp:.3f} absolute / {rel_imp:.2f}% relative improvement\n")
            
            # Retrieval time improvement
            if "mean_retrieval_time" in improvements:
                time_imp = improvements["mean_retrieval_time"]
                time_diff = time_imp.get("absolute_improvement", 0)
                time_pct = time_imp.get("relative_improvement_percent", 0)
                
                if time_diff > 0:
                    summary.append(f"- **Retrieval Time**: {time_diff:.3f}s faster ({time_pct:.2f}% improvement)\n")
                else:
                    summary.append(f"- **Retrieval Time**: {-time_diff:.3f}s slower ({-time_pct:.2f}% slower)\n")
        
        # Dimension reduction impact summary
        if metadata.get('enable_dimension_reduction') and "fine_tuned" in clean_results:
            summary.append("\n## Dimension Reduction Impact\n")
            
            orig_dim = clean_results["fine_tuned"].get("original_dimension", 0)
            target_dim = clean_results["fine_tuned"].get("target_dimension", 0)
            
            if orig_dim > 0 and target_dim > 0:
                dim_reduction_pct = ((orig_dim - target_dim) / orig_dim) * 100
                summary.append(f"- **Dimension Reduction**: {orig_dim}d → {target_dim}d ({dim_reduction_pct:.1f}% smaller)\n")
                
                # Performance impact
                mrr = clean_results["fine_tuned"].get("mean_reciprocal_rank", 0)
                p1 = clean_results["fine_tuned"].get("precision@1", 0)
                
                summary.append(f"- **Performance with Reduced Dimensions**:\n")
                summary.append(f"  - Mean Reciprocal Rank: {mrr:.3f}\n")
                summary.append(f"  - Precision@1: {p1:.3f}\n")
                
                # Add storage efficiency note
                summary.append(f"- **Storage Efficiency**: Approximately {dim_reduction_pct:.1f}% reduction in embedding storage requirements\n")
        
        # Conclusion
        summary.append("\n## Conclusion\n")
        
        if "original" in clean_results and "fine_tuned" in clean_results:
            # Compare MRR as overall metric
            orig_mrr = clean_results["original"].get("mean_reciprocal_rank", 0)
            ft_mrr = clean_results["fine_tuned"].get("mean_reciprocal_rank", 0)
            
            if ft_mrr > orig_mrr:
                rel_imp = (ft_mrr - orig_mrr) / orig_mrr * 100 if orig_mrr > 0 else 0
                summary.append(f"The fine-tuned model shows an overall improvement of {rel_imp:.2f}% in Mean Reciprocal Rank compared to the original model. ")
                
                # Add dimension reduction context if enabled
                if metadata.get('enable_dimension_reduction'):
                    orig_dim = clean_results["fine_tuned"].get("original_dimension", 0) 
                    target_dim = clean_results["fine_tuned"].get("target_dimension", 0)
                    
                    if orig_dim > 0 and target_dim > 0:
                        dim_reduction = ((orig_dim - target_dim) / orig_dim) * 100
                        summary.append(f"This improvement was achieved while reducing the embedding dimension by {dim_reduction:.1f}%, from {orig_dim}d to {target_dim}d. ")
                
                # Add context for improvement
                if rel_imp > 50:
                    summary.append("This represents a substantial improvement in retrieval quality for domain-specific queries. ")
                elif rel_imp > 20:
                    summary.append("This represents a significant improvement in retrieval quality for domain-specific queries. ")
                elif rel_imp > 5:
                    summary.append("This represents a modest improvement in retrieval quality for domain-specific queries. ")
                else:
                    summary.append("This represents a slight improvement in retrieval quality for domain-specific queries. ")
            elif ft_mrr < orig_mrr:
                summary.append("The fine-tuned model did not show improvement over the original model. Further fine-tuning or different training approaches may be needed. ")
            else:
                summary.append("The fine-tuned model performed similarly to the original model. ")
            
            # Add specific observations
            for k in [1, 5]:
                if k in metadata.get('k_values', [1, 3, 5, 10, 20]):
                    orig_p = clean_results["original"].get(f"precision@{k}", 0)
                    ft_p = clean_results["fine_tuned"].get(f"precision@{k}", 0)
                    
                    if ft_p > orig_p:
                        summary.append(f"Precision@{k} improved, indicating better retrieval accuracy for the top {k} results. ")
        
        return "".join(summary)


def run_benchmarking(config: BenchmarkConfig) -> Tuple[Dict, str]:
    """
    Run the benchmarking process with provided configuration
    
    Args:
        config: Benchmark configuration
    
    Returns:
        Tuple of (benchmark_results, summary)
    """
    # Initialize benchmarker
    benchmarker = EmbeddingBenchmarker(config)
    
    # Run benchmark
    results = benchmarker.run_benchmark()
    
    # Generate summary
    summary = benchmarker.get_summary(results)
    
    # Save summary
    summary_path = os.path.join(
        config.output_dir, 
        f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    logger.info(f"Benchmark summary saved to {summary_path}")
    
    return results, summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark embedding models for domain-specific retrieval")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to the test dataset JSON file")
    parser.add_argument("--original_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Original model name")
    parser.add_argument("--fine_tuned_model", type=str, default=None, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of evaluation queries")
    parser.add_argument("--target_dim", type=int, default=512, help="Target dimension size (e.g., 384, 512)")
    parser.add_argument("--reduction_method", type=str, default="pca", choices=["pca", "svd", "random_projection"], help="Dimension reduction method")
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        original_model_name=args.original_model,
        fine_tuned_model_path=args.fine_tuned_model,
        output_dir=args.output_dir,
        test_dataset_path=args.test_dataset,
        num_evaluation_queries=args.num_queries,
        enable_dimension_reduction=True,
        target_dimension=args.target_dim,
        reduction_method=args.reduction_method
    )
    
    results, summary = run_benchmarking(config)
    
    print("\nBenchmark Summary:")
    print("=" * 50)
    print(summary[:500] + "..." if len(summary) > 500 else summary)
    print("=" * 50)
    print(f"Full results and summary available in {args.output_dir}")