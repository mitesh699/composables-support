import os
import json
import logging
import asyncio
import numpy as np
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ragas_integration')

class RagasEvaluator:
    """
    Enhanced RAGAS integration for evaluating RAG pipeline performance
    
    This class supports multiple RAGAS metrics including:
    - Faithfulness
    - Response Relevancy
    - Context Precision
    - Context Recall
    - Context Entities Recall
    - Noise Sensitivity
    """
    
    def __init__(self, llm_name: str = "groq", use_llm_based: bool = True):
        """
        Initialize the RAGAS evaluator
        
        Args:
            llm_name: Name of the LLM to use (default: "groq")
            use_llm_based: Whether to use LLM-based metrics (default: True)
        """
        self.llm_name = llm_name
        self.use_llm_based = use_llm_based
        self.ragas_available = self._check_ragas_available()
        
        # Will be initialized when needed
        self.llm = None
        self.evaluator_llm = None
        self.evaluator_embeddings = None
    
    def _check_ragas_available(self) -> bool:
        """Check if RAGAS is available"""
        try:
            # Try to import RAGAS metrics
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                NoiseSensitivity
            )
            logger.info("RAGAS successfully imported")
            return True
        except ImportError as e:
            logger.error(f"RAGAS not available. Error: {e}")
            logger.error("Install with: pip install ragas datasets -U")
            return False
    
    def _initialize_llm(self):
        """Initialize LLM for evaluation"""
        if self.llm is not None:
            return
        
        # Get API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize LLM
        if self.llm_name == "groq":
            try:
                from llama_index.llms.groq import Groq
                self.llm = Groq(model="llama3-70b-8192", api_key=api_key)
                logger.info("Initialized Groq LLM for evaluation")
            except Exception as e:
                logger.error(f"Error initializing Groq LLM: {e}")
                raise
        else:
            raise ValueError(f"Unsupported LLM: {self.llm_name}")
    
    def _initialize_evaluator_components(self):
        """Initialize components needed for RAGAS evaluation"""
        if not self.ragas_available:
            raise ImportError("RAGAS not available")
        
        if self.evaluator_llm is not None:
            return
        
        try:
            # Initialize LLM if not already done
            self._initialize_llm()
            
            # Initialize LLM wrapper for RAGAS
            from ragas.llms import LlamaIndexLLMWrapper
            self.evaluator_llm = LlamaIndexLLMWrapper(llm=self.llm)
            
            # Initialize embeddings for metrics that need them
            from ragas.embeddings import LlamaIndexEmbeddings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.evaluator_embeddings = LlamaIndexEmbeddings(embed_model=embed_model)
            
            logger.info("Initialized RAGAS evaluation components")
        except Exception as e:
            logger.error(f"Error initializing RAGAS components: {e}")
            raise
    
    def prepare_single_turn_samples(self, qa_pairs: List[Dict], retrieved_contexts: Optional[List[List[str]]] = None):
        """
        Prepare RAGAS SingleTurnSample objects from QA pairs
        
        Args:
            qa_pairs: List of QA pair dictionaries
            retrieved_contexts: Optional list of retrieved contexts for each QA pair
            
        Returns:
            List of SingleTurnSample objects for RAGAS evaluation
        """
        if not self.ragas_available:
            raise ImportError("RAGAS not available")
        
        try:
            from ragas import SingleTurnSample
            
            samples = []
            for i, qa_pair in enumerate(qa_pairs):
                # Get basic information
                question = qa_pair.get('question', '')
                answer = qa_pair.get('answer', '')
                
                # Get retrieved contexts for this QA pair
                contexts = []
                if retrieved_contexts and i < len(retrieved_contexts):
                    contexts = retrieved_contexts[i]
                
                # Create sample
                sample = SingleTurnSample(
                    user_input=question,
                    response=answer,
                    reference=answer,  # Using answer as reference
                    retrieved_contexts=contexts
                )
                
                samples.append(sample)
            
            logger.info(f"Prepared {len(samples)} samples for RAGAS evaluation")
            return samples
        except Exception as e:
            logger.error(f"Error preparing RAGAS samples: {e}")
            raise
    
    async def evaluate_faithfulness(self, samples):
        """Evaluate faithfulness of the generated answers"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            if self.use_llm_based:
                from ragas.metrics import Faithfulness
                metric = Faithfulness(llm=self.evaluator_llm)
            else:
                from ragas.metrics import FaithfulnesswithHHEM
                metric = FaithfulnesswithHHEM()
            
            # Evaluate samples
            scores = []
            for sample in samples:
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating faithfulness for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "faithfulness": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating faithfulness: {e}")
            return {"error": str(e)}
    
    async def evaluate_response_relevancy(self, samples):
        """Evaluate how relevant the answers are to the questions"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            from ragas.metrics import ResponseRelevancy
            metric = ResponseRelevancy(
                llm=self.evaluator_llm,
                embeddings=self.evaluator_embeddings
            )
            
            # Evaluate samples
            scores = []
            for sample in samples:
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating response relevancy for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "response_relevancy": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating response relevancy: {e}")
            return {"error": str(e)}
    
    async def evaluate_context_precision(self, samples):
        """Evaluate precision of the retrieved contexts"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            if self.use_llm_based:
                from ragas.metrics import LLMContextPrecisionWithoutReference
                metric = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
            else:
                from ragas.metrics import NonLLMContextPrecisionWithReference
                metric = NonLLMContextPrecisionWithReference()
            
            # Evaluate samples
            scores = []
            for sample in samples:
                # Skip samples without contexts
                if not sample.retrieved_contexts:
                    scores.append(None)
                    continue
                
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating context precision for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "context_precision": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating context precision: {e}")
            return {"error": str(e)}
    
    async def evaluate_context_recall(self, samples):
        """Evaluate recall of the retrieved contexts"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            if self.use_llm_based:
                from ragas.metrics import LLMContextRecall
                metric = LLMContextRecall(llm=self.evaluator_llm)
            else:
                from ragas.metrics import NonLLMContextRecall
                metric = NonLLMContextRecall()
            
            # Evaluate samples
            scores = []
            for sample in samples:
                # Skip samples without contexts
                if not sample.retrieved_contexts:
                    scores.append(None)
                    continue
                
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating context recall for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "context_recall": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating context recall: {e}")
            return {"error": str(e)}
    
    async def evaluate_noise_sensitivity(self, samples):
        """Evaluate noise sensitivity of the generated answers"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            from ragas.metrics import NoiseSensitivity
            metric = NoiseSensitivity(llm=self.evaluator_llm)
            
            # Evaluate samples
            scores = []
            for sample in samples:
                # Skip samples without contexts
                if not sample.retrieved_contexts:
                    scores.append(None)
                    continue
                
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating noise sensitivity for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "noise_sensitivity": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating noise sensitivity: {e}")
            return {"error": str(e)}
    
    async def evaluate_context_entities_recall(self, samples):
        """Evaluate entity recall in the retrieved contexts"""
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Initialize components
            self._initialize_evaluator_components()
            
            # Import metrics
            from ragas.metrics import ContextEntityRecall
            metric = ContextEntityRecall(llm=self.evaluator_llm)
            
            # Evaluate samples
            scores = []
            for sample in samples:
                # Skip samples without contexts
                if not sample.retrieved_contexts:
                    scores.append(None)
                    continue
                
                try:
                    score = await metric.single_turn_ascore(sample)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Error evaluating context entity recall for a sample: {e}")
                    scores.append(None)
            
            # Calculate mean score
            valid_scores = [s for s in scores if s is not None]
            mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            return {
                "context_entity_recall": mean_score,
                "sample_scores": scores,
                "valid_samples": len(valid_scores),
                "total_samples": len(samples)
            }
        except Exception as e:
            logger.error(f"Error evaluating context entity recall: {e}")
            return {"error": str(e)}
    
    async def evaluate_all_metrics(self, qa_pairs: List[Dict], retrieved_contexts: Optional[List[List[str]]] = None):
        """
        Evaluate all RAGAS metrics
        
        Args:
            qa_pairs: List of QA pair dictionaries
            retrieved_contexts: Optional list of retrieved contexts for each QA pair
            
        Returns:
            Dictionary of evaluation results
        """
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        try:
            # Prepare samples
            samples = self.prepare_single_turn_samples(qa_pairs, retrieved_contexts)
            
            # Evaluate metrics
            results = {}
            
            # Faithfulness
            logger.info("Evaluating faithfulness...")
            faithfulness_results = await self.evaluate_faithfulness(samples)
            results["faithfulness"] = faithfulness_results
            
            # Response Relevancy
            logger.info("Evaluating response relevancy...")
            relevancy_results = await self.evaluate_response_relevancy(samples)
            results["response_relevancy"] = relevancy_results
            
            # Context metrics (if contexts are available)
            if retrieved_contexts:
                # Context Precision
                logger.info("Evaluating context precision...")
                precision_results = await self.evaluate_context_precision(samples)
                results["context_precision"] = precision_results
                
                # Context Recall
                logger.info("Evaluating context recall...")
                recall_results = await self.evaluate_context_recall(samples)
                results["context_recall"] = recall_results
                
                # Context Entity Recall
                logger.info("Evaluating context entity recall...")
                entity_recall_results = await self.evaluate_context_entities_recall(samples)
                results["context_entity_recall"] = entity_recall_results
                
                # Noise Sensitivity
                logger.info("Evaluating noise sensitivity...")
                noise_results = await self.evaluate_noise_sensitivity(samples)
                results["noise_sensitivity"] = noise_results
            
            # Add summary of results
            summary = {
                "faithfulness": faithfulness_results.get("faithfulness", 0.0),
                "response_relevancy": relevancy_results.get("response_relevancy", 0.0)
            }
            
            if retrieved_contexts:
                summary["context_precision"] = precision_results.get("context_precision", 0.0)
                summary["context_recall"] = recall_results.get("context_recall", 0.0)
                summary["context_entity_recall"] = entity_recall_results.get("context_entity_recall", 0.0)
                summary["noise_sensitivity"] = noise_results.get("noise_sensitivity", 0.0)
            
            results["summary"] = summary
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating RAGAS metrics: {e}")
            return {"error": str(e)}


# Function to run asyncio event loop in a thread-safe way
def run_async_function(async_func, *args, **kwargs):
    """
    Run an async function in a synchronous context
    
    Args:
        async_func: The async function to run
        *args, **kwargs: Arguments to pass to the async function
        
    Returns:
        The result of the async function
    """
    # Ensure we have an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in this thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Run the async function
    return loop.run_until_complete(async_func(*args, **kwargs))


# Function to integrate with the benchmarking module
def run_ragas_evaluation(qa_pairs, retrieved_contexts=None, llm_name="groq", use_llm_based=True):
    """
    Run RAGAS evaluation on QA pairs
    
    Args:
        qa_pairs: List of QA pair dictionaries
        retrieved_contexts: Optional list of retrieved contexts for each QA pair
        llm_name: Name of the LLM to use (default: "groq")
        use_llm_based: Whether to use LLM-based metrics (default: True)
        
    Returns:
        Dictionary of evaluation results
    """
    try:
        evaluator = RagasEvaluator(llm_name=llm_name, use_llm_based=use_llm_based)
        return run_async_function(evaluator.evaluate_all_metrics, qa_pairs, retrieved_contexts)
    except Exception as e:
        logger.error(f"Error running RAGAS evaluation: {e}")
        return {"error": str(e)}


# Example standalone usage
if __name__ == "__main__":
    # Sample data
    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris."
        }
    ]
    
    # Sample retrieved contexts
    contexts = [
        [
            "Paris is the capital and most populous city of France.",
            "France is a country located in Western Europe."
        ]
    ]
    
    # Run evaluation
    results = run_ragas_evaluation(qa_pairs, contexts)
    print(json.dumps(results, indent=2))