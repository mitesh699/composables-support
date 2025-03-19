import os
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AdamW
)
from sentence_transformers import SentenceTransformer, InputExample, losses
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding_finetuner')

@dataclass
class FineTuningConfig:
    """Configuration for embedding model fine-tuning"""
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Base model to fine-tune
    output_dir: str = "fine_tuned_models"  # Directory to save fine-tuned models
    learning_rate: float = 2e-5  # Learning rate for training
    train_batch_size: int = 16  # Batch size for training
    epochs: int = 3  # Number of training epochs
    max_seq_length: int = 256  # Maximum sequence length for tokenization
    use_amp: bool = True  # Use Automatic Mixed Precision for faster training
    evaluation_steps: int = 100  # Steps between evaluations
    save_steps: int = 500  # Steps between model checkpoint saves
    warmup_steps: int = 100  # Warmup steps for learning rate scheduler
    seed: int = 42  # Random seed for reproducibility


class FinancialDataset(Dataset):
    """Dataset class for financial question-answer pairs"""
    
    def __init__(self, examples: List[InputExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize texts
        encoding1 = self.tokenizer(
            example.texts[0], 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            example.texts[1], 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension
        return {
            'input_ids1': encoding1['input_ids'].squeeze(),
            'attention_mask1': encoding1['attention_mask'].squeeze(),
            'input_ids2': encoding2['input_ids'].squeeze(),
            'attention_mask2': encoding2['attention_mask'].squeeze(),
            'label': torch.tensor(example.label, dtype=torch.float)
        }


class EmbeddingFineTuner:
    """
    Fine-tunes embedding models on domain-specific data using contrastive learning
    """
    
    def __init__(self, config: Optional[FineTuningConfig] = None):
        """Initialize the fine-tuner with configuration"""
        self.config = config or FineTuningConfig()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set random seed for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        logger.info(f"Initializing EmbeddingFineTuner with model: {self.config.base_model_name}")
    
    def prepare_dataset_from_qa(self, 
                               qa_pairs: List[Dict], 
                               similarity_threshold: float = 0.7, 
                               negative_sampling_ratio: float = 1.0) -> Tuple[List[InputExample], Dict]:
        """
        Prepare a dataset for contrastive learning from question-answer pairs
        
        Args:
            qa_pairs: List of question-answer dictionaries
            similarity_threshold: Threshold to consider questions similar (for subtopic grouping)
            negative_sampling_ratio: Ratio of negative examples to positive examples
            
        Returns:
            List of InputExamples for sentence-transformers training
            Dictionary with dataset statistics
        """
        logger.info(f"Preparing dataset from {len(qa_pairs)} QA pairs")
        
        # Group questions by subtopic (financial domain categories)
        subtopic_groups = {}
        for item in qa_pairs:
            subtopic = item.get('subtopic', 'general_finance')
            if subtopic not in subtopic_groups:
                subtopic_groups[subtopic] = []
            subtopic_groups[subtopic].append(item)
        
        # Create positive pairs (questions from same subtopic with similar complexity)
        positive_examples = []
        
        # Create positive pairs from question-answer relationships
        qa_positive_examples = []
        for item in qa_pairs:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question and answer:
                # Direct question-answer pair (label 1.0 for high similarity)
                qa_positive_examples.append(InputExample(texts=[question, answer], label=1.0))
        
        # Create additional positive pairs from same subtopic questions
        subtopic_positive_examples = []
        for subtopic, items in subtopic_groups.items():
            if len(items) < 2:
                continue
                
            # Group by complexity within subtopic
            for i in range(len(items)):
                for j in range(i+1, min(i+5, len(items))):
                    item_i = items[i]
                    item_j = items[j]
                    
                    # Questions in same subtopic and similar complexity are positive pairs
                    # with slightly lower similarity score
                    if item_i.get('complexity') == item_j.get('complexity'):
                        similarity = 0.8  # Higher similarity for same complexity
                    else:
                        similarity = 0.6  # Lower similarity for different complexity
                        
                    subtopic_positive_examples.append(
                        InputExample(
                            texts=[item_i.get('question', ''), item_j.get('question', '')],
                            label=similarity
                        )
                    )
        
        # Combine the different types of positive examples
        positive_examples = qa_positive_examples + subtopic_positive_examples
        
        # Create negative pairs (questions from different subtopics)
        negative_examples = []
        subtopics = list(subtopic_groups.keys())
        
        # Determine how many negative examples to generate
        num_negative_examples = min(
            int(len(positive_examples) * negative_sampling_ratio),
            10000  # Cap to avoid generating too many negative examples
        )
        
        # Generate negative examples
        negative_pairs_seen = set()
        for _ in range(num_negative_examples):
            if len(subtopics) < 2:
                break
                
            # Select two different subtopics
            subtopic_i, subtopic_j = np.random.choice(subtopics, size=2, replace=False)
            
            # Select random items from each subtopic
            if subtopic_groups[subtopic_i] and subtopic_groups[subtopic_j]:
                item_i = np.random.choice(subtopic_groups[subtopic_i])
                item_j = np.random.choice(subtopic_groups[subtopic_j])
                
                # Create a unique key for this pair
                pair_key = (item_i.get('question', ''), item_j.get('question', ''))
                if pair_key in negative_pairs_seen:
                    continue
                    
                negative_pairs_seen.add(pair_key)
                
                # Different subtopics are negative pairs (low similarity)
                negative_examples.append(
                    InputExample(
                        texts=[item_i.get('question', ''), item_j.get('question', '')],
                        label=0.1  # Low similarity score for negative pairs
                    )
                )
        
        # Combine positive and negative examples
        all_examples = positive_examples + negative_examples
        
        # Shuffle the examples
        np.random.shuffle(all_examples)
        
        # Return examples and statistics
        stats = {
            "total_examples": len(all_examples),
            "qa_positive_examples": len(qa_positive_examples),
            "subtopic_positive_examples": len(subtopic_positive_examples),
            "negative_examples": len(negative_examples),
            "subtopics": list(subtopic_groups.keys())
        }
        
        logger.info(f"Created dataset with {stats['total_examples']} examples: "
                   f"{stats['qa_positive_examples']} QA pairs, "
                   f"{stats['subtopic_positive_examples']} subtopic pairs, "
                   f"{stats['negative_examples']} negative pairs")
        
        return all_examples, stats
    
    def create_triplet_dataset(self, qa_pairs: List[Dict]) -> Tuple[List[InputExample], Dict]:
        """
        Create triplet examples (anchor, positive, negative) for triplet loss training
        
        Args:
            qa_pairs: List of question-answer dictionaries
            
        Returns:
            List of InputExamples with triplets
            Dictionary with dataset statistics
        """
        logger.info(f"Creating triplet dataset from {len(qa_pairs)} QA pairs")
        
        # Group questions by subtopic
        subtopic_groups = {}
        for item in qa_pairs:
            subtopic = item.get('subtopic', 'general_finance')
            if subtopic not in subtopic_groups:
                subtopic_groups[subtopic] = []
            subtopic_groups[subtopic].append(item)
        
        triplet_examples = []
        
        # Create triplets: (question, same_subtopic_question, different_subtopic_question)
        for subtopic, items in subtopic_groups.items():
            if len(items) < 2 or len(subtopic_groups) < 2:
                continue
                
            other_subtopics = [s for s in subtopic_groups.keys() if s != subtopic]
            
            for anchor_item in items:
                # Skip if no question
                if not anchor_item.get('question'):
                    continue
                    
                anchor = anchor_item.get('question')
                
                # Find a positive example (same subtopic, different question)
                positive_candidates = [item for item in items if item != anchor_item]
                if not positive_candidates:
                    continue
                    
                positive_item = np.random.choice(positive_candidates)
                positive = positive_item.get('question')
                
                # Find a negative example (different subtopic)
                negative_subtopic = np.random.choice(other_subtopics)
                negative_candidates = subtopic_groups[negative_subtopic]
                if not negative_candidates:
                    continue
                    
                negative_item = np.random.choice(negative_candidates)
                negative = negative_item.get('question')
                
                # Create triplet example
                triplet_examples.append(
                    InputExample(texts=[anchor, positive, negative])
                )
        
        # Also create triplets for question-answer pairs
        for item in qa_pairs:
            question = item.get('question')
            answer = item.get('answer')
            
            if not question or not answer:
                continue
                
            # Find a negative example (different question's answer)
            negative_candidates = [qa for qa in qa_pairs if qa != item]
            if not negative_candidates:
                continue
                
            negative_item = np.random.choice(negative_candidates)
            negative = negative_item.get('answer')
            
            # Create triplet example
            triplet_examples.append(
                InputExample(texts=[question, answer, negative])
            )
        
        # Shuffle the examples
        np.random.shuffle(triplet_examples)
        
        # Return examples and statistics
        stats = {
            "total_triplets": len(triplet_examples),
            "subtopics": list(subtopic_groups.keys())
        }
        
        logger.info(f"Created triplet dataset with {stats['total_triplets']} examples")
        
        return triplet_examples, stats
    
    def fine_tune(self, 
                 examples: List[InputExample], 
                 training_type: str = "contrastive",
                 evaluation_examples: Optional[List[InputExample]] = None) -> str:
        """
        Fine-tune the embedding model using the prepared examples
        
        Args:
            examples: List of InputExamples for training
            training_type: Type of training ('contrastive', 'triplet', 'cosine')
            evaluation_examples: Optional separate evaluation examples
            
        Returns:
            Path to the saved fine-tuned model
        """
        if not examples:
            raise ValueError("No examples provided for fine-tuning")
            
        logger.info(f"Fine-tuning {self.config.base_model_name} with {len(examples)} examples")
        
        # Create unique model ID for this fine-tuning run
        model_id = f"ft_{os.path.basename(self.config.base_model_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = os.path.join(self.config.output_dir, model_id)
        
        # Initialize model from sentence-transformers
        model = SentenceTransformer(self.config.base_model_name)
        
        # Split examples into train and eval sets if evaluation_examples not provided
        if evaluation_examples is None:
            train_examples, eval_examples = train_test_split(
                examples, test_size=0.1, random_state=self.config.seed
            )
        else:
            train_examples = examples
            eval_examples = evaluation_examples
        
        logger.info(f"Training with {len(train_examples)} examples, evaluating with {len(eval_examples)} examples")
        
        # Configure training based on training type
        train_dataloader = None
        train_loss = None
        
        if training_type == "contrastive":
            # Contrastive loss for similarity learning
            train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=self.config.train_batch_size
            )
            train_loss = losses.CosineSimilarityLoss(model)
            
        elif training_type == "triplet":
            # Triplet loss for learning embeddings that keep similar items close and dissimilar items far
            train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=self.config.train_batch_size
            )
            train_loss = losses.TripletLoss(model, triplet_margin=1)
            
        elif training_type == "cosine":
            # Multiple Negatives Ranking Loss - efficient softmax loss
            train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=self.config.train_batch_size
            )
            train_loss = losses.CosineSimilarityLoss(model)
            
        else:
            raise ValueError(f"Unsupported training type: {training_type}")
        
        # Prepare evaluation examples
        evaluator = None
        if eval_examples:
            # Use cosine similarity evaluator
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
            
            # Filter examples with labels for evaluation
            labeled_eval_examples = [ex for ex in eval_examples if hasattr(ex, 'label')]
            
            if labeled_eval_examples:
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
                    labeled_eval_examples, 
                    name='eval'
                )
        
        # Train the model
        logger.info(f"Starting training with {training_type} loss")
        
        warmup_steps = int(len(train_dataloader) * self.config.epochs * 0.1) if self.config.warmup_steps <= 0 else self.config.warmup_steps
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.config.epochs,
            evaluation_steps=self.config.evaluation_steps,
            warmup_steps=warmup_steps,
            output_path=output_path,
            save_best_model=True,
            show_progress_bar=True
        )
        
        logger.info(f"Fine-tuning complete. Model saved to {output_path}")
        return output_path
    
    def load_from_json(self, file_path: str) -> List[Dict]:
        """Load datasets from JSON file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if data is in expected format
            if 'data' in data and isinstance(data['data'], list):
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Unexpected data format in {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise


# Add this function to your embedding_finetuner.py file
# Make sure to place it after the EmbeddingFineTuner class definition but before the __main__ block

def run_finetuning(
    dataset_path: str,
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    training_type: str = "contrastive",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    output_dir: str = "fine_tuned_models",
    target_dimension: int = None  # Added target_dimension parameter
):
    """
    Run the fine-tuning process from a dataset file
    
    Args:
        dataset_path: Path to the JSON dataset file
        base_model_name: Name of the base model to fine-tune
        training_type: Type of training ('contrastive', 'triplet', 'cosine')
        learning_rate: Learning rate for training
        batch_size: Batch size for training
        epochs: Number of training epochs
        output_dir: Directory to save fine-tuned models
        target_dimension: Target dimension for embeddings (optional)
    
    Returns:
        Path to the saved fine-tuned model
    """
    # Configure fine-tuning
    config = FineTuningConfig(
        base_model_name=base_model_name,
        output_dir=output_dir,
        learning_rate=learning_rate,
        train_batch_size=batch_size,
        epochs=epochs
    )
    
    # Initialize fine-tuner
    fine_tuner = EmbeddingFineTuner(config)
    
    # Load dataset
    qa_pairs = fine_tuner.load_from_json(dataset_path)
    
    # Prepare dataset based on training type
    if training_type == "triplet":
        examples, stats = fine_tuner.create_triplet_dataset(qa_pairs)
    else:
        examples, stats = fine_tuner.prepare_dataset_from_qa(qa_pairs)
    
    # Save dataset statistics
    stats_path = os.path.join(output_dir, f"dataset_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    # Fine-tune the model
    model_path = fine_tuner.fine_tune(examples, training_type=training_type)
    
    # Note: target_dimension is received but not currently used in the fine-tuning process
    # This would require additional implementation to support dimension reduction during fine-tuning
    if target_dimension:
        logger.info(f"Target dimension ({target_dimension}) noted for embedding model, but dimension reduction requires post-processing")
    
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune embedding models on financial domain data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Base model to fine-tune")
    parser.add_argument("--training_type", type=str, default="contrastive", choices=["contrastive", "triplet", "cosine"], help="Training approach")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default="fine_tuned_models", help="Output directory")
    
    args = parser.parse_args()
    
    model_path = run_finetuning(
        dataset_path=args.dataset,
        base_model_name=args.model,
        training_type=args.training_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir
    )
    
    print(f"Fine-tuned model saved to: {model_path}")