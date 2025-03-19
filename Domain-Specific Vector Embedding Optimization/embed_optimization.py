import os
import json
import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import platform
import asyncio
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embed_optimization')

file_handler = logging.FileHandler('pipeline_execution.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Setup asyncio event loop for Windows compatibility
def setup_asyncio_event_loop():
    """Set up the asyncio event loop based on platform"""
    if platform.system() == 'Windows':
        # On Windows, use the ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        # Enable ANSI color support on Windows
        os.system('color')
        
        logger.info("Set up Windows-compatible asyncio event loop")
    else:
        logger.info(f"Using default asyncio event loop for {platform.system()}")
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Domain-Specific Vector Embedding Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset generation arguments
    parser.add_argument("--pdf_dir", type=str, default="finance_pdfs",
                        help="Directory containing domain-specific PDFs")
    parser.add_argument("--urls", type=str, default=None,
                        help="Optional comma-separated list of URLs with domain content")
    parser.add_argument("--num_questions", type=int, default=5000,
                        help="Number of questions to generate")
    parser.add_argument("--dataset_output", type=str, default="financial_dataset.json",
                        help="Output file for the generated dataset")
    
    # Fine-tuning arguments
    parser.add_argument("--original_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Base embedding model to fine-tune")
    parser.add_argument("--fine_tune", action="store_true",
                        help="Fine-tune the embedding model")
    parser.add_argument("--training_type", type=str, default="contrastive", 
                        choices=["contrastive", "triplet", "cosine"],
                        help="Type of fine-tuning to perform")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training")
    
    # Benchmarking arguments
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark the fine-tuned model against the original")
    parser.add_argument("--test_dataset", type=str, default=None,
                        help="Test dataset for benchmarking (if different from generated)")
    parser.add_argument("--num_eval_queries", type=int, default=100,
                        help="Number of queries to use for evaluation")
    parser.add_argument("--benchmark_output", type=str, default="benchmark_results",
                        help="Output directory for benchmark results")
    
    # Pipeline control arguments
    parser.add_argument("--skip_dataset_generation", action="store_true",
                        help="Skip dataset generation (use existing dataset)")
    parser.add_argument("--existing_dataset", type=str, default=None,
                        help="Path to existing dataset (if skipping generation)")
    parser.add_argument("--existing_finetuned_model", type=str, default=None,
                        help="Path to existing fine-tuned model (if skipping fine-tuning)")
    parser.add_argument("--chunk_size", type=int, default=512,
                        help="Chunk size for document processing")
    parser.add_argument("--chunk_overlap", type=int, default=100,
                        help="Chunk overlap for document processing")
    parser.add_argument("--chunk_method", type=str, default="hybrid", 
                        choices=["hybrid", "sentence", "paragraph", "fixed"],
                        help="Method to use for document chunking")
    parser.add_argument("--llm_model", type=str, default="llama3-70b-8192",
                        help="LLM model to use for generation")
    
    # API key rotation arguments
    parser.add_argument("--api_keys_file", type=str, default=None,
                        help="JSON file containing API keys for rotation")
    parser.add_argument("--api_key_env_prefix", type=str, default="GROQ_API_KEY",
                        help="Environment variable prefix for API keys (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)")
    
    # Embedding dimension arguments - Setting fixed to 512
    parser.add_argument("--embedding_dim", type=int, default=512, 
                        help="Embedding dimension to use (fixed at 512)")
    
    # Answer evaluation arguments
    parser.add_argument("--evaluate_answers", action="store_true",
                        help="Evaluate answer quality before/after optimization")
    parser.add_argument("--answers_before", type=str, default=None,
                        help="JSON file with answers generated before optimization")
    parser.add_argument("--answers_after", type=str, default=None, 
                        help="JSON file with answers generated after optimization")
    parser.add_argument("--max_eval_samples", type=int, default=100,
                        help="Maximum number of answers to evaluate")
    parser.add_argument("--use_llm_judge", action="store_true", 
                        help="Use LLM as a judge for answer evaluation")
    parser.add_argument("--judge_model", type=str, default="llama3-8b-8192",
                        help="LLM model to use for judging answers (lightweight)")
    
    # Dashboard generation arguments
    parser.add_argument("--generate_dashboard", action="store_true",
                        help="Generate HTML dashboard with results")
    parser.add_argument("--dashboard_title", type=str, default="Domain-Specific Embedding Optimization Results",
                        help="Title for the dashboard")
    parser.add_argument("--dashboard_output", type=str, default="dashboard",
                        help="Output directory for the dashboard")
    
    # RAGAS evaluation arguments
    parser.add_argument("--use_ragas", action="store_true",
                        help="Use RAGAS for additional evaluation")
    parser.add_argument("--no_llm_ragas", action="store_true",
                        help="Use non-LLM based metrics for RAGAS (faster)")
    
    # Chroma DB arguments
    parser.add_argument("--chroma_dir", type=str, default="./chroma_db",
                        help="Directory for Chrome vector database")
    parser.add_argument("--clear_chroma", action="store_true",
                        help="Clear existing Chroma database before starting")
    parser.add_argument("--no_clear_chroma", action="store_false", dest="clear_chroma",
                        help="Don't clear existing Chroma database (for incremental runs)")
    
    # Add new argument for combined before/after answers
    parser.add_argument("--enable_combined_answers", action="store_true",
                        help="Generate combined before/after answers in a single file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Force embedding_dim to be 512 regardless of what was provided
    args.embedding_dim = 512
    
    # Disable dimension reduction completely
    args.enable_dimension_reduction = False
    
    # Set default for clear_chroma
    if not hasattr(args, 'clear_chroma'):
        args.clear_chroma = True
    
    return args


def setup_api_key_manager(args):
    """Set up API key manager for key rotation"""
    try:
        from api_key_manager import APIKeyManager
        from dotenv import load_dotenv
    except ImportError:
        logger.error("Missing required packages. Please install with: pip install python-dotenv")
        raise
    
    import os
    
    # Explicitly load environment variables from .env file
    dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path)
    
    # Debug: Print environment variables to confirm they're loaded
    logger.info("Checking Groq API keys in environment...")
    found_keys = False
    for i in range(1, 5):
        key_name = f"GROQ_API_KEY_{i}"
        if key_name in os.environ and os.environ[key_name].strip():
            logger.info(f"Found {key_name}")
            found_keys = True
    if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"].strip():
        logger.info("Found GROQ_API_KEY")
        found_keys = True
    
    if not found_keys:
        logger.warning("No Groq API keys found in environment variables!")
        logger.warning("Please check your .env file format and location.")
    
    # Initialize API key manager
    key_manager = APIKeyManager(
        api_keys=None,  # Will load from environment
        status_file="api_key_status.json",
        cooldown_minutes=60
    )
    
    # Check if we have active keys
    active_key = key_manager.get_active_key()
    if not active_key:
        logger.warning("No active API keys available. Set environment variables like GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")
    else:
        logger.info("API key manager initialized successfully")
    
    return key_manager


def generate_dataset(args, api_key_manager):
    """Generate synthetic dataset using RAG with API key rotation"""
    try:
        from document_processor import DocumentProcessor
        from question_generator import QuestionGenerator
        from answer_generator import AnswerGenerator
        from api_key_manager import RotatingAPIRateLimiter
        from dotenv import load_dotenv
    except ImportError:
        logger.error("Missing required packages. Please install dependencies first.")
        raise
    
    # Load environment variables
    load_dotenv()
    
    # Set up rotating rate limiter with key manager
    rate_limiter = RotatingAPIRateLimiter(
        key_manager=api_key_manager,
        requests_per_minute=25,  # Conservative rate limits
        requests_per_day=14400,
        tokens_per_minute=5000,
        tokens_per_day=500000
    )
    
    # Get initial API key (will rotate automatically as needed)
    wait_time, api_key = rate_limiter.wait_if_needed()
    if wait_time > 0:
        logger.info(f"Rate limited, waited {wait_time:.2f}s before starting")
    
    if not api_key:
        raise ValueError("No API key available")
    
    # Process URLs
    urls = []
    if args.urls:
        urls = [url.strip() for url in args.urls.split(",")]
    
    logger.info("=== Domain-Specific Dataset Generation ===")
    
    # Step 1: Process documents with specified chunking method
    logger.info(f"1. Processing documents with {args.chunk_method} chunking...")
    
    # Clear Chroma directory if requested
    if args.clear_chroma and os.path.exists(args.chroma_dir):
        logger.info(f"Clearing existing Chroma database: {args.chroma_dir}")
        shutil.rmtree(args.chroma_dir)
    
    processor = DocumentProcessor(
        pdf_dir=args.pdf_dir,
        url_sources=urls,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_method=args.chunk_method,
        embedding_dim=args.embedding_dim,  # Fixed at 512
        chroma_dir=args.chroma_dir
    )
    
    index, chunked_docs, vector_store = processor.process()
    
    logger.info(f"Processed {len(chunked_docs)} chunks from documents")
    
    # Step 2: Generate diverse questions with key rotation
    logger.info("2. Generating diverse questions...")
    question_generator = QuestionGenerator(
        index=index,
        api_key=api_key,  # Initial key, will be rotated as needed
        llm_model=args.llm_model,
        seed_questions_file="prompt.txt",
        max_tokens_per_question=40  # Use 40 tokens for questions
    )
    
    # Replace rate limiter with our rotating version
    question_generator.rate_limiter = rate_limiter
    
    # Generate questions
    questions = question_generator.generate_questions(args.num_questions)
    
    # Save questions backup
    questions_backup_path = f"{os.path.splitext(args.dataset_output)[0]}_questions.json"
    with open(questions_backup_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(questions)} questions to {questions_backup_path}")
    
    # Step 3: Generate and enhance answers with key rotation
    logger.info("3. Generating and enhancing answers...")
    
    # Get fresh API key (may have rotated during question generation)
    wait_time, api_key = rate_limiter.wait_if_needed()
    
    answer_generator = AnswerGenerator(
        index=index,
        api_key=api_key,
        llm_model=args.llm_model,
        max_tokens=200  # Use 200 tokens per answer
    )
    
    # Replace rate limiter with our rotating version
    answer_generator.rate_limiter = rate_limiter
    
    # Generate answers
    qa_pairs = answer_generator.generate_all_answers(questions)
    
    # Step 4: Save the final dataset
    logger.info("4. Saving the final dataset...")
    
    # If dataset is large, split into chunks
    if args.num_questions > 1000:
        chunk_size = 1000
        num_chunks = (args.num_questions + chunk_size - 1) // chunk_size
        
        dataset_files = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(qa_pairs))
            chunk_pairs = qa_pairs[start_idx:end_idx]
            
            # Create filename with chunk number
            base_name, ext = os.path.splitext(args.dataset_output)
            chunk_file = f"{base_name}_{i+1}{ext}"
            
            answer_generator.save_dataset(chunk_pairs, chunk_file)
            logger.info(f"Saved chunk {i+1}/{num_chunks} with {len(chunk_pairs)} QA pairs to {chunk_file}")
            dataset_files.append(chunk_file)
            
        return dataset_files, index, vector_store, chunked_docs
    else:
        answer_generator.save_dataset(qa_pairs, args.dataset_output)
        logger.info(f"Saved dataset with {len(qa_pairs)} QA pairs to {args.dataset_output}")
        return [args.dataset_output], index, vector_store, chunked_docs


def fine_tune_embeddings(args, dataset_files):
    """Fine-tune embedding model with domain-specific data - keeping at 512 dimensions"""
    logger.info("=== Domain-Specific Embedding Fine-tuning ===")
    
    # Import fine-tuning module
    try:
        from embedding_finetuner import run_finetuning
    except ImportError:
        # Fall back to original module if improved version is not available
        try:
            from embedding_finetuner import run_finetuning
            logger.warning("Using original embedding finetuner. For best results, use the improved version.")
        except ImportError:
            logger.error("Missing embedding_finetuner module! Please ensure it's available.")
            raise
    
    # Use first dataset file for fine-tuning (can be extended to use multiple files)
    dataset_path = dataset_files[0] if isinstance(dataset_files, list) else dataset_files
    
    logger.info(f"Fine-tuning {args.original_model} using dataset: {dataset_path}")
    logger.info(f"Training type: {args.training_type}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    logger.info(f"Using embedding dimension: 512 (fixed)")
    
    # Call fine-tuning with fixed 512 dimensions
    model_path = run_finetuning(
        dataset_path=dataset_path,
        base_model_name=args.original_model,
        training_type=args.training_type,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir="fine_tuned_models",
        target_dimension=512  # Fixed dimension
    )
    
    logger.info(f"Fine-tuning complete. Model saved to: {model_path}")
    return model_path

def benchmark_embeddings(args, dataset_files, fine_tuned_model_path):
    """Benchmark fine-tuned embedding model against original, with fixed 512 dimensions"""
    logger.info("=== Domain-Specific Embedding Benchmarking ===")
    
    # Import benchmarking module
    try:
        from embedding_benchmarker import run_benchmarking, BenchmarkConfig
    except ImportError:
        logger.error("Missing embedding_benchmarker module! Please ensure it's available.")
        raise
    
    # Determine test dataset
    test_dataset = args.test_dataset
    if not test_dataset:
        # Use the first dataset file if dataset_files is a list
        if isinstance(dataset_files, list) and len(dataset_files) > 0:
            test_dataset = dataset_files[0]
        else:
            test_dataset = dataset_files
    
    logger.info(f"Benchmarking using test dataset: {test_dataset}")
    logger.info(f"Comparing original model ({args.original_model}) with fine-tuned model")
    logger.info(f"Using fixed embedding dimension: 512")
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        original_model_name=args.original_model,
        fine_tuned_model_path=fine_tuned_model_path,
        output_dir=args.benchmark_output,
        test_dataset_path=test_dataset,
        num_evaluation_queries=args.num_eval_queries,
        enable_dimension_reduction=False,  # Disable dimension reduction
        target_dimension=512,              # Fixed at 512
        chroma_dir=args.chroma_dir
    )
    
    # Run benchmarking
    results, summary = run_benchmarking(config)
    
    # Print a preview of the summary
    logger.info("\nBenchmark Summary Preview:")
    logger.info("=" * 50)
    logger.info(summary[:500] + "..." if len(summary) > 500 else summary)
    logger.info("=" * 50)
    
    return results, summary



def generate_combined_answers(args, dataset_files, api_key_manager, fine_tuned_model_path):
    """
    Generate answers using both original and fine-tuned models and combine them
    
    Args:
        args: Command line arguments
        dataset_files: Path to dataset files
        api_key_manager: API key manager for key rotation
        fine_tuned_model_path: Path to fine-tuned model
        
    Returns:
        Path to the combined answers file
    """
    from answer_generator import AnswerGenerator
    from document_processor import DocumentProcessor
    from api_key_manager import RotatingAPIRateLimiter
    
    # Set up rotating rate limiter with key manager
    rate_limiter = RotatingAPIRateLimiter(
        key_manager=api_key_manager
    )
    
    # Get initial API key
    wait_time, api_key = rate_limiter.wait_if_needed()
    
    logger.info("=== Generating Combined Before/After Answers ===")
    
    # Extract questions from dataset
    test_dataset = args.test_dataset if args.test_dataset else (dataset_files[0] if isinstance(dataset_files, list) else dataset_files)
    
    logger.info(f"Using test dataset: {test_dataset}")
    
    with open(test_dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        if isinstance(data, dict) and 'data' in data:
            qa_pairs = data['data']
        elif isinstance(data, list):
            qa_pairs = data
        else:
            raise ValueError(f"Unexpected data format in {test_dataset}")
    
    # Extract questions only
    questions = []
    for qa in qa_pairs[:args.max_eval_samples]:
        question = qa.get('question')
        if question:
            questions.append({
                "question": question,
                "subtopic": qa.get('subtopic', 'general_finance'),
                "complexity": qa.get('complexity', 'basic')
            })
    
    logger.info(f"Generating answers for {len(questions)} questions using both models")
    
    # Process with original model and fine-tuned model simultaneously
    # And combine results immediately into one structure
    combined_answers = []
    
    # Check if existing ChromaDB is present 
    original_chroma_dir = f"{args.chroma_dir}_original"
    finetuned_chroma_dir = f"{args.chroma_dir}_finetuned"
    clear_chroma = args.clear_chroma  # Use the command line argument

    # Create the two indexes with different models
    logger.info(f"Processing with original model: {args.original_model}")
    processor_original = DocumentProcessor(
        pdf_dir=args.pdf_dir,
        url_sources=args.urls.split(",") if args.urls else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_method=args.chunk_method,
        embedding_model=args.original_model,
        embedding_dim=512,  # Fixed at 512
        chroma_dir=original_chroma_dir,
        clear_chroma=clear_chroma  # Pass the parameter
    )
    
    logger.info(f"Processing with fine-tuned model: {fine_tuned_model_path}")
    processor_finetuned = DocumentProcessor(
        pdf_dir=args.pdf_dir,
        url_sources=args.urls.split(",") if args.urls else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunk_method=args.chunk_method,
        embedding_model=fine_tuned_model_path,
        embedding_dim=512,  # Fixed at 512
        chroma_dir=finetuned_chroma_dir,
        clear_chroma=clear_chroma  # Pass the parameter
    )
    
    # Process documents and create both indexes
    logger.info("Processing documents with original model...")
    index_original, _, _ = processor_original.process()
    
    logger.info("Processing documents with fine-tuned model...")
    index_finetuned, _, _ = processor_finetuned.process()
    
    # Create answer generators
    answer_generator_original = AnswerGenerator(
        index=index_original,
        api_key=api_key,
        llm_model=args.llm_model,
        max_tokens=200
    )
    answer_generator_original.rate_limiter = rate_limiter
    
    answer_generator_finetuned = AnswerGenerator(
        index=index_finetuned,
        api_key=api_key,
        llm_model=args.llm_model,
        max_tokens=200
    )
    answer_generator_finetuned.rate_limiter = rate_limiter
    
    # Process each question with both models
    for i, question_item in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question_item['question'][:50]}...")
        
        # Generate answer with original model
        original_answer = answer_generator_original.generate_answer(question_item)
        
        # Generate answer with fine-tuned model
        finetuned_answer = answer_generator_finetuned.generate_answer(question_item)
        
        # Combine into a single item
        combined_item = {
            "question": question_item["question"],
            "subtopic": question_item["subtopic"],
            "complexity": question_item["complexity"],
            "answer_original": original_answer.get("answer", ""),
            "answer_finetuned": finetuned_answer.get("answer", ""),
            "token_count_original": original_answer.get("token_count", 0),
            "token_count_finetuned": finetuned_answer.get("token_count", 0)
        }
        
        # Add to results
        combined_answers.append(combined_item)
        
        # Save interim progress
        if (i+1) % 10 == 0:
            interim_file = f"interim_combined_answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "original_model": args.original_model,
                        "fine_tuned_model": fine_tuned_model_path,
                        "generated_at": datetime.now().isoformat(),
                        "embedding_dim": 512,  # Fixed at 512
                        "llm_model": args.llm_model,
                        "status": "in_progress",
                        "completed": i+1,
                        "total": len(questions)
                    },
                    "data": combined_answers
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved interim progress to {interim_file}")
    
    # Save final combined answers
    output_file = f"combined_answers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "metadata": {
                "original_model": args.original_model,
                "fine_tuned_model": fine_tuned_model_path,
                "generated_at": datetime.now().isoformat(),
                "embedding_dim": 512,  # Fixed at 512
                "llm_model": args.llm_model
            },
            "data": combined_answers
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(combined_answers)} combined answers to {output_file}")
    return output_file


def evaluate_combined_answers(args, combined_file, api_key_manager):
    """
    Evaluate answer quality from combined answers file with enhanced metrics
    
    Args:
        args: Command line arguments
        combined_file: Path to the combined answers file
        api_key_manager: API key manager for key rotation
        
    Returns:
        Evaluation results dictionary with additional metrics
    """
    logger.info("=== Answer Quality Evaluation ===")
    
    # Import evaluation module
    try:
        from answer_evaluator import HybridAnswerEvaluator
    except ImportError:
        logger.error("Missing answer_evaluator module! Please ensure it's available.")
        raise
    
    # Load combined answers
    with open(combined_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        combined_answers = data.get('data', [])
    
    logger.info(f"Loaded {len(combined_answers)} combined answers for evaluation")
    
    # Convert combined answers to separate before/after lists
    qa_before = []
    qa_after = []
    
    for item in combined_answers:
        # Create before item
        before_item = {
            "question": item.get('question', ''),
            "answer": item.get('answer_original', ''),
            "subtopic": item.get('subtopic', 'general_finance')
        }
        qa_before.append(before_item)
        
        # Create after item
        after_item = {
            "question": item.get('question', ''),
            "answer": item.get('answer_finetuned', ''),
            "subtopic": item.get('subtopic', 'general_finance')
        }
        qa_after.append(after_item)
    
    # Initialize evaluator
    evaluator = HybridAnswerEvaluator(
        api_key_manager=api_key_manager,
        judge_model=args.judge_model,
        output_dir="evaluation_results",
        use_llm=args.use_llm_judge
    )
    
    logger.info(f"Using LLM judge: {args.use_llm_judge}")
    if args.use_llm_judge:
        logger.info(f"Judge model: {args.judge_model}")
    
    # Run evaluation
    logger.info("Evaluating answers...")
    eval_results = evaluator.batch_evaluate(
        qa_pairs_before=qa_before,
        qa_pairs_after=qa_after
    )
    
    # Calculate additional metrics (if not already provided by evaluator)
    if "statistics" in eval_results:
        stats = eval_results["statistics"]
        
        # Handle both hybrid and direct evaluation formats
        llm_stats = stats.get("llm_evaluation", {}) if "llm_evaluation" in stats else stats
        
        # Calculate precision, recall, and F1 if we have the required data
        try:
            # Extract evaluation results
            results = eval_results.get("results", [])
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            # Define threshold for "good" answer
            good_threshold = 3.5  # Answers scored >= 3.5 are considered "good"
            
            for result in results:
                eval_data = result.get("evaluation", {})
                
                # Get scores based on evaluation format
                if "llm_evaluation" in eval_data:
                    score_before = eval_data["llm_evaluation"].get("answer_a_score", 0)
                    score_after = eval_data["llm_evaluation"].get("answer_b_score", 0)
                else:
                    score_before = eval_data.get("answer_a_score", 0)
                    score_after = eval_data.get("answer_b_score", 0)
                
                # Convert to binary classification: good/not good
                before_good = score_before >= good_threshold
                after_good = score_after >= good_threshold
                
                # Update counts for precision/recall calculation
                if after_good and before_good:
                    true_positives += 1
                elif after_good and not before_good:
                    false_positives += 1
                elif not after_good and before_good:
                    false_negatives += 1
            
            # Calculate precision, recall, and F1
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Add to statistics
            if "llm_evaluation" in stats:
                stats["llm_evaluation"]["precision"] = precision
                stats["llm_evaluation"]["recall"] = recall
                stats["llm_evaluation"]["f1_score"] = f1_score
            else:
                stats["precision"] = precision
                stats["recall"] = recall
                stats["f1_score"] = f1_score
                
            logger.info(f"Calculated additional metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
    
    # Print summary
    stats = eval_results.get("statistics", {})
    llm_stats = stats.get("llm_evaluation", {}) if "llm_evaluation" in stats else stats
    
    logger.info("\nEvaluation Results:")
    logger.info(f"Average score before: {llm_stats.get('average_score_before', 0):.2f}/5.0")
    logger.info(f"Average score after: {llm_stats.get('average_score_after', 0):.2f}/5.0")
    logger.info(f"Improvement: {llm_stats.get('absolute_improvement', 0):.2f} points")
    logger.info(f"Improved answers: {llm_stats.get('improvements', 0)} ({llm_stats.get('improvement_percentage', 0):.1f}%)")
    
    # Save enhanced evaluation results
    output_file = os.path.join(
        "evaluation_results",
        f"enhanced_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved enhanced evaluation results to {output_file}")
    
    return eval_results


def run_ragas_benchmarks(args, combined_file):
    """
    Run RAGAS benchmarks on the combined answers
    
    Args:
        args: Command line arguments
        combined_file: Path to the combined answers file
        
    Returns:
        RAGAS benchmark results
    """
    logger.info("=== Running RAGAS Benchmarks ===")
    
    # Import RAGAS benchmarking module
    try:
        from ragas_benchmarking_integration import run_enhanced_benchmarks
    except ImportError:
        logger.error("Missing ragas_benchmarking_integration module! Please ensure it's available.")
        logger.warning("Skipping RAGAS benchmarks")
        return None
    
    # Run benchmarks
    try:
        logger.info(f"Running RAGAS benchmarks on {combined_file}")
        results = run_enhanced_benchmarks(
            dataset_path=combined_file,
            output_dir=f"{args.benchmark_output}/ragas",
            llm_name=args.llm_model,
            use_llm_based=not args.no_llm_ragas,
            generate_markdown_report=True
        )
        
        logger.info("RAGAS benchmarks completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error running RAGAS benchmarks: {e}")
        return {"error": str(e)}


def generate_dashboard(args, benchmark_results, evaluation_results=None, ragas_results=None, combined_answers_file=None):
    """Generate HTML dashboard with all results"""
    logger.info("=== Generating Results Dashboard ===")
    
    # Import dashboard generator
    try:
        from dashboard_generator import DashboardGenerator
    except ImportError:
        logger.error("Missing dashboard_generator module! Please ensure it's available.")
        raise
    
    # Initialize dashboard generator
    dashboard = DashboardGenerator(
        output_dir=args.dashboard_output,
        title=args.dashboard_title
    )
    
    # Generate dashboard
    output_file = dashboard.generate_dashboard(
        benchmark_results=benchmark_results,
        evaluation_results=evaluation_results,
        ragas_results=ragas_results
    )
    
    logger.info(f"Generated dashboard at {output_file}")
    return output_file


def run_pipeline(args):
    """Run the complete pipeline based on arguments"""
    dataset_files = None
    fine_tuned_model_path = None
    benchmark_results = None
    combined_answers_file = None
    evaluation_results = None
    ragas_results = None
    
    # Set up API key manager for key rotation
    api_key_manager = setup_api_key_manager(args)
    
    # Step 1: Dataset Generation (if not skipped)
    if not args.skip_dataset_generation:
        logger.info("Starting dataset generation...")
        dataset_files, index, vector_store, chunked_docs = generate_dataset(args, api_key_manager)
    else:
        logger.info("Skipping dataset generation...")
        if args.existing_dataset:
            # Validate existing dataset
            if os.path.exists(args.existing_dataset):
                dataset_files = args.existing_dataset
                logger.info(f"Using existing dataset: {dataset_files}")
            else:
                raise FileNotFoundError(f"Existing dataset not found: {args.existing_dataset}")
        else:
            raise ValueError("Must specify --existing_dataset when using --skip_dataset_generation")
    
    # Step 2: Fine-tune Embeddings (if requested)
    if args.fine_tune:
        logger.info("Starting embedding fine-tuning...")
        fine_tuned_model_path = fine_tune_embeddings(args, dataset_files)
    elif args.existing_finetuned_model:
        # Validate existing fine-tuned model
        if os.path.exists(args.existing_finetuned_model):
            fine_tuned_model_path = args.existing_finetuned_model
            logger.info(f"Using existing fine-tuned model: {fine_tuned_model_path}")
        else:
            raise FileNotFoundError(f"Existing fine-tuned model not found: {args.existing_finetuned_model}")
    
    # Step 3: Benchmark Embeddings (if requested)
    if args.benchmark:
        if not fine_tuned_model_path:
            raise ValueError("Benchmarking requires a fine-tuned model. Either use --fine_tune or --existing_finetuned_model")
            
        logger.info("Starting embedding benchmarking...")
        benchmark_results, summary = benchmark_embeddings(args, dataset_files, fine_tuned_model_path)
        
        # Print key benchmark metrics
        if benchmark_results and "improvements" in benchmark_results:
            logger.info("\nKey Benchmark Metrics:")
            for metric, data in benchmark_results["improvements"].items():
                rel_imp = data.get("relative_improvement_percent", 0)
                logger.info(f"{metric}: {rel_imp:.2f}% improvement")
    
    # Step 4: Generate Combined Answers (if needed for evaluation, RAGAS, or explicitly requested)
    if args.evaluate_answers or args.use_ragas or args.enable_combined_answers:
        logger.info("Generating combined before/after answers...")
        
        if not fine_tuned_model_path:
            raise ValueError("Need fine-tuned model for answer comparison. Use --fine_tune or --existing_finetuned_model")
            
        combined_answers_file = generate_combined_answers(
            args, dataset_files, api_key_manager, fine_tuned_model_path
        )
    
    # Step 5: Answer Evaluation (if requested)
    if args.evaluate_answers:
        logger.info("Starting answer evaluation...")
        
        if combined_answers_file:
            evaluation_results = evaluate_combined_answers(
                args, combined_answers_file, api_key_manager
            )
            
            # Print enhanced evaluation results including all new metrics
            if evaluation_results and "statistics" in evaluation_results:
                stats = evaluation_results.get("statistics", {})
                llm_stats = stats.get("llm_evaluation", {}) if "llm_evaluation" in stats else stats
                
                logger.info("\nEnhanced Evaluation Results:")
                logger.info(f"Average score before: {llm_stats.get('average_score_before', 0):.2f}/5.0")
                logger.info(f"Average score after: {llm_stats.get('average_score_after', 0):.2f}/5.0")
                logger.info(f"Absolute improvement: {llm_stats.get('absolute_improvement', 0):.2f} points")
                logger.info(f"Improved answers: {llm_stats.get('improvements', 0)} ({llm_stats.get('improvement_percentage', 0):.1f}%)")
                
                # Add new metrics if available
                if "precision" in llm_stats:
                    logger.info(f"Precision: {llm_stats.get('precision', 0):.3f}")
                if "recall" in llm_stats:
                    logger.info(f"Recall: {llm_stats.get('recall', 0):.3f}")
                if "f1_score" in llm_stats:
                    logger.info(f"F1 Score: {llm_stats.get('f1_score', 0):.3f}")
        else:
            logger.warning("No combined answers file available for evaluation")
    
    # Step 6: RAGAS Benchmarks (if requested)
    if args.use_ragas:
        logger.info("Running RAGAS benchmarks...")
        
        if combined_answers_file:
            ragas_results = run_ragas_benchmarks(args, combined_answers_file)
            
            # Display RAGAS results summary
            if ragas_results and "summary" in ragas_results:
                logger.info("\nRAGAS Metrics Summary:")
                for metric, value in ragas_results["summary"].items():
                    logger.info(f"{metric}: {value:.4f}")
        else:
            logger.warning("No combined answers file available for RAGAS benchmarks")
    
    # Return summary of actions
    return {
        "dataset_files": dataset_files,
        "fine_tuned_model_path": fine_tuned_model_path,
        "benchmark_results": benchmark_results,
        "combined_answers_file": combined_answers_file,
        "evaluation_results": evaluation_results,
        "ragas_results": ragas_results
    }


def main():
    """Main entry point"""
    # Set up asyncio event loop for compatibility
    setup_asyncio_event_loop()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not args.skip_dataset_generation and not os.path.isdir(args.pdf_dir):
        logger.error(f"PDF directory not found: {args.pdf_dir}")
        return 1
    
    if args.benchmark and not (args.fine_tune or args.existing_finetuned_model):
        logger.error("Benchmarking requires fine-tuning. Enable --fine_tune or provide --existing_finetuned_model")
        return 1
    
    # Run the pipeline
    try:
        start_time = time.time()
        results = run_pipeline(args)
        end_time = time.time()
        
        logger.info("Pipeline completed successfully")
        logger.info(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")
        
        # Print summary
        logger.info("\nExecution Summary:")
        if isinstance(results.get("dataset_files"), list):
            logger.info(f"Generated {len(results['dataset_files'])} dataset files")
        else:
            logger.info(f"Dataset: {results.get('dataset_files')}")
            
        logger.info(f"Fine-tuned model: {results.get('fine_tuned_model_path', 'Not fine-tuned')}")
        
        if results.get("benchmark_results"):
            logger.info("Benchmarking completed. Results available in the benchmark output directory.")
        
        if results.get("combined_answers_file"):
            logger.info(f"Combined answers file: {results.get('combined_answers_file')}")
        
        if results.get("evaluation_results"):
            logger.info("Answer evaluation completed. Results available in the evaluation output directory.")
        
        if results.get("ragas_results"):
            logger.info("RAGAS benchmarks completed. Results available in the RAGAS output directory.")
        
        return 0
    except Exception as e:
        logger.exception(f"Error in pipeline execution: {e}")
        return 1

if __name__ == "__main__":
    exit(main())