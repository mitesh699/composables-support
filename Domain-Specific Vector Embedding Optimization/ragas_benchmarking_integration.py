import os
import json
import logging
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

# Import your existing benchmarking module
from benchmarking import RagBenchmarker, NumpyEncoder

# Import the RAGAS evaluator
from ragas_integration import run_ragas_evaluation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ragas_benchmarking')

class EnhancedRagBenchmarker(RagBenchmarker):
    """
    Enhanced version of RagBenchmarker with improved RAGAS integration
    
    This class extends RagBenchmarker to include:
    - All RAGAS metrics (Faithfulness, Response Relevancy, Context Precision, etc.)
    - Support for both LLM-based and non-LLM based evaluation
    - Detailed reporting and visualization
    """
    
    def __init__(self, output_dir: str = "benchmarks", llm_name: str = "groq", use_llm_based: bool = True):
        """Initialize the benchmarker"""
        # Initialize the parent class
        super().__init__(output_dir=output_dir, llm_name=llm_name)
        
        # RAGAS-specific settings
        self.use_llm_based = use_llm_based
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Enhanced RAG Benchmarker initialized:")
        logger.info(f"- Output directory: {output_dir}")
        logger.info(f"- LLM: {llm_name}")
        logger.info(f"- Using LLM-based metrics: {use_llm_based}")
    
    def evaluate_ragas(self, qa_pairs: List[Dict], 
                      query_engine: Optional[Any] = None,
                      retrieved_contexts: Optional[List[List[str]]] = None) -> Dict:
        """
        Evaluate using enhanced RAGAS metrics
        
        Args:
            qa_pairs: List of QA pairs
            query_engine: Optional LlamaIndex QueryEngine for live evaluation
            retrieved_contexts: Optional list of contexts for each QA pair
            
        Returns:
            Dictionary of RAGAS metric scores
        """
        if not self.ragas_available:
            return {"error": "RAGAS not available"}
        
        if not qa_pairs:
            return {}
        
        try:
            # Run RAGAS evaluation using the integrated function
            results = run_ragas_evaluation(
                qa_pairs=qa_pairs,
                retrieved_contexts=retrieved_contexts,
                llm_name=self.llm_name,
                use_llm_based=self.use_llm_based
            )
            
            # Check if we got an error
            if "error" in results:
                logger.error(f"RAGAS evaluation error: {results['error']}")
                # Fall back to original implementation if needed
                return super().evaluate_ragas(qa_pairs, query_engine, retrieved_contexts)
            
            return results
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def run_benchmarks(self, dataset_path: str, 
                      query_engine: Optional[Any] = None,
                      retrieved_contexts: Optional[List[List[str]]] = None) -> Dict:
        """
        Run all benchmarks on a dataset, including enhanced RAGAS metrics
        
        Args:
            dataset_path: Path to dataset file
            query_engine: Optional LlamaIndex QueryEngine for live evaluation
            retrieved_contexts: Optional list of retrieved contexts for each QA pair
            
        Returns:
            Dictionary of benchmark results
        """
        # Load dataset
        qa_pairs = self.load_dataset(dataset_path)
        
        if not qa_pairs:
            return {"error": "No data found or failed to load dataset"}
        
        results = {
            "dataset": os.path.basename(dataset_path),
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Run basic statistics
        results["metrics"]["basic_stats"] = self.basic_statistics(qa_pairs)
        
        # Run content analysis
        results["metrics"]["content_analysis"] = self.content_analysis(qa_pairs)
        
        # Run RAGAS evaluation if available
        if self.ragas_available:
            try:
                ragas_results = self.evaluate_ragas(
                    qa_pairs, 
                    query_engine=query_engine,
                    retrieved_contexts=retrieved_contexts
                )
                
                results["metrics"]["ragas"] = ragas_results
                
                # Extract summary for easier reference
                if "summary" in ragas_results:
                    results["ragas_summary"] = ragas_results["summary"]
                    
                    # Print a summary of the metrics
                    logger.info("\nRAGAS Metrics Summary:")
                    for metric, value in ragas_results["summary"].items():
                        logger.info(f"  {metric}: {value:.4f}")
            except Exception as e:
                logger.error(f"RAGAS evaluation failed: {e}")
                results["metrics"]["ragas"] = {"error": str(e)}
        else:
            results["metrics"]["ragas"] = {"error": "RAGAS not available"}
        
        # Run custom semantic evaluation as a fallback
        try:
            results["metrics"]["semantic"] = self.custom_semantic_evaluation(qa_pairs)
        except Exception as e:
            logger.error(f"Error during semantic evaluation: {e}")
            results["metrics"]["semantic"] = {"error": str(e)}
        
        # Save results
        output_file = os.path.join(
            self.output_dir,
            f"benchmark_{os.path.basename(dataset_path).replace('.json', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Benchmarking results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving benchmark results: {e}")
            self._save_simplified_backup(results)
        
        return results
    
    def _save_simplified_backup(self, results):
        """Save simplified backup of results"""
        try:
            simplified_results = {
                "dataset": results.get("dataset", "unknown"),
                "timestamp": results.get("timestamp", datetime.now().isoformat()),
                "error": "Error during serialization"
            }
            
            # Add any available RAGAS summary
            if "ragas_summary" in results:
                simplified_results["ragas_summary"] = results["ragas_summary"]
            
            backup_file = os.path.join(
                self.output_dir,
                f"backup_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f)
                
            logger.info(f"Simplified backup saved to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save even a simplified backup: {e}")
    
    def generate_report(self, results: Dict) -> str:
        """
        Generate a human-readable report from benchmark results
        
        Args:
            results: Dictionary of benchmark results
            
        Returns:
            Markdown-formatted report
        """
        if not results:
            return "No benchmark results available."
        
        report = []
        report.append("# RAG Benchmarking Report\n")
        report.append(f"**Dataset:** {results.get('dataset', 'Unknown')}")
        report.append(f"**Timestamp:** {results.get('timestamp', datetime.now().isoformat())}\n")
        
        # Basic statistics
        if "basic_stats" in results.get("metrics", {}):
            stats = results["metrics"]["basic_stats"]
            report.append("## Dataset Statistics\n")
            report.append(f"- **Total examples:** {stats.get('total_examples', 'N/A')}")
            
            if "question_length" in stats:
                q_len = stats["question_length"]
                report.append(f"- **Question length:** Avg {q_len.get('mean', 0):.1f} words (min {q_len.get('min', 0)}, max {q_len.get('max', 0)})")
            
            if "answer_length" in stats:
                a_len = stats["answer_length"]
                report.append(f"- **Answer length:** Avg {a_len.get('mean', 0):.1f} words (min {a_len.get('min', 0)}, max {a_len.get('max', 0)})")
            
            # Add subtopic distribution
            if "subtopic_distribution" in stats:
                report.append("\n### Subtopic Distribution\n")
                for subtopic, count in stats["subtopic_distribution"].items():
                    report.append(f"- **{subtopic}:** {count}")
        
        # Content analysis
        if "content_analysis" in results.get("metrics", {}):
            content = results["metrics"]["content_analysis"]
            if "problematic_counts" in content:
                counts = content["problematic_counts"]
                report.append("\n## Content Analysis\n")
                report.append(f"- **Context references:** {counts.get('context_references', 0)}")
                report.append(f"- **Too short answers:** {counts.get('too_short', 0)}")
                report.append(f"- **Too long answers:** {counts.get('too_long', 0)}")
                report.append(f"- **Uncertain answers:** {counts.get('uncertain', 0)}")
        
        # RAGAS metrics
        if "ragas_summary" in results:
            summary = results["ragas_summary"]
            report.append("\n## RAGAS Metrics\n")
            
            # Sort metrics by name but put noise_sensitivity last (lower is better)
            sorted_metrics = sorted([m for m in summary.keys() if m != "noise_sensitivity"])
            if "noise_sensitivity" in summary:
                sorted_metrics.append("noise_sensitivity")
            
            for metric in sorted_metrics:
                value = summary.get(metric, 0.0)
                # Add note for noise_sensitivity (lower is better)
                note = " (lower is better)" if metric == "noise_sensitivity" else ""
                report.append(f"- **{metric}:** {value:.4f}{note}")
            
            report.append("\n### Metrics Interpretation\n")
            report.append("- **Faithfulness:** Measures factual consistency between responses and retrieved contexts")
            report.append("- **Response Relevancy:** Evaluates how well responses address the original questions")
            report.append("- **Context Precision:** Measures relevance of retrieved chunks")
            report.append("- **Context Recall:** Assesses how well retrieved contexts cover reference information")
            report.append("- **Context Entity Recall:** Measures how well entities are recalled from reference")
            report.append("- **Noise Sensitivity:** Measures how often the system makes errors (lower is better)")
        
        # Semantic evaluation (fallback)
        if "semantic" in results.get("metrics", {}) and "semantic_relevance" in results["metrics"]["semantic"]:
            semantic = results["metrics"]["semantic"]["semantic_relevance"]
            report.append("\n## Semantic Relevance (Fallback Metric)\n")
            report.append(f"- **Mean:** {semantic.get('mean', 0):.4f}")
            report.append(f"- **Min:** {semantic.get('min', 0):.4f}")
            report.append(f"- **Max:** {semantic.get('max', 0):.4f}")
        
        return "\n".join(report)


def run_enhanced_benchmarks(
    dataset_path: str,
    query_engine: Optional[Any] = None,
    retrieved_contexts: Optional[List[List[str]]] = None,
    output_dir: str = "benchmarks",
    llm_name: str = "groq",
    use_llm_based: bool = True,
    generate_markdown_report: bool = True
):
    """
    Run enhanced benchmarks with RAGAS integration
    
    Args:
        dataset_path: Path to dataset file
        query_engine: Optional LlamaIndex QueryEngine for live evaluation
        retrieved_contexts: Optional list of retrieved contexts for each QA pair
        output_dir: Directory to save benchmark results
        llm_name: Name of LLM to use
        use_llm_based: Whether to use LLM-based RAGAS metrics
        generate_markdown_report: Whether to generate a markdown report
        
    Returns:
        Dictionary of benchmark results
    """
    # Initialize benchmarker
    benchmarker = EnhancedRagBenchmarker(
        output_dir=output_dir,
        llm_name=llm_name,
        use_llm_based=use_llm_based
    )
    
    # Run benchmarks
    results = benchmarker.run_benchmarks(
        dataset_path=dataset_path,
        query_engine=query_engine,
        retrieved_contexts=retrieved_contexts
    )
    
    # Generate and save markdown report if requested
    if generate_markdown_report:
        report = benchmarker.generate_report(results)
        report_file = os.path.join(
            output_dir,
            f"report_{os.path.basename(dataset_path).replace('.json', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"Benchmark report saved to {report_file}")
        except Exception as e:
            logger.error(f"Error saving benchmark report: {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run enhanced benchmarks with RAGAS integration")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset file")
    parser.add_argument("--output_dir", type=str, default="benchmarks", help="Directory to save benchmark results")
    parser.add_argument("--llm_name", type=str, default="groq", help="Name of LLM to use")
    parser.add_argument("--no_llm_based", action="store_true", help="Use non-LLM based RAGAS metrics")
    parser.add_argument("--no_report", action="store_true", help="Skip generating markdown report")
    
    args = parser.parse_args()
    
    results = run_enhanced_benchmarks(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        llm_name=args.llm_name,
        use_llm_based=not args.no_llm_based,
        generate_markdown_report=not args.no_report
    )
    
    print(f"\nBenchmarking complete. Results saved to {args.output_dir}")