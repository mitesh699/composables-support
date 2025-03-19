import os
import json
import time
import logging
import torch
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
from datetime import datetime
from sentence_transformers import SentenceTransformer
from api_key_manager import APIKeyManager, RotatingAPIRateLimiter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("answer_evaluator")

class HuggingFaceJudge:
    """
    Uses Hugging Face models to judge answer quality in controlled experiments
    
    Features:
    - Compare before/after fine-tuning results
    - Scale-based evaluation (1-5)
    - Concise explanations (limited to 80 tokens)
    - Supports multiple Hugging Face models
    """
    
    def __init__(
        self,
        judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct", # Lightweight model for judging
        huggingface_token: str = None,
        output_dir: str = "evaluation_results",
        max_tokens: int = 80,  # Limited to 80 tokens for evaluation responses
        temperature: float = 0.3  # Low temperature for consistent judgments
    ):
        """
        Initialize the Hugging Face Judge
        
        Args:
            judge_model: HF model to use for judging
            huggingface_token: HF API token (will check HF_TOKEN env var if None)
            output_dir: Directory to save evaluation results
            max_tokens: Maximum tokens for evaluation response (limited to 80)
            temperature: Temperature for generation
        """
        self.judge_model = judge_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_dir = output_dir
        
        # Get HF token from env if not provided
        self.hf_token = huggingface_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            logger.warning("No Hugging Face token found. Set HF_TOKEN environment variable.")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize rate limiting tracker
        self._init_rate_tracker()
        
        logger.info(f"Initialized Hugging Face Judge with model: {judge_model}")
    
    def _init_rate_tracker(self):
        """Initialize rate tracking for HuggingFace API"""
        self._hf_rate_tracker = {
            'minute_start': datetime.now(),
            'tokens_this_minute': 0,
            'requests_this_minute': 0
        }
    
    def _apply_rate_limit(self, prompt_tokens):
        """Apply token-aware rate limiting for HuggingFace API calls"""
        # Constants for HuggingFace rate limits
        MAX_TOKENS_PER_MINUTE = 10000
        MAX_REQUESTS_PER_MINUTE = 20
        
        # Get current timestamp
        now = datetime.now()
        
        # Reset counter if a minute has passed
        time_diff = (now - self._hf_rate_tracker['minute_start']).total_seconds()
        if time_diff > 60:
            self._hf_rate_tracker = {
                'minute_start': now,
                'tokens_this_minute': 0,
                'requests_this_minute': 0
            }
        
        # Check if we would exceed limits
        if (self._hf_rate_tracker['tokens_this_minute'] + prompt_tokens > MAX_TOKENS_PER_MINUTE or
            self._hf_rate_tracker['requests_this_minute'] >= MAX_REQUESTS_PER_MINUTE):
            # Calculate wait time until next minute
            wait_time = 60 - time_diff
            if wait_time > 0:
                logger.info(f"HuggingFace rate limit approached. Waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                # Reset counters after waiting
                self._hf_rate_tracker = {
                    'minute_start': datetime.now(),
                    'tokens_this_minute': 0,
                    'requests_this_minute': 0
                }
        
        # Update counters
        self._hf_rate_tracker['tokens_this_minute'] += prompt_tokens
        self._hf_rate_tracker['requests_this_minute'] += 1
    
    def evaluate_answer_pair(
        self,
        question: str,
        answer_before: str,
        answer_after: str,
        attempt: int = 0
    ) -> Dict:
        """
        Evaluate a pair of answers (before/after fine-tuning)
        
        Args:
            question: The question being answered
            answer_before: Answer before fine-tuning
            answer_after: Answer after fine-tuning
            attempt: Current attempt number (for retries)
            
        Returns:
            Evaluation results dictionary
        """
        max_retries = 3
        
        try:
            # Truncate answers if too long (to avoid token overflow)
            max_answer_len = 500
            if len(answer_before) > max_answer_len:
                answer_before = answer_before[:max_answer_len] + "..."
            if len(answer_after) > max_answer_len:
                answer_after = answer_after[:max_answer_len] + "..."
            
            # Create evaluation prompt (optimized for brevity)
            prompt = f"""Rate two answers to this question on a scale of 1-5 (5 is best). Provide very brief one-line feedback.

Question: {question}

Answer A: {answer_before}

Answer B: {answer_after}

JSON format only:
{{
  "answer_a_score": [1-5],
  "answer_b_score": [1-5],
  "answer_a_feedback": "brief feedback",
  "answer_b_feedback": "brief feedback"
}}
"""
            # Estimate token count for rate limiting
            estimated_tokens = len(prompt) // 4
            
            # Apply rate limiting
            self._apply_rate_limit(estimated_tokens)
            
            # Call Hugging Face API
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "return_full_text": False
                }
            }
            
            url = f"https://api-inference.huggingface.co/models/{self.judge_model}"
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code != 200:
                if response.status_code == 429:
                    # Rate limit hit
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited by Hugging Face. Retry after {retry_after}s")
                    time.sleep(retry_after)
                    raise ValueError(f"Rate limited by Hugging Face API")
                else:
                    # Other error
                    raise ValueError(f"HF API error: {response.status_code} - {response.text}")
            
            # Extract JSON from response
            response_json = response.json()
            if isinstance(response_json, list) and len(response_json) > 0:
                response_text = response_json[0].get("generated_text", "")
            else:
                response_text = response_json.get("generated_text", "")
            
            # Extract JSON from text response
            json_str = self._extract_json(response_text)
            evaluation = json.loads(json_str)
            
            # Validate evaluation format and fix if needed
            required_keys = ["answer_a_score", "answer_b_score", "answer_a_feedback", "answer_b_feedback"]
            for key in required_keys:
                if key not in evaluation:
                    if "score" in key:
                        evaluation[key] = 0
                    else:
                        evaluation[key] = "No feedback provided"
            
            # Ensure scores are integers 1-5
            for key in ["answer_a_score", "answer_b_score"]:
                try:
                    score = int(evaluation[key])
                    evaluation[key] = max(1, min(5, score))
                except (ValueError, TypeError):
                    evaluation[key] = 0
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"Error during evaluation: {e}")
            
            # Retry with backoff
            if attempt < max_retries:
                retry_delay = 2 ** attempt
                logger.info(f"Retrying in {retry_delay}s (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                return self.evaluate_answer_pair(question, answer_before, answer_after, attempt + 1)
            else:
                # Return a default response after max retries
                logger.error(f"Failed to evaluate after {max_retries} attempts")
                return {
                    "answer_a_score": 0,
                    "answer_b_score": 0,
                    "answer_a_feedback": "Evaluation failed",
                    "answer_b_feedback": "Evaluation failed",
                    "error": str(e)
                }
    
    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from text, handling various formats including HTML errors
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted JSON string
        """
        # Return default if text appears to be HTML content
        if "<!DOCTYPE html>" in text or "<html" in text:
            logger.warning("Received HTML response instead of JSON. Using default values.")
            return '{"answer_a_score": 0, "answer_b_score": 0, "answer_a_feedback": "Invalid response format", "answer_b_feedback": "Invalid response format", "error": "Response contained HTML"}'
        
        # Check if the entire text is valid JSON
        try:
            json.loads(text)
            return text
        except:
            pass
        
        # Try to find JSON between curly braces
        import re
        json_pattern = r'(\{.*\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        
        # Handle JSON with code blocks
        if "```json" in text:
            parts = text.split("```json")
            if len(parts) > 1:
                json_part = parts[1].split("```")[0].strip()
                try:
                    json.loads(json_part)
                    return json_part
                except:
                    pass
        
        if "```" in text:
            parts = text.split("```")
            if len(parts) > 1:
                json_part = parts[1].strip()
                try:
                    json.loads(json_part)
                    return json_part
                except:
                    pass
        
        # If all else fails, attempt to fix common JSON errors
        text = text.replace("'", '"')  # Replace single quotes with double quotes
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        
        # Last resort: find anything that looks like JSON
        json_pattern = r'(\{.*\})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                json.loads(match)
                return match
            except:
                continue
        
        # Give up and return a minimal valid JSON
        logger.warning(f"Could not extract valid JSON, returning default")
        return '{"answer_a_score": 0, "answer_b_score": 0, "answer_a_feedback": "Invalid response format", "answer_b_feedback": "Invalid response format", "error": "Failed to parse JSON"}'
    
    def batch_evaluate(
        self,
        qa_pairs_before: List[Dict],
        qa_pairs_after: List[Dict],
        max_pairs: int = None,
        output_file: str = None
    ) -> Dict:
        """
        Evaluate multiple QA pairs and compare before/after
        
        Args:
            qa_pairs_before: List of QA pairs before fine-tuning
            qa_pairs_after: List of QA pairs after fine-tuning
            max_pairs: Maximum number of pairs to evaluate
            output_file: File to save results (default: auto-generated)
            
        Returns:
            Evaluation results
        """
        # Validate inputs
        if not qa_pairs_before or not qa_pairs_after:
            raise ValueError("Both before and after QA pairs must be provided")
        
        # Create question-based lookup for after pairs
        after_lookup = {qa["question"]: qa for qa in qa_pairs_after}
        
        # Find pairs with matching questions
        matching_pairs = []
        for before_qa in qa_pairs_before:
            question = before_qa.get("question")
            if question in after_lookup:
                matching_pairs.append({
                    "question": question,
                    "answer_before": before_qa.get("answer", ""),
                    "answer_after": after_lookup[question].get("answer", "")
                })
        
        # Limit pairs if requested
        if max_pairs and len(matching_pairs) > max_pairs:
            logger.info(f"Limiting evaluation to {max_pairs} pairs (out of {len(matching_pairs)})")
            matching_pairs = matching_pairs[:max_pairs]
        
        logger.info(f"Evaluating {len(matching_pairs)} QA pairs with Hugging Face model: {self.judge_model}")
        
        # Evaluate all pairs
        results = []
        for i, pair in enumerate(tqdm(matching_pairs, desc="Evaluating")):
            evaluation = self.evaluate_answer_pair(
                pair["question"],
                pair["answer_before"],
                pair["answer_after"]
            )
            
            # Add pair data to evaluation
            result = {
                "question": pair["question"],
                "answer_before": pair["answer_before"],
                "answer_after": pair["answer_after"],
                "evaluation": evaluation
            }
            
            results.append(result)
            
            # Periodic save
            if i > 0 and i % 10 == 0:
                self._save_interim_results(results, i, len(matching_pairs))
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        # Prepare final output
        final_results = {
            "metadata": {
                "judge_model": self.judge_model,
                "evaluated_at": datetime.now().isoformat(),
                "pairs_evaluated": len(results),
                "original_before_count": len(qa_pairs_before),
                "original_after_count": len(qa_pairs_after),
                "max_tokens": self.max_tokens
            },
            "statistics": stats,
            "results": results
        }
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"evaluation_results_{timestamp}.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved evaluation results to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            
            # Try backup save
            backup_file = os.path.join(self.output_dir, f"backup_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved backup to {backup_file}")
        
        # Generate summary report
        report = self._generate_report(final_results)
        
        report_file = output_file.replace(".json", "_report.md")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved evaluation report to {report_file}")
        
        return final_results
    
    def _save_interim_results(self, results: List[Dict], current: int, total: int):
        """Save interim results during batch processing"""
        interim_file = os.path.join(self.output_dir, f"interim_eval_results_{current}of{total}.json")
        try:
            with open(interim_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "interim": True,
                        "progress": f"{current}/{total}",
                        "timestamp": datetime.now().isoformat()
                    },
                    "results": results
                }, f, indent=2)
            logger.info(f"Saved interim results ({current}/{total})")
        except Exception as e:
            logger.error(f"Error saving interim results: {e}")
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics from evaluation results"""
        if not results:
            return {}
        
        # Extract scores
        scores_before = []
        scores_after = []
        
        for result in results:
            eval_data = result.get("evaluation", {})
            score_a = eval_data.get("answer_a_score")
            score_b = eval_data.get("answer_b_score")
            
            # Skip invalid scores
            if not isinstance(score_a, (int, float)) or not isinstance(score_b, (int, float)):
                continue
                
            scores_before.append(score_a)
            scores_after.append(score_b)
        
        if not scores_before or not scores_after:
            return {"error": "No valid scores found"}
        
        # Calculate basic statistics
        avg_before = sum(scores_before) / len(scores_before)
        avg_after = sum(scores_after) / len(scores_after)
        
        # Count improvements, regressions, ties
        improvements = sum(1 for a, b in zip(scores_before, scores_after) if b > a)
        regressions = sum(1 for a, b in zip(scores_before, scores_after) if b < a)
        ties = sum(1 for a, b in zip(scores_before, scores_after) if b == a)
        
        # Calculate improvement percentage
        total_pairs = len(scores_before)
        improvement_pct = (improvements / total_pairs) * 100 if total_pairs > 0 else 0
        
        # Score distributions
        score_dist_before = {str(i): scores_before.count(i) for i in range(1, 6)}
        score_dist_after = {str(i): scores_after.count(i) for i in range(1, 6)}
        
        return {
            "average_score_before": avg_before,
            "average_score_after": avg_after,
            "improvement_percentage": improvement_pct,
            "absolute_improvement": avg_after - avg_before,
            "total_evaluated": total_pairs,
            "improvements": improvements,
            "regressions": regressions,
            "ties": ties,
            "score_distribution_before": score_dist_before,
            "score_distribution_after": score_dist_after
        }
    
    def _generate_report(self, results: Dict) -> str:
        """Generate a markdown report from evaluation results"""
        stats = results.get("statistics", {})
        metadata = results.get("metadata", {})
        
        # Format report
        report = [
            "# Answer Quality Evaluation Report",
            "",
            f"**Date:** {metadata.get('evaluated_at', datetime.now().isoformat())}",
            f"**Judge Model:** {metadata.get('judge_model', self.judge_model)}",
            f"**Pairs Evaluated:** {metadata.get('pairs_evaluated', 0)}",
            "",
            "## Summary",
            "",
            f"**Average Score Before Optimization:** {stats.get('average_score_before', 0):.2f} / 5.0",
            f"**Average Score After Optimization:** {stats.get('average_score_after', 0):.2f} / 5.0",
            f"**Absolute Improvement:** {stats.get('absolute_improvement', 0):.2f} points",
            f"**Relative Improvement:** {stats.get('improvement_percentage', 0):.1f}%",
            "",
            "## Results Breakdown",
            "",
            f"**Improved Answers:** {stats.get('improvements', 0)} ({(stats.get('improvements', 0) / stats.get('total_evaluated', 1) * 100):.1f}%)",
            f"**Unchanged Answers:** {stats.get('ties', 0)} ({(stats.get('ties', 0) / stats.get('total_evaluated', 1) * 100):.1f}%)",
            f"**Regression Answers:** {stats.get('regressions', 0)} ({(stats.get('regressions', 0) / stats.get('total_evaluated', 1) * 100):.1f}%)",
            "",
            "## Score Distribution",
            ""
        ]
        
        # Add score distribution
        before_dist = stats.get("score_distribution_before", {})
        after_dist = stats.get("score_distribution_after", {})
        
        report.append("| Score | Before | After | Change |")
        report.append("|-------|--------|-------|--------|")
        
        for score in range(1, 6):
            score_str = str(score)
            before_count = before_dist.get(score_str, 0)
            after_count = after_dist.get(score_str, 0)
            change = after_count - before_count
            
            if change > 0:
                change_str = f"+{change}"
            else:
                change_str = str(change)
                
            report.append(f"| {score} | {before_count} | {after_count} | {change_str} |")
        
        # Add sample evaluations
        report.extend([
            "",
            "## Sample Evaluations",
            ""
        ])
        
        # Add up to 5 sample evaluations (prioritize ones with improvement)
        eval_results = results.get("results", [])
        
        # Sort by improvement (score_after - score_before)
        eval_results.sort(
            key=lambda x: (x.get("evaluation", {}).get("answer_b_score", 0) - 
                         x.get("evaluation", {}).get("answer_a_score", 0)),
            reverse=True
        )
        
        # Take top 5
        samples = eval_results[:5]
        
        for i, sample in enumerate(samples):
            question = sample.get("question", "")
            eval_data = sample.get("evaluation", {})
            
            score_before = eval_data.get("answer_a_score", 0)
            score_after = eval_data.get("answer_b_score", 0)
            feedback_before = eval_data.get("answer_a_feedback", "")
            feedback_after = eval_data.get("answer_b_feedback", "")
            
            report.extend([
                f"### Sample {i+1}",
                "",
                f"**Question:** {question}",
                "",
                f"**Before Score:** {score_before}/5",
                f"**After Score:** {score_after}/5",
                "",
                f"**Before Feedback:** {feedback_before}",
                f"**After Feedback:** {feedback_after}",
                ""
            ])
        
        # Add conclusions
        avg_improvement = stats.get('absolute_improvement', 0)
        if avg_improvement > 0.5:
            conclusion = "The optimized model shows significant improvement in answer quality."
        elif avg_improvement > 0.2:
            conclusion = "The optimized model shows moderate improvement in answer quality."
        elif avg_improvement > 0:
            conclusion = "The optimized model shows slight improvement in answer quality."
        else:
            conclusion = "The optimization did not improve answer quality."
            
        report.extend([
            "## Conclusion",
            "",
            conclusion
        ])
        
        return "\n".join(report)


class SemanticEvaluator:
    """
    Evaluates answer quality using semantic similarity metrics
    
    Features:
    - Uses embedding models to calculate similarity
    - Provides objective metrics without LLM dependencies
    - Fast evaluation of large datasets
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "evaluation_results"
    ):
        """
        Initialize the semantic evaluator
        
        Args:
            embedding_model: Model to use for embeddings
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize embedding model
        try:
            self.model = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def evaluate_answer_pair(
        self,
        question: str,
        answer_before: str,
        answer_after: str
    ) -> Dict:
        """
        Evaluate a pair of answers using semantic similarity
        
        Args:
            question: The question being answered
            answer_before: Answer before fine-tuning
            answer_after: Answer after fine-tuning
            
        Returns:
            Evaluation results dictionary
        """
        # Generate embeddings
        question_emb = self.model.encode(question, convert_to_numpy=True)
        before_emb = self.model.encode(answer_before, convert_to_numpy=True)
        after_emb = self.model.encode(answer_after, convert_to_numpy=True)
        
        # Calculate similarities
        q_before_sim = self._cosine_similarity(question_emb, before_emb)
        q_after_sim = self._cosine_similarity(question_emb, after_emb)
        
        # Convert similarity to score (0-5 scale)
        before_score = min(5, max(1, round(q_before_sim * 5, 1)))
        after_score = min(5, max(1, round(q_after_sim * 5, 1)))
        
        # Generate feedback
        if q_after_sim > q_before_sim:
            after_feedback = "Improved semantic relevance to the question."
            improvement = "improved"
        elif q_after_sim < q_before_sim:
            after_feedback = "Decreased semantic relevance to the question."
            improvement = "decreased"
        else:
            after_feedback = "No significant change in semantic relevance."
            improvement = "unchanged"
        
        # Generate result
        return {
            "answer_a_score": before_score,
            "answer_b_score": after_score,
            "answer_a_feedback": f"Semantic similarity to question: {q_before_sim:.3f}",
            "answer_b_feedback": after_feedback,
            "semantic_metrics": {
                "question_before_similarity": float(q_before_sim),
                "question_after_similarity": float(q_after_sim),
                "relative_improvement": float(q_after_sim - q_before_sim),
                "improvement": improvement
            }
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def batch_evaluate(
        self,
        qa_pairs_before: List[Dict],
        qa_pairs_after: List[Dict],
        max_pairs: int = None,
        output_file: str = None
    ) -> Dict:
        """
        Evaluate multiple QA pairs and compare before/after
        
        Args:
            qa_pairs_before: List of QA pairs before fine-tuning
            qa_pairs_after: List of QA pairs after fine-tuning
            max_pairs: Maximum number of pairs to evaluate
            output_file: File to save results (default: auto-generated)
            
        Returns:
            Evaluation results
        """
        # Validate inputs
        if not qa_pairs_before or not qa_pairs_after:
            raise ValueError("Both before and after QA pairs must be provided")
        
        # Create question-based lookup for after pairs
        after_lookup = {qa["question"]: qa for qa in qa_pairs_after}
        
        # Find pairs with matching questions
        matching_pairs = []
        for before_qa in qa_pairs_before:
            question = before_qa.get("question")
            if question in after_lookup:
                matching_pairs.append({
                    "question": question,
                    "answer_before": before_qa.get("answer", ""),
                    "answer_after": after_lookup[question].get("answer", "")
                })
        
        # Limit pairs if requested
        if max_pairs and len(matching_pairs) > max_pairs:
            logger.info(f"Limiting evaluation to {max_pairs} pairs (out of {len(matching_pairs)})")
            matching_pairs = matching_pairs[:max_pairs]
        
        logger.info(f"Evaluating {len(matching_pairs)} QA pairs using semantic similarity")
        
        # Evaluate all pairs
        results = []
        for pair in tqdm(matching_pairs, desc="Evaluating"):
            evaluation = self.evaluate_answer_pair(
                pair["question"],
                pair["answer_before"],
                pair["answer_after"]
            )
            
            # Add pair data to evaluation
            result = {
                "question": pair["question"],
                "answer_before": pair["answer_before"],
                "answer_after": pair["answer_after"],
                "evaluation": evaluation
            }
            
            results.append(result)
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        # Prepare final output
        final_results = {
            "metadata": {
                "evaluator": "semantic",
                "embedding_model": self.model.get_sentence_embedding_dimension(),
                "evaluated_at": datetime.now().isoformat(),
                "pairs_evaluated": len(results),
                "original_before_count": len(qa_pairs_before),
                "original_after_count": len(qa_pairs_after)
            },
            "statistics": stats,
            "results": results
        }
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"semantic_evaluation_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        return final_results
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics from evaluation results"""
        if not results:
            return {}
        
        # Extract semantic metrics
        similarities_before = []
        similarities_after = []
        improvements = []
        
        for result in results:
            metrics = result.get("evaluation", {}).get("semantic_metrics", {})
            sim_before = metrics.get("question_before_similarity")
            sim_after = metrics.get("question_after_similarity")
            
            if sim_before is not None and sim_after is not None:
                similarities_before.append(sim_before)
                similarities_after.append(sim_after)
                improvements.append(sim_after - sim_before)
        
        if not similarities_before:
            return {"error": "No valid similarity metrics found"}
        
        # Calculate statistics
        avg_before = sum(similarities_before) / len(similarities_before)
        avg_after = sum(similarities_after) / len(similarities_after)
        avg_improvement = sum(improvements) / len(improvements)
        
        # Count improvements, regressions, ties
        improved_count = sum(1 for imp in improvements if imp > 0.01)
        unchanged_count = sum(1 for imp in improvements if abs(imp) <= 0.01)
        regressed_count = sum(1 for imp in improvements if imp < -0.01)
        
        # Calculate percentiles
        improvements.sort()
        median_improvement = improvements[len(improvements) // 2]
        p90_improvement = improvements[int(len(improvements) * 0.9)]
        p10_improvement = improvements[int(len(improvements) * 0.1)]
        
        return {
            "average_similarity_before": avg_before,
            "average_similarity_after": avg_after,
            "average_improvement": avg_improvement,
            "relative_improvement_percent": (avg_improvement / avg_before) * 100 if avg_before > 0 else 0,
            "median_improvement": median_improvement,
            "p90_improvement": p90_improvement,
            "p10_improvement": p10_improvement,
            "improved_count": improved_count,
            "unchanged_count": unchanged_count,
            "regressed_count": regressed_count,
            "total_evaluated": len(similarities_before)
        }


# Hybrid evaluator that combines HuggingFace and semantic approaches
class HybridAnswerEvaluator:
    """
    Comprehensive answer evaluator combining HuggingFace judgment and semantic metrics
    
    Features:
    - Uses both HuggingFace-based and embedding-based evaluation
    - Falls back to semantic evaluation when HuggingFace is unavailable
    - Provides rich evaluation metrics
    """
    
    def __init__(
        self,
        api_key_manager: Optional[APIKeyManager] = None,
        judge_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "evaluation_results",
        use_llm: bool = True,
        huggingface_token: str = None
    ):
        """
        Initialize the hybrid evaluator
        
        Args:
            api_key_manager: API key manager for Groq (not used with HuggingFace)
            judge_model: HuggingFace model for subjective evaluation
            embedding_model: Model for semantic evaluation
            output_dir: Directory to save results
            use_llm: Whether to use LLM-based evaluation
            huggingface_token: Token for HuggingFace API
        """
        self.output_dir = output_dir
        self.use_llm = use_llm
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize evaluators
        self.semantic_evaluator = SemanticEvaluator(
            embedding_model=embedding_model,
            output_dir=output_dir
        )
        
        if use_llm:
            self.llm_judge = HuggingFaceJudge(
                judge_model=judge_model,
                huggingface_token=huggingface_token,
                output_dir=output_dir,
                max_tokens=80  # Limit to 80 tokens as requested
            )
        else:
            self.llm_judge = None
        
        logger.info(f"Initialized hybrid evaluator with LLM: {use_llm}")
        if use_llm:
            logger.info(f"Using HuggingFace judge model: {judge_model} (limited to 80 tokens)")
    
    def evaluate_answer_pair(
        self,
        question: str,
        answer_before: str,
        answer_after: str
    ) -> Dict:
        """
        Evaluate a pair of answers using both HuggingFace and semantic approaches
        
        Args:
            question: The question being answered
            answer_before: Answer before optimization/fine-tuning
            answer_after: Answer after optimization/fine-tuning
            
        Returns:
            Combined evaluation results
        """
        # Always run semantic evaluation
        semantic_eval = self.semantic_evaluator.evaluate_answer_pair(
            question, answer_before, answer_after
        )
        
        # Run HuggingFace evaluation if enabled
        if self.use_llm and self.llm_judge:
            try:
                llm_eval = self.llm_judge.evaluate_answer_pair(
                    question, answer_before, answer_after
                )
                
                # Combine results
                return {
                    "llm_evaluation": llm_eval,
                    "semantic_evaluation": semantic_eval,
                    "answer_a_score": llm_eval.get("answer_a_score", 0),
                    "answer_b_score": llm_eval.get("answer_b_score", 0),
                    "answer_a_feedback": llm_eval.get("answer_a_feedback", ""),
                    "answer_b_feedback": llm_eval.get("answer_b_feedback", "")
                }
            except Exception as e:
                logger.warning(f"HuggingFace evaluation failed: {e}. Using semantic evaluation only.")
        
        # Fallback to semantic evaluation results
        return {
            "semantic_evaluation": semantic_eval,
            "answer_a_score": semantic_eval.get("answer_a_score", 0),
            "answer_b_score": semantic_eval.get("answer_b_score", 0),
            "answer_a_feedback": semantic_eval.get("answer_a_feedback", ""),
            "answer_b_feedback": semantic_eval.get("answer_b_feedback", ""),
            "note": "Using semantic evaluation only"
        }
    
    def batch_evaluate(
        self,
        qa_pairs_before: List[Dict],
        qa_pairs_after: List[Dict],
        max_pairs: int = None,
        output_file: str = None
    ) -> Dict:
        """
        Evaluate multiple QA pairs and compare before/after
        
        Args:
            qa_pairs_before: List of QA pairs before optimization/fine-tuning
            qa_pairs_after: List of QA pairs after optimization/fine-tuning
            max_pairs: Maximum number of pairs to evaluate
            output_file: File to save results (default: auto-generated)
            
        Returns:
            Evaluation results
        """
        # Validate inputs
        if not qa_pairs_before or not qa_pairs_after:
            raise ValueError("Both before and after QA pairs must be provided")
        
        # Create question-based lookup for after pairs
        after_lookup = {qa["question"]: qa for qa in qa_pairs_after}
        
        # Find pairs with matching questions
        matching_pairs = []
        for before_qa in qa_pairs_before:
            question = before_qa.get("question")
            if question in after_lookup:
                matching_pairs.append({
                    "question": question,
                    "answer_before": before_qa.get("answer", ""),
                    "answer_after": after_lookup[question].get("answer", "")
                })
        
        # Limit pairs if requested
        if max_pairs and len(matching_pairs) > max_pairs:
            logger.info(f"Limiting evaluation to {max_pairs} pairs (out of {len(matching_pairs)})")
            matching_pairs = matching_pairs[:max_pairs]
        
        logger.info(f"Evaluating {len(matching_pairs)} QA pairs")
        
        # Evaluate all pairs
        results = []
        for i, pair in enumerate(tqdm(matching_pairs, desc="Evaluating")):
            evaluation = self.evaluate_answer_pair(
                pair["question"],
                pair["answer_before"],
                pair["answer_after"]
            )
            
            # Add pair data to evaluation
            result = {
                "question": pair["question"],
                "answer_before": pair["answer_before"],
                "answer_after": pair["answer_after"],
                "evaluation": evaluation
            }
            
            results.append(result)
            
            # Periodic save
            if i > 0 and i % 10 == 0 and self.use_llm:
                self._save_interim_results(results, i, len(matching_pairs))
            
            # Small delay to avoid overwhelming the API
            if self.use_llm:
                time.sleep(0.5)
        
        # Calculate statistics
        stats = self._calculate_statistics(results)
        
        # Prepare final output
        final_results = {
            "metadata": {
                "evaluator": "hybrid",
                "llm_enabled": self.use_llm,
                "judge_model": self.llm_judge.judge_model if self.llm_judge else None,
                "embedding_model": self.semantic_evaluator.model.get_sentence_embedding_dimension(),
                "evaluated_at": datetime.now().isoformat(),
                "pairs_evaluated": len(results),
                "max_tokens": 80 if self.use_llm else None  # Record token limit
            },
            "statistics": stats,
            "results": results
        }
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"hybrid_evaluation_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation results to {output_file}")
        
        return final_results
    
    def _save_interim_results(self, results: List[Dict], current: int, total: int):
        """Save interim results during batch processing"""
        interim_file = os.path.join(self.output_dir, f"interim_hybrid_eval_{current}of{total}.json")
        with open(interim_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metadata": {
                    "interim": True,
                    "progress": f"{current}/{total}",
                    "timestamp": datetime.now().isoformat()
                },
                "results": results
            }, f, indent=2)
        logger.info(f"Saved interim results ({current}/{total})")
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics from evaluation results"""
        if not results:
            return {}
        
        # Extract scores
        llm_scores_before = []
        llm_scores_after = []
        semantic_scores_before = []
        semantic_scores_after = []
        
        for result in results:
            eval_data = result.get("evaluation", {})
            
            # Extract LLM scores if available
            llm_eval = eval_data.get("llm_evaluation", {})
            if llm_eval:
                llm_before = llm_eval.get("answer_a_score")
                llm_after = llm_eval.get("answer_b_score")
                
                if isinstance(llm_before, (int, float)) and isinstance(llm_after, (int, float)):
                    llm_scores_before.append(llm_before)
                    llm_scores_after.append(llm_after)
            
            # Extract semantic scores
            semantic_eval = eval_data.get("semantic_evaluation", {})
            sem_before = semantic_eval.get("answer_a_score")
            sem_after = semantic_eval.get("answer_b_score")
            
            if isinstance(sem_before, (int, float)) and isinstance(sem_after, (int, float)):
                semantic_scores_before.append(sem_before)
                semantic_scores_after.append(sem_after)
        
        # Calculate LLM statistics if available
        llm_stats = {}
        if llm_scores_before and llm_scores_after:
            llm_avg_before = sum(llm_scores_before) / len(llm_scores_before)
            llm_avg_after = sum(llm_scores_after) / len(llm_scores_after)
            
            llm_improvements = sum(1 for a, b in zip(llm_scores_before, llm_scores_after) if b > a)
            llm_regressions = sum(1 for a, b in zip(llm_scores_before, llm_scores_after) if b < a)
            llm_ties = sum(1 for a, b in zip(llm_scores_before, llm_scores_after) if b == a)
            
            llm_stats = {
                "average_score_before": llm_avg_before,
                "average_score_after": llm_avg_after,
                "absolute_improvement": llm_avg_after - llm_avg_before,
                "total_evaluated": len(llm_scores_before),
                "improvements": llm_improvements,
                "regressions": llm_regressions,
                "ties": llm_ties,
                "improvement_percentage": (llm_improvements / len(llm_scores_before)) * 100
            }
        
        # Calculate semantic statistics
        semantic_stats = {}
        if semantic_scores_before and semantic_scores_after:
            sem_avg_before = sum(semantic_scores_before) / len(semantic_scores_before)
            sem_avg_after = sum(semantic_scores_after) / len(semantic_scores_after)
            
            sem_improvements = sum(1 for a, b in zip(semantic_scores_before, semantic_scores_after) if b > a)
            sem_regressions = sum(1 for a, b in zip(semantic_scores_before, semantic_scores_after) if b < a)
            sem_ties = sum(1 for a, b in zip(semantic_scores_before, semantic_scores_after) if b == a)
            
            semantic_stats = {
                "average_score_before": sem_avg_before,
                "average_score_after": sem_avg_after,
                "absolute_improvement": sem_avg_after - sem_avg_before,
                "total_evaluated": len(semantic_scores_before),
                "improvements": sem_improvements,
                "regressions": sem_regressions,
                "ties": sem_ties,
                "improvement_percentage": (sem_improvements / len(semantic_scores_before)) * 100
            }
        
        # Combined statistics
        combined_stats = {
            "llm_evaluation": llm_stats,
            "semantic_evaluation": semantic_stats
        }
        
        # Overall judgment
        if llm_stats and semantic_stats:
            llm_improvement = llm_stats.get("absolute_improvement", 0)
            sem_improvement = semantic_stats.get("absolute_improvement", 0)
            
            # Weighted average (give more weight to LLM evaluation)
            weighted_improvement = (llm_improvement * 0.7) + (sem_improvement * 0.3)
            
            if weighted_improvement > 0.5:
                overall_assessment = "significant_improvement"
            elif weighted_improvement > 0.2:
                overall_assessment = "moderate_improvement"
            elif weighted_improvement > 0:
                overall_assessment = "slight_improvement"
            elif weighted_improvement == 0:
                overall_assessment = "no_change"
            else:
                overall_assessment = "regression"
                
            combined_stats["overall_assessment"] = overall_assessment
            combined_stats["weighted_improvement"] = weighted_improvement
        
        return combined_stats


# Example usage showing how to compare QA generation before/after optimization
if __name__ == "__main__":
    # Load example QA pairs from before optimization
    qa_before_file = "answers_before.json"
    qa_after_file = "answers_after.json"
    
    try:
        with open(qa_before_file, 'r') as f:
            before_data = json.load(f)
        with open(qa_after_file, 'r') as f:
            after_data = json.load(f)
        
        # Extract QA pairs
        qa_before = before_data.get('data', before_data)
        qa_after = after_data.get('data', after_data)
        
        # Initialize evaluator
        evaluator = HybridAnswerEvaluator(
            judge_model="mistralai/Mistral-7B-Instruct-v0.2",  # Use Hugging Face model
            use_llm=True,  # Set to False to use only semantic evaluation
        )
        
        # Evaluate a sample pair
        sample_result = evaluator.evaluate_answer_pair(
            qa_before[0]["question"],
            qa_before[0]["answer"],
            qa_after[0]["answer"]
        )
        
        print("Sample evaluation result:")
        print(json.dumps(sample_result, indent=2))
        
        print("\nNote: For full batch evaluation, use evaluator.batch_evaluate()")
    except FileNotFoundError:
        print(f"Example files not found. This is just a demonstration.")
        print("To use this module in your code:")
        print("1. Generate answers using your original model")
        print("2. Generate answers to the same questions using your optimized model")
        print("3. Compare them with the HybridAnswerEvaluator")