import json
import time
import re
from typing import List, Dict, Any
from datetime import datetime
from llama_index.core import VectorStoreIndex
from llama_index.llms.groq import Groq
from rate_limiter import RateLimiter  # Import the rate limiter

class AnswerGenerator:
    def __init__(
        self,
        index: VectorStoreIndex,
        api_key: str,
        llm_model: str = "llama3-70b-8192",
        max_tokens: int = 200  # Changed from 400 to 200 tokens
    ):
        self.index = index
        self.api_key = api_key
        self.llm_model = llm_model
        self.max_tokens = max_tokens
        
        # Set up LLM with error handling
        try:
            self.llm = Groq(model=llm_model, api_key=api_key)
            # Explicitly set the LLM for LlamaIndex to avoid using OpenAI
            from llama_index.core import Settings
            Settings.llm = self.llm
        except Exception as e:
            print(f"Warning: Error initializing Groq LLM: {e}")
            print("Will attempt to recover when generating answers")
            self.llm = None
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=30,
            requests_per_day=14400,
            tokens_per_minute=6000,
            tokens_per_day=500000
        )
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token for English
        return max(1, len(text) // 4)
    
    def enforce_token_limit(self, answer: str, max_tokens: int = None) -> str:
        """
        Ensure an answer stays within the token limit
        
        Args:
            answer: Original answer text
            max_tokens: Maximum tokens allowed (default: self.max_tokens)
            
        Returns:
            Truncated answer if needed
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
            
        estimated_tokens = self.estimate_tokens(answer)
        
        if estimated_tokens <= max_tokens:
            return answer
        
        # If over the limit, find a good cutoff point
        # Try to keep complete sentences up to the limit
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        result = ""
        
        for sentence in sentences:
            if self.estimate_tokens(result + sentence) <= max_tokens - 3:  # Leave room for "..."
                result += sentence + " "
            else:
                break
        
        return result.strip() + "..."
    
    def enforce_rate_limit(self, text=None, expected_response_tokens=None):
        """
        Advanced rate limiting with token awareness
        
        Args:
            text: The input text (prompt or query)
            expected_response_tokens: Expected tokens in response (default: self.max_tokens)
        """
        if expected_response_tokens is None:
            expected_response_tokens = self.max_tokens
            
        # Calculate estimated tokens
        estimated_tokens = 0
        if text:
            estimated_tokens = self.estimate_tokens(text)
        total_tokens = estimated_tokens + expected_response_tokens
        
        # Wait if needed - now handling tuple return
        result = self.rate_limiter.wait_if_needed(total_tokens)
        # Check if this is the new tuple format
        if isinstance(result, tuple):
            wait_time, api_key = result
            # Update the API key
            self.api_key = api_key
        else:
            # Old format for backward compatibility
            wait_time = result
        
        if wait_time > 0:
            print(f"Rate limit applied: waited {wait_time:.2f}s")
    
    def generate_answer(self, question_item: Dict, context_list: List[str] = None) -> Dict:
        """
        Generate a high-quality answer for a question using RAG
        
        Args:
            question_item: Question dictionary with metadata
            context_list: Optional list to store retrieved contexts for benchmarking
            
        Returns:
            Dictionary with question, answer and metadata
        """
        question_text = question_item["question"]
        question_type = question_item.get("type", "general")
        complexity = question_item.get("complexity", "basic")
        
        try:
            # Configure query engine based on question type and complexity
            if self.index:
                # Make sure we're using Groq for all operations
                from llama_index.core import Settings
                Settings.llm = self.llm
                
                similarity_top_k = 5  # Increased from 3
                if complexity in ["advanced", "complex"] or question_type in ["multi_part", "relationship"]:
                    similarity_top_k = 7  # Increased from 4
                    response_mode = "tree_summarize"
                else:
                    response_mode = "compact"
                
                # Build query engine
                query_engine = self.index.as_query_engine(
                    similarity_top_k=similarity_top_k,
                    response_mode=response_mode
                )
                
                # Create instruction based on question type
                type_instructions = {
                    "factual": "Provide accurate, concrete information with key facts and details.",
                    "conceptual": "Explain the theoretical concepts clearly with focus on principles.",
                    "relationship": "Explore the connections between concepts, showing how they influence each other.",
                    "application": "Show how these concepts apply in practical scenarios with examples.",
                    "evaluative": "Assess the effectiveness, strengths, and limitations of the approaches.",
                    "multi_part": "Address each part of the question separately while maintaining coherence."
                }
                
                instruction = type_instructions.get(question_type, "Provide a focused answer.")
                
                # Add formatting guidance for multi-part questions
                if question_type == "multi_part":
                    instruction += " Structure your answer to clearly address each component."
                
                # Add token limit instruction - NEW
                token_instruction = f" Limit your answer to approximately {self.max_tokens} tokens (about {self.max_tokens * 4} characters)."
                instruction += token_instruction
                
                # Format query with instruction
                query_text = f"{question_text}\n\n{instruction}"
                
                # Apply rate limiting
                self.enforce_rate_limit(query_text, self.max_tokens)
                
                # Query the index
                response = query_engine.query(query_text)
                answer = str(response)
                
                # Collect contexts if requested
                if context_list is not None and hasattr(response, 'source_nodes'):
                    for node in response.source_nodes:
                        context_list.append(node.node.text)
                
                # Check if answer quality is sufficient
                if len(answer.split()) < 30 or "no information" in answer.lower():
                    print(f"Low quality answer detected for: {question_text[:50]}...")
                    
                    # Try with expanded parameters
                    self.enforce_rate_limit(query_text, self.max_tokens)
                    retry_engine = self.index.as_query_engine(
                        similarity_top_k=similarity_top_k + 2,
                        response_mode="tree_summarize"
                    )
                    
                    retry_response = retry_engine.query(
                        f"{question_text}\n\nProvide a comprehensive, detailed answer based on all available information. {token_instruction}"
                    )
                    
                    retry_answer = str(retry_response)
                    
                    # Collect contexts from retry if requested
                    if context_list is not None and hasattr(retry_response, 'source_nodes'):
                        for node in retry_response.source_nodes:
                            if node.node.text not in context_list:
                                context_list.append(node.node.text)
                    
                    # Use retry answer if it's better
                    if len(retry_answer.split()) > len(answer.split()):
                        answer = retry_answer
            else:
                # Fallback if no index available - generate with just the LLM
                print(f"No vector index available, generating answer using only LLM")
                
                # Make sure LLM is initialized
                if self.llm is None:
                    self.llm = Groq(model=self.llm_model, api_key=self.api_key)
                
                # Create instruction based on question type
                instructions = {
                    "factual": "Provide a direct, factual answer with specific details.",
                    "conceptual": "Explain the concept clearly, focusing on core principles.",
                    "relationship": "Explain how these concepts connect and influence each other.",
                    "application": "Show how this concept can be applied with specific examples.",
                    "evaluative": "Assess the effectiveness with balanced consideration of strengths and limitations.",
                    "multi_part": "Address each part of the question separately in your answer."
                }
                
                instruction = instructions.get(question_type, "Provide a clear, comprehensive answer.")
                
                # Add token limit instruction - NEW
                token_instruction = f"Limit your answer to {self.max_tokens} tokens (about {self.max_tokens * 4} characters)."
                
                # Construct the prompt
                prompt = f"""
                Question: {question_text}

                Instructions: {instruction} {token_instruction}

                Answer the question focusing on financial domain expertise. Be concise but thorough, and provide a well-structured response.
                """
                
                # Apply rate limiting
                self.enforce_rate_limit(prompt, self.max_tokens)
                
                # Generate answer
                response = self.llm.complete(prompt)
                answer = str(response).strip()
            
            # Post-process the answer for quality
            answer = self.post_process_answer(answer)
            
            # Create result
            result = {
                "question": question_text,
                "answer": answer,
                "subtopic": question_item.get("subtopic", "general_finance"),
                "token_count": self.estimate_tokens(answer)  # Store token count - NEW
            }
            
            return result
            
        except Exception as e:
            print(f"Error generating answer for '{question_text[:50]}...': {e}")
            time.sleep(5)  # Wait before continuing
            
            # Try to generate a fallback answer using just the LLM
            try:
                if self.llm is None:
                    self.llm = Groq(model=self.llm_model, api_key=self.api_key)
                
                # Include token limit in prompt
                prompt = f"As a financial expert, please answer this question: {question_text}\n\nProvide a concise, informative answer. Limit your response to {self.max_tokens} tokens."
                
                self.enforce_rate_limit(prompt, self.max_tokens)
                
                fallback_response = self.llm.complete(prompt)
                fallback_answer = self.post_process_answer(str(fallback_response).strip())
                
                return {
                    "question": question_text,
                    "answer": fallback_answer,
                    "subtopic": question_item.get("subtopic", "general_finance"),
                    "token_count": self.estimate_tokens(fallback_answer)  # Store token count - NEW
                }
            except Exception as e2:
                print(f"Fallback answer generation also failed: {e2}")
                
                # Return placeholder as last resort
                return {
                    "question": question_text,
                    "answer": "Unable to generate answer due to technical issues.",
                    "subtopic": question_item.get("subtopic", "general_finance"),
                    "token_count": 0  # Zero tokens for error message
                }
    
    def post_process_answer(self, answer: str) -> str:
        """
        Improve answer quality through post-processing
        
        Args:
            answer: Raw answer text
            
        Returns:
            Improved answer text
        """
        # Remove phrases that reference the knowledge base or context
        phrases_to_remove = [
            r"Based on the (?:provided|given) (?:context|information|knowledge base)",
            r"According to the (?:provided|given) (?:context|information|knowledge base)",
            r"The (?:provided|given) (?:context|information|knowledge base) (?:states|mentions|discusses|indicates)",
            r"From the (?:provided|given) (?:context|information|knowledge base)",
            r"In the (?:provided|given) (?:context|information|knowledge base)"
        ]
        
        for phrase in phrases_to_remove:
            answer = re.sub(phrase, "", answer, flags=re.IGNORECASE)
        
        # Fix "unfortunately" phrasing
        answer = re.sub(
            r"(?:Unfortunately|However),\s+(?:the provided context|the context|the information) does not (?:provide|mention|contain)",
            "The information available does not explicitly cover",
            answer,
            flags=re.IGNORECASE
        )
        
        # Remove uncertainty phrases
        uncertainty_phrases = [
            r"It is unclear from the context",
            r"The context does not specify",
            r"It's not clear from the context",
            r"The context does not provide details",
            r"Without more information",
            r"Based on the available information"
        ]
        
        for phrase in uncertainty_phrases:
            answer = re.sub(phrase, "", answer, flags=re.IGNORECASE)
        
        # Clean up whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure the answer starts with a capital letter
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Remove any trailing sentences that reference the context
        sentences = answer.split('. ')
        filtered_sentences = []
        
        for sentence in sentences:
            if not any(phrase in sentence.lower() for phrase in ['the context', 'the provided information']):
                filtered_sentences.append(sentence)
                
        # Reconstruct the answer
        answer = '. '.join(filtered_sentences)
        if not answer.endswith('.') and not answer.endswith('?') and not answer.endswith('!'):
            answer += '.'
        
        # Enforce token limit - NEW
        answer = self.enforce_token_limit(answer, self.max_tokens)
        
        return answer
    
    def generate_all_answers(self, questions: List[Dict], retrieved_contexts: List[List[str]] = None) -> List[Dict]:
        """
        Generate answers for all questions
        
        Args:
            questions: List of question dictionaries
            retrieved_contexts: Optional list to store retrieved contexts for benchmarking
            
        Returns:
            List of QA pairs with metadata
        """
        qa_pairs = []
        total = len(questions)
        
        print(f"Generating answers for {total} questions...")
        
        # Track token statistics
        total_tokens = 0
        min_tokens = float('inf')
        max_tokens = 0
        
        for i, question in enumerate(questions):
            print(f"Processing question {i+1}/{total}: {question['question'][:50]}...")
            
            # If collecting contexts for benchmarking, pass a context list to generate_answer
            context_list = [] if retrieved_contexts is not None else None
            qa_pair = self.generate_answer(question, context_list)
            qa_pairs.append(qa_pair)
            
            # Store retrieved contexts if we're collecting them
            if retrieved_contexts is not None and context_list:
                retrieved_contexts.append(context_list)
            
            # Update token statistics
            token_count = qa_pair.get('token_count', 0)
            total_tokens += token_count
            min_tokens = min(min_tokens, token_count) if token_count > 0 else min_tokens
            max_tokens = max(max_tokens, token_count)
            
            # Progress update
            if (i+1) % 10 == 0:
                print(f"Completed {i+1}/{total} questions")
                avg_tokens = total_tokens / (i+1)
                print(f"Token statistics so far: Avg={avg_tokens:.1f}, Min={min_tokens}, Max={max_tokens}")
                
                # Save intermediate progress
                self.save_progress(qa_pairs, f"progress_{i+1}.json")
            
            # Add delay between requests to avoid rate limits
            time.sleep(2)
        
        # Print final token statistics
        if len(qa_pairs) > 0:
            avg_tokens = total_tokens / len(qa_pairs)
            print(f"\nFinal token statistics:")
            print(f"Average tokens per answer: {avg_tokens:.1f}")
            print(f"Min tokens: {min_tokens}")
            print(f"Max tokens: {max_tokens}")
            print(f"Total tokens used: {total_tokens}")
        
        return qa_pairs
    
    def save_progress(self, qa_pairs: List[Dict], filename: str):
        """Save intermediate progress"""
        output = {
            "metadata": {
                "model": self.llm_model,
                "timestamp": datetime.now().isoformat(),
                "count": len(qa_pairs),
                "status": "in_progress",
                "max_tokens": self.max_tokens  # Add token limit info - NEW
            },
            "data": qa_pairs
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
                
            print(f"Saved progress to {filename}")
        except Exception as e:
            print(f"Error saving progress: {e}")
    
    def save_dataset(self, qa_pairs: List[Dict], output_file: str):
        """
        Save the final dataset to a JSON file with simplified format
        
        Args:
            qa_pairs: List of QA pairs
            output_file: Output file path
        """
        # Calculate token statistics
        total_tokens = sum(pair.get('token_count', 0) for pair in qa_pairs)
        token_counts = [pair.get('token_count', 0) for pair in qa_pairs if pair.get('token_count', 0) > 0]
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        
        # Format for output
        output = {
            "metadata": {
                "model": self.llm_model,
                "generated_at": datetime.now().isoformat(),
                "count": len(qa_pairs),
                "domain": "finance",
                "max_token_limit": self.max_tokens,  # Add token limit info - NEW
                "token_statistics": {
                    "total": total_tokens,
                    "average": avg_tokens,
                    "max": max_tokens,
                    "min": min_tokens
                }
            },
            "data": qa_pairs
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
                
            print(f"Saved {len(qa_pairs)} QA pairs to {output_file}")
            print(f"Token statistics: Avg={avg_tokens:.1f}, Min={min_tokens}, Max={max_tokens}, Total={total_tokens}")
        except Exception as e:
            print(f"Error saving dataset: {e}")
            
            # Try to save with a different filename as backup
            backup_file = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"Saved backup to {backup_file}")
            except:
                print("Failed to save backup")