import time
import logging
import threading
from typing import Optional, Dict
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rate_limiter")

class RateLimiter:
    """
    Advanced rate limiter for LLM API calls
    
    Features:
    - Multiple constraint levels (requests per minute, requests per day)
    - Token-aware rate limiting
    - Thread-safe implementation
    - Automatic wait times
    - Detailed logging
    
    Usage:
        rate_limiter = RateLimiter()
        # Before making API call
        rate_limiter.wait_if_needed(estimated_tokens)
        # Make API call
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        requests_per_day: int = 14400,
        tokens_per_minute: int = 6000,
        tokens_per_day: int = 500000,
        safety_factor: float = 0.9  # Use 90% of limits to be safe
    ):
        """
        Initialize rate limiter with configured limits
        
        Args:
            requests_per_minute: Maximum requests per minute (default: 30)
            requests_per_day: Maximum requests per day (default: 14400)
            tokens_per_minute: Maximum tokens per minute (default: 6000)
            tokens_per_day: Maximum tokens per day (default: 500000)
            safety_factor: Percentage of limits to use (default: 0.9 = 90%)
        """
        self.rpm_limit = int(requests_per_minute * safety_factor)
        self.rpd_limit = int(requests_per_day * safety_factor)
        self.tpm_limit = int(tokens_per_minute * safety_factor) 
        self.tpd_limit = int(tokens_per_day * safety_factor)
        
        # Track request timestamps within the last minute and day
        self.minute_requests = deque()
        self.day_requests = deque()
        
        # Track token usage
        self.minute_tokens = deque()
        self.day_tokens = deque()
        
        # For thread safety
        self.lock = threading.RLock()
        
        # Calculate time intervals
        self.minute_interval = timedelta(minutes=1)
        self.day_interval = timedelta(days=1)
        
        logger.info(f"Rate limiter initialized with limits: "
                   f"{self.rpm_limit} RPM, {self.rpd_limit} RPD, "
                   f"{self.tpm_limit} TPM, {self.tpd_limit} TPD")
    
    def _clean_old_entries(self, now: datetime):
        """
        Remove entries older than the time windows
        
        Args:
            now: Current datetime
        """
        with self.lock:
            # Clean minute windows
            minute_cutoff = now - self.minute_interval
            while self.minute_requests and self.minute_requests[0] < minute_cutoff:
                self.minute_requests.popleft()
            
            while self.minute_tokens and self.minute_tokens[0][0] < minute_cutoff:
                self.minute_tokens.popleft()
            
            # Clean day windows
            day_cutoff = now - self.day_interval
            while self.day_requests and self.day_requests[0] < day_cutoff:
                self.day_requests.popleft()
                
            while self.day_tokens and self.day_tokens[0][0] < day_cutoff:
                self.day_tokens.popleft()
    
    def _get_current_usage(self) -> Dict[str, int]:
        """
        Get current usage counts
        
        Returns:
            Dictionary with current usage counts
        """
        with self.lock:
            # Count requests
            rpm_count = len(self.minute_requests)
            rpd_count = len(self.day_requests)
            
            # Sum tokens
            tpm_count = sum(tokens for _, tokens in self.minute_tokens)
            tpd_count = sum(tokens for _, tokens in self.day_tokens)
            
            return {
                "rpm": rpm_count,
                "rpd": rpd_count,
                "tpm": tpm_count,
                "tpd": tpd_count
            }
    
    def wait_if_needed(self, tokens: int = 0) -> float:
        """
        Check rate limits and wait if necessary before allowing a request.
        
        Args:
            tokens: Number of tokens in the request (prompt + expected completion)
            
        Returns:
            Wait time in seconds (0 if no wait was needed)
        """
        now = datetime.now()
        wait_time = 0
        
        with self.lock:
            # Clean old entries
            self._clean_old_entries(now)
            
            # Get current usage
            usage = self._get_current_usage()
            
            # Check if we're at any limit
            at_rpm_limit = usage["rpm"] >= self.rpm_limit
            at_rpd_limit = usage["rpd"] >= self.rpd_limit
            at_tpm_limit = tokens > 0 and (usage["tpm"] + tokens) >= self.tpm_limit
            at_tpd_limit = tokens > 0 and (usage["tpd"] + tokens) >= self.tpd_limit
            
            # Determine if and how long to wait
            if at_rpm_limit or at_tpm_limit:
                # Wait until the oldest minute entry expires
                if self.minute_requests:
                    oldest = self.minute_requests[0]
                    expires_at = oldest + self.minute_interval
                    wait_seconds = max(0, (expires_at - now).total_seconds())
                    wait_seconds += 0.1  # Add a small buffer
                    wait_time = wait_seconds
                    
                    if at_rpm_limit:
                        logger.warning(f"RPM limit reached ({usage['rpm']}/{self.rpm_limit}). Waiting {wait_seconds:.2f}s.")
                    if at_tpm_limit:
                        logger.warning(f"TPM limit reached ({usage['tpm'] + tokens}/{self.tpm_limit}). Waiting {wait_seconds:.2f}s.")
            
            if at_rpd_limit or at_tpd_limit:
                # This is more serious - we're hitting daily limits
                if self.day_requests:
                    oldest = self.day_requests[0]
                    expires_at = oldest + self.day_interval
                    wait_seconds = max(0, (expires_at - now).total_seconds())
                    
                    if at_rpd_limit:
                        logger.error(f"RPD limit reached ({usage['rpd']}/{self.rpd_limit})! Would need to wait {wait_seconds:.2f}s.")
                    if at_tpd_limit:
                        logger.error(f"TPD limit reached ({usage['tpd'] + tokens}/{self.tpd_limit})! Would need to wait {wait_seconds:.2f}s.")
                    
                    # For daily limits, sleep a shorter time than the full wait
                    # and then pause again later, to avoid extremely long waits
                    wait_time = min(wait_seconds, 60)  # Wait at most a minute
                    logger.warning(f"Hitting daily limits. Waiting {wait_time}s before retrying.")
        
        # Wait if needed
        if wait_time > 0:
            time.sleep(wait_time)
            
            # Recursive call after waiting to make sure we're under limits
            # This handles cases where multiple threads are waiting
            return wait_time + self.wait_if_needed(tokens)
        
        # If we're good to go, record this request
        with self.lock:
            now = datetime.now()  # Get fresh timestamp after any waiting
            self.minute_requests.append(now)
            self.day_requests.append(now)
            
            if tokens > 0:
                self.minute_tokens.append((now, tokens))
                self.day_tokens.append((now, tokens))
        
        return wait_time
    
    def estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate the number of tokens in a text string.
        This is an approximation - GPT/LLaMA models use subword tokenization.
        
        Args:
            text: The input text
            
        Returns:
            Estimated token count
        """
        # Very rough approximation: ~4 chars per token for English
        return max(1, len(text) // 4)
    
    def estimate_tokens_with_completion(self, text: str, expected_completion_tokens: int = 200) -> int:
        """
        Estimate tokens for a full request including completion
        
        Args:
            text: The input prompt text
            expected_completion_tokens: Expected number of tokens in completion
            
        Returns:
            Estimated total token count
        """
        prompt_tokens = self.estimate_tokens(text)
        return prompt_tokens + expected_completion_tokens


# Example integration with question/answer generators:
"""
from rate_limiter import RateLimiter

class QuestionGenerator:
    def __init__(self, ...):
        # Create rate limiter
        self.rate_limiter = RateLimiter()
        
    def generate_questions(self, ...):
        # Before making API call
        prompt = self.create_question_prompt(...)
        
        # Estimate tokens and apply rate limiting
        estimated_prompt_tokens = self.rate_limiter.estimate_tokens(prompt)
        expected_completion_tokens = 200  # For generated questions
        
        wait_time = self.rate_limiter.wait_if_needed(
            estimated_prompt_tokens + expected_completion_tokens
        )
        
        if wait_time > 0:
            print(f"Applied rate limiting: waited {wait_time:.2f}s")
            
        # Make API call
        response = self.llm.complete(prompt)
        
        # Continue processing response...
"""

# Standalone usage example:
if __name__ == "__main__":
    # Create rate limiter
    limiter = RateLimiter(
        requests_per_minute=30,
        requests_per_day=14400,
        tokens_per_minute=6000, 
        tokens_per_day=500000
    )
    
    # Simulate API calls
    for i in range(40):  # Try to exceed RPM limit
        prompt = f"This is test prompt {i}" * 20  # Longer prompt
        estimated_tokens = limiter.estimate_tokens_with_completion(prompt)
        
        wait_time = limiter.wait_if_needed(estimated_tokens)
        
        if wait_time > 0:
            print(f"Test {i}: Rate limited, waited {wait_time:.2f}s")
        else:
            print(f"Test {i}: API call would be made now with {estimated_tokens} tokens")
            
        # Simulate shorter API calls to demonstrate the effect
        time.sleep(0.1)