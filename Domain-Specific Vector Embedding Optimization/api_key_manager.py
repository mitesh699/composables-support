import os
import json
import time
import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_key_manager")

class APIKeyManager:
    """
    Manages multiple Groq API keys with rotation to avoid rate limiting
    
    Features:
    - Store and rotate multiple API keys
    - Track usage and cooldown periods
    - Automatic fallback when hitting rate limits
    """
    
    def __init__(
        self,
        api_keys: List[str] = None,
        status_file: str = "api_key_status.json",
        cooldown_minutes: int = 60
    ):
        """
        Initialize the API key manager
        
        Args:
            api_keys: List of API keys (if None, will check environment variables)
            status_file: File to store key status
            cooldown_minutes: Minutes to wait before using a rate-limited key again
        """
        self.status_file = status_file
        self.cooldown_minutes = cooldown_minutes
        
        # Initialize key registry
        self.key_registry = {}
        
        # Load API keys from parameters or environment variables
        self._load_api_keys(api_keys)
        
        # Load existing status if available
        self._load_status()
        
        logger.info(f"Initialized API Key Manager with {len(self.key_registry)} Groq keys")
    
    def _load_api_keys(self, api_keys: Optional[List[str]]):
        """Load API keys from parameters or environment variables"""
        if api_keys and len(api_keys) > 0:
            # Use provided keys
            for i, key in enumerate(api_keys):
                if key and key.strip():
                    self.key_registry[f"key_{i}"] = {
                        "key": key,
                        "active": True,
                        "last_used": None,
                        "cooldown_until": None,
                        "failure_count": 0
                    }
        else:
            # Load from environment variables (GROQ_API_KEY_1, GROQ_API_KEY_2, etc.)
            for i in range(1, 10):  # Try up to 9 keys
                env_var = f"GROQ_API_KEY_{i}"
                if env_var in os.environ and os.environ[env_var].strip():
                    self.key_registry[f"key_{i-1}"] = {
                        "key": os.environ[env_var],
                        "active": True,
                        "last_used": None,
                        "cooldown_until": None,
                        "failure_count": 0
                    }
            
            # Also check the default GROQ_API_KEY
            default_env_var = "GROQ_API_KEY"
            if default_env_var in os.environ and os.environ[default_env_var].strip():
                self.key_registry["key_default"] = {
                    "key": os.environ[default_env_var],
                    "active": True,
                    "last_used": None,
                    "cooldown_until": None,
                    "failure_count": 0
                }
        
        # Check if we have any keys
        if not self.key_registry:
            logger.warning(f"No Groq API keys found. Please add them to your .env file as GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")
    
    def _load_status(self):
        """Load the status of keys from file if available"""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    stored_status = json.load(f)
                
                # Update existing keys with stored status
                if "groq" in stored_status:
                    provider_status = stored_status["groq"]
                    for key_id, key_info in self.key_registry.items():
                        # Use key as identifier since key_id might change
                        api_key = key_info["key"]
                        for stored_key_id, stored_info in provider_status.items():
                            if stored_info.get("key") == api_key:
                                # Update relevant fields
                                key_info["active"] = stored_info.get("active", True)
                                key_info["failure_count"] = stored_info.get("failure_count", 0)
                                
                                # Parse timestamps if they exist
                                if stored_info.get("last_used"):
                                    key_info["last_used"] = datetime.fromisoformat(stored_info["last_used"])
                                
                                if stored_info.get("cooldown_until"):
                                    cooldown = datetime.fromisoformat(stored_info["cooldown_until"])
                                    # Only keep cooldown if it's in the future
                                    if cooldown > datetime.now():
                                        key_info["cooldown_until"] = cooldown
                                    else:
                                        key_info["cooldown_until"] = None
                
                logger.info(f"Loaded status for {len(self.key_registry)} keys")
            except Exception as e:
                logger.error(f"Error loading key status: {e}")
    
    def _save_status(self):
        """Save the current status of all keys to file"""
        try:
            # Load existing file if it exists
            stored_status = {}
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    stored_status = json.load(f)
            
            # Prepare status for Groq provider
            provider_status = {}
            for key_id, key_info in self.key_registry.items():
                # Convert datetime objects to ISO format strings
                info_copy = key_info.copy()
                if info_copy.get("last_used"):
                    info_copy["last_used"] = info_copy["last_used"].isoformat()
                if info_copy.get("cooldown_until"):
                    info_copy["cooldown_until"] = info_copy["cooldown_until"].isoformat()
                
                provider_status[key_id] = info_copy
            
            # Update the stored status with Groq provider
            stored_status["groq"] = provider_status
            
            # Save to file
            with open(self.status_file, 'w') as f:
                json.dump(stored_status, f, indent=2)
            
            logger.debug(f"Saved key status to {self.status_file}")
        except Exception as e:
            logger.error(f"Error saving key status: {e}")
    
    def get_active_key(self) -> Optional[str]:
        """
        Get an active API key, rotating if necessary
        
        Returns:
            An active API key, or None if no active keys available
        """
        now = datetime.now()
        
        # Update status of keys in cooldown
        for key_id, key_info in self.key_registry.items():
            if not key_info["active"] and key_info["cooldown_until"] and now > key_info["cooldown_until"]:
                logger.info(f"Key {key_id[:5]}... is out of cooldown and now active")
                key_info["active"] = True
                key_info["cooldown_until"] = None
        
        # Get all active keys
        active_keys = [
            key_id for key_id, key_info in self.key_registry.items()
            if key_info["active"]
        ]
        
        if not active_keys:
            logger.warning("No active Groq API keys available!")
            return None
        
        # Create a safer sort function that handles None values properly
        def safe_sort_key(k):
            last_used = self.key_registry[k].get("last_used")
            return datetime.min if last_used is None else last_used
        
        # Sort using our safe sort function
        active_keys.sort(key=safe_sort_key)
        
        selected_key_id = active_keys[0]
        selected_key = self.key_registry[selected_key_id]["key"]
        
        # Update last used time
        self.key_registry[selected_key_id]["last_used"] = now
        self._save_status()
        
        logger.info(f"Using Groq API key: {selected_key_id[:5]}...")
        return selected_key
    
    def report_success(self, api_key: str):
        """
        Report successful API call with a key
        
        Args:
            api_key: The API key that was used successfully
        """
        # Find the key in the registry
        for key_id, key_info in self.key_registry.items():
            if key_info["key"] == api_key:
                # Reset failure count on success
                if key_info["failure_count"] > 0:
                    key_info["failure_count"] = 0
                    self._save_status()
                return
    
    def report_failure(self, api_key: str, rate_limited: bool = False):
        """
        Report a failed API call, optionally specifying if it was rate limited
        
        Args:
            api_key: The API key that failed
            rate_limited: Whether the failure was due to rate limiting
        """
        now = datetime.now()
        
        # Find the key in the registry
        for key_id, key_info in self.key_registry.items():
            if key_info["key"] == api_key:
                if rate_limited:
                    # Key is rate limited, put it in cooldown
                    logger.warning(f"Key {key_id[:5]}... is rate limited, putting in cooldown")
                    key_info["active"] = False
                    key_info["cooldown_until"] = now + timedelta(minutes=self.cooldown_minutes)
                else:
                    # Increment failure count
                    key_info["failure_count"] += 1
                    logger.warning(f"Key {key_id[:5]}... failed, failure count: {key_info['failure_count']}")
                    
                    # If too many failures, put in cooldown
                    if key_info["failure_count"] >= 3:
                        logger.warning(f"Key {key_id[:5]}... has too many failures, putting in cooldown")
                        key_info["active"] = False
                        key_info["cooldown_until"] = now + timedelta(minutes=self.cooldown_minutes // 2)
                
                self._save_status()
                return
    
    def add_key(self, api_key: str) -> bool:
        """
        Add a new API key to the manager
        
        Args:
            api_key: The API key to add
            
        Returns:
            True if added successfully, False if the key already exists
        """
        # Check if key already exists
        for key_info in self.key_registry.values():
            if key_info["key"] == api_key:
                logger.warning(f"API key already exists in registry")
                return False
        
        # Add new key
        new_key_id = f"key_{len(self.key_registry)}"
        self.key_registry[new_key_id] = {
            "key": api_key,
            "active": True,
            "last_used": None,
            "cooldown_until": None,
            "failure_count": 0
        }
        
        logger.info(f"Added new Groq API key with ID {new_key_id[:5]}...")
        self._save_status()
        return True
    
    def get_status_summary(self) -> Dict:
        """
        Get a summary of the current key status
        
        Returns:
            Dictionary with key status summary
        """
        total_keys = len(self.key_registry)
        active_keys = sum(1 for info in self.key_registry.values() if info["active"])
        
        return {
            "total_keys": total_keys,
            "active_keys": active_keys,
            "inactive_keys": total_keys - active_keys,
            "keys": [
                {
                    "id": key_id,
                    "active": info["active"],
                    "failure_count": info["failure_count"],
                    "last_used": info["last_used"].isoformat() if info["last_used"] else None,
                    "cooldown_until": info["cooldown_until"].isoformat() if info["cooldown_until"] else None
                }
                for key_id, info in self.key_registry.items()
            ]
        }


# Integration with the RateLimiter
class RotatingAPIRateLimiter:
    """
    Enhanced rate limiter with API key rotation capabilities
    
    Extends the RateLimiter functionality with key rotation to handle multiple API keys.
    """
    
    def __init__(
        self,
        key_manager: APIKeyManager,
        requests_per_minute: int = 30,
        requests_per_day: int = 14400,
        tokens_per_minute: int = 6000,
        tokens_per_day: int = 500000,
        safety_factor: float = 0.9
    ):
        """
        Initialize with key manager and rate limits
        
        Args:
            key_manager: API key manager for key rotation
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day
            tokens_per_minute: Maximum tokens per minute
            tokens_per_day: Maximum tokens per day
            safety_factor: Percentage of limits to use
        """
        # Import the base RateLimiter for composition
        from rate_limiter import RateLimiter
        
        self.key_manager = key_manager
        self.current_key = None
        
        # Create a RateLimiter instance
        self.rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            requests_per_day=requests_per_day,
            tokens_per_minute=tokens_per_minute,
            tokens_per_day=tokens_per_day,
            safety_factor=safety_factor
        )
        
        # Track rate limit errors to trigger key rotation
        self.consecutive_rate_limit_errors = 0
        
        logger.info("Initialized RotatingAPIRateLimiter for Groq API")
    
    def get_current_key(self) -> Optional[str]:
        """
        Get the current active API key, rotating if necessary
        
        Returns:
            Current API key or None if no keys available
        """
        # If we have a rate limit error or no current key, get a new one
        if self.consecutive_rate_limit_errors > 0 or self.current_key is None:
            self.current_key = self.key_manager.get_active_key()
            self.consecutive_rate_limit_errors = 0
        
        return self.current_key
    
    def rotate_key(self, rate_limited: bool = True):
        """
        Force rotation to a new API key
        
        Args:
            rate_limited: Whether rotation is due to rate limiting
        """
        if self.current_key:
            # Report the issue with the current key
            self.key_manager.report_failure(self.current_key, rate_limited)
            
            # Get a new key
            old_key = self.current_key
            self.current_key = self.key_manager.get_active_key()
            
            if self.current_key == old_key:
                logger.warning("No alternative keys available, using the same key")
            else:
                logger.info("Rotated to a new Groq API key")
        else:
            # No current key, just get a new one
            self.current_key = self.key_manager.get_active_key()
    
    def wait_if_needed(self, tokens: int = 0) -> Tuple[float, str]:
        """
        Check rate limits and wait if necessary before allowing a request.
        
        Args:
            tokens: Number of tokens in the request
            
        Returns:
            Tuple of (wait time in seconds, API key to use)
        """
        wait_time = 0
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            # Get current API key
            api_key = self.get_current_key()
            
            if not api_key:
                logger.error("No Groq API keys available")
                raise ValueError("No Groq API keys available for use")
            
            try:
                # Try waiting with the current rate limiter
                wait_time = self.rate_limiter.wait_if_needed(tokens)
                
                # If successful, reset errors
                self.consecutive_rate_limit_errors = 0
                
                # Report success to key manager
                self.key_manager.report_success(api_key)
                
                # Return wait time and API key
                return wait_time, api_key
                
            except Exception as e:
                logger.warning(f"Rate limit error: {e}")
                
                # Increment error counter
                self.consecutive_rate_limit_errors += 1
                
                # Rotate key if too many errors
                self.rotate_key(rate_limited=True)
                retries += 1
                
                # Add a small delay before retry
                time.sleep(1)
        
        # If we get here, we've exhausted our retries
        logger.error("Rate limit errors with all keys, giving up")
        raise RuntimeError("Unable to find an available Groq API key after multiple attempts")
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text string
        
        Args:
            text: The input text
            
        Returns:
            Estimated token count
        """
        return self.rate_limiter.estimate_tokens(text)


# Setup instructions for .env file
SETUP_INSTRUCTIONS = """
To set up multiple Groq API keys for rotation, create a .env file with the following format:

```
GROQ_API_KEY_1=your-first-api-key
GROQ_API_KEY_2=your-second-api-key
GROQ_API_KEY_3=your-third-api-key
...
```

You can also include the standard GROQ_API_KEY which will be used as well:

```
GROQ_API_KEY=your-default-api-key
```

The API Key Manager will automatically rotate between these keys when rate limits are encountered.
"""

# Example use
if __name__ == "__main__":
    # Print setup instructions
    print(SETUP_INSTRUCTIONS)
    
    # Test the key manager
    manager = APIKeyManager()
    key = manager.get_active_key()
    
    if key:
        print(f"Got active key: {key[:5]}...")
        
        # Test reporting
        manager.report_success(key)
        
        # Test rotation
        manager.report_failure(key, rate_limited=True)
        new_key = manager.get_active_key()
        print(f"After failure, got key: {new_key[:5] if new_key else 'None'}...")
    else:
        print("No Groq API keys available. Set environment variables like GROQ_API_KEY_1, GROQ_API_KEY_2, etc.")