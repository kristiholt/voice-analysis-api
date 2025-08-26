"""
Utility functions for authentication, hashing, and ID generation.
"""

import hashlib
import secrets
import os
from typing import Optional, Tuple
import time


def generate_request_id() -> str:
    """Generate unique request ID in format vx_<token>"""
    token = secrets.token_urlsafe(16)
    return f"vx_{token}"


def hash_api_key(raw_key: str) -> str:
    """Generate SHA256 hash of raw API key"""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def compute_audio_hash(audio_bytes: bytes) -> str:
    """Compute SHA256 hash of audio file content"""
    return hashlib.sha256(audio_bytes).hexdigest()


def parse_auth_header(authorization: Optional[str]) -> Optional[str]:
    """
    Parse Authorization header and extract bearer token.
    
    Args:
        authorization: Authorization header value
        
    Returns:
        Raw token if valid Bearer format, None otherwise
    """
    if not authorization:
        return None
    
    parts = authorization.split(' ')
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]


def get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Get environment variable with optional default and required validation.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required
        
    Returns:
        Environment variable value
        
    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


def timing_decorator(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = int((end_time - start_time) * 1000)  # Convert to milliseconds
        return result, execution_time
    return wrapper


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """Validate file size against maximum limit"""
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove directory traversal attempts and keep only safe characters
    import re
    safe_name = re.sub(r'[^\w\-_\.]', '_', filename)
    return safe_name[:100]  # Limit length


def create_test_api_key() -> Tuple[str, str]:
    """
    Generate a test API key pair for development.
    
    Returns:
        Tuple of (raw_token, hashed_token)
    """
    raw_token = secrets.token_urlsafe(32)
    hashed_token = hash_api_key(raw_token)
    return raw_token, hashed_token
