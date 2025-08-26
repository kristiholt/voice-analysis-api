"""
Utility functions for authentication, hashing, ID generation, and audio downloads.
"""

import hashlib
import secrets
import os
from typing import Optional, Tuple
import time
import boto3
import pandas as pd
import logging

logger = logging.getLogger(__name__)


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


def download_audio_from_csv(csv_path: str, output_dir: str, bucket_name: str = 'vibeonix-production') -> bool:
    """
    Download audio files from S3 bucket using a CSV file with metadata.
    
    Args:
        csv_path: Path to CSV file containing S3_url and voice_file_name columns
        output_dir: Directory to save downloaded audio files
        bucket_name: S3 bucket name (default: 'vibeonix-production')
        
    Returns:
        True if all downloads succeeded, False otherwise
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV is missing required columns
    """
    try:
        # Validate CSV file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV and validate columns
        df = pd.read_csv(csv_path)
        required_columns = ['S3_url', 'voice_file_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        total_files = len(df)
        successful_downloads = 0
        failed_downloads = []
        
        logger.info(f"Starting download of {total_files} audio files from {bucket_name}")
        
        for index, row in df.iterrows():
            filename = None
            try:
                filename = str(row['voice_file_name'])
                safe_filename = sanitize_filename(filename)
                output_path = os.path.join(output_dir, safe_filename)
                
                # Skip if file already exists
                if os.path.exists(output_path):
                    logger.debug(f"File already exists, skipping: {safe_filename}")
                    successful_downloads += 1
                    continue
                
                # Download from S3
                s3_key = f'voices/{filename}'
                s3.download_file(bucket_name, s3_key, output_path)
                
                logger.debug(f"Downloaded: {safe_filename}")
                successful_downloads += 1
                
            except Exception as e:
                filename_str = filename or f"row_{index}"
                error_msg = f"Failed to download {filename_str}: {str(e)}"
                logger.error(error_msg)
                failed_downloads.append(error_msg)
                continue
        
        # Log summary
        success_rate = (successful_downloads / total_files) * 100
        logger.info(f"Download complete: {successful_downloads}/{total_files} files ({success_rate:.1f}%)")
        
        if failed_downloads:
            logger.warning(f"Failed downloads ({len(failed_downloads)}):")
            for error in failed_downloads:
                logger.warning(f"  - {error}")
        
        return len(failed_downloads) == 0
        
    except Exception as e:
        logger.error(f"Error in download_audio_from_csv: {str(e)}")
        return False
