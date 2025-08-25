"""
Supabase client and database operations.
Handles API keys, request logging, results storage, and normalization baselines.
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
import logging

from .utils import get_env_var

logger = logging.getLogger(__name__)


class SupabaseStorage:
    """Supabase database operations client"""
    
    def __init__(self):
        self.url = get_env_var("SUPABASE_URL", default="")
        self.service_key = get_env_var("SUPABASE_SERVICE_KEY", default="")
        self.dev_mode = not (self.url and self.service_key)
        
        if self.dev_mode:
            logger.warning("Running in development mode without Supabase - database operations will be mocked")
            self.client = None
        else:
            self.client: Client = create_client(self.url, self.service_key)
    
    async def verify_api_key(self, key_hash: str) -> Optional[Dict[str, Any]]:
        """
        Verify API key hash against database.
        
        Args:
            key_hash: SHA256 hash of the raw API key
            
        Returns:
            API key record if valid, None otherwise
        """
        if self.dev_mode:
            # Mock API key validation in development
            logger.info(f"Mock: Validating API key hash {key_hash[:8]}...")
            return {
                'id': 'dev-key-1',
                'key_hash': key_hash,
                'label': 'Development Key',
                'is_active': True,
                'rate_limit': 1000
            }
        
        try:
            response = self.client.table('api_keys').select('*').eq('key_hash', key_hash).eq('is_active', True).execute()
            
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None
    
    async def log_request(self, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Log API request to database.
        
        Args:
            request_data: Request information
            
        Returns:
            Inserted record ID or None on error
        """
        if self.dev_mode:
            logger.info(f"Mock: Logging request {request_data.get('request_id', 'unknown')}")
            return f"req_{request_data.get('request_id', 'unknown')}"
        
        try:
            response = self.client.table('requests').insert(request_data).execute()
            if response.data:
                return response.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error logging request: {e}")
            return None
    
    async def store_result(self, result_data: Dict[str, Any]) -> Optional[str]:
        """
        Store analysis result to database.
        
        Args:
            result_data: Analysis result data
            
        Returns:
            Inserted record ID or None on error
        """
        if self.dev_mode:
            logger.info(f"Mock: Storing result for {result_data.get('request_id', 'unknown')}")
            return f"result_{result_data.get('request_id', 'unknown')}"
        
        try:
            response = self.client.table('results').insert(result_data).execute()
            if response.data:
                return response.data[0]['id']
            return None
        except Exception as e:
            logger.error(f"Error storing result: {e}")
            return None
    
    async def get_cached_result(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result by content hash.
        
        Args:
            content_hash: SHA256 hash of audio content
            
        Returns:
            Cached result data or None if not found
        """
        if self.dev_mode:
            logger.info(f"Mock: Checking cache for {content_hash[:8]}... (no cache in dev mode)")
            return None
        
        try:
            # Check cache TTL
            cache_ttl_seconds = int(get_env_var("CACHE_TTL_SECONDS", "2592000") or "2592000")
            
            response = self.client.table('results').select('*').eq('content_hash', content_hash).order('created_at', desc=True).limit(1).execute()
            
            if response.data:
                result = response.data[0]
                
                # Check if result is within cache TTL
                created_at = datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                age_seconds = (now - created_at).total_seconds()
                
                if age_seconds <= cache_ttl_seconds:
                    return result
            
            return None
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    async def get_normalization_baselines(self, scheme: str) -> Optional[Dict[str, Any]]:
        """
        Get normalization baselines for the specified scheme.
        
        Args:
            scheme: Normalization scheme name
            
        Returns:
            Baseline data or None if not found
        """
        if self.dev_mode:
            logger.info(f"Mock: Getting normalization baselines for {scheme} (no baselines in dev mode)")
            return None
        
        try:
            response = self.client.table('normalization_baselines').select('*').eq('scheme', scheme).order('updated_at', desc=True).limit(1).execute()
            
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting normalization baselines: {e}")
            return None
    
    async def upsert_normalization_baselines(self, baseline_data: Dict[str, Any]) -> bool:
        """
        Upsert normalization baseline data.
        
        Args:
            baseline_data: Baseline statistics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.table('normalization_baselines').upsert(baseline_data).execute()
            return bool(response.data)
        except Exception as e:
            logger.error(f"Error upserting normalization baselines: {e}")
            return False
    
    async def store_audio_blob(self, content_hash: str, audio_data: bytes, filename: str) -> Optional[str]:
        """
        Store audio blob in Supabase storage (if enabled).
        
        Args:
            content_hash: Content hash for the file
            audio_data: Raw audio bytes
            filename: Original filename
            
        Returns:
            Storage path or None on error
        """
        store_audio = get_env_var("STORE_AUDIO", "false").lower() == "true"
        if not store_audio:
            return None
        
        try:
            bucket = get_env_var("AUDIO_BUCKET", "voice-uploads")
            file_path = f"{content_hash[:2]}/{content_hash[2:4]}/{content_hash}.wav"
            
            response = self.client.storage.from_(bucket).upload(file_path, audio_data)
            
            if response:
                # Log to audio_blobs table
                blob_data = {
                    'content_hash': content_hash,
                    'storage_path': file_path,
                    'original_filename': filename,
                    'file_size': len(audio_data),
                    'bucket': bucket
                }
                self.client.table('audio_blobs').insert(blob_data).execute()
                return file_path
            
            return None
        except Exception as e:
            logger.error(f"Error storing audio blob: {e}")
            return None
    
    async def get_recent_results_for_normalization(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent results for normalization baseline computation.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent result records
        """
        try:
            response = self.client.table('results').select('scores, created_at').gte('created_at', f'now() - interval \'{days} days\'').execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            return []


# Global storage instance
storage = SupabaseStorage()
