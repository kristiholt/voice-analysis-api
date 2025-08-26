"""
Supabase client and database operations.
Handles API keys, request logging, results storage, and normalization baselines.
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Union
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
            self.client: Optional[Client] = None
        else:
            # Ensure we have valid non-empty strings before creating client
            if self.url and self.service_key:
                self.client: Optional[Client] = create_client(self.url, self.service_key)
            else:
                logger.error("Invalid Supabase credentials - missing URL or service key")
                self.client = None
                self.dev_mode = True
    
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
        
        if not self.client:
            logger.error("Cannot verify API key: Supabase client not available")
            return None
            
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
        
        if not self.client:
            logger.error("Cannot log request: Supabase client not available")
            return None
            
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
        
        if not self.client:
            logger.error("Cannot store result: Supabase client not available")
            return None
            
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
        
        if not self.client:
            logger.error("Cannot get cached result: Supabase client not available")
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
        
        if not self.client:
            logger.error("Cannot get normalization baselines: Supabase client not available")
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
        if not self.client:
            logger.error("Cannot upsert normalization baselines: Supabase client not available")
            return False
            
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
        store_audio_val = get_env_var("STORE_AUDIO", "false")
        store_audio = (store_audio_val or "false").lower() == "true"
        if not store_audio:
            return None
        
        try:
            bucket = get_env_var("AUDIO_BUCKET", "voice-uploads") or "voice-uploads"
            file_path = f"{content_hash[:2]}/{content_hash[2:4]}/{content_hash}.wav"
            
            if not self.client:
                logger.error("Cannot upload audio blob: Supabase client not available")
                return None
                
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
        if self.dev_mode:
            logger.info(f"Mock: Getting recent results for normalization (dev mode)")
            return []
        
        if not self.client:
            logger.error("Cannot get recent results: Supabase client not available")
            return []
            
        try:
            response = self.client.table('results').select('scores, created_at').gte('created_at', f'now() - interval \'{days} days\'').execute()
            return response.data or []
        except Exception as e:
            logger.error(f"Error getting recent results: {e}")
            return []
    
    async def store_recording(self, recording_data: Dict[str, Any]) -> Optional[int]:
        """
        Store wellness recording with scores to database.
        
        Args:
            recording_data: Recording information including wellness scores
            
        Returns:
            Recording ID or None on error
        """
        if self.dev_mode:
            # Mock recording storage in development
            import random
            recording_id = random.randint(1000, 9999)
            logger.info(f"Mock: Storing recording with ID {recording_id}")
            return recording_id
        
        try:
            if not self.client:
                logger.error("Supabase client not available for recording storage")
                return None
            
            # Create record for recordings table
            response = self.client.table('recordings').insert(recording_data).execute()
            
            if response.data:
                return response.data[0]['id']  # Return the sequential ID
            return None
        except Exception as e:
            logger.error(f"Error storing recording: {e}")
            return None
    
    async def get_cached_recording(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get cached recording by content hash.
        
        Args:
            content_hash: SHA256 hash of audio content
            
        Returns:
            Recording data with wellness scores or None if not found
        """
        if self.dev_mode:
            logger.info(f"Mock: Looking for cached recording {content_hash[:8]}...")
            return None  # No cache in dev mode
        
        try:
            if not self.client:
                logger.error("Supabase client not available for cached recording lookup")
                return None
                
            response = self.client.table('recordings').select('*').eq('content_hash', content_hash).execute()
            
            if response.data:
                return response.data[0]  # Return the most recent match
            return None
        except Exception as e:
            logger.error(f"Error getting cached recording: {e}")
            return None
    
    async def get_recording_by_id(self, recording_id: int) -> Optional[Dict[str, Any]]:
        """
        Get recording by ID for GET /assessment/{recordingId} endpoint.
        
        Args:
            recording_id: Recording ID
            
        Returns:
            Recording data with wellness scores or None if not found
        """
        if self.dev_mode:
            logger.info(f"Mock: Getting recording {recording_id}")
            # Mock recording data for development
            return {
                'id': recording_id,
                'filepath': f'recordings/mock_{recording_id}.wav',
                'user_id': f'user_mock_{recording_id}',
                'created_at': '2025-08-25T12:00:00.000Z',
                'mood_score': 65,
                'anxiety_score': 30,
                'stress_score': 25,
                'happiness_score': 70,
                'loneliness_score': 20,
                'resilience_score': 75,
                'energy_score': 80,
                'emo_id': None,
                'intro_id': None,
                'char_id': None,
                'pers_id': None
            }
        
        try:
            if not self.client:
                logger.error("Supabase client not available for recording lookup")
                return None
                
            response = self.client.table('recordings').select('*').eq('id', recording_id).execute()
            
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Error getting recording by ID: {e}")
            return None
    
    async def get_user_statistics(self, user_id: str, project_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get user statistics and trends for GET /users/{userId} endpoint.
        
        Args:
            user_id: User ID
            project_id: Optional project filter
            
        Returns:
            User statistics with trends and averages or None if not found
        """
        if self.dev_mode:
            logger.info(f"Mock: Getting user statistics for {user_id}")
            # Mock user statistics for development
            return {
                'uuid': user_id,
                'recording_ids': [1001, 1002, 1003, 1004, 1005],
                'first_recording_date': '2025-01-15T09:12:33.001Z',
                'latest_recording_date': '2025-08-25T09:12:33.001Z',
                'wellness_indicator_id': 6,
                'mood_avg': 65,
                'anxiety_avg': 35,
                'stress_avg': 30,
                'happiness_avg': 70,
                'loneliness_avg': 25,
                'resilience_avg': 75,
                'energy_avg': 78,
                'mood_trend': 1,
                'anxiety_trend': -1,
                'stress_trend': 0,
                'happiness_trend': 1,
                'loneliness_trend': -1,
                'resilience_trend': 1,
                'energy_trend': 1
            }
        
        try:
            if not self.client:
                logger.error("Supabase client not available for user statistics lookup")
                return None
            
            # Get user statistics
            query = self.client.table('user_statistics').select('*').eq('user_id', user_id)
            if project_id:
                query = query.eq('project_id', project_id)
            
            stats_response = query.execute()
            
            if not stats_response.data:
                return None
            
            user_stats = stats_response.data[0]
            
            # Get recording IDs for this user
            recordings_query = self.client.table('recordings').select('id').eq('user_id', user_id)
            if project_id:
                recordings_query = recordings_query.eq('project_id', project_id)
            
            recordings_response = recordings_query.execute()
            recording_ids = [r['id'] for r in recordings_response.data] if recordings_response.data else []
            
            # Build response in wellness format
            return {
                'uuid': user_id,
                'recording_ids': recording_ids,
                'first_recording_date': user_stats.get('first_recording_date'),
                'latest_recording_date': user_stats.get('latest_recording_date'),
                'wellness_indicator_id': user_stats.get('wellness_indicator_id', 5),
                'mood_avg': user_stats.get('mood_avg'),
                'anxiety_avg': user_stats.get('anxiety_avg'),
                'stress_avg': user_stats.get('stress_avg'),
                'happiness_avg': user_stats.get('happiness_avg'),
                'loneliness_avg': user_stats.get('loneliness_avg'),
                'resilience_avg': user_stats.get('resilience_avg'),
                'energy_avg': user_stats.get('energy_avg'),
                'mood_trend': user_stats.get('mood_trend', 0),
                'anxiety_trend': user_stats.get('anxiety_trend', 0),
                'stress_trend': user_stats.get('stress_trend', 0),
                'happiness_trend': user_stats.get('happiness_trend', 0),
                'loneliness_trend': user_stats.get('loneliness_trend', 0),
                'resilience_trend': user_stats.get('resilience_trend', 0),
                'energy_trend': user_stats.get('energy_trend', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return None
    
    async def get_all_indicators(self) -> List[Dict[str, Any]]:
        """
        Get all wellness indicators for GET /indicators endpoint.
        
        Returns:
            List of indicator descriptions and tips
        """
        if self.dev_mode:
            logger.info("Mock: Getting all wellness indicators")
            # Return mock indicators for development
            return [
                {
                    'id': 1,
                    'name': 'Mood',
                    'description': 'Overall emotional state and disposition',
                    'tips': ["Practice gratitude daily", "Engage in activities you enjoy"],
                    'positive_components': ["happiness", "contentment", "optimism"],
                    'negative_components': ["sadness", "depression", "irritability"]
                },
                # Add more mock indicators as needed
            ]
        
        try:
            if not self.client:
                logger.error("Supabase client not available for indicators lookup")
                return []
                
            response = self.client.table('indicators').select('*').execute()
            
            if response.data:
                # Convert JSONB fields to lists
                indicators = []
                for indicator in response.data:
                    indicators.append({
                        'id': indicator['id'],
                        'name': indicator['name'],
                        'description': indicator.get('description'),
                        'tips': indicator.get('tips', []),
                        'positive_components': indicator.get('positive_components', []),
                        'negative_components': indicator.get('negative_components', [])
                    })
                return indicators
            return []
        except Exception as e:
            logger.error(f"Error getting indicators: {e}")
            return []
    
    async def update_user_statistics(self, user_id: str, project_id: str, wellness_scores: Dict[str, int]):
        """
        Update user statistics with new wellness scores (rolling averages and trends).
        
        Args:
            user_id: User ID
            project_id: Project ID
            wellness_scores: New wellness scores to incorporate
        """
        if self.dev_mode:
            logger.info(f"Mock: Updating user statistics for {user_id} with scores: {wellness_scores}")
            return
        
        try:
            if not self.client:
                logger.error("Supabase client not available for user statistics update")
                return
            
            from datetime import datetime
            current_time = datetime.now()
            
            # Get current statistics or create new record
            existing_stats = await self.get_user_statistics(user_id, project_id)
            
            if existing_stats:
                # Update existing statistics (simplified - would need more complex rolling average logic)
                new_stats = {
                    'latest_recording_date': current_time.isoformat(),
                    'total_recordings': (existing_stats.get('total_recordings', 0) or 0) + 1,
                    'mood_avg': int((existing_stats.get('mood_avg', 50) + wellness_scores['mood_score']) / 2),
                    'anxiety_avg': int((existing_stats.get('anxiety_avg', 50) + wellness_scores['anxiety_score']) / 2),
                    'stress_avg': int((existing_stats.get('stress_avg', 50) + wellness_scores['stress_score']) / 2),
                    'happiness_avg': int((existing_stats.get('happiness_avg', 50) + wellness_scores['happiness_score']) / 2),
                    'loneliness_avg': int((existing_stats.get('loneliness_avg', 50) + wellness_scores['loneliness_score']) / 2),
                    'resilience_avg': int((existing_stats.get('resilience_avg', 50) + wellness_scores['resilience_score']) / 2),
                    'energy_avg': int((existing_stats.get('energy_avg', 50) + wellness_scores['energy_score']) / 2),
                    'updated_at': current_time.isoformat()
                }
                
                # Update record
                self.client.table('user_statistics').update(new_stats).eq('user_id', user_id).eq('project_id', project_id).execute()
            else:
                # Create new statistics record
                new_stats = {
                    'user_id': user_id,
                    'project_id': project_id,
                    'total_recordings': 1,
                    'first_recording_date': current_time.isoformat(),
                    'latest_recording_date': current_time.isoformat(),
                    'wellness_indicator_id': 5,  # Default neutral
                    'mood_avg': wellness_scores['mood_score'],
                    'anxiety_avg': wellness_scores['anxiety_score'],
                    'stress_avg': wellness_scores['stress_score'],
                    'happiness_avg': wellness_scores['happiness_score'],
                    'loneliness_avg': wellness_scores['loneliness_score'],
                    'resilience_avg': wellness_scores['resilience_score'],
                    'energy_avg': wellness_scores['energy_score'],
                    'updated_at': current_time.isoformat()
                }
                
                # Insert new record
                self.client.table('user_statistics').insert(new_stats).execute()
                
        except Exception as e:
            logger.error(f"Error updating user statistics: {e}")


# Global storage instance
storage = SupabaseStorage()
