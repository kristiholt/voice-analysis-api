"""
FastAPI main application.
Handles API routes, authentication, caching, and request persistence.
"""

import time
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import logging
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
import uvicorn

from .schema import (
    AnalyzeResponse, ErrorResponse, HealthResponse, ScoresModel, PillarsModel, NormalizationModel, MetaModel,
    WellnessRecordingResponse, WellnessUserResponse, WellnessIndicatorResponse
)
from .utils import (
    generate_request_id, 
    hash_api_key, 
    compute_audio_hash, 
    parse_auth_header,
    get_env_var,
    validate_file_size
)
from .storage import storage
from .audio_io import audio_processor
from .features import feature_extractor
from .enhanced_models import enhanced_model_manager as model_manager
from .postproc import post_processor
from .vibeonix_mapper import wellness_mapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup rate limiting with robust fallback
redis_url = get_env_var("REDIS_URL")
try:
    if redis_url and redis_url.strip() and "${{" not in redis_url:
        # Only use Redis if we have a valid URL (not a template variable)
        limiter = Limiter(
            key_func=get_remote_address,
            storage_uri=redis_url,
            default_limits=["100/minute"]
        )
        logger.info(f"✅ Redis rate limiting configured: {redis_url}")
    else:
        # Use in-memory rate limiting if no valid Redis URL
        limiter = Limiter(
            key_func=get_remote_address,
            default_limits=["100/minute"]
        )
        logger.info("✅ Using in-memory rate limiting (no Redis URL)")
except Exception as e:
    logger.warning(f"Redis failed, using in-memory rate limiting: {e}")
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100/minute"]
    )

# Create FastAPI app
app = FastAPI(
    title="X Voice API",
    description="Advanced voice analysis API by Voxcentia",
    version="1.0.0",
    docs_url="/docs" if get_env_var("APP_ENV", "dev") == "dev" else None
)

# Add rate limiting
app.state.limiter = limiter

# Custom rate limit handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    response = JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"}
    )
    # Add basic rate limit headers
    response.headers["X-RateLimit-Limit"] = "60"
    response.headers["Retry-After"] = "60"
    return response

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configuration
MAX_FILE_SIZE_MB = int(get_env_var("MAX_FILE_SIZE_MB", "10") or "10")
ENABLE_CACHE = (get_env_var("ENABLE_CACHE", "true") or "true").lower() == "true"
API_VERSION = "1.0.0"


async def verify_auth(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Verify bearer token authentication.
    
    Args:
        authorization: Authorization header
        
    Returns:
        API key record
        
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
            headers={"X-Error-Code": "UNAUTHORIZED"}
        )
    
    raw_token = parse_auth_header(authorization)
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
            headers={"X-Error-Code": "UNAUTHORIZED"}
        )
    
    # Hash token and verify against database
    key_hash = hash_api_key(raw_token)
    api_key_record = await storage.verify_api_key(key_hash)
    
    if not api_key_record:
        raise HTTPException(
            status_code=401,
            detail="Invalid or inactive API key",
            headers={"X-Error-Code": "UNAUTHORIZED"}
        )
    
    return api_key_record


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


@app.get("/")
@app.head("/")
async def root():
    """Root endpoint - optimized for deployment health checks"""
    import time
    return {
        "status": "healthy",
        "service": "X Voice API", 
        "timestamp": int(time.time()),
        "version": API_VERSION
    }


@app.get("/health", response_model=HealthResponse)
@app.head("/health")
async def health_check():
    """Health check endpoint"""
    import time
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=int(time.time())
    )


@app.get("/health/", response_model=HealthResponse)
async def health_check_trailing_slash():
    """Health check endpoint with trailing slash"""
    import time
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=int(time.time())
    )


@app.get("/healthz", response_model=HealthResponse)
@app.head("/healthz")
async def health_check_k8s():
    """Kubernetes-style health check endpoint"""
    import time
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=int(time.time())
    )


@app.get("/ready")
@app.head("/ready")
async def readiness_check():
    """Readiness check for deployment platforms"""
    try:
        # You can add more comprehensive checks here if needed
        # For now, just ensure the basic components are loaded
        return {"status": "ready", "version": API_VERSION}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


@app.get("/live")
@app.head("/live")
async def liveness_check():
    """Liveness check for deployment platforms"""
    return {"status": "alive", "version": API_VERSION}


@app.get("/ping")
@app.head("/ping")
async def ping():
    """Simple ping endpoint for load balancers"""
    return {"status": "pong"}


@app.post("/v1/voice/analyze", response_model=AnalyzeResponse)
@limiter.limit("60/minute")  # Per-IP rate limit, API key limits handled separately
async def analyze_voice(
    request: Request,
    audio: UploadFile = File(...),
    api_key: Dict[str, Any] = Depends(verify_auth),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    x_no_store: Optional[str] = Header(None, alias="X-No-Store")
):
    """
    Analyze voice audio for emotions and personality traits.
    
    Args:
        audio: Audio file (WAV/FLAC/MP3, ≤20s, ≤10MB)
        api_key: Verified API key record
        
    Returns:
        Analysis results with emotions, traits, and metadata
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate file
        if not audio.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not audio_processor.validate_audio_format(audio.filename):
            raise HTTPException(
                status_code=400,
                detail="Invalid audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC",
                headers={"X-Error-Code": "INVALID_FORMAT", "X-Request-ID": request_id}
            )
        
        # Read file content
        audio_bytes = await audio.read()
        
        # Validate file size
        if not validate_file_size(len(audio_bytes), MAX_FILE_SIZE_MB):
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds 10MB limit",
                headers={"X-Error-Code": "FILE_TOO_LARGE", "X-Request-ID": request_id}
            )
        
        # Compute content hash for caching
        content_hash = compute_audio_hash(audio_bytes)
        
        # Check cache if enabled
        cached_result = None
        if ENABLE_CACHE:
            cached_result = await storage.get_cached_result(content_hash)
            if cached_result:
                logger.info(f"Cache hit for request {request_id}, returning cached result")
                
                # Update response with new request_id but preserve cached scores
                cached_scores = cached_result.get('scores', {})
                if isinstance(cached_scores, str):
                    cached_scores = json.loads(cached_scores)
                
                processing_time = int((time.time() - start_time) * 1000)
                
                # Log this request
                await _log_request(request_id, api_key, audio.filename, content_hash, len(audio_bytes), True)
                
                # Extract pillar scores from cached result or provide defaults
                cached_pillars = cached_result.get('pillars', {
                    'self_awareness': 50.0,
                    'empathy': 50.0, 
                    'self_expression': 50.0,
                    'self_management': 50.0
                })
                
                return AnalyzeResponse(
                    request_id=request_id,
                    version=API_VERSION,
                    processing_ms=processing_time,
                    audio_ms=cached_result.get('audio_ms', 0),
                    scores=ScoresModel(**cached_scores),
                    pillars=PillarsModel(**cached_pillars),
                    normalization=NormalizationModel(**cached_result.get('normalization', {})),
                    meta=MetaModel(**cached_result.get('meta', {})),
                    warnings=cached_result.get('warnings', [])
                )
        
        # Process audio
        try:
            audio_data, sample_rate, audio_metadata = audio_processor.decode_audio_file(
                audio_bytes, audio.filename
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail="Audio processing failed",
                headers={"X-Error-Code": "PROCESSING_ERROR", "X-Request-ID": request_id}
            )
        
        # Extract features
        try:
            features, feature_metadata = feature_extractor.extract_features(audio_data, sample_rate)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Audio processing failed",
                headers={"X-Error-Code": "PROCESSING_ERROR", "X-Request-ID": request_id}
            )
        
        # Run model predictions
        try:
            emotion_scores, trait_scores = model_manager.predict_all(features, audio_metadata)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Audio processing failed",
                headers={"X-Error-Code": "PROCESSING_ERROR", "X-Request-ID": request_id}
            )
        
        # Apply post-processing and normalization
        try:
            normalized_emotions, normalized_traits, normalization_info = await post_processor.normalize_scores(
                emotion_scores, trait_scores
            )
        except Exception as e:
            logger.warning(f"Normalization failed, using raw scores: {e}")
            normalized_emotions = emotion_scores
            normalized_traits = trait_scores
            normalization_info = {'scheme': 'none', 'window_days': 0, 'baseline_date': None}
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Extract pillar scores (keeping them in emotions dict for Vibeonix compatibility)
        pillars_data = PillarsModel(
            self_awareness=normalized_emotions.get('emo15', 0.0),  # alertness
            empathy=normalized_emotions.get('emo29', 0.0),         # positive_overall  
            self_expression=normalized_emotions.get('emo31', 0.0), # valence_overall
            self_management=normalized_emotions.get('emo32', 0.0)  # control_overall
        )
        
        # Ensure we have all expected emotions (emo1-emo38) for Vibeonix compatibility
        # Fill missing emotions with neutral values (50.0 on 0-100 scale)
        expected_emotions = {}
        for i in range(1, 39):  # emo1 to emo38
            emo_key = f'emo{i}'
            expected_emotions[emo_key] = normalized_emotions.get(emo_key, 50.0)
        
        # Keep all emotions including pillars (no removal for Vibeonix compatibility)
        clean_emotions = expected_emotions
        
        scores = ScoresModel(emotions=clean_emotions, traits=normalized_traits)
        normalization = NormalizationModel(**normalization_info)
        
        model_info = model_manager.get_model_info()
        meta = MetaModel(
            audio_format=audio_metadata['format'],
            sample_rate=audio_metadata['sample_rate'],
            channels=audio_metadata['channels'],
            features_version=feature_metadata['version'],
            model_version=model_info['version']
        )
        
        warnings = []
        if audio_metadata['duration_seconds'] > 20:
            warnings.append("Audio duration exceeds recommended 20 seconds")
        
        response = AnalyzeResponse(
            request_id=request_id,
            version=API_VERSION,
            processing_ms=processing_time,
            audio_ms=audio_metadata['duration_ms'],
            scores=scores,
            pillars=pillars_data,
            normalization=normalization,
            meta=meta,
            warnings=warnings
        )
        
        # Log request and store result asynchronously
        asyncio.create_task(_log_and_store_result(
            request_id, api_key, audio.filename, content_hash, 
            len(audio_bytes), False, response, audio_bytes if (get_env_var("STORE_AUDIO", "false") or "false").lower() == "true" else None
        ))
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_voice: {e}", exc_info=True)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Try to log the failed request
        try:
            filename = "unknown"
            if 'audio' in locals() and audio and audio.filename:
                filename = audio.filename
            await _log_request(request_id, api_key, filename, "", 0, False, error=str(e))
        except:
            pass
        
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/recording", response_model=WellnessRecordingResponse)
@limiter.limit("60/minute")  # Per-IP rate limit, API key limits handled separately
async def post_recording(
    request: Request,
    recording: UploadFile = File(...),
    api_key: Dict[str, Any] = Depends(verify_auth),
    user_id: Optional[str] = None  # Will be extracted from API key context
):
    """
    Upload voice recording and analyze wellness indicators.
    
    Args:
        recording: Audio file (WAV/FLAC/MP3, ≤20s, ≤10MB)
        api_key: Verified API key record
        
    Returns:
        Recording with simple wellness scores (1-100 scale)
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    try:
        # Validate file
        if not recording.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not audio_processor.validate_audio_format(recording.filename):
            raise HTTPException(
                status_code=400, 
                detail="Unsupported audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC"
            )
        
        # Read file content
        audio_bytes = await recording.read()
        
        # Validate file size
        if not validate_file_size(len(audio_bytes), MAX_FILE_SIZE_MB):
            raise HTTPException(
                status_code=413,
                detail="Audio file exceeds 10MB limit",
                headers={"X-Error-Code": "FILE_TOO_LARGE", "X-Request-ID": request_id}
            )
        
        # Compute content hash for caching
        content_hash = compute_audio_hash(audio_bytes)
        
        # Use user_id from API key context if not provided
        if not user_id:
            user_id = api_key.get('user_id', f"user_{api_key.get('id', 'unknown')}")
        
        # Ensure user_id is not None
        if user_id is None:
            user_id = f"user_{api_key.get('id', 'unknown')}"
        
        # Check for cached wellness scores
        cached_recording = None
        if ENABLE_CACHE:
            cached_recording = await storage.get_cached_recording(content_hash)
            if cached_recording:
                logger.info(f"Cache hit for recording {request_id}, returning cached wellness scores")
                processing_time = int((time.time() - start_time) * 1000)
                
                # Log this request
                await _log_request(request_id, api_key, recording.filename, content_hash, len(audio_bytes), True)
                
                return WellnessRecordingResponse(
                    id=cached_recording['id'],
                    filepath=cached_recording.get('filepath', ''),
                    creator_id=user_id or '',
                    created_at=cached_recording.get('created_at', ''),
                    emo_id=None,
                    intro_id=None,
                    char_id=None,
                    pers_id=None,
                    mood_score=cached_recording.get('mood_score', 50),
                    anxiety_score=cached_recording.get('anxiety_score', 50),
                    stress_score=cached_recording.get('stress_score', 50),
                    happiness_score=cached_recording.get('happiness_score', 50),
                    loneliness_score=cached_recording.get('loneliness_score', 50),
                    resilience_score=cached_recording.get('resilience_score', 50),
                    energy_score=cached_recording.get('energy_score', 50)
                )
        
        # Process audio - same pipeline as v1/voice/analyze
        try:
            audio_data, sample_rate, audio_metadata = audio_processor.decode_audio_file(
                audio_bytes, recording.filename
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
        # Extract features
        try:
            features, feature_metadata = feature_extractor.extract_features(audio_data, sample_rate)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Audio processing failed",
                headers={"X-Error-Code": "PROCESSING_ERROR", "X-Request-ID": request_id}
            )
        
        # Run model predictions
        try:
            emotion_scores, trait_scores = model_manager.predict_all(features, audio_metadata)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Audio processing failed",
                headers={"X-Error-Code": "PROCESSING_ERROR", "X-Request-ID": request_id}
            )
        
        # Convert to wellness scores using our mapper
        try:
            wellness_scores = wellness_mapper.map_to_wellness_scores(emotion_scores, trait_scores)
            logger.info(f"✅ Converted to wellness scores: {wellness_scores}")
        except Exception as e:
            logger.error(f"Wellness score mapping failed: {e}")
            # Fallback to neutral scores
            wellness_scores = {
                "mood_score": 50,
                "anxiety_score": 50,
                "stress_score": 50,
                "happiness_score": 50,
                "loneliness_score": 50,
                "resilience_score": 50,
                "energy_score": 50,
            }
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Store recording in database
        try:
            recording_data = {
                'user_id': user_id,
                'project_id': api_key.get('project_id'),
                'api_key_id': api_key.get('id'),
                'filename': recording.filename,
                'filepath': f"recordings/{content_hash}_{recording.filename}",  # Mock S3 path
                'content_hash': content_hash,
                'audio_ms': audio_metadata['duration_ms'],
                'processing_ms': processing_time,
                'full_scores': {
                    'emotions': emotion_scores,
                    'traits': trait_scores
                },
                **wellness_scores
            }
            
            recording_id = await storage.store_recording(recording_data)
            logger.info(f"✅ Stored recording with ID: {recording_id}")
            
        except Exception as e:
            logger.error(f"Failed to store recording: {e}")
            # Continue without storing - still return the analysis
            recording_id = None
        
        # Build wellness response
        response = WellnessRecordingResponse(
            id=recording_id or 0,
            filepath=f"recordings/{content_hash}_{recording.filename}",
            creator_id=user_id or '',
            created_at=time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
            emo_id=None,
            intro_id=None,
            char_id=None,
            pers_id=None,
            **wellness_scores
        )
        
        # Update user statistics asynchronously
        if recording_id and user_id and api_key.get('project_id'):
            project_id = api_key.get('project_id', '')
            if project_id:
                asyncio.create_task(_update_user_statistics(user_id, project_id, wellness_scores))
        
        # Log the request
        asyncio.create_task(_log_request(request_id, api_key, recording.filename, content_hash, len(audio_bytes), False))
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in post_recording: {e}", exc_info=True)
        processing_time = int((time.time() - start_time) * 1000)
        
        # Try to log the failed request
        try:
            filename = "unknown"
            if 'recording' in locals() and recording and recording.filename:
                filename = recording.filename
            await _log_request(request_id, api_key, filename, "", 0, False, error=str(e))
        except:
            pass
        
        raise HTTPException(status_code=500, detail="Internal server error")


async def _update_user_statistics(user_id: str, project_id: str, wellness_scores: Dict[str, int]):
    """Update user statistics with new wellness scores asynchronously"""
    try:
        await storage.update_user_statistics(user_id, project_id, wellness_scores)
    except Exception as e:
        logger.error(f"Failed to update user statistics: {e}")


async def _log_request(request_id: str, api_key: Dict[str, Any], filename: str, 
                      content_hash: str, file_size: int, cache_hit: bool, error: Optional[str] = None):
    """Log API request to database"""
    try:
        request_data = {
            'request_id': request_id,
            'api_key_id': api_key.get('id'),
            'filename': filename,
            'content_hash': content_hash,
            'file_size': file_size,
            'cache_hit': cache_hit,
            'error': error,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        await storage.log_request(request_data)
    except Exception as e:
        logger.error(f"Failed to log request: {e}")


async def _log_and_store_result(request_id: str, api_key: Dict[str, Any], filename: str,
                               content_hash: str, file_size: int, cache_hit: bool,
                               response: AnalyzeResponse, audio_bytes: Optional[bytes] = None):
    """Log request and store result to database"""
    try:
        # Log request
        await _log_request(request_id, api_key, filename, content_hash, file_size, cache_hit)
        
        # Store result
        result_data = {
            'request_id': request_id,
            'content_hash': content_hash,
            'scores': response.scores.dict(),
            'normalization': response.normalization.dict(),
            'meta': response.meta.dict(),
            'processing_ms': response.processing_ms,
            'audio_ms': response.audio_ms,
            'warnings': response.warnings,
            'version': response.version,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        
        await storage.store_result(result_data)
        
        # Store audio blob if enabled
        if audio_bytes:
            await storage.store_audio_blob(content_hash, audio_bytes, filename)
            
    except Exception as e:
        logger.error(f"Failed to store result: {e}")


@app.get("/v1/usage")
async def get_usage_analytics(
    project_id: Optional[str] = None,
    hours: int = 24,
    api_key: Dict[str, Any] = Depends(verify_auth)
):
    """
    Get usage analytics from usage_hourly table.
    Admin endpoint for project usage statistics.
    """
    try:
        # For now, return mock analytics since we need to implement storage.execute_query
        return {
            "message": "Usage analytics endpoint ready",
            "project_id": project_id,
            "period_hours": hours,
            "note": "Implement storage.execute_query for full functionality"
        }
        
    except Exception as e:
        logger.error(f"Error fetching usage analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch usage data")


@app.get("/assessment/{recording_id}", response_model=WellnessRecordingResponse)
async def get_recording_by_id(
    recording_id: int,
    api_key: Dict[str, Any] = Depends(verify_auth)
):
    """
    Get wellness recording analysis by ID.
    
    Args:
        recording_id: Recording ID
        
    Returns:
        Recording with wellness scores
    """
    try:
        recording = await storage.get_recording_by_id(recording_id)
        
        if not recording:
            raise HTTPException(status_code=404, detail="Recording not found")
        
        return WellnessRecordingResponse(
            id=recording['id'],
            filepath=recording.get('filepath', ''),
            creator_id=recording.get('user_id', ''),
            created_at=recording.get('created_at', ''),
            emo_id=recording.get('emo_id'),
            intro_id=recording.get('intro_id'),
            char_id=recording.get('char_id'),
            pers_id=recording.get('pers_id'),
            mood_score=recording.get('mood_score', 50),
            anxiety_score=recording.get('anxiety_score', 50),
            stress_score=recording.get('stress_score', 50),
            happiness_score=recording.get('happiness_score', 50),
            loneliness_score=recording.get('loneliness_score', 50),
            resilience_score=recording.get('resilience_score', 50),
            energy_score=recording.get('energy_score', 50)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recording: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/users/{user_id}", response_model=WellnessUserResponse)
async def get_user_statistics(
    user_id: str,
    project_id: Optional[str] = None,
    api_key: Dict[str, Any] = Depends(verify_auth)
):
    """
    Get user wellness statistics and trends.
    
    Args:
        user_id: User ID
        project_id: Optional project filter
        
    Returns:
        User statistics with wellness trends and averages
    """
    try:
        user_stats = await storage.get_user_statistics(user_id, project_id or '')
        
        if not user_stats:
            # Return default stats for new users
            user_stats = {
                'uuid': user_id,
                'recording_ids': [],
                'first_recording_date': None,
                'latest_recording_date': None,
                'wellness_indicator_id': 5,
                'mood_avg': 50,
                'anxiety_avg': 50,
                'stress_avg': 50,
                'happiness_avg': 50,
                'loneliness_avg': 50,
                'resilience_avg': 50,
                'energy_avg': 50,
                'mood_trend': 0,
                'anxiety_trend': 0,
                'stress_trend': 0,
                'happiness_trend': 0,
                'loneliness_trend': 0,
                'resilience_trend': 0,
                'energy_trend': 0
            }
        
        return WellnessUserResponse(**user_stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/indicators", response_model=List[WellnessIndicatorResponse])
async def get_all_indicators(
    api_key: Dict[str, Any] = Depends(verify_auth)
):
    """
    Get all available wellness indicators.
    
    Returns:
        List of wellness indicators with descriptions and tips
    """
    try:
        indicators = await storage.get_all_indicators()
        
        # Convert to wellness indicator response format
        result = []
        for indicator in indicators:
            result.append(WellnessIndicatorResponse(
                id=indicator['id'],
                name=indicator['name'],
                description=indicator.get('description', ''),
                tips=indicator.get('tips', []),
                positive_components=indicator.get('positive_components', []),
                negative_components=indicator.get('negative_components', [])
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting indicators: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/admin/preload-models")
async def preload_models(api_key: Dict[str, Any] = Depends(verify_auth)):
    """
    Preload models after deployment for better performance.
    Admin endpoint to trigger lazy loading of ML models.
    """
    try:
        start_time = time.time()
        
        # Trigger model loading by accessing model properties
        logger.info("Starting model preloading...")
        
        # Access emotion model to trigger loading
        if hasattr(model_manager, 'emotion_model'):
            _ = model_manager.emotion_model
            logger.info("✅ Emotion model loaded")
        
        # Access trait model to trigger loading
        if hasattr(model_manager, 'trait_model'):
            _ = model_manager.trait_model
            logger.info("✅ Trait model loaded")
            
        # Access enhanced model manager components
        if hasattr(model_manager, 'emotion_model'):
            _ = model_manager.emotion_model
            logger.info("✅ Enhanced emotion model accessed")
            
        if hasattr(model_manager, 'trait_model'):
            _ = model_manager.trait_model
            logger.info("✅ Enhanced trait model accessed")
            
        # Trigger a prediction to ensure all models are loaded
        try:
            import numpy as np
            dummy_features = np.random.rand(13)  # Sample MFCC features
            _, _ = model_manager.predict_all(dummy_features, {})
            logger.info("✅ All models loaded via prediction test")
        except Exception as e:
            logger.warning(f"Model prediction test failed: {e}")
        
        loading_time = int((time.time() - start_time) * 1000)
        logger.info(f"Model preloading completed in {loading_time}ms")
        
        return {
            "status": "models loaded",
            "loading_time_ms": loading_time,
            "timestamp": int(time.time()),
            "models_loaded": [
                "emotion_model",
                "trait_model", 
                "enhanced_emotion_model",
                "enhanced_trait_model"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")
        raise HTTPException(status_code=500, detail=f"Model preloading failed: {str(e)}")


if __name__ == "__main__":
    port = int(get_env_var("PORT", "8000") or "8000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=get_env_var("APP_ENV", "dev") == "dev"
    )
