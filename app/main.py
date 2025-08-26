"""
FastAPI main application.
Handles API routes, authentication, caching, and request persistence.
"""

import time
import asyncio
from typing import Optional, Dict, Any
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

from .schema import AnalyzeResponse, ErrorResponse, HealthResponse, ScoresModel, NormalizationModel, MetaModel
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup rate limiting
redis_url = get_env_var("REDIS_URL", "redis://localhost:6379")
try:
    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=redis_url,
        default_limits=["100/minute"]
    )
except Exception as e:
    logger.warning(f"Redis not available, using in-memory rate limiting: {e}")
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
            detail="Authorization header is required"
        )
    
    raw_token = parse_auth_header(authorization)
    if not raw_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>"
        )
    
    # Hash token and verify against database
    key_hash = hash_api_key(raw_token)
    api_key_record = await storage.verify_api_key(key_hash)
    
    if not api_key_record:
        raise HTTPException(
            status_code=403,
            detail="Invalid or inactive API key"
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
async def root():
    """Root endpoint with API information"""
    return {
        "name": "X Voice API",
        "version": API_VERSION,
        "description": "Advanced voice analysis API by Voxcentia",
        "endpoints": {
            "health": "/health",
            "analyze": "/v1/voice/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=API_VERSION
    )


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
                detail="Unsupported audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC"
            )
        
        # Read file content
        audio_bytes = await audio.read()
        
        # Validate file size
        if not validate_file_size(len(audio_bytes), MAX_FILE_SIZE_MB):
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB"
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
                
                return AnalyzeResponse(
                    request_id=request_id,
                    version=API_VERSION,
                    processing_ms=processing_time,
                    audio_ms=cached_result.get('audio_ms', 0),
                    scores=ScoresModel(**cached_scores),
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
            raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")
        
        # Extract features
        try:
            features, feature_metadata = feature_extractor.extract_features(audio_data, sample_rate)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")
        
        # Run model predictions
        try:
            emotion_scores, trait_scores = model_manager.predict_all(features, audio_metadata)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
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
        
        # Build response
        scores = ScoresModel(emotions=normalized_emotions, traits=normalized_traits)
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
            'created_at': time.time()
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
            'created_at': time.time()
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


if __name__ == "__main__":
    port = int(get_env_var("PORT", "8000") or "8000")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=get_env_var("APP_ENV", "dev") == "dev"
    )
