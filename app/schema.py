"""
Pydantic schemas for API request/response models.
Advanced voice analysis API by Voxcentia.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import time


class ScoresModel(BaseModel):
    """Emotion and trait scores container"""
    emotions: Dict[str, float] = Field(
        description="Emotion scores emo1-emo26"
    )
    traits: Dict[str, float] = Field(
        description="Trait scores char1-char94"
    )


class NormalizationModel(BaseModel):
    """Normalization metadata"""
    scheme: str = Field(description="Normalization scheme used")
    window_days: int = Field(description="Rolling window size in days")
    baseline_date: Optional[str] = Field(description="Date of baseline data")


class MetaModel(BaseModel):
    """Request metadata"""
    audio_format: str = Field(description="Detected audio format")
    sample_rate: int = Field(description="Audio sample rate")
    channels: int = Field(description="Number of audio channels")
    features_version: str = Field(description="Feature extraction version")
    model_version: str = Field(description="Model version used")


class AnalyzeResponse(BaseModel):
    """Main API response model - exact legacy schema"""
    request_id: str = Field(description="Unique request identifier")
    version: str = Field(description="API version")
    processing_ms: int = Field(description="Processing time in milliseconds")
    audio_ms: int = Field(description="Audio duration in milliseconds")
    scores: ScoresModel = Field(description="Emotion and trait scores")
    normalization: NormalizationModel = Field(description="Normalization info")
    meta: MetaModel = Field(description="Request metadata")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")


class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str = Field(description="Error description")
    error_code: Optional[str] = Field(description="Error code")
    request_id: Optional[str] = Field(description="Request ID if available")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(description="Service status")
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    version: str = Field(description="API version")
