"""
Pydantic schemas for API request/response models.
Advanced voice analysis API by Voxcentia.
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import time


class PillarsModel(BaseModel):
    """Customer pillar scores with semantic names"""
    self_awareness: float = Field(description="Self-awareness score (0-100)")
    empathy: float = Field(description="Empathy score (0-100)")  
    self_expression: float = Field(description="Self-expression score (0-100)")
    self_management: float = Field(description="Self-management score (0-100)")


class ScoresModel(BaseModel):
    """Emotion and trait scores container"""
    emotions: Dict[str, float] = Field(
        description="Emotion scores (excluding pillar emotions to avoid duplication)"
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
    """Main API response model - clean format with separate pillars"""
    request_id: str = Field(description="Unique request identifier")
    version: str = Field(description="API version")
    processing_ms: int = Field(description="Processing time in milliseconds")
    audio_ms: int = Field(description="Audio duration in milliseconds")
    scores: ScoresModel = Field(description="Emotion and trait scores")
    pillars: PillarsModel = Field(description="Four key pillar scores")
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


# Wellness API Models

class WellnessRecordingResponse(BaseModel):
    """Wellness API recording response model"""
    id: int = Field(description="Recording ID")
    filepath: str = Field(description="File path (S3 URL)")
    creator_id: Union[str, int] = Field(description="User/creator ID")
    created_at: str = Field(description="Creation timestamp (ISO format)")
    
    # Optional emotion/trait IDs (for compatibility)
    emo_id: Optional[int] = Field(None, description="Emotion analysis ID")
    intro_id: Optional[int] = Field(None, description="Introspection analysis ID") 
    char_id: Optional[int] = Field(None, description="Character analysis ID")
    pers_id: Optional[int] = Field(None, description="Personality analysis ID")
    
    # Wellness scores (1-100 scale)
    mood_score: int = Field(description="Mood score (1-100)")
    anxiety_score: int = Field(description="Anxiety score (1-100)")
    stress_score: int = Field(description="Stress score (1-100)")
    happiness_score: int = Field(description="Happiness score (1-100)")
    loneliness_score: int = Field(description="Loneliness score (1-100)")
    resilience_score: int = Field(description="Resilience score (1-100)")
    energy_score: int = Field(description="Energy score (1-100)")


class WellnessUserResponse(BaseModel):
    """Wellness API user statistics response model"""
    uuid: str = Field(description="User UUID")
    recording_ids: Optional[List[int]] = Field(description="List of recording IDs")
    first_recording_date: Optional[str] = Field(description="First recording date (ISO format)")
    latest_recording_date: Optional[str] = Field(description="Latest recording date (ISO format)")
    wellness_indicator_id: int = Field(description="Overall wellness indicator (1-10)")
    
    # Average scores
    mood_avg: int = Field(description="Average mood score")
    anxiety_avg: int = Field(description="Average anxiety score")
    stress_avg: int = Field(description="Average stress score") 
    happiness_avg: int = Field(description="Average happiness score")
    loneliness_avg: int = Field(description="Average loneliness score")
    resilience_avg: int = Field(description="Average resilience score")
    energy_avg: int = Field(description="Average energy score")
    
    # Trend indicators (-1=declining, 0=stable, 1=improving)
    mood_trend: int = Field(description="Mood trend indicator")
    anxiety_trend: int = Field(description="Anxiety trend indicator")
    stress_trend: int = Field(description="Stress trend indicator")
    happiness_trend: int = Field(description="Happiness trend indicator")
    loneliness_trend: int = Field(description="Loneliness trend indicator")
    resilience_trend: int = Field(description="Resilience trend indicator")
    energy_trend: int = Field(description="Energy trend indicator")


class WellnessIndicatorResponse(BaseModel):
    """Wellness API indicator response model"""
    id: int = Field(description="Indicator ID")
    name: str = Field(description="Indicator name")
    description: Optional[str] = Field(description="Indicator description")
    tips: Optional[List[str]] = Field(description="Helpful tips")
    positive_components: Optional[List[str]] = Field(description="Positive components")
    negative_components: Optional[List[str]] = Field(description="Negative components")
