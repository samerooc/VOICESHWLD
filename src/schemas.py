"""
Pydantic Request & Response Schemas for VoiceShield FastAPI Service (Phase 13).
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", example="ok")
    service: str = Field(default="voiceshield-api", example="voiceshield-api")


class MetadataResponse(BaseModel):
    model_version: str = Field(..., example="2.0.0")
    feature_version: str = Field(..., example="1.0.0")
    backbone: str = Field(default="acoustic_spectral_net", example="acoustic_spectral_net")
    class_mapping: Dict[str, str] = Field(
        default_factory=lambda: {"0": "bona_fide", "1": "spoof"},
        example={"0": "bona_fide", "1": "spoof"},
    )
    supported_format: str = Field(default="wav, mp3, mp4, m4a, ogg, flac", example="wav, mp3, mp4, m4a, ogg, flac")
    audio_saved: bool = Field(default=False, example=False)
    disclaimer: str = Field(
        default="Experimental decision-support prototype; not identity proof.",
        example="Experimental decision-support prototype; not identity proof.",
    )


class PredictResponse(BaseModel):
    model_version: str = Field(..., example="2.0.0")
    backbone: str = Field(default="acoustic_spectral_net", example="acoustic_spectral_net")
    feature_version: str = Field(..., example="1.0.0")
    class_mapping: Dict[str, str] = Field(
        default_factory=lambda: {"0": "bona_fide", "1": "spoof"},
        example={"0": "bona_fide", "1": "spoof"},
    )
    original_format: str = Field(default="wav", example="wav")
    detected_codec: str = Field(default="pcm", example="pcm")
    raw_model_score: float = Field(..., ge=0.0, le=1.0, example=0.34)
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, example=0.34)
    bona_fide_probability: float = Field(..., ge=0.0, le=1.0, example=0.66)
    spoof_probability: float = Field(..., ge=0.0, le=1.0, example=0.34)
    risk_score: int = Field(..., ge=0, le=100, example=34)
    risk_band: str = Field(..., example="review")
    uncertainty: bool = Field(default=False, example=False)
    quality_flags: List[str] = Field(default_factory=list, example=[])
    valid_window_count: int = Field(default=1, example=1)
    total_window_count: int = Field(default=1, example=1)
    explanation: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(default_factory=list)
    processing_ms: float = Field(..., ge=0.0, example=45.2)
    audio_saved: bool = Field(default=False, example=False)
    reliability_notice: str = Field(
        default="Prediction reliability depends on audio quality and similarity to evaluation data.",
        example="Prediction reliability depends on audio quality and similarity to evaluation data.",
    )
    disclaimer: str = Field(
        default="Experimental decision-support prototype; not identity proof.",
        example="Experimental decision-support prototype; not identity proof.",
    )


class ErrorResponse(BaseModel):
    detail: str = Field(..., example="Validation error: Unsupported audio format.")
