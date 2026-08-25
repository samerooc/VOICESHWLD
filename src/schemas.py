"""
VoiceShield Phase 5 — Pydantic v2 Data Contracts & Strict Validation Models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health, device routing, and uptime telemetry."""
    status: str = Field(default="ok", example="ok")
    healthy: bool = Field(default=True, example=True)
    service: str = Field(default="voiceshield-api", example="voiceshield-api")
    device: str = Field(..., example="cpu")
    model_name: str = Field(..., example="garystafford/wav2vec2-deepfake-voice-detector")
    target_sr: int = Field(default=16000, example=16000)
    uptime_sec: float = Field(..., example=124.5)


class MetadataResponse(BaseModel):
    """Model architecture, calibrated thresholds, and supported wire formats."""
    status: str = Field(default="ok")
    service: str = Field(default="voiceshield-api")
    service_name: str = Field(default="VoiceShield Enterprise Deepfake & Clone Defense Platform")
    version: str = Field(default="3.0.0")
    model_version: str = Field(default="3.0.0")
    feature_version: str = Field(default="3.0.0")
    architecture: str = Field(default="Tri-Tier Adaptive Consensus (Transformer + LPC Physics + DSP Biomechanics)")
    backbone: str = Field(default="garystafford/wav2vec2-deepfake-voice-detector")
    class_mapping: Dict[str, str] = Field(default={"0": "bona_fide", "1": "spoof"})
    active_spoof_index: int = Field(default=1, example=1)
    temperature: float = Field(default=1.35, example=1.35)
    sample_rate_hz: int = Field(default=16000, example=16000)
    supported_format: str = Field(default="wav, mp3, m4a, flac, ogg, webm, aac")
    supported_formats: List[str] = Field(
        default=["WAV", "MP3", "M4A", "FLAC", "OGG", "WebM", "AAC", "G.711 mu-law", "PCM16", "Float32"]
    )
    supported_audio_formats: List[str] = Field(
        default=["WAV", "MP3", "M4A", "FLAC", "OGG", "WebM", "AAC", "G.711 mu-law"]
    )
    risk_thresholds: Dict[str, Any] = Field(
        default={
            "low_risk_max": 25,
            "review_required_range": [26, 60],
            "high_risk_min": 61,
        }
    )
    audio_saved: bool = Field(default=False)
    disclaimer: str = Field(
        default="Advisory forensic risk assessment. Not conclusive proof of human identity."
    )


class AudioDiagnostics(BaseModel):
    """Audio signal physical properties and quality flags."""
    original_sr: int = Field(default=16000)
    duration_sec: float = Field(default=0.0)
    voiced_sec: float = Field(default=0.0)
    snr_db: float = Field(default=0.0)
    is_clipped: bool = Field(default=False)
    is_silent: bool = Field(default=False)
    voiced_ratio: Optional[float] = Field(default=0.0)
    rms_energy: Optional[float] = Field(default=0.0)
    sample_rate: Optional[int] = Field(default=16000)
    num_samples: Optional[int] = Field(default=0)


class ForensicBreakdown(BaseModel):
    """Fine-grained multi-tier physical and neural forensic metrics."""
    transformer_spoof_prob: Optional[float] = Field(default=0.5)
    active_model_id: Optional[str] = Field(default="Native Acoustic Backbone")
    lpc_anomaly_score: Optional[float] = Field(default=0.5)
    lpc_kurtosis: Optional[float] = Field(default=3.0)
    phase_entropy: Optional[float] = Field(default=0.5)
    residual_flatness: Optional[float] = Field(default=0.5)
    dsp_physics_prob: Optional[float] = Field(default=0.5)
    glottal_spoof_prob: Optional[float] = Field(default=0.5)
    lfcc_spoof_prob: Optional[float] = Field(default=0.5)
    spectral_spoof_prob: Optional[float] = Field(default=0.5)
    jitter_local: Optional[float] = Field(default=0.01)
    shimmer_local: Optional[float] = Field(default=0.05)
    hnr_db: Optional[float] = Field(default=15.0)
    lfcc_variance: Optional[float] = Field(default=1.0)
    hf_cutoff_ratio: Optional[float] = Field(default=0.0)
    snr_db: Optional[float] = Field(default=20.0)
    snr_weight_mode: Optional[str] = Field(default="clean")
    voiced_ratio: Optional[float] = Field(default=1.0)


class PredictionResponse(BaseModel):
    """Comprehensive forensic assessment payload for batch and single-file inspection."""
    prediction_label: str = Field(..., example="AUTHENTIC HUMAN VOICE")
    spoof_probability: float = Field(..., example=0.12)
    human_probability: float = Field(..., example=0.88)
    bona_fide_probability: Optional[float] = Field(default=0.88, example=0.88)
    risk_score: int = Field(..., ge=0, le=100, example=12)
    risk_band: str = Field(..., example="Low Risk (Human Voice)")
    risk_band_key: Optional[str] = Field(default="low", example="low")
    badge_class: Optional[str] = Field(default="badge-low", example="badge-low")
    risk_description: str = Field(..., example="Natural vocal-fold micro-perturbations detected.")
    diagnostics: AudioDiagnostics
    forensic_breakdown: Optional[Dict[str, Any]] = Field(default_factory=dict)
    latency_ms: float = Field(..., example=42.5)
    processing_ms: Optional[float] = Field(default=42.5, example=42.5)
    uncertainty: Optional[float] = Field(default=0.05, example=0.05)
    explanation: Optional[List[str]] = Field(default_factory=list)
    model_version: str = Field(default="3.0.0")
    feature_version: str = Field(default="3.0.0")
    is_realtime_compliant: bool = Field(default=True, example=True)
    disclaimer: str = Field(
        default="⚠️ ADVISORY FORENSIC REPORT: Probability assessment based on biomechanical and neural feature synthesis."
    )
    audio_saved: bool = Field(default=False, example=False)
    window_breakdown: Optional[List[Dict[str, Any]]] = Field(default_factory=list)


class StreamingTelemetryFrame(BaseModel):
    """Real-time streaming telemetry frame emitted over WebSockets."""
    session_id: Optional[str] = Field(default=None)
    timestamp_sec: float = Field(..., example=1.5)
    window_index: int = Field(default=1, example=1)
    instantaneous_prob: float = Field(..., example=0.45)
    instantaneous_score: Optional[int] = Field(default=45, example=45)
    ema_prob: Optional[float] = Field(default=0.42, example=0.42)
    top_k_prob: Optional[float] = Field(default=0.48, example=0.48)
    smoothed_risk_score: int = Field(..., ge=0, le=100, example=46)
    risk_score: int = Field(..., ge=0, le=100, example=46)
    risk_band: str = Field(..., example="Review Required (Borderline Evidence)")
    risk_band_key: Optional[str] = Field(default="review", example="review")
    badge_class: Optional[str] = Field(default="badge-review", example="badge-review")
    is_alert_held: bool = Field(default=False, example=False)
    alert_hold_counter: Optional[int] = Field(default=0, example=0)
    forensic_breakdown: Optional[Dict[str, Any]] = Field(default_factory=dict)
    diagnostics: Optional[Dict[str, Any]] = Field(default_factory=dict)
    latency_ms: float = Field(..., example=14.2)
    is_realtime_compliant: bool = Field(default=True, example=True)
    disclaimer: Optional[str] = Field(
        default="⚠️ SANDBOX SIMULATION — NOT A LIVE CALL: Advisory forensic scoring demonstration."
    )


# Alias for backward compatibility
StreamRiskFrame = StreamingTelemetryFrame
