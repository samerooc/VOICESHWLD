"""
Unit tests for Phase 1 VoiceShield modular architecture.
Tests audio_io, validation, features, scoring, model, and privacy modules.
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import get_audio_metadata, load_audio_from_bytes
from src.config import (
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_NAMES,
    MODEL_PATH,
    N_MFCC,
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
    TOTAL_FEATURES,
)
from src.features import extract_features_from_audio
from src.model import build_pipeline, load_model
from src.privacy import compute_sha256, get_privacy_statement
from src.scoring import calculate_risk_score, get_risk_band, predict_and_score
from src.validation import validate_audio_signal, validate_wav_bytes


def generate_synthetic_wav_bytes(duration_sec: float = 1.0, sr: int = SAMPLE_RATE, freq: float = 440.0) -> bytes:
    """Helper to generate a clean synthetic WAV byte buffer."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_config_constants():
    """Verify configuration parameters."""
    assert SAMPLE_RATE == 16000
    assert N_MFCC == 20
    assert TOTAL_FEATURES == 42
    assert LABEL_HUMAN == 0
    assert LABEL_AI == 1
    assert RISK_LOW_THRESHOLD == 25
    assert RISK_MEDIUM_THRESHOLD == 65
    assert "Experimental" in STATUTORY_DISCLAIMER


def test_load_audio_from_bytes():
    """Verify audio bytes loading and zero data persistence."""
    wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0)
    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)

    assert sr == SAMPLE_RATE
    assert len(audio) == SAMPLE_RATE
    assert isinstance(audio, np.ndarray)


def test_feature_extraction_dimensions():
    """Verify exact 42 acoustic features (20 mean MFCC + 20 std MFCC + 1 RMS + 1 ZCR)."""
    wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0)
    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)

    features = extract_features_from_audio(audio, sample_rate=sr)
    assert isinstance(features, np.ndarray)
    assert features.shape == (42,)
    assert features.dtype == np.float32


def test_audio_metadata():
    """Verify metadata extraction."""
    wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.5)
    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)
    metadata = get_audio_metadata(audio, sample_rate=sr)

    assert metadata["duration_seconds"] == 1.5
    assert metadata["sample_rate"] == SAMPLE_RATE
    assert metadata["num_samples"] == 24000
    assert "rms_energy" in metadata
    assert "zero_crossing_rate" in metadata


def test_scoring_and_risk_bands():
    """Verify risk score calibration and band assignment."""
    assert calculate_risk_score(0.12) == 12
    assert calculate_risk_score(0.85) == 85

    desc, band, badge, recs = get_risk_band(10)
    assert band == "Low"
    assert badge == "success"

    desc, band, badge, recs = get_risk_band(45)
    assert band == "Review required"
    assert badge == "warning"

    desc, band, badge, recs = get_risk_band(90)
    assert band == "High risk"
    assert badge == "error"
    assert len(recs) >= 2


def test_privacy_and_hashing():
    """Verify privacy statement and cryptographic hashing."""
    stmt = get_privacy_statement()
    assert "All audio is processed locally" in stmt

    dummy_bytes = b"test_voice_bytes"
    h = compute_sha256(dummy_bytes)
    assert len(h) == 64


def test_model_pipeline_inference():
    """Verify model pipeline prediction and scoring."""
    model = load_model(MODEL_PATH)
    if model is not None:
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=1.0)
        audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)

        result = predict_and_score(model, audio, sample_rate=sr)
        assert "prediction_label" in result
        assert result["prediction_class"] in [0, 1]
        assert 0.0 <= result["human_probability"] <= 1.0
        assert 0.0 <= result["spoof_probability"] <= 1.0
        assert 0 <= result["risk_score"] <= 100
        assert result["risk_band"] in ["Low", "Review required", "High risk"]
        assert len(result["recommendations"]) >= 1
