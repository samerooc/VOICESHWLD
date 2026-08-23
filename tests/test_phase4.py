"""
Unit tests for Phase 4: SIH-Quality Streamlit Dashboard & Robust Audio Handling.
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import load_audio_from_bytes
from src.config import (
    MIN_AUDIO_DURATION_SEC,
    MIN_AUDIO_RMS_ENERGY,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SAMPLE_RATE,
)
from src.features import extract_features_from_audio
from src.predict import (
    PHASE4_DISCLAIMER,
    load_metadata,
    load_model,
    predict_audio,
)


def generate_wav_bytes(duration_sec: float = 1.0, sr: int = SAMPLE_RATE, amplitude: float = 0.5) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_too_short_audio_rejection():
    """Verify that audio < 0.5s is rejected with descriptive error."""
    short_wav = generate_wav_bytes(duration_sec=0.2)
    with pytest.raises(ValueError, match="too short"):
        load_audio_from_bytes(short_wav, sample_rate=SAMPLE_RATE)


def test_silent_audio_rejection():
    """Verify that silent audio (near-zero amplitude) is rejected."""
    silent_wav = generate_wav_bytes(duration_sec=1.0, amplitude=0.0)
    with pytest.raises(ValueError, match="completely silent"):
        load_audio_from_bytes(silent_wav, sample_rate=SAMPLE_RATE)


def test_exact_phase4_risk_bands():
    """Verify exact Phase 4 risk bands: Low (0-25), Review required (26-65), High risk (66-100)."""
    model = load_model(MODEL_PATH)
    if model is not None:
        valid_wav = generate_wav_bytes(duration_sec=1.0, amplitude=0.5)
        audio, sr = load_audio_from_bytes(valid_wav, sample_rate=SAMPLE_RATE)
        res = predict_audio(model, audio, sample_rate=sr)

        assert res["risk_band"] in ["Low", "Review required", "High risk"]
        assert "disclaimer" in res
        assert res["disclaimer"] == PHASE4_DISCLAIMER
        assert 0 <= res["risk_score"] <= 100


def test_metadata_and_disclaimer():
    """Verify metadata loading and statutory disclaimer presence."""
    meta = load_metadata(MODEL_METADATA_PATH)
    assert meta is not None
    assert "model_name" in meta
    assert "production_reliability_disclaimer" in meta
