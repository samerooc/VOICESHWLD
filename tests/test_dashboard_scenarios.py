"""
Unit & Integration Tests for SIH-Ready Streamlit Dashboard Scenarios.
Tests:
- bona-fide WAV
- spoof WAV
- silent WAV (rejected)
- corrupt WAV (rejected)
- short WAV (rejected)
- uncertain sample (0.40 <= spoof_prob <= 0.60)
"""

import io
import numpy as np
import pytest
import soundfile as sf

from src.audio_io import load_audio_from_bytes
from src.config import MODEL_PATH, SAMPLE_RATE, TOTAL_FEATURES
from src.explainability import build_explainability_report, compute_signal_diagnostics
from src.model import load_model
from src.scoring import predict_and_score


def create_sine_wav(duration: float, freq: float = 440.0, amplitude: float = 0.5, sr: int = SAMPLE_RATE) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_bona_fide_wav_scenario():
    """Verify processing of bona-fide genuine human WAV sample."""
    model = load_model(MODEL_PATH)
    assert model is not None

    with open("data/test/human/01.wav", "rb") as f:
        wav_bytes = f.read()

    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)
    assert len(audio) > 0

    res = predict_and_score(model, audio, sample_rate=sr, decision_threshold=0.40)
    assert res["prediction_label"] in ["Likely Human Voice", "Ambiguous / Review Required"]
    assert res["risk_score"] < 65

    diag = compute_signal_diagnostics(audio, sr)
    assert diag["duration"] > 0
    assert diag["audio_quality"] in ["Standard Broadcast / Clear Audio", "Bandwidth-Limited (Telephony 8kHz Narrowband)", "Degraded (Faint / Heavy Silence)"]


def test_spoof_wav_scenario():
    """Verify processing of spoof synthetic WAV sample."""
    model = load_model(MODEL_PATH)
    assert model is not None

    with open("data/test/ai_voice/1.wav", "rb") as f:
        wav_bytes = f.read()

    audio, sr = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)
    res = predict_and_score(model, audio, sample_rate=sr, decision_threshold=0.40)

    assert res["spoof_probability"] >= 0.40
    assert res["risk_band"] in ["Review required", "High risk"]


def test_silent_wav_scenario():
    """Verify that silent WAV (< 1e-5 RMS) is caught and rejected."""
    silent_bytes = create_sine_wav(duration=1.5, amplitude=0.0)
    with pytest.raises(ValueError, match="completely silent"):
        load_audio_from_bytes(silent_bytes, target_sr=SAMPLE_RATE)


def test_corrupt_wav_scenario():
    """Verify that malformed/corrupted bytes are rejected gracefully."""
    corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPT_DATA_HEADER"
    with pytest.raises(Exception):
        load_audio_from_bytes(corrupt_bytes, target_sr=SAMPLE_RATE)


def test_short_wav_scenario():
    """Verify that short WAV (< 0.5s) is rejected."""
    short_bytes = create_sine_wav(duration=0.2, amplitude=0.5)
    with pytest.raises(ValueError, match="too short"):
        load_audio_from_bytes(short_bytes, target_sr=SAMPLE_RATE)


def test_uncertain_sample_scenario():
    """Verify that borderline sample (0.40 <= spoof_prob <= 0.60) enters UNCERTAIN state."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    fake_pred_uncertain = {
        "prediction_class": 0,
        "prediction_label": "Ambiguous / Review Required",
        "human_probability": 0.52,
        "spoof_probability": 0.48,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }

    report = build_explainability_report(model, audio, SAMPLE_RATE, fake_pred_uncertain)
    assert report["is_uncertain"] is True
    assert "UNCERTAIN — MANUAL REVIEW REQUIRED" in report["uncertainty_banner"]
    assert report["distance_from_threshold"] == 0.08
    assert "Confidence is not calibrated" in report["confidence_status"]
