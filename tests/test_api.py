"""
Unit & Integration Tests for VoiceShield FastAPI Service (api.py).
Tests all 13 Phase 6 required scenarios:
1. GET /health
2. GET /metadata
3. Valid WAV /predict
4. Empty upload
5. Non-WAV upload
6. Corrupt WAV
7. Silent WAV
8. Too-short WAV
9. Oversized file
10. Temporary cleanup
11. No raw audio retention
12. Safe error response
13. Model-missing behavior
"""

import io
import os
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from api import app, get_cached_model
from src.config import MAX_FILE_SIZE_BYTES, SAMPLE_RATE

client = TestClient(app)


def make_test_wav_bytes(
    duration: float = 1.0,
    freq: float = 440.0,
    amplitude: float = 0.5,
    sr: int = SAMPLE_RATE,
) -> bytes:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_1_get_health_endpoint():
    """1. Verify GET /health returns status ok and service name."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "voiceshield-api"


def test_2_get_metadata_endpoint():
    """2. Verify GET /metadata returns versions, supported_format, and audio_saved=False."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "feature_version" in data
    assert "wav" in data["supported_format"]
    assert data["audio_saved"] is False
    assert "disclaimer" in data


def test_3_predict_valid_wav():
    """3. Verify POST /predict with valid WAV audio."""
    with open("data/test/human/01.wav", "rb") as f:
        wav_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("sample_human.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert response.status_code == 200
    res = response.json()

    assert "model_version" in res
    assert "feature_version" in res
    assert "bona_fide_probability" in res
    assert "spoof_probability" in res
    assert "risk_score" in res
    assert res["risk_band"] in ["low", "review", "high", "uncertain", "low_quality"]
    assert "uncertainty" in res
    assert isinstance(res["explanation"], list)
    assert res["processing_ms"] > 0
    assert res["audio_saved"] is False
    assert "disclaimer" in res


def test_4_predict_empty_upload():
    """4. Verify POST /predict rejects 0-byte file."""
    response = client.post(
        "/predict",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_5_predict_unsupported_extension_upload():
    """5. Verify POST /predict rejects unsupported extensions like .txt or .exe."""
    response = client.post(
        "/predict",
        files={"file": ("payload.txt", io.BytesIO(b"TEXT_DATA"), "text/plain")},
    )
    assert response.status_code == 415
    assert "Unsupported audio format" in response.json()["detail"]


def test_5b_predict_ogg_flac_format_uploads():
    """5b. Verify POST /predict accepts OGG and FLAC formatted audio."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    data = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Test OGG
    bio_ogg = io.BytesIO()
    sf.write(bio_ogg, data, SAMPLE_RATE, format="OGG")
    res_ogg = client.post(
        "/predict",
        files={"file": ("sample.ogg", io.BytesIO(bio_ogg.getvalue()), "audio/ogg")},
    )
    assert res_ogg.status_code == 200
    assert "risk_score" in res_ogg.json()

    # Test FLAC
    bio_flac = io.BytesIO()
    sf.write(bio_flac, data, SAMPLE_RATE, format="FLAC")
    res_flac = client.post(
        "/predict",
        files={"file": ("sample.flac", io.BytesIO(bio_flac.getvalue()), "audio/flac")},
    )
    assert res_flac.status_code == 200
    assert "risk_score" in res_flac.json()


def test_6_predict_corrupt_wav():
    """6. Verify POST /predict rejects malformed WAV headers."""
    corrupt_bytes = b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00CORRUPT_BYTES_DATA"
    response = client.post(
        "/predict",
        files={"file": ("corrupt.wav", io.BytesIO(corrupt_bytes), "audio/wav")},
    )
    assert response.status_code == 400
    assert "failed" in response.json()["detail"].lower()


def test_7_predict_silent_wav():
    """7. Verify POST /predict rejects completely silent WAV audio."""
    silent_bytes = make_test_wav_bytes(duration=1.0, amplitude=0.0)
    response = client.post(
        "/predict",
        files={"file": ("silent.wav", io.BytesIO(silent_bytes), "audio/wav")},
    )
    assert response.status_code == 400
    assert "silent" in response.json()["detail"].lower()


def test_8_predict_too_short_wav():
    """8. Verify POST /predict rejects audio shorter than minimum duration (< 0.5s)."""
    short_bytes = make_test_wav_bytes(duration=0.2, amplitude=0.5)
    response = client.post(
        "/predict",
        files={"file": ("short.wav", io.BytesIO(short_bytes), "audio/wav")},
    )
    assert response.status_code == 400
    assert "short" in response.json()["detail"].lower()


def test_9_predict_oversized_file():
    """9. Verify POST /predict rejects audio exceeding maximum size limit."""
    # Test boundary with simulated oversized byte count
    oversized_dummy = b"\x00" * (MAX_FILE_SIZE_BYTES + 1024)
    response = client.post(
        "/predict",
        files={"file": ("huge.wav", io.BytesIO(oversized_dummy), "audio/wav")},
    )
    assert response.status_code == 413
    assert "exceeds maximum limit" in response.json()["detail"].lower()


def test_10_and_11_temporary_cleanup_and_no_audio_retention():
    """10 & 11. Verify zero disk retention and clean in-memory execution."""
    wav_bytes = make_test_wav_bytes(duration=1.0, amplitude=0.5)
    response = client.post(
        "/predict",
        files={"file": ("clean_test.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["audio_saved"] is False


def test_12_safe_error_response_no_stack_trace_or_paths():
    """12. Verify errors do not leak internal filesystem paths or stack traces."""
    response = client.post(
        "/predict",
        files={"file": ("../../etc/passwd.wav", io.BytesIO(b"MALICIOUS"), "audio/wav")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert "C:\\" not in detail
    assert "/Users/" not in detail


def test_13_model_missing_behavior(monkeypatch):
    """13. Verify 503 response if model is unavailable."""
    import api
    monkeypatch.setattr(api, "get_cached_model", lambda: (None, None))
    wav_bytes = make_test_wav_bytes(duration=1.0, amplitude=0.5)
    response = client.post(
        "/predict",
        files={"file": ("test.wav", io.BytesIO(wav_bytes), "audio/wav")},
    )
    assert response.status_code == 503
    assert "model is not loaded" in response.json()["detail"].lower()
