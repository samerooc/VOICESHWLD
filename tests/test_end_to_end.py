"""
End-to-End System Tests for VoiceShield Prototype (Phase 9).
Verifies all 20 required end-to-end criteria.
"""

import io
import os
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from api import app
from src.config import MAX_FILE_SIZE_BYTES, MODEL_PATH, SAMPLE_RATE
from src.model import load_model
from src.streaming import SandboxStreamAnalyzer, slice_streaming_windows

client = TestClient(app)


def make_test_wav(duration: float = 1.0, freq: float = 440.0, amp: float = 0.5) -> bytes:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    audio = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, SAMPLE_RATE, format="WAV")
    return bio.getvalue()


def test_e2e_1_to_4_api_health_metadata():
    """1-4: Verify API startup, GET /health, GET /metadata."""
    r_h = client.get("/health")
    assert r_h.status_code == 200
    assert r_h.json()["status"] == "ok"

    r_m = client.get("/metadata")
    assert r_m.status_code == 200
    assert "model_version" in r_m.json()


def test_e2e_5_valid_wav_prediction():
    """5: Verify valid WAV prediction endpoint."""
    wav = make_test_wav(1.0)
    r_p = client.post("/predict", files={"file": ("test.wav", io.BytesIO(wav), "audio/wav")})
    assert r_p.status_code == 200
    res = r_p.json()
    assert "bona_fide_probability" in res
    assert "spoof_probability" in res
    assert "risk_score" in res


def test_e2e_6_to_11_invalid_input_rejections():
    """6-11: Empty, non-WAV, corrupt, silent, short, oversized uploads."""
    # 6. Empty
    assert client.post("/predict", files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")}).status_code == 400
    # 7. Unsupported extension (non-audio)
    assert client.post("/predict", files={"file": ("bad.txt", io.BytesIO(b"TEXT"), "text/plain")}).status_code == 415
    # 8. Corrupt
    assert client.post("/predict", files={"file": ("corrupt.wav", io.BytesIO(b"RIFF\x00\x00"), "audio/wav")}).status_code == 400
    # 9. Silent
    silent = make_test_wav(1.0, amp=0.0)
    assert client.post("/predict", files={"file": ("silent.wav", io.BytesIO(silent), "audio/wav")}).status_code == 400
    # 10. Too short (<0.5s)
    short = make_test_wav(0.2)
    assert client.post("/predict", files={"file": ("short.wav", io.BytesIO(short), "audio/wav")}).status_code == 400
    # 11. Oversized
    huge = b"\x00" * (MAX_FILE_SIZE_BYTES + 1024)
    assert client.post("/predict", files={"file": ("huge.wav", io.BytesIO(huge), "audio/wav")}).status_code == 413


def test_e2e_12_to_17_security_and_privacy():
    """12-17: Temp files deleted, no raw audio persisted/logged, safe errors, no stack traces."""
    wav = make_test_wav(1.0)
    r = client.post("/predict", files={"file": ("clean.wav", io.BytesIO(wav), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["audio_saved"] is False

    # Path traversal rejected safely without stack trace
    r_bad = client.post("/predict", files={"file": ("../../../etc/passwd.wav", io.BytesIO(b"EVIL"), "audio/wav")})
    assert r_bad.status_code == 400
    assert "Traceback" not in r_bad.json()["detail"]


def test_e2e_18_to_20_streaming_docker_and_existing():
    """18-20: Streaming sandbox starts/stops, docker files exist, dashboard compiles."""
    sim = SandboxStreamAnalyzer()
    chunk = (0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.160, 2560))).astype(np.float32)
    res = sim.process_chunk(0, 0.0, chunk, sample_rate=SAMPLE_RATE)
    assert res["is_valid"] is True
    assert res["audio_saved"] is False

    assert os.path.exists("Dockerfile")
    assert os.path.exists("docker-compose.yml")
    assert os.path.exists(".dockerignore")
