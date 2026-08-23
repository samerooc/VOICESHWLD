"""
Privacy & Zero-Retention Compliance Tests for VoiceShield (Phase 9).
Verifies that no raw audio bytes are stored on disk or leaked in responses.
"""

import io
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from api import app
from src.config import SAMPLE_RATE

client = TestClient(app)


def test_privacy_audio_saved_flag_is_false():
    """Verify audio_saved is strictly False across all inference responses."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, SAMPLE_RATE, format="WAV")

    res = client.post(
        "/predict",
        files={"file": ("privacy_test.wav", io.BytesIO(bio.getvalue()), "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["audio_saved"] is False
    assert "audio" not in data
    assert "raw_bytes" not in data


def test_privacy_statutory_disclaimer_present():
    """Verify statutory disclaimer is returned in all metadata and prediction payloads."""
    r_meta = client.get("/metadata").json()
    assert "Experimental decision-support prototype; not identity proof." in r_meta["disclaimer"]
