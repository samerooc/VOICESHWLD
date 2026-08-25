"""
Unit and Integration Tests for VoiceShield Neural Step 5 WebSocket & REST Endpoints.
"""

import base64
import io
import os
import sys
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api import app, decode_mulaw_to_16k_pcm
from src.config import SAMPLE_RATE

client = TestClient(app)


def test_decode_mulaw_to_16k_pcm():
    """Verify G.711 mu-law base64 decoding and 2x upsampling to 16kHz linear PCM."""
    # 80 bytes of silence/mu-law
    dummy_payload = base64.b64encode(b"\xff" * 80).decode("utf-8")
    pcm_16k = decode_mulaw_to_16k_pcm(dummy_payload)

    assert pcm_16k.dtype == np.float32
    assert len(pcm_16k) == 160  # 80 samples @ 8kHz -> 160 samples @ 16kHz
    assert np.allclose(pcm_16k, 0.0, atol=1e-3)


def test_websocket_live_stream_binary_pcm():
    """Verify /ws/live-stream endpoint accepts 16-bit linear PCM binary chunks and returns assessment."""
    with client.websocket_connect("/ws/live-stream") as websocket:
        # Generate 400ms of 16-bit 16kHz audio (6400 samples = 12800 bytes)
        t = np.linspace(0, 0.40, int(SAMPLE_RATE * 0.40), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        pcm_bytes = (audio * 32767.0).astype(np.int16).tobytes()

        # Send in two 200ms chunks
        chunk_len = len(pcm_bytes) // 2
        websocket.send_bytes(pcm_bytes[:chunk_len])
        websocket.send_bytes(pcm_bytes[chunk_len:])

        # Receive assessment message
        response = websocket.receive_json()
        assert response["event"] == "assessment"
        assert "smoothed_risk_score" in response
        assert "latency_ms" in response
        assert "audio_flags" in response
        assert response["latency_ms"] < 2500.0


def test_websocket_twilio_media_stream():
    """Verify /ws/twilio-media-stream handles connected, start, media, and stop events."""
    with client.websocket_connect("/ws/twilio-media-stream") as websocket:
        # 1. Connected
        websocket.send_json({"event": "connected", "protocol": "Call"})
        res_conn = websocket.receive_json()
        assert res_conn["event"] == "connected_ack"

        # 2. Start
        websocket.send_json({
            "event": "start",
            "streamSid": "MZ1234567890",
            "start": {"callSid": "CA1234567890", "streamSid": "MZ1234567890"},
        })
        res_start = websocket.receive_json()
        assert res_start["event"] == "start_ack"

        # 3. Media: Send 250ms of audio (2000 mu-law bytes @ 8kHz)
        # 2000 samples @ 8kHz = 4000 samples @ 16kHz (> 200ms eval stride)
        payload = base64.b64encode(b"\x00\xff" * 1000).decode("utf-8")
        websocket.send_json({
            "event": "media",
            "streamSid": "MZ1234567890",
            "media": {"payload": payload, "chunk": "1"},
        })

        res_media = websocket.receive_json()
        assert res_media["event"] == "assessment"
        assert res_media["streamSid"] == "MZ1234567890"
        assert "smoothed_risk_score" in res_media

        # 4. Stop
        websocket.send_json({"event": "stop", "streamSid": "MZ1234567890"})


def test_rest_health_and_metadata():
    """Verify REST /health and /metadata endpoints."""
    res_h = client.get("/health")
    assert res_h.status_code == 200
    assert res_h.json()["status"] == "ok"

    res_m = client.get("/metadata")
    assert res_m.status_code == 200
    meta = res_m.json()
    assert "model_version" in meta
    assert "backbone" in meta
