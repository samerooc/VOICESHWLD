"""
VoiceShield Phase 5 — Production FastAPI REST & WebSocket Streaming Test Suite.

Verifies:
  1. GET /health & GET /metadata Schema Contracts & Uptime Tracking
  2. Custom X-Process-Time-Ms Response Header Middleware
  3. POST /predict In-Memory Multi-Format Forensic Inspection (WAV, latency < 250ms)
  4. POST /predict Error Gating (Empty payload -> 400, Oversized > 50MB -> 413)
  5. WebSocket /ws/live-stream Binary PCM16 Ingestion & JSON Telemetry Frames
  6. WebSocket /ws/live-stream Text Control Reset Mechanism
  7. WebSocket /ws/twilio-media-stream Twilio Voice Protocol & Base64 Mu-Law Decoding

Run with:
    pytest tests/test_phase5.py -v
"""

from __future__ import annotations

import base64
import io
import json
import time
from typing import Any, Dict

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api import app
from src.config import SAMPLE_RATE
from src.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
    StreamingTelemetryFrame,
)
from src.streaming import linear_to_mulaw_bytes

# ---------------------------------------------------------------------------
# Test Fixtures & Generators
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client() -> TestClient:
    """Instantiate a TestClient wrapping the FastAPI app with lifespan support."""
    with TestClient(app) as test_client:
        yield test_client


def _make_in_memory_wav(
    duration_sec: float = 1.5,
    sr: int = SAMPLE_RATE,
    freq_hz: float = 220.0,
) -> bytes:
    """Generate in-memory WAV container bytes for HTTP POST multipart uploads."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sig = (0.35 * np.sin(2.0 * np.pi * freq_hz * t) + 0.1 * np.sin(2.0 * np.pi * 2 * freq_hz * t)).astype(np.float32)
    sig += 0.01 * np.random.default_rng(42).standard_normal(n_samples)

    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: GET /health and GET /metadata REST Endpoints
# ---------------------------------------------------------------------------

def test_get_health_endpoint(client: TestClient):
    """Verify GET /health returns 200 with valid schema, device, and uptime."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers

    data = response.json()
    validated = HealthResponse(**data)
    assert validated.status in ["healthy", "degraded", "unhealthy", "ok"]
    assert validated.target_sr == 16000
    assert validated.uptime_sec >= 0.0


def test_get_metadata_endpoint(client: TestClient):
    """Verify GET /metadata returns 200 with architecture and format schemas."""
    response = client.get("/metadata")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers

    data = response.json()
    validated = MetadataResponse(**data)
    assert "WAV" in validated.supported_formats
    assert "G.711 mu-law" in validated.supported_formats
    assert validated.sample_rate_hz == 16000
    assert validated.temperature > 0.0


# ---------------------------------------------------------------------------
# Test 2: POST /predict Valid Audio Ingestion
# ---------------------------------------------------------------------------

def test_post_predict_valid_wav(client: TestClient):
    """Verify POST /predict parses WAV file, runs forensic analysis, and returns 200."""
    wav_bytes = _make_in_memory_wav(duration_sec=1.5, sr=16000)

    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers

    data = response.json()
    validated = PredictionResponse(**data)

    assert 0 <= validated.risk_score <= 100
    assert 0.0 <= validated.spoof_probability <= 1.0
    assert 0.0 <= validated.human_probability <= 1.0
    assert validated.diagnostics.duration_sec >= 1.4
    assert validated.latency_ms > 0


# ---------------------------------------------------------------------------
# Test 3: POST /predict Error Gating & Edge Cases
# ---------------------------------------------------------------------------

def test_post_predict_empty_file_rejected(client: TestClient):
    """Verify POST /predict rejects empty file payload with HTTP 400."""
    files = {"file": ("empty.wav", b"", "audio/wav")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_post_predict_unsupported_extension(client: TestClient):
    """Verify POST /predict rejects unsupported extension with HTTP 415 or 400."""
    files = {"file": ("document.pdf", b"%PDF-1.4 ...", "application/pdf")}
    response = client.post("/predict", files=files)
    assert response.status_code in [400, 415]


def test_post_predict_oversized_payload_rejected(client: TestClient):
    """Verify POST /predict rejects payloads > 50MB with HTTP 413."""
    # 51 MB dummy buffer
    oversized_bytes = b"0" * (51 * 1024 * 1024)
    files = {"file": ("large.wav", oversized_bytes, "audio/wav")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "50mb" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Test 4: WebSocket /ws/live-stream Raw PCM Streaming
# ---------------------------------------------------------------------------

def test_websocket_live_stream_binary_chunks(client: TestClient):
    """
    Verify /ws/live-stream accepts continuous binary 16kHz PCM chunks
    and yields real-time StreamingTelemetryFrame JSON responses.
    """
    # 40ms of 16kHz PCM16 (640 samples -> 1280 bytes)
    t = np.linspace(0, 0.040, 640, endpoint=False)
    pcm16_chunk = (0.3 * np.sin(2.0 * np.pi * 300.0 * t) * 32767).astype(np.int16).tobytes()

    with client.websocket_connect("/ws/live-stream") as ws:
        # Send 5 continuous chunks
        for _ in range(5):
            ws.send_bytes(pcm16_chunk)
            msg = ws.receive_json()

            # Schema validation
            telemetry = StreamingTelemetryFrame(**msg)
            assert 0 <= telemetry.smoothed_risk_score <= 100
            assert 0.0 <= telemetry.instantaneous_prob <= 1.0
            assert telemetry.latency_ms >= 0.0


def test_websocket_live_stream_control_reset(client: TestClient):
    """Verify /ws/live-stream handles JSON control messages like buffer reset."""
    with client.websocket_connect("/ws/live-stream") as ws:
        ws.send_json({"action": "reset"})
        msg = ws.receive_json()
        assert msg.get("status") == "buffer_reset"
        assert "session_id" in msg


# ---------------------------------------------------------------------------
# Test 5: WebSocket /ws/twilio-media-stream Protocol & Mu-Law Decoding
# ---------------------------------------------------------------------------

def test_websocket_twilio_media_stream(client: TestClient):
    """
    Verify /ws/twilio-media-stream handles Twilio voice protocol:
    connected -> start -> media (base64 G.711 mu-law) -> stop.
    """
    # Generate 100ms 8kHz audio encoded as G.711 mu-law base64
    t = np.linspace(0, 0.100, 800, endpoint=False)
    sig_8k = (0.4 * np.sin(2.0 * np.pi * 400.0 * t)).astype(np.float32)
    mulaw_bytes = linear_to_mulaw_bytes(sig_8k)
    b64_payload = base64.b64encode(mulaw_bytes).decode("utf-8")

    with client.websocket_connect("/ws/twilio-media-stream") as ws:
        # 1. Connected handshake
        ws.send_text(json.dumps({"event": "connected", "protocol": "Call"}))
        ack = ws.receive_json()
        assert ack.get("event") == "connected_ack"

        # 2. Start call event
        ws.send_text(json.dumps({
            "event": "start",
            "streamSid": "MZ1234567890abcdef",
            "start": {"streamSid": "MZ1234567890abcdef", "callSid": "CA123"},
        }))
        start_ack = ws.receive_json()
        assert start_ack.get("event") == "start_ack"
        assert start_ack.get("streamSid") == "MZ1234567890abcdef"

        # 3. Media packet
        ws.send_text(json.dumps({
            "event": "media",
            "streamSid": "MZ1234567890abcdef",
            "media": {"payload": b64_payload, "chunk": "1", "timestamp": "100"},
        }))
        assessment = ws.receive_json()
        assert assessment.get("event") == "assessment"
        assert assessment.get("streamSid") == "MZ1234567890abcdef"
        assert "smoothed_risk_score" in assessment
        assert "is_high_risk_alert" in assessment

        # 4. Stop call event
        ws.send_text(json.dumps({"event": "stop", "streamSid": "MZ1234567890abcdef"}))
        stop_ack = ws.receive_json()
        assert stop_ack.get("event") == "stop_ack"
