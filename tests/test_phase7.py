"""
VoiceShield Phase 7 — Streamlit SOC Analyst Operations Dashboard Test Suite.

Verifies:
  1. app.py Script Compilation & Import Validity
  2. Custom Plotly Circular Gauge Rendering & Color Band Routing
  3. Interactive Waveform Envelope & VAD Boundary Visualization
  4. Log Mel-Spectrogram & 5.5 kHz High-Frequency Cutoff Overlay
  5. Equal Error Rate (EER) ROC / DET Benchmark Curve Rendering
  6. Confusion Matrix Heatmap Rendering
  7. Cryptographically Verifiable Forensic Audit Report Generation (SHA-256)
  8. Offline Backend Health & Metadata Fallback Resilience

Run with:
    pytest tests/test_phase7.py -v
"""

from __future__ import annotations

import ast
import json
from typing import Any, Dict

import numpy as np
import plotly.graph_objects as go
import pytest

from app import (
    fetch_backend_health,
    fetch_backend_metadata,
    generate_forensic_audit_report,
    render_circular_gauge,
    render_confusion_matrix,
    render_melspectrogram_cutoff,
    render_roc_det_curve,
    render_waveform_vad,
)
from src.config import SAMPLE_RATE


# ---------------------------------------------------------------------------
# Test 1: Compilation & AST Validity of app.py
# ---------------------------------------------------------------------------

def test_app_script_compiles_cleanly():
    """Verify that app.py parses cleanly without any syntax errors."""
    with open("app.py", "r", encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    assert tree is not None
    assert len(tree.body) > 0


# ---------------------------------------------------------------------------
# Test 2: Custom Circular Gauge Rendering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "risk_score,spoof_prob,expected_title",
    [
        (15, 0.15, "LOW RISK"),
        (45, 0.45, "REVIEW REQUIRED"),
        (85, 0.85, "HIGH RISK"),
    ],
)
def test_render_circular_gauge(risk_score: int, spoof_prob: float, expected_title: str):
    """Verify circular gauge returns valid Plotly Figure with expected risk levels."""
    fig = render_circular_gauge(risk_score, spoof_prob)
    assert isinstance(fig, go.Figure)
    data = fig.to_dict()
    assert "data" in data
    assert len(data["data"]) > 0
    indicator = data["data"][0]
    assert indicator["type"] == "indicator"
    assert indicator["value"] == risk_score
    assert expected_title in indicator["title"]["text"].upper()


# ---------------------------------------------------------------------------
# Test 3: Waveform Envelope & VAD Shading
# ---------------------------------------------------------------------------

def test_render_waveform_vad():
    """Verify waveform VAD visualizer handles active speech and silence without NaN."""
    # 1. Normal voiced speech
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
    fig = render_waveform_vad(audio, sr=SAMPLE_RATE, voiced_ratio=0.85)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Waveform line + envelope dot line

    # 2. Empty / silence audio
    fig_empty = render_waveform_vad(np.array([], dtype=np.float32), sr=SAMPLE_RATE, voiced_ratio=0.0)
    assert isinstance(fig_empty, go.Figure)


# ---------------------------------------------------------------------------
# Test 4: Mel-Spectrogram & Cutoff Overlay
# ---------------------------------------------------------------------------

def test_render_melspectrogram_cutoff():
    """Verify log Mel-spectrogram visualizer with 5.5 kHz cutoff line."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    fig = render_melspectrogram_cutoff(audio, sr=SAMPLE_RATE)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    assert fig.data[0].type == "heatmap"

    # Verify cutoff annotation / line is in layout
    layout_dict = fig.layout.to_plotly_json()
    assert "shapes" in layout_dict or "annotations" in layout_dict


# ---------------------------------------------------------------------------
# Test 5: Benchmark Visualizations (ROC / DET & Confusion Matrix)
# ---------------------------------------------------------------------------

def test_render_roc_det_and_confusion_matrix():
    """Verify ROC curve and Confusion Matrix figures return valid Plotly charts."""
    fig_roc = render_roc_det_curve()
    assert isinstance(fig_roc, go.Figure)
    assert len(fig_roc.data) >= 2  # VoiceShield + Baseline

    fig_cm = render_confusion_matrix()
    assert isinstance(fig_cm, go.Figure)
    assert fig_cm.data[0].type == "heatmap"


# ---------------------------------------------------------------------------
# Test 6: Compliance Audit Report Generator (SHA-256)
# ---------------------------------------------------------------------------

def test_generate_forensic_audit_report():
    """Verify forensic audit report JSON structure, SHA-256 hash, and metadata."""
    sample_bytes = b"VOICESHIELD_DUMMY_AUDIO_TEST_PAYLOAD"
    prediction_mock = {
        "prediction_label": "AI VOICE CLONE DETECTED",
        "risk_score": 88,
        "risk_band": "High Risk (Likely AI / Cloned Voice)",
        "spoof_probability": 0.88,
        "human_probability": 0.12,
        "risk_description": "Synthetic vocoder spectral holes detected.",
        "diagnostics": {"duration_sec": 2.5, "snr_db": 18.2},
        "forensic_breakdown": {"lpc_kurtosis": 6.8, "phase_entropy": 0.82},
    }

    report = generate_forensic_audit_report(
        audio_bytes=sample_bytes,
        prediction=prediction_mock,
        session_id="test-session-1234",
    )

    assert report["audit_report_id"] == "test-session-1234"
    assert report["file_size_bytes"] == len(sample_bytes)
    assert len(report["audio_sha256"]) == 64  # SHA-256 hex string length
    assert report["forensic_verdict"]["risk_score"] == 88
    assert report["forensic_breakdown"]["lpc_kurtosis"] == 6.8

    # Must be JSON serializable
    json_str = json.dumps(report)
    assert "test-session-1234" in json_str


# ---------------------------------------------------------------------------
# Test 7: Offline Backend Health & Metadata Fallback Resilience
# ---------------------------------------------------------------------------

def test_backend_offline_fallback():
    """Verify dashboard resilience when connecting to offline backend port."""
    health = fetch_backend_health(api_url="http://localhost:59999")
    assert "status" in health
    assert "device" in health
    assert health["target_sr"] == 16000

    metadata = fetch_backend_metadata(api_url="http://localhost:59999")
    assert "architecture" in metadata
    assert "supported_formats" in metadata
    assert "WAV" in metadata["supported_formats"]
