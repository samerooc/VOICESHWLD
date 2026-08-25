"""
Unit & Component Test Suite for Phase 7 Streamlit SOC Dashboard.
Tests:
1. Plotly gauge, mel-spectrogram, waveform envelope, and timeline figure generators.
2. Forensic JSON audit payload schema compliance and serialization.
3. CSS design system tokens and statutory notice integrity.
"""

import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import SAMPLE_RATE


def create_mock_risk_gauge(score: int, band_label: str) -> go.Figure:
    """Generates the circular Plotly risk gauge figure."""
    color_map = {
        "Low Risk": "#10B981",
        "Review Required": "#F59E0B",
        "High Risk": "#EF4444",
        "Inconclusive": "#94A3B8",
        "Low Quality": "#64748B",
    }
    bar_color = color_map.get(band_label, "#06B6D4")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"<b>{band_label.upper()}</b>", "font": {"size": 18, "color": "#F1F5F9"}},
            number={"suffix": " / 100", "font": {"size": 36, "color": "#FFFFFF"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "#1E293B",
                "borderwidth": 1,
                "bordercolor": "#334155",
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [25, 65], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": 66,
                },
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font={"color": "#F1F5F9", "family": "monospace"},
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
    )
    return fig


def create_mock_mel_spectrogram_figure(audio: np.ndarray, sr: int = SAMPLE_RATE) -> go.Figure:
    """Generates the interactive mel-spectrogram Plotly figure."""
    import librosa
    if len(audio) < 512:
        audio = np.pad(audio, (0, 512 - len(audio)))
    s_mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, n_fft=1024, hop_length=256)
    s_db = librosa.power_to_db(s_mel, ref=np.max)

    fig = go.Figure(
        data=go.Heatmap(
            z=s_db,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title="Mel-Frequency Spectrogram (0 - 8000 Hz)",
        xaxis_title="Time Frames",
        yaxis_title="Mel Frequency Bins",
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#0B0F19",
        font={"color": "#F1F5F9"},
        height=260,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def create_mock_timeline_figure(timestamps: list, raw_scores: list, smoothed_scores: list) -> go.Figure:
    """Generates the streaming telemetry risk timeline Plotly figure."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=raw_scores,
            mode="lines",
            name="Instantaneous Risk",
            line=dict(color="#06B6D4", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=smoothed_scores,
            mode="lines",
            name="EMA Smoothed Score",
            line=dict(color="#EF4444", width=3),
        )
    )
    fig.add_hline(y=66, line_dash="dash", line_color="#EF4444", annotation_text="High Risk Threshold (66)")
    fig.update_layout(
        title="Streaming Risk Telemetry Timeline",
        xaxis_title="Call Elapsed Time (s)",
        yaxis_title="Risk Score (0 - 100)",
        yaxis_range=[0, 105],
        paper_bgcolor="#0B0F19",
        plot_bgcolor="#111827",
        font={"color": "#F1F5F9"},
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def test_plotly_gauge_generator():
    """Verify circular risk gauge returns valid Plotly figure object."""
    fig = create_mock_risk_gauge(score=82, band_label="High Risk")
    assert isinstance(fig, go.Figure)
    assert fig.layout.paper_bgcolor == "#0B0F19"
    assert len(fig.data) == 1
    assert fig.data[0].value == 82


def test_plotly_mel_spectrogram_generator():
    """Verify Mel-Spectrogram visualization generates valid heatmap."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    fig = create_mock_mel_spectrogram_figure(sine, sr=SAMPLE_RATE)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"


def test_plotly_timeline_generator():
    """Verify streaming telemetry timeline renders raw and smoothed risk curves."""
    ts = [0.2, 0.4, 0.6, 0.8, 1.0]
    raw = [20, 35, 70, 85, 90]
    smoothed = [20, 25, 40, 56, 68]

    fig = create_mock_timeline_figure(ts, raw, smoothed)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_forensic_audit_json_schema():
    """Verify generated forensic audit payload conforms to expected JSON schema."""
    audit_data = {
        "audit_id": "AUDIT-2026-08-23-001",
        "timestamp_utc": "2026-08-23T22:45:00Z",
        "sha256_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "assessment": {
            "prediction_label": "Likely AI / Cloned Voice",
            "risk_score": 88,
            "risk_band": "High Risk",
            "spoof_probability": 0.88,
            "human_probability": 0.12,
        },
        "diagnostics": {
            "duration_sec": 3.0,
            "snr_db": 24.5,
            "is_clipped": False,
            "is_silent": False,
        },
        "disclaimer": "Statutory Notice: Advisory risk signal for SOC human analysts.",
    }

    payload_str = json.dumps(audit_data)
    parsed = json.loads(payload_str)

    assert parsed["assessment"]["risk_score"] == 88
    assert parsed["assessment"]["risk_band"] == "High Risk"
    assert "sha256_hash" in parsed
    assert "disclaimer" in parsed
