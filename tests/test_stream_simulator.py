"""
Unit Tests for VoiceShield Sandbox Streaming Simulator (simulate_stream.py).
"""

import numpy as np
import pytest

from scripts.simulate_stream import (
    SANDBOX_BANNER,
    StreamSimulator,
    chunk_audio_stream,
)
from src.config import SAMPLE_RATE


def test_chunk_audio_stream_window_and_stride():
    """Verify window size and stride calculations for 160ms/40ms."""
    # 1.0 second of audio = 16000 samples
    audio = np.ones(SAMPLE_RATE, dtype=np.float32) * 0.5
    chunks = list(chunk_audio_stream(audio, sample_rate=SAMPLE_RATE, window_ms=160, stride_ms=40))

    window_samples = int(SAMPLE_RATE * 0.160)  # 2560
    assert len(chunks) > 0
    for idx, timestamp, chunk in chunks:
        assert len(chunk) == window_samples
        assert timestamp >= 0.0


def test_stream_simulator_silent_chunk_handling():
    """Verify that silent chunks are flagged with descriptive skipped reasons."""
    sim = StreamSimulator()
    silent_chunk = np.zeros(2560, dtype=np.float32)
    res = sim.process_window(0, 0.0, silent_chunk, sample_rate=SAMPLE_RATE)

    assert res["is_valid"] is False
    assert "Silence" in res["skipped_reason"]
    assert res["instantaneous_score"] is None


def test_stream_simulator_valid_chunk_and_rolling_score():
    """Verify that active voice chunks update rolling score and calculate latency."""
    sim = StreamSimulator()
    t = np.linspace(0, 0.160, 2560, endpoint=False)
    active_chunk = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    res = sim.process_window(0, 0.0, active_chunk, sample_rate=SAMPLE_RATE)

    assert res["is_valid"] is True
    assert res["instantaneous_score"] is not None
    assert 0.0 <= res["rolling_score"] <= 100.0
    assert res["processing_ms"] > 0.0
    assert res["risk_band"].lower() in ["low", "low risk", "review required", "review", "high risk", "high-risk signal", "high"]


def test_stream_simulator_max_windows_control():
    """Verify that simulation halts cleanly at max_windows limit."""
    sim = StreamSimulator()
    results = sim.run_simulation(
        audio_file="data/test/human/01.wav",
        window_ms=160,
        stride_ms=40,
        max_windows=5,
        simulated_delay_sec=0.0,
    )

    assert len(results) == 5
    assert sim.is_running is False


def test_sandbox_simulation_banner():
    """Verify that sandbox disclaimer is clearly highlighted."""
    assert "SANDBOX SIMULATION — NOT A LIVE CALL" in SANDBOX_BANNER
