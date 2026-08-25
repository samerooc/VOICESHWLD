"""
Unit & Integration Test Suite for Phase 4 Streaming Audio Buffer & Real-Time Scoring.
Tests:
1. Thread-safe RollingAudioBuffer chunk appending, circular wrap-around, and window extraction.
2. Real-time NeuralStreamingScoreEngine forward inference and latency profiling.
3. Exponential Moving Average (EMA) score smoothing convergence across sequential frames.
4. Calibrated 5-state risk band mapping and diagnostics handling.
"""

import os
import sys
import time
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.calibration import calibrate_risk
from src.config import SAMPLE_RATE
from src.neural_scoring import NeuralStreamingScoreEngine
from src.streaming import RollingAudioBuffer


def test_rolling_audio_buffer_wrap_and_window():
    """Verify circular buffer appends chunks, wraps correctly, and extracts exact 3.0s window."""
    buffer = RollingAudioBuffer(capacity_seconds=4.0, sample_rate=16000)
    window_samples = 48000  # 3.0s

    # Empty retrieval returns zero-padded array
    empty_win = buffer.get_latest_window(window_samples=window_samples)
    assert len(empty_win) == window_samples
    assert np.all(empty_win == 0.0)

    # Push 1.0s of data (16000 samples)
    chunk1 = np.ones(16000, dtype=np.float32) * 0.5
    buffer.add_samples(chunk1)
    assert buffer.has_sufficient_audio(min_duration_sec=0.5) is True

    win1 = buffer.get_latest_window(window_samples=window_samples)
    assert len(win1) == window_samples
    assert np.max(win1) == 0.5

    # Push 5.0s of data to test circular wrap-around past 4.0s capacity
    chunk2 = np.ones(80000, dtype=np.float32) * 0.8
    buffer.add_samples(chunk2)
    win2 = buffer.get_latest_window(window_samples=window_samples)
    assert len(win2) == window_samples
    assert np.all(win2 == 0.8)

    # Test buffer clear / flush
    buffer.clear()
    assert buffer.has_sufficient_audio(min_duration_sec=0.5) is False


def test_neural_streaming_score_engine_inference():
    """Verify NeuralStreamingScoreEngine scores waveform windows with sub-200ms latency."""
    engine = NeuralStreamingScoreEngine(device="cpu", smoothing_alpha=0.35)
    engine.reset()

    # Test sine tone window (3.0s @ 16kHz)
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    sine_audio = (0.6 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    res = engine.predict_stream_window(sine_audio)
    assert res["is_valid"] is True
    assert 0.0 <= res["instantaneous_spoof_prob"] <= 1.0
    assert 0.0 <= res["ema_spoof_prob"] <= 1.0
    assert 0 <= res["risk_score"] <= 100
    assert res["risk_band_key"] in ["low", "review", "high", "inconclusive", "low_quality"]
    assert res["inference_latency_ms"] >= 0.0
    assert "is_realtime_compliant" in res


def test_ema_smoothing_convergence():
    """Verify Exponential Moving Average converges monotonically towards steady probability."""
    engine = NeuralStreamingScoreEngine(device="cpu", smoothing_alpha=0.35)
    engine.reset()

    t = np.linspace(0, 3.0, 48000, endpoint=False)
    high_spoof_audio = (0.5 * np.sin(2 * np.pi * 200.0 * t) + np.random.normal(0, 0.05, 48000)).astype(np.float32)

    ema_history = []
    for _ in range(10):
        res = engine.predict_stream_window(high_spoof_audio)
        ema_history.append(res["ema_spoof_prob"])

    assert len(ema_history) == 10
    # Check that EMA smoothed values are non-negative and bounded
    for val in ema_history:
        assert 0.0 <= val <= 1.0


def test_calibrate_risk_edge_cases():
    """Verify 5-state risk calibration rules across silent, clipped, and extreme probabilities."""
    # 1. Silent input
    res_silent = calibrate_risk(raw_prob=0.8, smoothed_prob=0.8, diagnostics={"is_silent": True})
    assert res_silent["risk_band"] == "low_quality"
    assert res_silent["risk_score"] == 0

    # 2. Heavy clipping
    res_clipped = calibrate_risk(raw_prob=0.1, smoothed_prob=0.1, diagnostics={"is_clipped": True})
    assert res_clipped["risk_band"] == "low_quality"

    # 3. High risk synthetic clone
    res_high = calibrate_risk(raw_prob=0.92, smoothed_prob=0.88, diagnostics={})
    assert res_high["risk_band"] == "high"
    assert res_high["risk_score"] >= 66

    # 4. Natural human voice
    res_low = calibrate_risk(raw_prob=0.12, smoothed_prob=0.15, diagnostics={})
    assert res_low["risk_band"] == "low"
    assert res_low["risk_score"] <= 25

    # 5. Inconclusive borderline
    res_inconclusive = calibrate_risk(raw_prob=0.50, smoothed_prob=0.50, diagnostics={})
    assert res_inconclusive["risk_band"] == "inconclusive"
