"""
Tests for VoiceShield Neural Step 4: Real-Time Neural Inference & Streaming Score Engine.
"""

import os
import sys
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.neural_scoring import (
    DEFAULT_NEURAL_CHECKPOINT,
    NeuralStreamingScoreEngine,
    RollingAudioBuffer,
    prepare_waveform_tensor,
)


def test_rolling_audio_buffer_circular_retrieval():
    """Verify that RollingAudioBuffer appends chunks and retrieves fixed 48k sample windows."""
    sr = 16000
    buffer = RollingAudioBuffer(capacity_seconds=4.0, sample_rate=sr)

    # 1. Empty buffer returns 48000 zeros
    empty_win = buffer.get_latest_window(window_samples=48000)
    assert empty_win.shape == (48000,)
    assert np.all(empty_win == 0.0)

    # 2. Add 1.0s chunk (16000 samples)
    chunk_1s = np.ones(16000, dtype=np.float32) * 0.5
    buffer.add_samples(chunk_1s)
    win_padded = buffer.get_latest_window(window_samples=48000)
    assert win_padded.shape == (48000,)
    # Symmetrical zero padding should preserve 16k samples in center
    assert np.sum(win_padded > 0) == 16000

    # 3. Add 4.0s of data (exceeding capacity)
    chunk_4s = np.ones(64000, dtype=np.float32) * 0.8
    buffer.add_samples(chunk_4s)
    win_full = buffer.get_latest_window(window_samples=48000)
    assert win_full.shape == (48000,)
    assert np.allclose(win_full, 0.8)


def test_prepare_waveform_tensor_shape_and_norm():
    """Verify in-memory waveform tensor adapter zero-padding and center cropping."""
    # Sub-length audio (padding)
    short_audio = np.random.randn(24000).astype(np.float32)
    tensor_short = prepare_waveform_tensor(short_audio, target_samples=48000)
    assert tensor_short.shape == (1, 48000)
    assert isinstance(tensor_short, torch.Tensor)

    # Exact length audio
    exact_audio = np.random.randn(48000).astype(np.float32)
    tensor_exact = prepare_waveform_tensor(exact_audio, target_samples=48000)
    assert tensor_exact.shape == (1, 48000)

    # Oversized audio (center cropping)
    long_audio = np.random.randn(64000).astype(np.float32)
    tensor_long = prepare_waveform_tensor(long_audio, target_samples=48000)
    assert tensor_long.shape == (1, 48000)


def test_neural_score_engine_inference_and_latency():
    """Verify forward pass latency < 200ms and 5 risk bands mapping."""
    engine = NeuralStreamingScoreEngine(device="cpu")
    engine.reset()

    # Generate synthetic active audio window (3.0s @ 16kHz)
    t = np.linspace(0, 3, 48000, endpoint=False)
    sine_wave = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    res = engine.predict_stream_window(sine_wave)

    assert "instantaneous_spoof_prob" in res
    assert "ema_spoof_prob" in res
    assert "risk_score" in res
    assert "risk_band" in res
    assert "inference_latency_ms" in res
    assert "is_realtime_compliant" in res

    # Check SLA compliance
    assert res["inference_latency_ms"] < 200.0
    assert res["is_realtime_compliant"] is True
    assert 0 <= res["risk_score"] <= 100


def test_neural_score_engine_silence_handling():
    """Verify silent/empty buffer returns low_quality risk band."""
    engine = NeuralStreamingScoreEngine(device="cpu")
    engine.reset()

    silent_window = np.zeros(48000, dtype=np.float32)
    res = engine.predict_stream_window(silent_window)

    assert res["is_valid"] is False
    assert res["is_silent"] is True
    assert res["risk_band_key"] == "low_quality"
