"""
Unit & Integration Tests for VoiceShield Sandbox Streaming Engine (src/streaming.py).
Tests:
- valid WAV
- short WAV (padding)
- silent window (skipping)
- missing / empty frame
- malformed input
- rolling score (EMA)
- stop behavior
- cleanup behavior (audio_saved: false)
"""

import numpy as np
import pytest

from src.config import SAMPLE_RATE
from src.streaming import (
    STREAMING_DISCLAIMER,
    SandboxStreamAnalyzer,
    slice_streaming_windows,
)


def test_streaming_valid_wav_slicing():
    """1. Verify windowing and stride over 1 second valid audio."""
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, SAMPLE_RATE))).astype(np.float32)
    windows = list(slice_streaming_windows(audio, sample_rate=SAMPLE_RATE, window_ms=160, stride_ms=40))

    assert len(windows) > 0
    window_samples = int(SAMPLE_RATE * 0.160)
    for idx, ts, chunk in windows:
        assert len(chunk) == window_samples
        assert ts >= 0.0


def test_streaming_short_wav_padding():
    """2. Verify that audio shorter than 160ms is safely zero-padded."""
    short_audio = np.ones(500, dtype=np.float32) * 0.5  # 500 samples < 2560
    windows = list(slice_streaming_windows(short_audio, sample_rate=SAMPLE_RATE, window_ms=160, stride_ms=40))

    assert len(windows) == 1
    idx, ts, chunk = windows[0]
    assert len(chunk) == int(SAMPLE_RATE * 0.160)
    assert ts == 0.0


def test_streaming_silent_window_skipping():
    """3. Verify silent window is skipped with clear descriptive reason."""
    analyzer = SandboxStreamAnalyzer()
    silent_chunk = np.zeros(2560, dtype=np.float32)

    res = analyzer.process_chunk(0, 0.0, silent_chunk, sample_rate=SAMPLE_RATE)
    assert res["is_valid"] is False
    assert "Silence" in res["skipped_reason"]
    assert res["audio_saved"] is False


def test_streaming_missing_empty_frame():
    """4. Verify missing or empty frame is handled gracefully."""
    analyzer = SandboxStreamAnalyzer()
    res = analyzer.process_chunk(0, 0.0, np.array([], dtype=np.float32), sample_rate=SAMPLE_RATE)

    assert res["is_valid"] is False
    assert "Empty" in res["skipped_reason"]
    assert res["audio_saved"] is False


def test_streaming_malformed_input():
    """5. Verify malformed array (e.g. NaN/Inf) is handled gracefully without crash."""
    analyzer = SandboxStreamAnalyzer()
    malformed_chunk = np.full(2560, np.nan, dtype=np.float32)

    res = analyzer.process_chunk(0, 0.0, malformed_chunk, sample_rate=SAMPLE_RATE)
    assert res is not None
    assert res["audio_saved"] is False


def test_streaming_rolling_score_ema():
    """6. Verify Exponential Moving Average (EMA) rolling score updates properly."""
    analyzer = SandboxStreamAnalyzer(smoothing_alpha=0.50)
    t = np.linspace(0, 0.160, 2560, endpoint=False)
    chunk = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    res1 = analyzer.process_chunk(0, 0.0, chunk, sample_rate=SAMPLE_RATE)
    assert res1["is_valid"] is True
    assert 0.0 <= res1["rolling_risk_score"] <= 100.0

    res2 = analyzer.process_chunk(1, 0.040, chunk, sample_rate=SAMPLE_RATE)
    assert res2["is_valid"] is True
    assert res2["window_number"] == 2


def test_streaming_stop_and_reset_behavior():
    """7. Verify analyzer reset and processed window tracking."""
    analyzer = SandboxStreamAnalyzer()
    t = np.linspace(0, 0.160, 2560, endpoint=False)
    chunk = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    analyzer.process_chunk(0, 0.0, chunk, sample_rate=SAMPLE_RATE)
    assert analyzer.processed_windows == 1

    analyzer.reset()
    assert analyzer.processed_windows == 0
    assert analyzer.rolling_score == 0.0


def test_streaming_cleanup_and_disclaimer():
    """8. Verify ephemeral cleanup flag and visible disclaimer."""
    analyzer = SandboxStreamAnalyzer()
    chunk = np.ones(2560, dtype=np.float32) * 0.2
    res = analyzer.process_chunk(0, 0.0, chunk, sample_rate=SAMPLE_RATE)

    assert res["audio_saved"] is False
    assert "SANDBOX SIMULATION — NOT A LIVE CALL" in STREAMING_DISCLAIMER
