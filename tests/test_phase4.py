"""
VoiceShield Phase 4 — High-Throughput Streaming Engine Integration Test Suite.

Verifies:
  1. Thread-Safe RollingAudioBuffer High-Frequency Ingestion (40ms chunks @ 25Hz without memory growth)
  2. G.711 Mu-Law Telephony Ingestion (8kHz mulaw -> 16kHz float32 linear PCM)
  3. Linear PCM16 & Float32 Ingestion with arbitrary chunk sizes (20ms to 200ms)
  4. Top-K (85th Percentile) & Temporal EMA Score Aggregation
  5. Hold-and-Decay Security Alert Gate (1.0s synthetic burst triggers 3.0s hold)
  6. Sub-150ms GPU / Sub-250ms Multi-Threaded CPU Latency Compliance
  7. Multi-Threaded Concurrent Writer / Reader Thread Safety
  8. Stateful Session Isolation & Reset Verification

Run with:
    pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import SAMPLE_RATE
from src.streaming import (
    LiveStreamingEngine,
    RollingAudioBuffer,
    decode_mulaw_bytes,
    linear_to_mulaw_bytes,
)

# ---------------------------------------------------------------------------
# Synthetic Audio Generators for Streaming Tests
# ---------------------------------------------------------------------------

def _make_clean_human_chunk(duration_sec: float = 0.040, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate 40ms of natural voiced harmonic audio (low spoof probability)."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sig = 0.25 * np.sin(2.0 * np.pi * 180.0 * t) + 0.12 * np.sin(2.0 * np.pi * 360.0 * t)
    sig += 0.005 * np.random.default_rng(42).standard_normal(n_samples)
    return sig.astype(np.float32)


def _make_synthetic_burst_chunk(duration_sec: float = 0.040, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate 40ms of robotic / pure sine tone (triggers high spoof probability)."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    # Pure tone with minimal jitter -> typical vocoder artifact
    return (0.35 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: RollingAudioBuffer High-Frequency Ingestion (25Hz / 40ms)
# ---------------------------------------------------------------------------

def test_rolling_buffer_high_frequency_writes():
    """
    Verify that RollingAudioBuffer handles 100 continuous 40ms chunks @ 25Hz
    without memory growth, capping exactly at 6.0 seconds (96,000 samples).
    """
    buffer = RollingAudioBuffer(capacity_seconds=6.0, sample_rate=16000)
    chunk_40ms = _make_clean_human_chunk(0.040, 16000)  # 640 samples

    # Ingest 200 chunks (8.0s of audio into a 6.0s capacity buffer)
    for _ in range(200):
        buffer.add_samples(chunk_40ms)

    # Buffer must clamp to exactly capacity (96,000 samples = 6.0s)
    assert len(buffer._buffer) == 96000
    assert buffer.get_current_duration() == 6.0
    assert buffer.has_sufficient_audio(min_duration_sec=3.0) is True

    # Check 3.0s window extraction geometry
    win = buffer.get_analysis_window(3.0)
    assert len(win) == 48000
    assert isinstance(win, np.ndarray)
    assert win.dtype == np.float32


# ---------------------------------------------------------------------------
# Test 2: G.711 Mu-Law Telephony Ingestion & Resampling
# ---------------------------------------------------------------------------

def test_mulaw_telephony_ingestion():
    """
    Verify that 8kHz G.711 mu-law telephony chunks are decoded and
    accurately resampled to 16kHz float32 linear PCM.
    """
    buffer = RollingAudioBuffer(capacity_seconds=4.0, sample_rate=16000)

    # 1. Generate 8kHz audio (100ms = 800 samples)
    t_8k = np.linspace(0, 0.100, 800, endpoint=False)
    sig_8k = (0.4 * np.sin(2.0 * np.pi * 300.0 * t_8k)).astype(np.float32)
    mulaw_bytes = linear_to_mulaw_bytes(sig_8k)

    # 2. Verify standalone mu-law decoder
    decoded = decode_mulaw_bytes(mulaw_bytes)
    assert len(decoded) == 800
    assert decoded.dtype == np.float32
    assert -1.0 <= decoded.min() <= 1.0
    assert -1.0 <= decoded.max() <= 1.0

    # 3. Add to rolling buffer with 8kHz input_sr
    buffer.add_pcm_chunk(mulaw_bytes, format="mulaw", input_sr=8000)

    # 4. Ingested 800 samples @ 8kHz -> 1600 samples @ 16kHz
    assert len(buffer._buffer) == 1600
    assert np.isclose(buffer.get_current_duration(), 0.100, atol=0.005)


# ---------------------------------------------------------------------------
# Test 3: Multi-Format Linear PCM16 & Float32 Ingestion
# ---------------------------------------------------------------------------

def test_multi_format_pcm_ingestion():
    """Verify ingestion of int16 PCM and float32 PCM across various chunk sizes."""
    buffer = RollingAudioBuffer(capacity_seconds=3.0, sample_rate=16000)

    # 1. Ingest 20ms of PCM16 (320 samples -> 640 bytes)
    samples_20ms = (np.sin(np.linspace(0, 10, 320)) * 0.5).astype(np.float32)
    pcm16_bytes = (samples_20ms * 32767).astype(np.int16).tobytes()
    buffer.add_pcm_chunk(pcm16_bytes, format="pcm16", input_sr=16000)
    assert len(buffer._buffer) == 320

    # 2. Ingest 200ms of Float32 (3200 samples -> 12800 bytes)
    samples_200ms = (np.sin(np.linspace(0, 100, 3200)) * 0.3).astype(np.float32)
    f32_bytes = samples_200ms.tobytes()
    buffer.add_pcm_chunk(f32_bytes, format="float32", input_sr=16000)
    assert len(buffer._buffer) == 320 + 3200


# ---------------------------------------------------------------------------
# Test 4: Top-K Percentile & Temporal EMA Smoothing
# ---------------------------------------------------------------------------

def test_topk_and_ema_smoothing():
    """
    Verify Top-K (85th percentile) and EMA aggregation:
      EMA_t = 0.35 * P_t + 0.65 * EMA_{t-1}
      Score_live = 0.70 * TopK_85 + 0.30 * EMA_t
    """
    mock_detector = MagicMock()
    # Step 1: P_1 = 0.20
    # Step 2: P_2 = 0.80
    mock_detector.predict.side_effect = [
        {"spoof_probability": 0.20, "forensic_breakdown": {}, "diagnostics": {}},
        {"spoof_probability": 0.80, "forensic_breakdown": {}, "diagnostics": {}},
    ]

    engine = LiveStreamingEngine(detector=mock_detector, ema_alpha=0.35)

    # Ingest 3.0s of active audio
    engine.ingest_pcm_chunk(np.ones(48000, dtype=np.float32) * 0.2)

    # Step 1
    res1 = engine.process_streaming_step()
    assert res1["instantaneous_prob"] == 0.20
    assert res1["ema_prob"] == 0.20
    assert res1["top_k_prob"] == 0.20
    assert res1["smoothed_risk_score"] == 20

    # Step 2: EMA = 0.35 * 0.80 + 0.65 * 0.20 = 0.28 + 0.13 = 0.41
    # Top-K (85th of [0.20, 0.80]) = 0.71
    # Combined = 0.70 * 0.71 + 0.30 * 0.41 = 0.497 + 0.123 = 0.62 -> 62 (High Risk)
    res2 = engine.process_streaming_step()
    assert res2["instantaneous_prob"] == 0.80
    assert np.isclose(res2["ema_prob"], 0.41, atol=0.01)
    assert res2["smoothed_risk_score"] >= 61
    assert res2["is_alert_held"] is True
    assert res2["alert_hold_counter"] == 6


# ---------------------------------------------------------------------------
# Test 5: Hold-and-Decay Security Alert Gate
# ---------------------------------------------------------------------------

def test_hold_and_decay_security_alert_gate():
    """
    Verify that a 1.0s synthetic burst triggers the High-Risk alert,
    maintains it across Top-K and the full hold period (6 steps), and then decays.
    """
    mock_detector = MagicMock()

    # Sequence of probabilities:
    # 1 High-Risk step (0.95), followed by 12 Low-Risk steps (0.05)
    mock_detector.predict.side_effect = [
        {"spoof_probability": 0.95, "forensic_breakdown": {}, "diagnostics": {}},
    ] + [
        {"spoof_probability": 0.05, "forensic_breakdown": {}, "diagnostics": {}}
    ] * 12

    engine = LiveStreamingEngine(
        detector=mock_detector,
        hold_steps=6,
        decay_rate=0.05,
    )

    # Ingest 3.0s audio
    engine.ingest_pcm_chunk(np.ones(48000, dtype=np.float32) * 0.2)

    # Step 1: High-Risk burst (0.95) -> Triggers alert lock & hold_counter = 6
    res1 = engine.process_streaming_step()
    assert res1["smoothed_risk_score"] >= 61
    assert res1["is_alert_held"] is True
    assert res1["alert_hold_counter"] == 6

    # Steps 2 to 9: Score and hold state are maintained (>= 61)
    for step_num in range(2, 10):
        res = engine.process_streaming_step()
        assert res["is_alert_held"] is True, f"Alert dropped early at step {step_num}"
        assert res["smoothed_risk_score"] >= 61, f"Score dropped below high-risk boundary at step {step_num}"

    # Step 10+: Hold counter has now reached 0, so decay begins
    res10 = engine.process_streaming_step()
    assert res10["is_alert_held"] is False
    assert res10["alert_hold_counter"] == 0
    assert res10["smoothed_risk_score"] < res1["smoothed_risk_score"]


# ---------------------------------------------------------------------------
# Test 6: Sub-250ms Latency Compliance & Telemetry Contract
# ---------------------------------------------------------------------------

def test_streaming_step_latency_and_telemetry_schema():
    """Verify live inference latency is compliant and telemetry payload is complete."""
    engine = LiveStreamingEngine()

    # Feed 3.0s audio
    audio = _make_clean_human_chunk(3.0, 16000)
    engine.ingest_pcm_chunk(audio)

    res = engine.process_streaming_step()

    # Schema contract
    required_keys = [
        "session_id", "timestamp_sec", "window_index", "instantaneous_prob",
        "ema_prob", "top_k_prob", "smoothed_risk_score", "risk_score",
        "risk_band", "badge_class", "is_alert_held", "alert_hold_counter",
        "latency_ms", "is_realtime_compliant", "disclaimer",
    ]
    for key in required_keys:
        assert key in res, f"Missing telemetry key: {key}"

    assert res["latency_ms"] > 0
    assert res["is_realtime_compliant"] is True
    assert 0 <= res["smoothed_risk_score"] <= 100


# ---------------------------------------------------------------------------
# Test 7: Multi-Threaded Concurrent Writer & Reader Safety
# ---------------------------------------------------------------------------

def test_concurrent_writer_and_reader_thread_safety():
    """
    Verify that concurrent writer thread pushing 40ms chunks and reader
    thread executing streaming steps run without deadlocks, exceptions, or race conditions.
    """
    engine = LiveStreamingEngine()
    stop_event = threading.Event()
    exceptions = []

    def writer_worker():
        try:
            chunk = _make_clean_human_chunk(0.040, 16000)
            for _ in range(50):
                if stop_event.is_set():
                    break
                engine.ingest_pcm_chunk(chunk)
                time.sleep(0.005)
        except Exception as e:
            exceptions.append(e)

    def reader_worker():
        try:
            for _ in range(10):
                if stop_event.is_set():
                    break
                res = engine.process_streaming_step()
                assert "smoothed_risk_score" in res
                time.sleep(0.020)
        except Exception as e:
            exceptions.append(e)

    writer_t = threading.Thread(target=writer_worker)
    reader_t = threading.Thread(target=reader_worker)

    writer_t.start()
    reader_t.start()

    writer_t.join(timeout=5.0)
    reader_t.join(timeout=5.0)
    stop_event.set()

    assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"
    assert engine.total_chunks_received > 0
    assert engine.processed_windows > 0


# ---------------------------------------------------------------------------
# Test 8: Stateful Session Isolation & Reset
# ---------------------------------------------------------------------------

def test_session_reset_and_isolation():
    """Verify reset() cleanly clears buffers and counters back to initial baseline."""
    engine = LiveStreamingEngine(session_id="test-session-001")
    engine.ingest_pcm_chunk(np.ones(16000, dtype=np.float32) * 0.3)
    engine.process_streaming_step()

    assert engine.total_chunks_received == 1
    assert engine.processed_windows == 1
    assert engine.buffer.get_current_duration() > 0

    engine.reset()

    assert engine.total_chunks_received == 0
    assert engine.processed_windows == 0
    assert engine.buffer.get_current_duration() == 0.0
    assert engine.ema_score is None
    assert engine.alert_hold_counter == 0
    assert engine.held_peak_score == 0.0
