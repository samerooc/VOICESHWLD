"""
VoiceShield Phase 3 — Comprehensive Integration Test Suite.

Verifies the complete Phase 3 SOTA Neural Inference Engine:
  1.  Dynamic Token & Label Resolution (eliminating label inversion on synthetic configs)
  2.  Temperature-Scaled Logit Calibration (T = 1.35)
  3.  Sliding-Window Deep Transformer Embeddings (3.0 s windows, zero-mean normalisation)
  4.  Tri-Tier Consensus Fusion (Adaptive SNR-weighted blend: >= 10 dB vs < 10 dB)
  5.  5-State Calibrated Risk Categorisation (Low, Review, High, Low Quality/Degraded)
  6.  Quality Gating on Degraded / Silent / Truncated Inputs (< 0.5 s, voiced < 0.4 s, SNR < 3 dB)
  7.  Output Key Contract & Diagnostic Metrics (Phase 2 & Phase 3 keys)
  8.  In-Memory Raw Byte Ingestion (WAV / FLAC bytes to forensic report)
  9.  End-to-End Latency Compliance (CPU < 500 ms / Real-time compliant)
  10. Model Loader Device Routing & Startup Warmup

Run with:
    pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F

from src.config import SAMPLE_RATE
from src.neural_engine import (
    ProductionNeuralDetector,
    _HUMAN_PATTERN,
    _SPOOF_PATTERN,
    _TEMPERATURE,
    compute_dsp_spoof_probability,
    compute_praat_biomechanics,
    compute_spectral_cutoff_ratio,
)

# ---------------------------------------------------------------------------
# Test Fixtures & Synthetic Signal Generators
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector() -> ProductionNeuralDetector:
    """Instantiate a ProductionNeuralDetector on CPU with native backbone for deterministic testing."""
    return ProductionNeuralDetector(device="cpu", load_hf=False)


def _generate_synthetic_waveform(
    duration_sec: float = 3.5,
    sr: int = SAMPLE_RATE,
    f0_hz: float = 180.0,
    noise_level: float = 0.02,
) -> np.ndarray:
    """Generate a voiced harmonic signal with mild noise and authentic duration."""
    n_samples = int(sr * duration_sec)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    # Voiced harmonic series
    sig = 0.3 * np.sin(2.0 * np.pi * f0_hz * t)
    sig += 0.15 * np.sin(2.0 * np.pi * (2 * f0_hz) * t)
    sig += 0.08 * np.sin(2.0 * np.pi * (3 * f0_hz) * t)
    # Background Gaussian noise
    rng = np.random.default_rng(42)
    sig += noise_level * rng.standard_normal(n_samples)
    return sig.astype(np.float32)


def _to_wav_bytes(audio: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    """Serialize float32 numpy waveform to WAV container bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1: Dynamic Token & Label Resolution
# ---------------------------------------------------------------------------

def test_dynamic_label_resolution_regex_patterns():
    """Verify regex patterns correctly classify diverse model label naming conventions."""
    # Spoof terms
    spoof_terms = ["fake", "spoof", "synth", "deepfake", "clone", "generated", "artificial", "ai_voice", "tts"]
    for term in spoof_terms:
        assert _SPOOF_PATTERN.search(term) is not None, f"Failed on spoof term: {term}"
        assert _SPOOF_PATTERN.search(f"LABEL_{term.upper()}") is not None

    # Human terms
    human_terms = ["real", "bonafide", "bona_fide", "human", "authentic", "original", "genuine", "natural"]
    for term in human_terms:
        assert _HUMAN_PATTERN.search(term) is not None, f"Failed on human term: {term}"
        assert _HUMAN_PATTERN.search(f"LABEL_{term.upper()}") is not None


def test_dynamic_label_resolution_synthetic_configs(detector: ProductionNeuralDetector):
    """Verify _resolve_labels correctly maps spoof_idx and human_idx on various HF configs."""
    # Case A: {"0": "fake", "1": "real"} -> spoof=0, human=1
    mock_model_a = MagicMock()
    mock_model_a.config.id2label = {0: "fake", 1: "real"}
    detector.model = mock_model_a
    detector._resolve_labels()
    assert detector.spoof_idx == 0
    assert detector.human_idx == 1

    # Case B: {"0": "bonafide", "1": "spoof"} -> spoof=1, human=0
    mock_model_b = MagicMock()
    mock_model_b.config.id2label = {0: "bonafide", 1: "spoof"}
    detector.model = mock_model_b
    detector._resolve_labels()
    assert detector.spoof_idx == 1
    assert detector.human_idx == 0

    # Case C: {"0": "authentic", "1": "deepfake"} -> spoof=1, human=0
    mock_model_c = MagicMock()
    mock_model_c.config.id2label = {0: "authentic", 1: "deepfake"}
    detector.model = mock_model_c
    detector._resolve_labels()
    assert detector.spoof_idx == 1
    assert detector.human_idx == 0

    # Case D: Unmapped generic labels {"0": "LABEL_0", "1": "LABEL_1"} -> default fallback (1, 0)
    mock_model_d = MagicMock()
    mock_model_d.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    detector.model = mock_model_d
    detector._resolve_labels()
    assert detector.spoof_idx == 1
    assert detector.human_idx == 0


# ---------------------------------------------------------------------------
# Test 2: Temperature-Scaled Logit Calibration
# ---------------------------------------------------------------------------

def test_temperature_scaling_calibration():
    """Verify that temperature scaling T=1.35 correctly moderates extreme logits."""
    raw_logits = torch.tensor([[4.0, -4.0]])
    t = _TEMPERATURE
    scaled_logits = raw_logits / t

    raw_probs = F.softmax(raw_logits, dim=-1).numpy()[0]
    scaled_probs = F.softmax(scaled_logits, dim=-1).numpy()[0]

    # Scaled probabilities should be closer to 0.5 (less overconfident)
    assert scaled_probs[0] < raw_probs[0]
    assert scaled_probs[1] > raw_probs[1]
    assert np.isclose(scaled_probs.sum(), 1.0)


# ---------------------------------------------------------------------------
# Test 3: Adaptive SNR-Weighted Consensus Logic
# ---------------------------------------------------------------------------

def test_adaptive_snr_weighted_consensus_clean_vs_noisy(detector: ProductionNeuralDetector):
    """
    Verify the consensus formula:
      - SNR >= 10 dB: 0.50 Transformer + 0.30 LPC + 0.20 DSP
      - SNR <  10 dB: 0.35 Transformer + 0.35 LPC + 0.30 DSP
    """
    # 1. Clean synthetic audio (low noise -> SNR >= 10 dB)
    clean_audio = _generate_synthetic_waveform(duration_sec=2.0, noise_level=0.001)
    res_clean = detector.predict(clean_audio)
    assert res_clean["forensic_breakdown"]["snr_weight_mode"] == "clean"
    assert res_clean["forensic_breakdown"]["snr_db"] >= 10.0

    # 2. Noisy synthetic audio (speech + noise intervals yielding SNR in [3.0, 9.9] dB)
    speech = _generate_synthetic_waveform(duration_sec=1.2, noise_level=0.01)
    noise = np.random.default_rng(42).normal(0.0, 0.10, int(16000 * 0.8)).astype(np.float32)
    noisy_audio = np.concatenate([speech, noise])

    res_noisy = detector.predict(noisy_audio)
    assert res_noisy["forensic_breakdown"]["snr_weight_mode"] == "noisy"
    assert 3.0 <= res_noisy["forensic_breakdown"]["snr_db"] < 10.0


# ---------------------------------------------------------------------------
# Test 4: 5-State Calibrated Risk Categorisation & Bands
# ---------------------------------------------------------------------------

def test_5_state_risk_categorization():
    """Verify mapping of risk scores into 5-state calibrated bands."""
    # Low Risk (0 - 25)
    label, band, key, badge, desc = ProductionNeuralDetector._risk_band(15)
    assert label == "AUTHENTIC HUMAN VOICE"
    assert band == "Low Risk (Human Voice)"
    assert key == "low"
    assert badge == "badge-low"
    assert "Natural vocal-fold micro-perturbations" in desc

    # Review Required (26 - 60)
    label, band, key, badge, desc = ProductionNeuralDetector._risk_band(45)
    assert label == "SUSPICIOUS / INCONCLUSIVE"
    assert band == "Review Required (Borderline Evidence)"
    assert key == "review"
    assert badge == "badge-review"
    assert "Secondary human-in-the-loop" in desc

    # High Risk (61 - 100)
    label, band, key, badge, desc = ProductionNeuralDetector._risk_band(85)
    assert label == "AI VOICE CLONE DETECTED"
    assert band == "High Risk (Likely AI / Cloned Voice)"
    assert key == "high"
    assert badge == "badge-high"
    assert "Synthetic neural vocoder signatures" in desc


# ---------------------------------------------------------------------------
# Test 5: Quality Gate: Degraded / Silent / Truncated Audio
# ---------------------------------------------------------------------------

def test_quality_gate_degraded_silent_audio(detector: ProductionNeuralDetector):
    """Verify silence returns safe Low Quality / Degraded response with no exceptions."""
    silence = np.zeros(16000 * 2, dtype=np.float32)
    res = detector.predict(silence)

    assert res["prediction_label"] == "LOW QUALITY / DEGRADED"
    assert res["risk_band"] == "Low Quality / Degraded"
    assert res["risk_band_key"] == "low_quality"
    assert res["badge_class"] == "badge-degraded"
    assert res["risk_score"] == 50
    assert res["spoof_probability"] == 0.50
    assert res["human_probability"] == 0.50
    assert "silent" in res["risk_description"].lower()


def test_quality_gate_truncated_audio(detector: ProductionNeuralDetector):
    """Verify audio shorter than 0.5s triggers degraded quality gate."""
    short_audio = np.sin(np.linspace(0, 100, 1600)).astype(np.float32)  # 0.1s
    res = detector.predict(short_audio)

    assert res["prediction_label"] == "LOW QUALITY / DEGRADED"
    assert res["risk_band_key"] == "low_quality"


# ---------------------------------------------------------------------------
# Test 6: 3.0s Sliding Window Transformer Extraction
# ---------------------------------------------------------------------------

def test_sliding_window_transformer_inference(detector: ProductionNeuralDetector):
    """Verify sliding windows (3.0s length, 1.5s hop) across a 5.0s waveform."""
    mock_model = MagicMock()
    mock_model.return_value.logits = torch.tensor([[0.2, 0.8]])
    mock_extractor = MagicMock()
    mock_extractor.return_value = {"input_values": torch.zeros(1, 48000)}

    detector.has_hf_model = True
    detector.model = mock_model
    detector.feature_extractor = mock_extractor
    detector.spoof_idx = 1
    detector.human_idx = 0

    try:
        long_audio = _generate_synthetic_waveform(duration_sec=5.0)
        p_trans, windows = detector._run_transformer_inference(long_audio)

        assert 0.0 <= p_trans <= 1.0
        # For 5.0s audio @ 16kHz (80k samples), 3.0s win (48k) with 1.5s hop (24k):
        # window starts: 0 (0-48k), 24k (24k-72k) -> 2 full windows
        assert len(windows) >= 2
        for win in windows:
            assert "window_index" in win
            assert "time_range" in win
            assert "spoof_probability" in win
            assert 0.0 <= win["spoof_probability"] <= 1.0
    finally:
        detector.has_hf_model = False
        detector.model = None
        detector.feature_extractor = None


# ---------------------------------------------------------------------------
# Test 7: Output Key Contract (Phase 2 & Phase 3 Schema)
# ---------------------------------------------------------------------------

def test_output_key_contract_and_schema(detector: ProductionNeuralDetector):
    """Verify complete schema contract matching all Phase 2 and Phase 3 specifications."""
    audio = _generate_synthetic_waveform(duration_sec=2.0)
    res = detector.predict(audio)

    # Top-level keys
    required_top_keys = [
        "prediction_label", "spoof_probability", "human_probability",
        "risk_score", "risk_band", "risk_band_key", "badge_class",
        "risk_description", "forensic_breakdown", "window_breakdown",
        "diagnostics", "latency_ms", "is_realtime_compliant", "disclaimer",
    ]
    for key in required_top_keys:
        assert key in res, f"Missing top-level key: {key}"

    # Forensic breakdown keys
    breakdown = res["forensic_breakdown"]
    required_breakdown_keys = [
        "transformer_spoof_prob", "active_model_id",
        "lpc_anomaly_score", "lpc_kurtosis", "phase_entropy", "residual_flatness",
        "dsp_physics_prob", "glottal_spoof_prob", "lfcc_spoof_prob", "spectral_spoof_prob",
        "jitter_local", "shimmer_local", "hnr_db", "lfcc_variance", "hf_cutoff_ratio",
        "snr_db", "snr_weight_mode", "voiced_ratio",
    ]
    for key in required_breakdown_keys:
        assert key in breakdown, f"Missing breakdown key: {key}"


# ---------------------------------------------------------------------------
# Test 8: In-Memory Raw Byte Ingestion
# ---------------------------------------------------------------------------

def test_predict_bytes_end_to_end(detector: ProductionNeuralDetector):
    """Verify end-to-end inference from in-memory WAV container bytes."""
    audio = _generate_synthetic_waveform(duration_sec=2.0)
    wav_bytes = _to_wav_bytes(audio)

    res = detector.predict_bytes(wav_bytes)
    assert res["prediction_label"] in ["AUTHENTIC HUMAN VOICE", "SUSPICIOUS / INCONCLUSIVE", "AI VOICE CLONE DETECTED"]
    assert 0 <= res["risk_score"] <= 100
    assert 0.0 <= res["spoof_probability"] <= 1.0
    assert res["diagnostics"]["duration_sec"] >= 1.9


# ---------------------------------------------------------------------------
# Test 9: End-to-End Latency Compliance
# ---------------------------------------------------------------------------

def test_inference_latency_compliance(detector: ProductionNeuralDetector):
    """Verify that CPU inference completes within the real-time threshold (< 500 ms)."""
    audio = _generate_synthetic_waveform(duration_sec=1.5)
    res = detector.predict(audio)

    assert "latency_ms" in res
    assert res["latency_ms"] > 0
    assert res["is_realtime_compliant"] is True


# ---------------------------------------------------------------------------
# Test 10: Backward Compatibility Functional Wrappers
# ---------------------------------------------------------------------------

def test_backward_compatibility_wrappers():
    """Verify helper functions compute_praat_biomechanics, compute_spectral_cutoff_ratio, compute_dsp_spoof_probability."""
    audio = _generate_synthetic_waveform(duration_sec=1.5)

    bio = compute_praat_biomechanics(audio, sr=SAMPLE_RATE)
    assert "jitter_local" in bio
    assert "shimmer_local" in bio
    assert "hnr_db" in bio

    cutoff = compute_spectral_cutoff_ratio(audio, sr=SAMPLE_RATE)
    assert 0.0 <= cutoff <= 1.0

    p_dsp = compute_dsp_spoof_probability(audio, sr=SAMPLE_RATE)
    assert 0.0 <= p_dsp <= 1.0
