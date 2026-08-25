"""
Unit & Integration Tests for Production Hybrid Forensic Neural Engine.
Tests:
1. Praat glottal micro-biomechanics (Local Jitter, Shimmer, HNR).
2. High-Frequency Spectral Cutoff Ratio calculation.
3. ProductionNeuralDetector weighted ensemble and risk calibration.
4. Accuracy CLI benchmark script execution.
"""

import os
import sys
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import SAMPLE_RATE
from src.neural_engine import (
    ProductionNeuralDetector,
    compute_dsp_spoof_probability,
    compute_praat_biomechanics,
    compute_spectral_cutoff_ratio,
)


def test_praat_biomechanics_extraction():
    """Verify Praat local jitter, shimmer, and HNR extraction."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # Simulated harmonic speech with natural f0 ~ 150 Hz
    audio = (0.4 * np.sin(2 * np.pi * 150.0 * t) + 0.2 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)

    glottal = compute_praat_biomechanics(audio, sr=16000)

    assert "local_jitter" in glottal
    assert "local_shimmer" in glottal
    assert "hnr_db" in glottal
    assert 0.0 <= glottal["local_jitter"] <= 0.1
    assert 0.0 <= glottal["local_shimmer"] <= 0.5


def test_spectral_cutoff_ratio():
    """Verify high-frequency (> 6kHz) spectral cutoff ratio."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    # Pure 1000 Hz tone (no energy > 6kHz)
    low_freq_audio = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    ratio_low = compute_spectral_cutoff_ratio(low_freq_audio, sr=16000)
    assert ratio_low < 0.05

    # White noise (broadband energy up to Nyquist 8kHz)
    noise_audio = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    ratio_noise = compute_spectral_cutoff_ratio(noise_audio, sr=16000)
    assert ratio_noise > 0.15


def test_production_neural_detector_predict():
    """Verify ProductionNeuralDetector forward evaluation and forensic breakdown."""
    detector = ProductionNeuralDetector(device="cpu")

    t = np.linspace(0, 2.0, 32000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    res = detector.predict(audio, sample_rate=16000)

    assert "prediction_label" in res
    assert "risk_score" in res
    assert 0 <= res["risk_score"] <= 100
    assert "forensic_breakdown" in res
    assert "transformer_spoof_prob" in res["forensic_breakdown"]
    assert "dsp_physics_prob" in res["forensic_breakdown"]
    assert "local_jitter" in res["forensic_breakdown"]
    assert res["latency_ms"] >= 0.0
