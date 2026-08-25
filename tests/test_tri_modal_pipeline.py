"""
Unit & Integration Test Suite for Tri-Modal Architecture (Acoustic + Physics + NLP Intent).
Tests:
1. Higher-Order DSP Physics & Bispectrum extraction.
2. Real-Time NLP Intent & Voice Phishing heuristic scoring.
3. VoiceShieldTriModalEngine forward pass shapes and end-to-end tri-modal prediction.
4. TriModalMultiTaskLoss multi-objective gradient backpropagation.
5. TriModalStreamingEngine sliding window telemetry evaluation.
"""

import os
import sys
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.losses import TriModalMultiTaskLoss
from src.nlp_intent import score_text_intent_heuristics
from src.physics_dsp import (
    PHYSICS_RAW_DIM,
    compute_bispectrum_coupling_metric,
    compute_cepstral_peak_prominence,
    extract_raw_physics_vector,
)
from src.streaming_engine import TriModalStreamingEngine
from src.tri_modal_model import VoiceShieldTriModalEngine


def test_physics_dsp_feature_extraction():
    """Verify higher-order DSP & bispectrum feature extractor."""
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)

    qpc = compute_bispectrum_coupling_metric(sine)
    assert 0.0 <= qpc <= 1.0

    cpp = compute_cepstral_peak_prominence(sine, sr=16000)
    assert 0.0 <= cpp <= 1.0

    phys_vec = extract_raw_physics_vector(sine, sr=16000)
    assert isinstance(phys_vec, np.ndarray)
    assert phys_vec.shape == (PHYSICS_RAW_DIM,)
    assert not np.isnan(phys_vec).any()


def test_nlp_intent_phishing_heuristics():
    """Verify NLP intent scoring identifies social engineering and urgency pressure."""
    urgent_text = "Please verify your account password and OTP immediately to avoid account suspension."
    res_urgent = score_text_intent_heuristics(urgent_text)

    assert res_urgent["fraud_intent_score"] > 0.50
    assert len(res_urgent["threat_categories"]) >= 2
    assert "Credential / OTP Harvesting" in res_urgent["threat_categories"]
    assert "Urgency & Coercion Pressure" in res_urgent["threat_categories"]

    clean_text = "Hi, I am calling regarding our appointment tomorrow afternoon at two."
    res_clean = score_text_intent_heuristics(clean_text)
    assert res_clean["fraud_intent_score"] <= 0.20
    assert len(res_clean["threat_categories"]) == 0


def test_tri_modal_engine_forward_and_prediction():
    """Verify VoiceShieldTriModalEngine processes multi-modal inputs and returns sub-scores."""
    engine = VoiceShieldTriModalEngine(backbone_name="lightweight", device=torch.device("cpu"))
    engine.eval()

    batch_size = 2
    dummy_wav = torch.randn(batch_size, 48000, dtype=torch.float32)
    dummy_phys = torch.randn(batch_size, 32, dtype=torch.float32)
    dummy_nlp = torch.randn(batch_size, 16, dtype=torch.float32)

    with torch.no_grad():
        out = engine(dummy_wav, dummy_phys, dummy_nlp)

    assert "fused_logit" in out
    assert out["fused_logit"].shape == (batch_size,)
    assert "modality_gates" in out
    assert out["modality_gates"].shape == (batch_size, 3)

    # Test end-to-end prediction interface
    t = np.linspace(0, 3.0, 48000, endpoint=False)
    test_audio = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    phishing_text = "Authorize the wire transfer right now to prevent frozen assets."

    pred = engine.predict_tri_modal(test_audio, transcription=phishing_text, sr=16000)
    assert "combined_risk_score" in pred
    assert 0 <= pred["combined_risk_score"] <= 100
    assert "breakdown" in pred
    assert "neural_acoustic_score" in pred["breakdown"]
    assert "dsp_phase_anomaly" in pred["breakdown"]
    assert "nlp_intent_threat" in pred["breakdown"]
    assert "modality_weights" in pred


def test_tri_modal_multitask_loss():
    """Verify TriModalMultiTaskLoss computes loss and propagates gradients."""
    criterion = TriModalMultiTaskLoss(focal_gamma=2.0, focal_alpha=0.25)

    batch_size = 3
    model_outputs = {
        "fused_logit": torch.tensor([1.5, -1.0, 2.0], requires_grad=True),
        "aux_acoustic_logit": torch.tensor([1.2, -0.8, 1.8], requires_grad=True),
        "aux_physics_logit": torch.tensor([0.9, -1.2, 1.5], requires_grad=True),
        "fused_embedding": torch.randn(batch_size, 256, requires_grad=True),
    }
    targets = torch.tensor([1.0, 0.0, 1.0])

    loss = criterion(model_outputs, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    loss.backward()
    assert model_outputs["fused_logit"].grad is not None
    assert not torch.isnan(model_outputs["fused_logit"].grad).any()


def test_tri_modal_streaming_engine():
    """Verify TriModalStreamingEngine evaluates sliding windows and telemetry."""
    stream_engine = TriModalStreamingEngine(device="cpu", smoothing_alpha=0.35)
    stream_engine.reset()

    t = np.linspace(0, 3.0, 48000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sample_text = "Confirming your one time verification password."

    res = stream_engine.process_stream_window(audio, transcription=sample_text, timestamp=1.5, sample_rate=16000)

    assert res["timestamp"] == 1.5
    assert 0 <= res["combined_risk_score"] <= 100
    assert "breakdown" in res
    assert res["latency_ms"] >= 0.0
    assert res["is_silent"] is False
