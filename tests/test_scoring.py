"""
Unit Tests for VoiceShield Risk Scoring, States & Advisory Guidance (src/scoring.py).
Tests:
- calculate_risk_score
- get_risk_band across all risk states (LOW RISK, REVIEW REQUIRED, HIGH-RISK SIGNAL, UNCERTAIN)
- Probability boundaries at 0.40 and 0.60
- Insufficient evidence message on uncertain probabilities
- Safe wording assertions
- Predict and score functionality
"""

import numpy as np
import pytest

from src.config import MODEL_PATH, SAMPLE_RATE, STATUTORY_DISCLAIMER
from src.model import load_model
from src.scoring import calculate_risk_score, get_risk_band, predict_and_score


def test_calculate_risk_score_bounds():
    """Verify probability to 0-100 risk score mapping and clipping."""
    assert calculate_risk_score(0.0) == 0
    assert calculate_risk_score(0.25) == 25
    assert calculate_risk_score(0.50) == 50
    assert calculate_risk_score(0.65) == 65
    assert calculate_risk_score(1.0) == 100
    assert calculate_risk_score(-0.1) == 0
    assert calculate_risk_score(1.2) == 100


def test_risk_states_and_uncertainty_bounds():
    """Verify risk states: LOW RISK, REVIEW REQUIRED, HIGH-RISK SIGNAL, UNCERTAIN."""
    # 1. LOW RISK (score <= 25)
    desc_low, band_low, _, recs_low = get_risk_band(15, spoof_prob=0.15)
    assert band_low == "Low"
    assert "LOW RISK" in desc_low

    # 2. REVIEW REQUIRED (26 <= score <= 39)
    desc_rev, band_rev, _, recs_rev = get_risk_band(35, spoof_prob=0.35)
    assert band_rev == "Review required"
    assert "REVIEW REQUIRED" in desc_rev

    # 3. UNCERTAIN (0.40 <= spoof_prob <= 0.60)
    for p in [0.40, 0.45, 0.50, 0.55, 0.60]:
        score = int(round(p * 100))
        desc_unc, band_unc, _, recs_unc = get_risk_band(score, spoof_prob=p)
        assert band_unc in ["Review required", "UNCERTAIN"]
        assert "Insufficient evidence — manual verification required" in desc_unc

    # 4. HIGH-RISK SIGNAL (score >= 66)
    desc_high, band_high, _, recs_high = get_risk_band(85, spoof_prob=0.85)
    assert band_high == "High risk"
    assert "HIGH-RISK SIGNAL" in desc_high


def test_safe_wording_compliance():
    """Verify safe wording compliance across all recommendations."""
    prohibited_words = [
        "confirmed ai",
        "confirmed fraudster",
        "genuine person confirmed",
        "proof of identity",
        "guaranteed detection",
    ]

    for score, prob in [(10, 0.10), (35, 0.35), (50, 0.50), (90, 0.90)]:
        desc, band, _, recs = get_risk_band(score, spoof_prob=prob)
        combined_text = (desc + " " + " ".join(recs)).lower()
        for pw in prohibited_words:
            assert pw not in combined_text, f"Prohibited phrase '{pw}' found in advisory text."


def test_predict_and_score_execution():
    """Verify predict_and_score with trained model pipeline."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    res = predict_and_score(model, audio, sample_rate=SAMPLE_RATE, decision_threshold=0.40)
    assert "prediction_class" in res
    assert "prediction_label" in res
    assert "human_probability" in res
    assert "spoof_probability" in res
    assert "risk_score" in res
    assert "risk_band" in res
    assert "recommendations" in res
    assert "disclaimer" in res
    assert res["disclaimer"] == STATUTORY_DISCLAIMER


def test_predict_and_score_none_model_error():
    """Verify ValueError when model is None."""
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    with pytest.raises(ValueError, match="Model is None"):
        predict_and_score(None, audio, sample_rate=SAMPLE_RATE)
