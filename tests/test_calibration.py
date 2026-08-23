"""
VoiceShield Calibration Tests.
Verifies temperature scaling and standard 5-state risk mapping.
"""

import pytest
from src.calibration import calibrate_probability, compute_risk_state


def test_calibrate_probability_identity_at_temp_1():
    p = 0.80
    assert pytest.approx(calibrate_probability(p, temperature=1.0), 0.001) == 0.80


def test_compute_risk_state_low():
    score, band, code, details = compute_risk_state(0.15)
    assert score == 15
    assert band == "low"
    assert code == "success"
    assert details["is_uncertain"] is False


def test_compute_risk_state_uncertain():
    score, band, code, details = compute_risk_state(0.50)
    assert score == 50
    assert band == "uncertain"
    assert details["is_uncertain"] is True


def test_compute_risk_state_high():
    score, band, code, details = compute_risk_state(0.90)
    assert score == 90
    assert band == "high"
    assert code == "error"


def test_compute_risk_state_low_quality():
    score, band, code, details = compute_risk_state(0.10, quality_flag="clipped")
    assert band == "low_quality"
    assert details["is_uncertain"] is True
