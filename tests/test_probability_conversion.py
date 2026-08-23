"""
VoiceShield Probability Conversion Tests.
Ensures probability outputs are finite, bounded [0, 1], normalized, and free of double-sigmoid/softmax artifacts.
"""

import pytest
import numpy as np
from src.model_contract import validate_model_probabilities


def test_probability_normalization_and_boundaries():
    raw_probs = np.array([0.85, 0.15])
    p_human, p_spoof = validate_model_probabilities(raw_probs)
    assert p_human == 0.85
    assert p_spoof == 0.15
    assert pytest.approx(p_human + p_spoof, 0.0001) == 1.0


def test_probability_floating_point_slight_drift_recovery():
    # Test that minor float32 drift (e.g. 0.700001 + 0.300001) normalizes cleanly
    drift_probs = np.array([0.70001, 0.30001])
    p_human, p_spoof = validate_model_probabilities(drift_probs)
    assert pytest.approx(p_human + p_spoof, 0.00001) == 1.0


def test_invalid_raw_probabilities():
    # NaN
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_model_probabilities(np.array([np.nan, 0.5]))

    # Out of bounds (<0)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_model_probabilities(np.array([-0.05, 1.05]))

    # Out of bounds (>1)
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_model_probabilities(np.array([1.2, 0.0]))
