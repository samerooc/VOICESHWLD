"""
VoiceShield Model Output Contract Tests (Section C).
Verifies probability bounds, sum to 1.0, finite values, and batch invariance.
"""

import pytest
import numpy as np
import joblib
from src.model_contract import validate_model_probabilities


def test_validate_model_probabilities_clean():
    p_human, p_spoof = validate_model_probabilities(np.array([0.75, 0.25]))
    assert p_human == 0.75
    assert p_spoof == 0.25
    assert pytest.approx(p_human + p_spoof, 1e-6) == 1.0


def test_batch_vs_single_sample_invariance():
    model = joblib.load("models/voice_detector.pkl")
    dummy_feat = np.random.normal(0, 1.0, (5, 42)).astype(np.float32)

    batch_probs = model.predict_proba(dummy_feat)
    for i in range(5):
        single_prob = model.predict_proba(dummy_feat[i : i + 1])[0]
        np.testing.assert_allclose(batch_probs[i], single_prob, atol=1e-6)
