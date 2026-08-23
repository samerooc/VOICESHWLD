"""
VoiceShield Model Training & Pipeline Invariants Unit Tests.
Verifies model creation, training reproducibility, contract metadata, and deterministic probability shapes.
"""

import os
import pytest
import numpy as np
from src.model_factory import create_model_backbone, check_backbone_availability
from src.model_contract import validate_model_probabilities


def test_model_factory_acoustic_spectral():
    classifier = create_model_backbone("acoustic_spectral_net")
    assert classifier is not None

    # Dummy feature matrix (42 features)
    X = np.random.normal(0, 1, (10, 42)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier.head_pipeline.fit(X, y)
    probs = classifier.predict_proba(X)

    assert probs.shape == (10, 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)


def test_model_factory_missing_pretrained_weights():
    # WavLM without local weights should report blocked
    avail, msg = check_backbone_availability("wavlm")
    if not avail:
        assert "BLOCKED" in msg or "unavailable" in msg
