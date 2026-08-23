"""
VoiceShield Model Factory & Candidate Architecture Tests.
Verifies factory instantiation, backbone availability checks, and probability outputs.
"""

import pytest
import numpy as np
from src.model_factory import (
    create_model_backbone,
    check_backbone_availability,
    SUPPORTED_BACKBONES,
)


def test_supported_backbones_list():
    assert "wavlm" in SUPPORTED_BACKBONES
    assert "wav2vec2" in SUPPORTED_BACKBONES
    assert "hubert" in SUPPORTED_BACKBONES


def test_check_backbone_availability_acoustic_net():
    avail, msg = check_backbone_availability("acoustic_spectral_net")
    assert avail is True


def test_model_factory_creation_and_predict_proba():
    classifier = create_model_backbone("acoustic_spectral_net")
    dummy_X = np.random.normal(0, 1.0, (10, 42)).astype(np.float32)
    dummy_y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    classifier.head_pipeline.fit(dummy_X, dummy_y)
    probs = classifier.predict_proba(dummy_X)

    assert probs.shape == (10, 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)
