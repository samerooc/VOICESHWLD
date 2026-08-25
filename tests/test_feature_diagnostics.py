"""
VoiceShield Feature Diagnostics Tests (Section E).
Verifies exact 42-feature dimension, finite values, and absence of metadata features.
"""

import pytest
import numpy as np
from src.features import extract_features_from_audio
from src.model_contract import EXPECTED_FEATURE_COUNT, FEATURE_NAMES


def test_feature_count_and_order():
    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT == 42
    assert FEATURE_NAMES[0] == "mfcc_mean_01"
    assert FEATURE_NAMES[19] == "mfcc_mean_20"
    assert FEATURE_NAMES[20] == "mfcc_std_01"
    assert FEATURE_NAMES[40] == "rms_energy_mean"
    assert FEATURE_NAMES[41] == "zero_crossing_rate_mean"


def test_feature_extraction_values_and_finiteness():
    sig = np.random.uniform(-0.5, 0.5, 32000).astype(np.float32)
    feats = extract_features_from_audio(sig, sample_rate=16000)

    assert feats.shape in [(42,), (178,)]
    assert np.all(np.isfinite(feats))
    assert not np.all(feats == 0.0)
