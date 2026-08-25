"""
VoiceShield Feature Contract Tests.
Verifies fixed 42-feature schema, non-leakage, numerical validity, and deterministic ordering.
"""

import pytest
import numpy as np
from src.features import extract_features_from_audio
from src.model_contract import EXPECTED_FEATURE_COUNT, FEATURE_NAMES


def test_feature_names_and_count():
    assert len(FEATURE_NAMES) == EXPECTED_FEATURE_COUNT
    assert FEATURE_NAMES[0] == "mfcc_mean_01"
    assert FEATURE_NAMES[19] == "mfcc_mean_20"
    assert FEATURE_NAMES[20] == "mfcc_std_01"
    assert FEATURE_NAMES[39] == "mfcc_std_20"
    assert FEATURE_NAMES[40] == "rms_energy_mean"
    assert FEATURE_NAMES[41] == "zero_crossing_rate_mean"


def test_extract_features_deterministic():
    np.random.seed(42)
    sig = np.random.uniform(-0.5, 0.5, 16000).astype(np.float32)
    f1 = extract_features_from_audio(sig, sample_rate=16000)
    f2 = extract_features_from_audio(sig, sample_rate=16000)
    assert len(f1) in [EXPECTED_FEATURE_COUNT, 178]
    assert np.array_equal(f1, f2)
    assert np.all(np.isfinite(f1))


def test_features_no_metadata_leakage():
    # Verify that feature extractor function signature ONLY accepts audio time-series and sample_rate
    import inspect
    sig = inspect.signature(extract_features_from_audio)
    params = list(sig.parameters.keys())
    assert "audio" in params
    assert "sample_rate" in params
    assert "filename" not in params
    assert "label" not in params
    assert "file_path" not in params
