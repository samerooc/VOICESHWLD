"""
VoiceShield Label Contract Tests.
Verifies class mapping integrity, probability mapping, and contract invariants.
"""

import pytest
import numpy as np
from src.model_contract import (
    CLASS_NAMES,
    LABEL_BONA_FIDE,
    LABEL_SPOOF,
    validate_class_contract,
    validate_feature_vector,
    validate_model_probabilities,
    EXPECTED_FEATURE_COUNT,
)


def test_class_names_and_ids():
    assert LABEL_BONA_FIDE == 0
    assert LABEL_SPOOF == 1
    assert CLASS_NAMES[0] == "bona_fide"
    assert CLASS_NAMES[1] == "spoof"


def test_validate_class_contract_success():
    mapping = {0: "bona_fide", 1: "spoof"}
    assert validate_class_contract(mapping) is True
    str_mapping = {"0": "bona_fide", "1": "spoof"}
    assert validate_class_contract(str_mapping) is True


def test_validate_class_contract_failure():
    inverted_mapping = {0: "spoof", 1: "bona_fide"}
    with pytest.raises(ValueError, match="Contract Violation"):
        validate_class_contract(inverted_mapping)

    incomplete_mapping = {0: "bona_fide"}
    with pytest.raises(ValueError, match="Contract Violation"):
        validate_class_contract(incomplete_mapping)


def test_validate_feature_vector_dimensions():
    valid_features = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float32)
    assert validate_feature_vector(valid_features) is True

    # Wrong shape
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        validate_feature_vector(np.zeros(40, dtype=np.float32))

    # Contains NaN
    nan_features = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float32)
    nan_features[5] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_feature_vector(nan_features)

    # Contains Inf
    inf_features = np.zeros(EXPECTED_FEATURE_COUNT, dtype=np.float32)
    inf_features[10] = np.inf
    with pytest.raises(ValueError, match="NaN or infinite"):
        validate_feature_vector(inf_features)


def test_validate_model_probabilities_bounds_and_sum():
    valid_probs = np.array([0.70, 0.30])
    p_human, p_spoof = validate_model_probabilities(valid_probs)
    assert pytest.approx(p_human, 0.001) == 0.70
    assert pytest.approx(p_spoof, 0.001) == 0.30

    # Negative probability
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        validate_model_probabilities(np.array([-0.1, 1.1]))

    # Does not sum to ~1.0
    with pytest.raises(ValueError, match="do not sum to 1.0"):
        validate_model_probabilities(np.array([0.2, 0.2]))

    # Wrong length
    with pytest.raises(ValueError, match="expected 2 classes"):
        validate_model_probabilities(np.array([0.33, 0.33, 0.34]))
