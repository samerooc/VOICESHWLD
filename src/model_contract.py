"""
VoiceShield Model Contract Module.
Enforces strict immutable data structures, class taxonomy, feature schema,
input validation limits, and verification tests across training and inference.
"""

from typing import Any, Dict, List, Tuple
import numpy as np

# Class Taxonomy & Mapping (Standard ASVspoof protocol)
LABEL_BONA_FIDE: int = 0
LABEL_SPOOF: int = 1

CLASS_NAMES: Dict[int, str] = {
    LABEL_BONA_FIDE: "bona_fide",
    LABEL_SPOOF: "spoof",
}

HUMAN_READABLE_LABELS: Dict[int, str] = {
    LABEL_BONA_FIDE: "Likely Human Voice",
    LABEL_SPOOF: "Likely Spoof / AI Voice",
}

# Expected Audio Specifications
EXPECTED_SAMPLE_RATE: int = 16000
EXPECTED_CHANNELS: int = 1  # Mono
MIN_AUDIO_DURATION_SEC: float = 0.50
MAX_AUDIO_DURATION_SEC: float = 300.00
MIN_AUDIO_RMS_ENERGY: float = 1e-5

# Feature Vector Specification (42 features)
N_MFCC: int = 20
EXPECTED_FEATURE_COUNT: int = 42

FEATURE_NAMES: List[str] = (
    [f"mfcc_mean_{i+1:02d}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i+1:02d}" for i in range(N_MFCC)]
    + ["rms_energy_mean", "zero_crossing_rate_mean"]
)

# Output Model Contract
EXPECTED_MODEL_OUTPUT_TYPE: str = "probability_distribution"  # predict_proba returns [P(bona_fide), P(spoof)]
DEFAULT_DECISION_THRESHOLD: float = 0.500


def validate_class_contract(class_mapping: Dict[Any, str]) -> bool:
    """
    Validates that class mapping strictly adheres to the standard:
    0 = bona_fide, 1 = spoof.
    """
    if str(class_mapping.get(0, class_mapping.get("0", ""))).lower() != "bona_fide":
        raise ValueError(f"Contract Violation: Class 0 must map to 'bona_fide', got {class_mapping.get(0)}")
    if str(class_mapping.get(1, class_mapping.get("1", ""))).lower() != "spoof":
        raise ValueError(f"Contract Violation: Class 1 must map to 'spoof', got {class_mapping.get(1)}")
    return True


def validate_feature_vector(features: np.ndarray) -> bool:
    """
    Validates feature vector dimensions, non-emptiness, and finite numeric values.
    """
    if features is None:
        raise ValueError("Feature vector is None.")
    if not isinstance(features, np.ndarray):
        raise TypeError(f"Features must be a numpy ndarray, got {type(features)}.")
    if features.ndim != 1 or len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Feature dimension mismatch: expected 1D array of shape ({EXPECTED_FEATURE_COUNT},), got shape {features.shape}."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature vector contains NaN or infinite values.")
    return True


def validate_model_probabilities(probs: np.ndarray) -> Tuple[float, float]:
    """
    Validates raw model output probabilities:
    - Array shape must be (2,)
    - Values must be finite and within [0.0, 1.0]
    - Values must sum to approximately 1.0
    Returns (p_bona_fide, p_spoof).
    """
    if probs is None:
        raise ValueError("Model probability output is None.")
    arr = np.asarray(probs, dtype=np.float64).flatten()
    if len(arr) != 2:
        raise ValueError(f"Model output probability shape mismatch: expected 2 classes, got {len(arr)}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Model probabilities contain NaN or Inf values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"Model probabilities outside [0, 1] range: {arr}.")
    total = np.sum(arr)
    if not (0.98 <= total <= 1.02):
        raise ValueError(f"Model probabilities do not sum to 1.0 (sum={total:.4f}).")

    # Normalize to exact 1.0
    p_human = float(arr[0] / total)
    p_spoof = float(arr[1] / total)
    return p_human, p_spoof
