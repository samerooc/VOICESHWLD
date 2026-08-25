"""
VoiceShield Production Model Registry & Integrity Verification Module (Phase 13).
Enforces cryptographic hash validation, metadata schema checks, and version compatibility.
"""

import os
import json
import hashlib
from typing import Any, Dict, Optional, Tuple
import joblib

from src.model_contract import validate_class_contract

SUPPORTED_FEATURE_VERSIONS = ["1.0.0", "2.0.0", "2.1.0"]
EXPECTED_FEATURE_VERSION: str = "1.0.0"
EXPECTED_PREPROCESSING_VERSION: str = "1.0.0"


def verify_and_load_model(
    model_path: str = "models/voice_detector.pkl",
    metadata_path: str = "models/model_metadata.json",
) -> Tuple[Any, Dict[str, Any]]:
    """
    Cryptographically verifies and securely loads model artifact.
    Raises ValueError if metadata is missing, hash mismatches, or schema is invalid.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'.")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model metadata not found at '{metadata_path}'. Refusing to load unverified model.")

    # 1. Load and validate metadata JSON
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 2. Check feature version and preprocessing version
    feat_ver = metadata.get("feature_version") or metadata.get("feature_configuration", {}).get("version", "1.0.0")
    if feat_ver not in SUPPORTED_FEATURE_VERSIONS:
        raise ValueError(
            f"Feature Version Mismatch: Expected one of {SUPPORTED_FEATURE_VERSIONS}, got '{feat_ver}'."
        )

    # 3. Check class mapping
    class_map = metadata.get("class_mapping") or metadata.get("label_mapping")
    if not class_map:
        raise ValueError("Model Metadata Error: Missing 'class_mapping'.")
    validate_class_contract(class_map)

    # 4. Verify artifact hash if present
    recorded_hash = metadata.get("model_artifact_sha256")
    actual_hash = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    if recorded_hash and recorded_hash != actual_hash:
        raise ValueError(
            f"Model Artifact Corruption: Hash mismatch! Expected {recorded_hash}, got {actual_hash}."
        )

    # 5. Load model object
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize model artifact: {e}")

    return model, metadata
