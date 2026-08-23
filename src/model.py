"""
VoiceShield Model Management & Inference Pipeline (Phase 1).
Manages StandardScaler + RandomForest Pipeline loading, serialization, and raw inference.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_BASELINE_V1_PATH, MODEL_METADATA_PATH, MODEL_PATH


def build_pipeline(
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    random_state: int = 42,
) -> Pipeline:
    """
    Builds the Baseline v1 Pipeline: StandardScaler + Balanced RandomForestClassifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state,
                class_weight="balanced",
            ),
        ),
    ])


def load_model(model_path: str = MODEL_PATH) -> Optional[Pipeline]:
    """Loads serialized pipeline from disk."""
    paths_to_try = [model_path, MODEL_BASELINE_V1_PATH] if model_path == MODEL_PATH else [model_path]
    for p in paths_to_try:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                print(f"Warning: Failed to load model from '{p}': {e}")
    return None


def load_metadata(metadata_path: str = MODEL_METADATA_PATH) -> Optional[Dict[str, Any]]:
    """Loads model metadata JSON from disk."""
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load metadata from '{metadata_path}': {e}")
        return None


def load_model_and_metadata(
    model_path: str = MODEL_PATH,
    metadata_path: str = MODEL_METADATA_PATH,
) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    """Loads both model pipeline and metadata."""
    return load_model(model_path), load_metadata(metadata_path)


def save_model(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: str = MODEL_PATH,
    metadata_path: str = MODEL_METADATA_PATH,
) -> None:
    """Saves pipeline and metadata atomically."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    if model_path != MODEL_BASELINE_V1_PATH:
        try:
            joblib.dump(pipeline, MODEL_BASELINE_V1_PATH)
        except Exception:
            pass
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
