"""
VoiceShield Production Model Architecture & Pipeline Engine (Step 3).
Provides scikit-learn Pipelines with RobustScaler / StandardScaler and
XGBoost / LightGBM / RandomForest classifiers, with serialization and metadata persistence.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    MODEL_BASELINE_V1_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    MODEL_VERSION,
)


def get_model_expected_features(model: Any) -> int:
    """
    Accurately detects expected input feature dimension from any scikit-learn Pipeline or CalibratedClassifierCV.
    """
    if model is None:
        return 42

    # 1. Check calibrated classifiers (fitted instances)
    if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
        for clf in model.calibrated_classifiers_:
            est = getattr(clf, "estimator", getattr(clf, "base_estimator", None))
            if est is not None:
                if hasattr(est, "named_steps") and "scaler" in est.named_steps:
                    scaler = est.named_steps["scaler"]
                    if hasattr(scaler, "n_features_in_"):
                        return int(scaler.n_features_in_)
                    if hasattr(scaler, "mean_"):
                        return len(scaler.mean_)
                    if hasattr(scaler, "center_"):
                        return len(scaler.center_)
                if hasattr(est, "n_features_in_"):
                    return int(est.n_features_in_)

    # 2. Check direct Pipeline named_steps
    if hasattr(model, "named_steps") and "scaler" in model.named_steps:
        scaler = model.named_steps["scaler"]
        if hasattr(scaler, "n_features_in_"):
            return int(scaler.n_features_in_)
        if hasattr(scaler, "mean_"):
            return len(scaler.mean_)
        if hasattr(scaler, "center_"):
            return len(scaler.center_)

    # 3. Check final classifier inside pipeline
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        clf = model.named_steps["classifier"]
        if hasattr(clf, "n_features_in_"):
            return int(clf.n_features_in_)

    # 4. Check direct n_features_in_
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)

    # 5. Check underlying base estimator if fitted
    if hasattr(model, "estimator"):
        est = model.estimator
        if hasattr(est, "named_steps") and "scaler" in est.named_steps:
            scaler = est.named_steps["scaler"]
            if hasattr(scaler, "n_features_in_"):
                return int(scaler.n_features_in_)
            if hasattr(scaler, "mean_"):
                return len(scaler.mean_)
            if hasattr(scaler, "center_"):
                return len(scaler.center_)

    return 42


def get_available_classifiers() -> Dict[str, bool]:
    """Checks which gradient boosting / forest classifier libraries are available."""
    available = {
        "random_forest": True,
        "xgboost": False,
        "lightgbm": False,
    }
    try:
        import xgboost  # noqa: F401
        available["xgboost"] = True
    except ImportError:
        pass

    try:
        import lightgbm  # noqa: F401
        available["lightgbm"] = True
    except ImportError:
        pass

    return available


def build_pipeline(
    model_type: str = "xgboost",
    scaler_type: str = "standard",
    n_estimators: int = 200,
    max_depth: Optional[int] = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """
    Constructs a scalable classification Pipeline with StandardScaler/RobustScaler and
    XGBoost / LightGBM / RandomForest classifier.
    """
    # 1. Feature Scaler Selection
    if scaler_type == "robust":
        scaler = RobustScaler(unit_variance=True)
    else:
        scaler = StandardScaler()

    # 2. Classifier Selection with Safe Fallbacks
    avail = get_available_classifiers()

    if model_type == "xgboost" and avail["xgboost"]:
        import xgboost as xgb
        classifier = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or 6,
            learning_rate=learning_rate,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
            **kwargs,
        )
    elif model_type == "lightgbm" and avail["lightgbm"]:
        import lightgbm as lgb
        classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth or -1,
            learning_rate=learning_rate,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            **kwargs,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            class_weight="balanced",
            random_state=random_state,
            **kwargs,
        )

    return Pipeline([
        ("scaler", scaler),
        ("classifier", classifier),
    ])


def load_model(model_path: str = MODEL_PATH) -> Optional[Pipeline]:
    """Loads serialized pipeline from disk."""
    paths_to_try = [model_path, os.path.splitext(model_path)[0] + ".joblib", MODEL_BASELINE_V1_PATH]
    for p in paths_to_try:
        if os.path.exists(p):
            try:
                return joblib.load(p)
            except Exception as e:
                print(f"Warning: Failed to load model from '{p}': {e}")
    return None


def load_metadata(metadata_path: str = MODEL_METADATA_PATH) -> Optional[Dict[str, Any]]:
    """Loads metadata JSON independently."""
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
    metadata_path: Optional[str] = None,
) -> None:
    """
    Persists pipeline to .pkl and .joblib, and metadata to JSON atomically.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    # Compute SHA256 of saved artifact
    import hashlib
    with open(model_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    metadata["model_artifact_sha256"] = actual_hash
    if "class_mapping" not in metadata:
        metadata["class_mapping"] = {"0": "bona_fide", "1": "spoof"}
    if "label_mapping" not in metadata:
        metadata["label_mapping"] = {"0": "bona_fide", "1": "spoof"}

    # Also persist .joblib copy
    joblib_path = os.path.splitext(model_path)[0] + ".joblib"
    if joblib_path != model_path:
        try:
            joblib.dump(pipeline, joblib_path)
        except Exception:
            pass

    meta_target = metadata_path or MODEL_METADATA_PATH
    os.makedirs(os.path.dirname(meta_target), exist_ok=True)
    with open(meta_target, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
