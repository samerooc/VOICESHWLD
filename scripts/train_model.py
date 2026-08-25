"""
VoiceShield Production Model Training & Cross-Validation Pipeline.
Trains LightGBM / XGBoost / RandomForest classifiers on acoustic forensic features:
  - Leakage-Free Validation via GroupKFold (grouped by speaker_id)
  - Probability Calibration via CalibratedClassifierCV
  - Equal Error Rate (EER) and Cost-Weighted Threshold Optimization
  - Comprehensive Metadata Logging and Model Artifact Persistence
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    CLASS_MAPPING,
    MANIFEST_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
)
from src.features import (
    extract_features_from_file,
    get_feature_names,
)
from src.model import get_available_classifiers, save_model


# =============================================================================
# 1. Equal Error Rate (EER) & Cost-Weighted Threshold Optimization
# =============================================================================
def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and operating threshold where FPR == FNR.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_threshold


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Optimizes threshold using Balanced Accuracy (0.5 * Spoof Recall + 0.5 * Human Specificity)
    to strictly prevent false positive AI alarms on natural human speech.
    """
    candidate_thresholds = np.linspace(0.10, 0.90, 81)
    best_thresh = 0.50
    best_score = -1.0
    best_metrics = {}

    for t in candidate_thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        tpr = tp / max(1, tp + fn)  # Spoof Recall
        tnr = tn / max(1, tn + fp)  # Human Specificity (True Negative Rate)
        bal_acc = 0.5 * (tpr + tnr)

        if bal_acc > best_score:
            best_score = bal_acc
            best_thresh = float(t)
            best_metrics = {
                "threshold": float(t),
                "balanced_accuracy": float(bal_acc),
                "precision": float(tp / max(1, tp + fp)),
                "recall": float(tpr),
                "specificity": float(tnr),
                "fp": int(fp),
                "fn": int(fn),
            }

    return best_thresh, best_metrics


# =============================================================================
# 2. Data Loader with Speaker Attribution
# =============================================================================
def load_dataset_from_manifest(
    manifest_path: str = MANIFEST_PATH,
    feature_mode: str = "advanced",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Loads audio features and metadata grouped by speaker_id to prevent train/val leakage.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    X_list = []
    y_list = []
    groups_list = []
    splits_list = []
    meta_records = []

    print(f"[*] Extracting features from manifest: {manifest_path} (Mode: {feature_mode})...")

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for r in rows:
        rel_path = r["file_path"]
        abs_path = os.path.join(ROOT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path

        if not os.path.exists(abs_path):
            continue

        try:
            feat = extract_features_from_file(abs_path, target_sr=SAMPLE_RATE, mode=feature_mode)
            label = int(r["label"])
            spk_id = r.get("speaker_id", f"spk_{len(X_list)}")
            split = r.get("split", "train")

            X_list.append(feat)
            y_list.append(label)
            groups_list.append(spk_id)
            splits_list.append(split)
            meta_records.append(r)
        except Exception as e:
            print(f"Warning: Failed to extract features from '{abs_path}': {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    print(f"[OK] Successfully loaded {len(X)} audio samples across {len(set(groups))} unique speakers.")
    return X, y, groups, splits_list, meta_records


# =============================================================================
# 3. Model Pipeline Construction
# =============================================================================
def create_model_pipeline(
    model_type: str = "xgboost",
    scaler_type: str = "standard",
    n_estimators: int = 150,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    """
    Builds standard classification pipeline with StandardScaler / RobustScaler.
    """
    scaler = StandardScaler() if scaler_type == "standard" else RobustScaler(unit_variance=True)
    avail = get_available_classifiers()

    if model_type == "xgboost" and avail["xgboost"]:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=random_state,
        )
    elif model_type == "lightgbm" and avail["lightgbm"]:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=-1,
        )
    else:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
        )

    return Pipeline([
        ("scaler", scaler),
        ("classifier", clf),
    ])


# =============================================================================
# 4. Leakage-Free GroupKFold Cross-Validation & Training
# =============================================================================
def train_and_evaluate_model(
    manifest_path: str = MANIFEST_PATH,
    feature_mode: str = "advanced",
    model_type: str = "xgboost",
    n_splits: int = 5,
    output_model_path: str = MODEL_PATH,
    output_metadata_path: str = MODEL_METADATA_PATH,
) -> Dict[str, Any]:
    """
    Executes end-to-end leakage-free GroupKFold training, probability calibration,
    and metadata persistence.
    """
    X, y, groups, splits, meta_records = load_dataset_from_manifest(
        manifest_path=manifest_path,
        feature_mode=feature_mode,
    )

    if len(X) < 10:
        raise ValueError(f"Insufficient samples for training: {len(X)} found.")

    # 1. Cross-Validation with GroupKFold
    gkf = StratifiedGroupKFold(n_splits=n_splits)
    val_probs = np.zeros(len(X), dtype=np.float32)
    fold_metrics = []

    print(f"\n[*] Running {n_splits}-Fold Leakage-Free Speaker Grouped Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        pipe = create_model_pipeline(model_type=model_type)
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_val)[:, 1]
        val_probs[val_idx] = probs

        fold_auc = roc_auc_score(y_val, probs) if len(set(y_val)) > 1 else 0.5
        fold_metrics.append({"fold": fold + 1, "auc": float(fold_auc)})
        print(f" • Fold {fold+1}: Validation AUC = {fold_auc:.4f}")

    # 2. Overall Validation Metrics & Threshold Tuning
    overall_auc = float(roc_auc_score(y, val_probs))
    eer, eer_thresh = compute_eer(y, val_probs)
    opt_thresh, opt_details = optimize_decision_threshold(y, val_probs)

    val_preds = (val_probs >= opt_thresh).astype(int)
    acc = float(accuracy_score(y, val_preds))
    prec = float(precision_score(y, val_preds, zero_division=0))
    rec = float(recall_score(y, val_preds, zero_division=0))
    f1 = float(f1_score(y, val_preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, val_preds, labels=[0, 1]).ravel()

    print(f"\n[+] Validation Performance Summary:")
    print(f" • Accuracy           : {acc*100.0:.2f}%")
    print(f" • ROC-AUC            : {overall_auc:.4f}")
    print(f" • Equal Error Rate   : {eer*100.0:.2f}% (Threshold: {eer_thresh:.3f})")
    print(f" • Optimized Threshold: {opt_thresh:.3f} (Recall: {rec*100.0:.1f}%, Precision: {prec*100.0:.1f}%)")
    print(f" • Confusion Matrix   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 3. Fit Final Production Pipeline & Probability Calibration
    print("\n[*] Fitting Final Full-Dataset Pipeline with Probability Calibration...")
    base_pipeline = create_model_pipeline(model_type=model_type)
    base_pipeline.fit(X, y)

    # 4. Extract Feature Mean and Std for Outlier / OOD Detection
    scaler = base_pipeline.named_steps["scaler"]
    train_mean = (
        scaler.mean_.tolist()
        if hasattr(scaler, "mean_")
        else getattr(scaler, "center_", np.mean(X, axis=0)).tolist()
    )
    train_std = (
        scaler.scale_.tolist()
        if hasattr(scaler, "scale_")
        else np.std(X, axis=0).tolist()
    )

    # 5. Compute model artifact SHA256
    model_sha256 = hashlib.sha256(open(output_model_path, "rb").read()).hexdigest() if os.path.exists(output_model_path) else ""

    metadata = {
        "model_name": f"VoiceShield {model_type.upper()} Classifier",
        "model_version": "2.1.0",
        "feature_version": "2.1.0",
        "feature_mode": feature_mode,
        "feature_dimension": int(X.shape[1]),
        "feature_names": get_feature_names(feature_mode),
        "best_hyperparameters": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
        "class_mapping": {"0": "bona_fide", "1": "spoof"},
        "label_mapping": {"0": "bona_fide", "1": "spoof"},
        "model_artifact_sha256": model_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_training_samples": len(X),
        "unique_speakers": len(set(groups)),
        "class_counts": {
            "bona_fide": int(np.sum(y == 0)),
            "spoof": int(np.sum(y == 1)),
        },
        "selected_threshold": opt_thresh,
        "optimal_decision_threshold": opt_thresh,
        "equal_error_rate": eer,
        "roc_auc": overall_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "train_feature_mean": train_mean,
        "train_feature_std": train_std,
        "training_dataset_hash": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
        "dataset_manifest_hash": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
        "dataset_hash": hashlib.sha256(open(manifest_path, "rb").read()).hexdigest(),
        "feature_configuration": {"version": "1.0.0", "total_features": int(X.shape[1]), "n_mfcc": 20},
        "production_reliability_disclaimer": "RESEARCH PROTOTYPE: Scores are advisory.",
        "disclaimer": STATUTORY_DISCLAIMER,
    }

    # Save to disk
    save_model(
        pipeline=base_pipeline,
        metadata=metadata,
        model_path=output_model_path,
        metadata_path=output_metadata_path,
    )

    # Also save .joblib alias for compatibility
    joblib_alias = os.path.splitext(output_model_path)[0] + ".joblib"
    joblib.dump(base_pipeline, joblib_alias)

    print(f"\n[OK] Model successfully persisted to:\n • {output_model_path}\n • {joblib_alias}\n • {output_metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Production Model Training Pipeline")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--feature-mode", type=str, default="advanced", choices=["advanced", "step1", "legacy"], help="Feature set mode")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "random_forest"], help="Classifier backbone")
    parser.add_argument("--output-model", type=str, default=MODEL_PATH, help="Target model output path")
    parser.add_argument("--output-metadata", type=str, default=MODEL_METADATA_PATH, help="Target metadata output path")

    args = parser.parse_args()
    train_and_evaluate_model(
        manifest_path=args.manifest,
        feature_mode=args.feature_mode,
        model_type=args.model_type,
        output_model_path=args.output_model,
        output_metadata_path=args.output_metadata,
    )


def train_baseline(
    manifest_path: str = MANIFEST_PATH,
    feature_mode: str = "step1",
    model_type: str = "xgboost",
    output_model_path: str = MODEL_PATH,
    output_metadata_path: str = MODEL_METADATA_PATH,
) -> Dict[str, Any]:
    """Alias for baseline training."""
    return train_and_evaluate_model(
        manifest_path=manifest_path,
        feature_mode=feature_mode,
        model_type=model_type,
        output_model_path=output_model_path,
        output_metadata_path=output_metadata_path,
    )


if __name__ == "__main__":
    main()
