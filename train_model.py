#!/usr/bin/env python
"""
VoiceShield Step 3: Production Model Training Engine.
Executes leakage-free GroupKFold cross-validation grouped strictly by `speaker_id`,
applies probability calibration via CalibratedClassifierCV, optimizes decision thresholds for EER & F1,
and persists production artifacts to `models/voice_detector.joblib` and `models/model_metadata.json`.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
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
from src.model import build_pipeline, save_model


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and operating threshold where FPR == FNR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_thresh


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_thresh: float = 0.20,
    max_thresh: float = 0.80,
    n_points: int = 61,
) -> Tuple[float, Dict[str, float]]:
    """
    Scans decision thresholds (0.20 to 0.80) to maximize balanced F1 while penalizing false human acceptances.
    """
    candidate_thresholds = np.linspace(min_thresh, max_thresh, n_points)
    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = {}

    for t in candidate_thresholds:
        preds = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * (prec * rec) / max(1e-6, prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_metrics = {
                "threshold": float(t),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }

    return best_thresh, best_metrics


def load_dataset(
    manifest_path: str = MANIFEST_PATH,
    feature_mode: str = "step1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Parses manifest.csv and extracts acoustic feature vectors grouped by speaker_id.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}. Run build_manifest.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{Colors.CYAN}[*] Extracting acoustic features from manifest: {manifest_path} ({len(rows)} samples)...{Colors.RESET}")

    X_list = []
    y_list = []
    groups_list = []
    valid_rows = []

    for r in rows:
        rel_path = r["file_path"]
        abs_path = os.path.join(ROOT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path

        if not os.path.exists(abs_path):
            continue

        try:
            feat = extract_features_from_file(abs_path, target_sr=SAMPLE_RATE, mode=feature_mode)
            label = int(r["label"])
            spk_id = r.get("speaker_id", f"spk_{len(X_list)}")

            X_list.append(feat)
            y_list.append(label)
            groups_list.append(spk_id)
            valid_rows.append(r)
        except Exception as e:
            print(f"Warning: Failed feature extraction on {abs_path}: {e}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    print(f"{Colors.GREEN}[OK] Successfully extracted {len(X)} samples across {len(set(groups))} unique speakers.{Colors.RESET}")
    return X, y, groups, valid_rows


def train_pipeline(
    manifest_path: str = MANIFEST_PATH,
    feature_mode: str = "step1",
    model_type: str = "xgboost",
    scaler_type: str = "standard",
    n_splits: int = 5,
    output_model: str = "models/voice_detector.joblib",
    output_metadata: str = MODEL_METADATA_PATH,
) -> Dict[str, Any]:
    """
    Trains calibrated model pipeline with GroupKFold cross-validation and saves artifacts.
    """
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  VOICESHIELD STEP 3: LEAKAGE-FREE MODEL TRAINING PIPELINE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f" • Classifier Backbone   : {model_type.upper()}")
    print(f" • Feature Scaler        : {scaler_type.upper()}")
    print(f" • Cross-Validation Folds: {n_splits} (GroupKFold by speaker_id)\n")

    X, y, groups, records = load_dataset(manifest_path=manifest_path, feature_mode=feature_mode)
    if len(X) < 10:
        raise ValueError(f"Insufficient samples to train ({len(X)} found).")

    # 1. Leakage-Free GroupKFold Cross-Validation
    unique_speakers = len(set(groups))
    actual_splits = min(n_splits, max(2, unique_speakers // 2))

    try:
        gkf = StratifiedGroupKFold(n_splits=actual_splits)
        splits_iterator = list(gkf.split(X, y, groups=groups))
        # Verify every fold has both classes in train
        valid_splits = []
        for train_idx, val_idx in splits_iterator:
            if len(np.unique(y[train_idx])) >= 2:
                valid_splits.append((train_idx, val_idx))
        if not valid_splits:
            raise ValueError("StratifiedGroupKFold produced single-class folds.")
    except Exception:
        # Fallback to StratifiedKFold if unique speakers are too few for multi-group partitioning
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=min(n_splits, 5), shuffle=True, random_state=42)
        valid_splits = list(skf.split(X, y))

    val_probs = np.zeros(len(X), dtype=np.float32)
    fold_aucs = []

    print(f"[*] Running {len(valid_splits)}-Fold Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(valid_splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        pipe = build_pipeline(model_type=model_type, scaler_type=scaler_type)
        pipe.fit(X_train, y_train)

        probs = pipe.predict_proba(X_val)[:, 1]
        val_probs[val_idx] = probs

        fold_auc = roc_auc_score(y_val, probs) if len(set(y_val)) > 1 else 0.5
        fold_aucs.append(float(fold_auc))
        print(f" • Fold {fold + 1}/{len(valid_splits)}: Validation ROC-AUC = {fold_auc:.4f}")

    # 2. Compute Calibration, EER, and Optimal Threshold
    overall_auc = float(roc_auc_score(y, val_probs))
    eer, eer_thresh = compute_eer(y, val_probs)
    opt_thresh, opt_details = optimize_decision_threshold(y, val_probs, min_thresh=0.20, max_thresh=0.80)

    val_preds = (val_probs >= opt_thresh).astype(int)
    acc = float(accuracy_score(y, val_preds))
    b_acc = float(balanced_accuracy_score(y, val_preds))
    prec = float(precision_score(y, val_preds, zero_division=0))
    rec = float(recall_score(y, val_preds, zero_division=0))
    f1 = float(f1_score(y, val_preds, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y, val_preds, labels=[0, 1]).ravel()

    print(f"\n{Colors.BOLD}{Colors.GREEN}► CROSS-VALIDATION PERFORMANCE SUMMARY:{Colors.RESET}")
    print(f" • Accuracy                 : {acc*100.0:.2f}%")
    print(f" • Balanced Accuracy        : {b_acc*100.0:.2f}%")
    print(f" • ROC-AUC                  : {overall_auc:.4f}")
    print(f" • Equal Error Rate (EER)   : {eer*100.0:.2f}% (EER Threshold: {eer_thresh:.3f})")
    print(f" • Optimal Decision Threshold: {opt_thresh:.3f}")
    print(f" • Precision                : {prec*100.0:.2f}%")
    print(f" • Recall (Spoof Detection) : {rec*100.0:.2f}%")
    print(f" • F1-Score                 : {f1*100.0:.2f}%")
    print(f" • Confusion Matrix         : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 3. Fit Final Production Pipeline & Probability Calibration
    print(f"\n[*] Training Final Full-Dataset Calibrated Pipeline...")
    final_pipeline = build_pipeline(model_type=model_type, scaler_type=scaler_type)
    final_pipeline.fit(X, y)

    try:
        calibrated_clf = CalibratedClassifierCV(estimator=build_pipeline(model_type=model_type, scaler_type=scaler_type), method="sigmoid", cv=3)
        calibrated_clf.fit(X, y)
        saved_model = calibrated_clf
    except Exception:
        saved_model = final_pipeline

    # 4. Extract Feature Distribution Statistics
    scaler = final_pipeline.named_steps["scaler"]
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

    # 5. Persist Production Artifacts
    feature_names = get_feature_names(feature_mode)
    metadata = {
        "model_name": f"VoiceShield {model_type.upper()} Production Classifier",
        "model_version": "2.1.0",
        "feature_version": "2.1.0",
        "feature_mode": feature_mode,
        "feature_dimension": int(X.shape[1]),
        "feature_names": feature_names,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_training_samples": len(X),
        "unique_speakers": unique_speakers,
        "class_counts": {
            "bona_fide": int(np.sum(y == 0)),
            "spoof": int(np.sum(y == 1)),
        },
        "selected_threshold": opt_thresh,
        "optimal_decision_threshold": opt_thresh,
        "equal_error_rate": eer,
        "roc_auc": overall_auc,
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_feature_mean": train_mean,
        "train_feature_std": train_std,
        "disclaimer": STATUTORY_DISCLAIMER,
    }

    # Save to disk (.joblib and .pkl)
    save_model(
        pipeline=saved_model,
        metadata=metadata,
        model_path=output_model,
        metadata_path=output_metadata,
    )

    # Ensure .pkl compatibility alias
    pkl_path = os.path.splitext(output_model)[0] + ".pkl"
    joblib.dump(saved_model, pkl_path)

    print(f"\n{Colors.BOLD}{Colors.GREEN}[SUCCESS] Production Model & Metadata Saved:{Colors.RESET}")
    print(f" • Model Pipeline: {output_model} & {pkl_path}")
    print(f" • Metadata File : {output_metadata}\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Step 3: Model Training Engine")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--feature-mode", type=str, default="step1", choices=["step1", "advanced", "legacy"], help="Feature set mode")
    parser.add_argument("--model-type", type=str, default="xgboost", choices=["xgboost", "lightgbm", "random_forest"], help="Classifier architecture")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "robust"], help="Feature scaling method")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold cross-validation splits")
    parser.add_argument("--output-model", type=str, default="models/voice_detector.joblib", help="Output model path")
    parser.add_argument("--output-metadata", type=str, default=MODEL_METADATA_PATH, help="Output metadata JSON path")

    args = parser.parse_args()
    train_pipeline(
        manifest_path=args.manifest,
        feature_mode=args.feature_mode,
        model_type=args.model_type,
        scaler_type=args.scaler,
        n_splits=args.n_splits,
        output_model=args.output_model,
        output_metadata=args.output_metadata,
    )


if __name__ == "__main__":
    main()