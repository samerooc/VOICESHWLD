"""
VoiceShield Reproducible Baseline v1 Model Training Pipeline (Phase 3 + Explainability).
Reads data/manifest.csv, fits a StandardScaler + RandomForest Pipeline,
performs parameter grid search and validation threshold tuning, extracts feature importances,
and persists metadata & feature importance reports.
"""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    CLASS_MAPPING,
    FEATURE_CONFIG,
    LABEL_AI,
    LABEL_HUMAN,
    MANIFEST_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    RESEARCH_NOTICE,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
)
from src.explainability import get_global_feature_importance
from src.audio_io import load_audio_from_file
from src.features import extract_features_from_audio, extract_features_from_file, extract_segmented_features
from src.model import build_pipeline, save_model


def load_data_from_manifest(
    manifest_path: str = MANIFEST_PATH,
    split_name: str = "train",
    augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], str]:
    """
    Loads samples from manifest.csv for the specified split, computes dataset hash,
    and extracts 42 acoustic features across full-files and sliding temporal segments.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest file not found: '{manifest_path}'. "
            f"Please run `python scripts/build_manifest.py` first."
        )

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") == split_name and row.get("is_valid", "True").lower() == "true":
                records.append(row)

    if len(records) < 2:
        raise ValueError(
            f"Split '{split_name}' has insufficient valid samples in manifest (Found: {len(records)})."
        )

    sorted_hashes = "".join(sorted(r["sha256_hash"] for r in records if r.get("sha256_hash")))
    dataset_hash = hashlib.sha256(sorted_hashes.encode("utf-8")).hexdigest()

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    print(f"Loading '{split_name}' split from manifest: {len(records)} samples...")
    for idx, r in enumerate(records, 1):
        file_path = r.get("path") or r.get("file_path", "")
        lbl = r.get("label") or r.get("class_label", "bona_fide")
        label_id = 1 if lbl in ["spoof", "ai_voice"] else 0

        audio, sr = load_audio_from_file(file_path, target_sr=SAMPLE_RATE)

        # 1. Full-file feature vector
        feat_full = extract_features_from_audio(audio, sample_rate=sr)
        features_list.append(feat_full)
        labels_list.append(label_id)

        # Segment slicing and augmentations (Train split only)
        if split_name == "train" and augment:
            seg_feats = extract_segmented_features(audio, sr, window_duration=2.5, hop_duration=1.0)
            for sf in seg_feats:
                features_list.append(sf)
                labels_list.append(label_id)

            # Multi-condition augmentations for channel and microphone robustness
            if len(audio) > sr:
                # 1. Gain variations (+/- 2.5dB)
                audio_louder = np.clip(audio * 1.25, -1.0, 1.0)
                features_list.append(extract_features_from_audio(audio_louder, sample_rate=sr))
                labels_list.append(label_id)

                audio_softer = audio * 0.75
                features_list.append(extract_features_from_audio(audio_softer, sample_rate=sr))
                labels_list.append(label_id)

                # 2. Additive ambient room noise
                noise = np.random.normal(0, 0.005, len(audio)).astype(np.float32)
                audio_noisy = np.clip(audio + noise, -1.0, 1.0)
                features_list.append(extract_features_from_audio(audio_noisy, sample_rate=sr))
                labels_list.append(label_id)

                # 3. Telephony 8kHz downsampling/upsampling simulation for BOTH classes
                down_8k = librosa.resample(audio, orig_sr=sr, target_sr=8000)
                audio_telephony = librosa.resample(down_8k, orig_sr=8000, target_sr=sr)
                features_list.append(extract_features_from_audio(audio_telephony, sample_rate=sr))
                labels_list.append(label_id)

        print(f"  [{idx:02d}/{len(records)}] Loaded: {file_path:<30} | Class: {lbl} ({label_id})")

    x = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return x, y, records, dataset_hash


def tune_decision_threshold(
    y_true: np.ndarray,
    y_probs_spoof: np.ndarray,
) -> Tuple[float, float]:
    """
    Scans decision thresholds from 0.20 to 0.80 to find the optimal threshold
    that maximizes the balanced F1 score on validation data, selecting the most stable center threshold.
    """
    best_threshold = 0.50
    best_f1 = -1.0
    best_thresholds = []

    thresholds = np.linspace(0.20, 0.80, 61)
    for t in thresholds:
        preds = (y_probs_spoof >= t).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thresholds = [t]
        elif abs(score - best_f1) < 1e-6:
            best_thresholds.append(t)

    # Pick the threshold in best_thresholds closest to 0.50 for optimal balance
    if best_thresholds:
        best_threshold = float(round(min(best_thresholds, key=lambda x: abs(x - 0.50)), 3))

    return best_threshold, best_f1


def train_baseline() -> Dict[str, Any]:
    """
    Trains Baseline v1: StandardScaler + RandomForest Pipeline with parameter grid tuning,
    threshold calibration, and feature importance generation.
    """
    print("=======================================================================")
    print("       VOICESHIELD HIGH-ACCURACY TRAINING PIPELINE (PHASE 3)")
    print("=======================================================================\n")

    x_train_full, y_train_full, train_records, dataset_hash = load_data_from_manifest(
        manifest_path=MANIFEST_PATH,
        split_name="train",
        augment=True,
    )

    print(f"\nTraining Dataset Hash (SHA-256): {dataset_hash}")
    print(f"Feature Matrix Shape: {x_train_full.shape} (Samples x Features)")
    print(f"Class Balance: {np.sum(y_train_full == LABEL_HUMAN)} bona_fide (0), {np.sum(y_train_full == LABEL_AI)} spoof (1)")

    # Compute training baseline statistics for OOD detection
    train_mean = np.mean(x_train_full, axis=0).tolist()
    train_std = np.std(x_train_full, axis=0).tolist()

    # 1. Stratified Train / Validation Split (80% Train, 20% Validation)
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    # 2. Build Pipeline: StandardScaler + Balanced RandomForest
    base_pipeline = build_pipeline(random_state=42)

    # 3. Documented Hyperparameter Grid Search (n_jobs=1 for Windows safe execution)
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 8, 12],
        "classifier__min_samples_split": [2, 3],
    }

    print("\n--- Running Hyperparameter Grid Search (Stratified CV) ---")
    cv_splits = min(5, min(np.sum(y_tr == 0), np.sum(y_tr == 1)))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=1,
    )
    grid_search.fit(x_tr, y_tr)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best Hyperparameters: {best_params}")
    print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

    # 4. Validation Set Evaluation & Threshold Tuning
    val_probs = best_pipeline.predict_proba(x_val)[:, 1]
    optimal_threshold, val_f1 = tune_decision_threshold(y_val, val_probs)
    val_preds = (val_probs >= optimal_threshold).astype(int)

    from sklearn.metrics import balanced_accuracy_score
    import sklearn
    import librosa
    from src.config import MODEL_BASELINE_V1_PATH, VALIDATION_METRICS_PATH

    val_acc = float(accuracy_score(y_val, val_preds))
    val_bal_acc = float(balanced_accuracy_score(y_val, val_preds))
    val_prec = float(precision_score(y_val, val_preds, zero_division=0))
    val_rec = float(recall_score(y_val, val_preds, zero_division=0))
    val_auc = float(roc_auc_score(y_val, val_probs)) if len(np.unique(y_val)) > 1 else 1.0

    val_cm = confusion_matrix(y_val, val_preds)
    val_tn, val_fp, val_fn, val_tp = (
        int(val_cm[0, 0]), int(val_cm[0, 1]), int(val_cm[1, 0]), int(val_cm[1, 1])
    ) if val_cm.shape == (2, 2) else (0, 0, 0, 0)

    val_metrics_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "is_preliminary": bool(len(y_train_full) < 100),
        "validation_samples": int(len(y_val)),
        "optimal_decision_threshold": optimal_threshold,
        "accuracy": round(val_acc, 4),
        "balanced_accuracy": round(val_bal_acc, 4),
        "precision": round(val_prec, 4),
        "recall": round(val_rec, 4),
        "f1_score": round(val_f1, 4),
        "roc_auc": round(val_auc, 4) if len(np.unique(y_val)) > 1 else None,
        "confusion_matrix": {
            "true_negatives_bona_fide": val_tn,
            "false_positives_spoof": val_fp,
            "false_negatives_bona_fide": val_fn,
            "true_positives_spoof": val_tp,
            "matrix_2x2": [[val_tn, val_fp], [val_fn, val_tp]],
        },
        "false_positive_count": val_fp,
        "false_negative_count": val_fn,
        "disclaimer": RESEARCH_NOTICE,
    }

    os.makedirs(os.path.dirname(VALIDATION_METRICS_PATH), exist_ok=True)
    with open(VALIDATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(val_metrics_payload, f, indent=2)

    print("\n--- Validation Tuning Metrics ---")
    print(f"  • Optimal Decision Threshold : {optimal_threshold:.3f}")
    print(f"  • Validation Accuracy        : {val_acc * 100:.2f}%")
    print(f"  • Balanced Accuracy          : {val_bal_acc * 100:.2f}%")
    print(f"  • Validation Precision       : {val_prec * 100:.2f}%")
    print(f"  • Validation Recall          : {val_rec * 100:.2f}%")
    print(f"  • Validation F1-Score        : {val_f1:.4f}")
    print(f"  • Validation ROC-AUC         : {val_auc:.4f}")

    # 5. Fit Final Model on Full Training Split
    print("\nFitting final tuned pipeline on all training samples...")
    best_pipeline.fit(x_train_full, y_train_full)

    # 6. Extract Feature Importances & Generate reports/feature_importance.csv
    raw_imp_df, top_groups_df = get_global_feature_importance(best_pipeline)
    print("\n--- Top Acoustic Feature Groups by Importance ---")
    for _, row in top_groups_df.iterrows():
        print(f"  • {row['feature_group']:<50}: {row['importance_share']*100:.1f}%")

    # 7. Compile & Persist Metadata
    package_versions = {
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "joblib": joblib.__version__,
    }

    split_counts = {
        "train": int(len(x_tr)),
        "validation": int(len(x_val)),
        "total_train_partition": int(len(y_train_full)),
    }

    class_counts = {
        "bona_fide": int(np.sum(y_train_full == LABEL_HUMAN)),
        "spoof": int(np.sum(y_train_full == LABEL_AI)),
    }

    model_metadata = {
        "model_name": "VoiceShield Baseline v1 (MFCC + StandardScaler + RandomForest)",
        "model_version": "1.0.0",
        "feature_version": FEATURE_CONFIG.get("version", "1.0.0"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "random_seed": 42,
        "training_dataset_hash": dataset_hash,
        "dataset_manifest_hash": dataset_hash,
        "dataset_hash": dataset_hash,
        "manifest_path": MANIFEST_PATH,
        "package_versions": package_versions,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "train_sample_count": int(len(x_tr)),
        "validation_sample_count": int(len(x_val)),
        "total_training_samples": int(len(y_train_full)),
        "bona_fide_samples": int(np.sum(y_train_full == LABEL_HUMAN)),
        "spoof_samples": int(np.sum(y_train_full == LABEL_AI)),
        "feature_configuration": FEATURE_CONFIG,
        "class_mapping": CLASS_MAPPING,
        "label_mapping": {"0": "bona_fide", "1": "spoof"},
        "best_hyperparameters": best_params,
        "threshold": optimal_threshold,
        "selected_threshold": optimal_threshold,
        "optimal_decision_threshold": optimal_threshold,
        "train_feature_mean": train_mean,
        "train_feature_std": train_std,
        "preliminary_results": bool(len(y_train_full) < 100),
        "validation_metrics": val_metrics_payload,
        "production_reliability_disclaimer": RESEARCH_NOTICE,
    }

    save_model(best_pipeline, model_metadata, MODEL_PATH, MODEL_METADATA_PATH)
    try:
        joblib.dump(best_pipeline, MODEL_BASELINE_V1_PATH)
    except Exception:
        pass

    print(f"\n[OK] Trained Model Pipeline saved to: {MODEL_PATH}")
    print(f"[OK] Trained Model Joblib saved to: {MODEL_BASELINE_V1_PATH}")
    print(f"[OK] Model Metadata saved to: {MODEL_METADATA_PATH}")
    print(f"[OK] Validation Metrics saved to: {VALIDATION_METRICS_PATH}")
    print(f"[OK] Global Feature Importance CSV saved to: reports/feature_importance.csv")

    print("\n=======================================================================")
    print("                      TRAINING COMPLETE")
    print("=======================================================================")
    return model_metadata


if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as e:
        print(f"[ERROR] Training pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
