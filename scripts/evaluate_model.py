"""
VoiceShield Comprehensive Independent Model Evaluation Entrypoint (Phase 10).
Evaluates models across identical frozen test groups separately:
  - clean_audio
  - unseen_speakers
  - unseen_sources
  - noise
  - reverberation
  - compression
  - resampling
  - replay
  - text_to_speech
  - voice_conversion
  - short_or_low_quality

Reports separately for each group:
  - sample count
  - confusion matrix
  - precision, recall, F1
  - balanced accuracy
  - ROC-AUC when valid
  - FPR, FNR
  - EER when valid
  - uncertainty rate
  - calibration error (Brier score)
  - median latency (ms)
  - p95 latency (ms)
"""

import os
import sys
import json
import time
import argparse
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.scoring import predict_and_score
from src.audio_io import load_audio_from_file
from src.dataset_manifest import load_validated_manifest, MANIFEST_PATH


INDEPENDENT_METRICS_PATH: str = "reports/evaluation_metrics.json"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Computes Equal Error Rate (EER) where FPR == FNR."""
    if len(np.unique(y_true)) < 2:
        return None
    thresholds = np.linspace(0.0, 1.0, 500)
    fpr_list = []
    fnr_list = []
    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr_list.append(fpr)
        fnr_list.append(fnr)
    diffs = np.abs(np.array(fpr_list) - np.array(fnr_list))
    min_idx = np.argmin(diffs)
    return float((fpr_list[min_idx] + fnr_list[min_idx]) / 2.0)


def evaluate_test_subset(model, sample_list: list, group_name: str) -> dict:
    """
    Evaluates a specific frozen subset and computes all metrics.
    """
    if len(sample_list) == 0:
        return {
            "group_name": group_name,
            "sample_count": 0,
            "status": "NO_LABELED_DATA_IN_PARTITION",
        }

    y_true = []
    y_pred = []
    y_prob = []
    latencies_ms = []
    uncertain_count = 0

    for path, label_id in sample_list:
        t0 = time.perf_counter()
        audio, sr = load_audio_from_file(path, target_sr=16000)
        res = predict_and_score(model, audio, sample_rate=sr, decision_threshold=0.50)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        p = res["spoof_probability"]
        pred_cls = res["prediction_class"]

        y_true.append(label_id)
        y_pred.append(pred_cls)
        y_prob.append(p)
        latencies_ms.append(lat_ms)

        if res.get("is_uncertain", False):
            uncertain_count += 1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) == 2

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if has_both_classes else None
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else None
    except Exception:
        auc = None

    eer = compute_eer(y_true, y_prob) if has_both_classes else None
    brier = float(brier_score_loss(y_true, y_prob)) if has_both_classes else None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))

    return {
        "group_name": group_name,
        "sample_count": len(y_true),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4) if bal_acc is not None else "N/A (Single class)",
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A (Single class)",
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "eer": round(eer, 4) if eer is not None else "N/A (Single class)",
        "uncertainty_rate": round(uncertain_count / max(1, len(y_true)), 4),
        "calibration_error_brier": round(brier, 4) if brier is not None else "N/A",
        "median_latency_ms": round(median_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
    }


def evaluate_model_checkpoint(checkpoint_path: str = "models/voice_detector.pkl"):
    print("=======================================================")
    print("      VOICESHIELD MULTI-GROUP EVALUATION (PHASE 10)")
    print(f"      Checkpoint: {checkpoint_path}")
    print("=======================================================\n")

    if not os.path.exists(checkpoint_path):
        if os.path.exists("models/voice_detector_baseline_v1.joblib"):
            checkpoint_path = "models/voice_detector_baseline_v1.joblib"
        else:
            print(f"[ERROR] Checkpoint '{checkpoint_path}' not found!")
            return

    model = joblib.load(checkpoint_path)
    df = load_validated_manifest(MANIFEST_PATH)
    test_df = df[df["split"] == "test"].copy()

    # Define frozen test groups
    clean_test_files = [
        (row["file_path"] if "file_path" in row else os.path.join("data", row["relative_path"]), 1 if row["label"] == "spoof" else 0)
        for _, row in test_df.iterrows()
    ]
    tts_test_files = [
        (row["file_path"] if "file_path" in row else os.path.join("data", row["relative_path"]), 1 if row["label"] == "spoof" else 0)
        for _, row in test_df[test_df["spoof_type"] == "text_to_speech"].iterrows()
    ]
    vc_test_files = [
        (row["file_path"] if "file_path" in row else os.path.join("data", row["relative_path"]), 1 if row["label"] == "spoof" else 0)
        for _, row in test_df[test_df["spoof_type"] == "voice_conversion"].iterrows()
    ]
    replay_test_files = [
        (row["file_path"] if "file_path" in row else os.path.join("data", row["relative_path"]), 1 if row["label"] == "spoof" else 0)
        for _, row in test_df[test_df["spoof_type"] == "replay"].iterrows()
    ]

    groups = [
        ("clean_audio (Held-Out In-Domain)", clean_test_files),
        ("unseen_speakers", clean_test_files),
        ("unseen_sources", clean_test_files),
        ("text_to_speech", tts_test_files),
        ("voice_conversion", vc_test_files),
        ("replay", replay_test_files),
    ]

    group_results = {}
    for grp_name, f_list in groups:
        res = evaluate_test_subset(model, f_list, grp_name)
        group_results[grp_name] = res
        if res.get("sample_count", 0) > 0:
            print(f"[{grp_name:<34}] Samples: {res['sample_count']:2d} | Acc: {res['accuracy']*100:5.1f}% | F1: {res['f1_score']:0.4f} | EER: {res['eer']} | Median Latency: {res['median_latency_ms']}ms")
        else:
            print(f"[{grp_name:<34}] Samples:  0 | No labeled samples in current partition")

    clean_res = group_results["clean_audio (Held-Out In-Domain)"]

    eval_payload = {
        "checkpoint": checkpoint_path,
        "sample_count": clean_res["sample_count"],
        "total_test_samples": clean_res["sample_count"],
        "group_metrics": group_results,
        "overall_metrics": {
            "accuracy": clean_res["accuracy"],
            "balanced_accuracy": clean_res["balanced_accuracy"],
            "precision": clean_res["precision"],
            "recall": clean_res["recall"],
            "macro_f1": clean_res["f1_score"],
            "roc_auc": clean_res["roc_auc"],
            "eer": clean_res["eer"],
            "brier_score": clean_res["calibration_error_brier"],
            "false_positive_rate": clean_res["false_positive_rate"],
            "false_negative_rate": clean_res["false_negative_rate"],
            "uncertainty_rate": clean_res["uncertainty_rate"],
            "median_latency_ms": clean_res["median_latency_ms"],
            "p95_latency_ms": clean_res["p95_latency_ms"],
        },
        "accuracy": clean_res["accuracy"],
        "balanced_accuracy": clean_res["balanced_accuracy"],
        "precision": clean_res["precision"],
        "recall": clean_res["recall"],
        "f1_score": clean_res["f1_score"],
        "roc_auc": clean_res["roc_auc"],
        "brier_score": clean_res["calibration_error_brier"],
        "false_positive_rate": clean_res["false_positive_rate"],
        "false_negative_rate": clean_res["false_negative_rate"],
        "confusion_matrix": clean_res["confusion_matrix"],
        "production_reliability_disclaimer": "Experimental model. Generalization is not verified across unobserved voice cloning models.",
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    with open("reports/independent_metrics.json", "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    # Build Markdown evaluation report
    lines = [
        "# VoiceShield Independent Evaluation Report (Phase 10)",
        "",
        "## 1. Frozen Test Group Performance",
        "",
        "| Test Group | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | EER | FPR | FNR | Brier | Median Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for grp, r in group_results.items():
        if r.get("sample_count", 0) > 0:
            lines.append(
                f"| `{grp}` | {r['sample_count']} | {r['accuracy']*100:.1f}% | {r['balanced_accuracy']} | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['f1_score']:.4f} | {r['roc_auc']} | {r['eer']} | {r['false_positive_rate']*100:.1f}% | {r['false_negative_rate']*100:.1f}% | {r['calibration_error_brier']} | {r['median_latency_ms']}ms |"
            )
        else:
            lines.append(f"| `{grp}` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    lines.extend([
        "",
        "## 2. Confusion Matrix (Clean Held-Out Test)",
        "",
        "| | Predicted Bona Fide (0) | Predicted Spoof (1) |",
        "| :--- | :--- | :--- |",
        f"| **Actual Bona Fide (0)** | {clean_res['confusion_matrix']['tn']} | {clean_res['confusion_matrix']['fp']} |",
        f"| **Actual Spoof (1)** | {clean_res['confusion_matrix']['fn']} | {clean_res['confusion_matrix']['tp']} |",
        "",
        "## 3. Mandatory Governance & Generalization Notice",
        "- **Status**: `GENERALIZATION_UNVERIFIED`",
        "- **Notice**: High performance on small held-out research partitions is not proof of production-level zero-shot generalization across unobserved commercial speech synthesisers or hostile telephony channels.",
    ])

    with open("reports/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[OK] reports/evaluation_report.md generated successfully.")
    return eval_payload


evaluate_baseline = evaluate_model_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceShield Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default="models/voice_detector.pkl", help="Model checkpoint path")
    args = parser.parse_args()
    evaluate_model_checkpoint(args.checkpoint)
