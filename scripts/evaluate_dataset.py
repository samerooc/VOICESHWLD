#!/usr/bin/env python
"""
VoiceShield Step 3: Model Benchmark & Held-Out Evaluation Tool.
Evaluates production model pipeline against held-out test splits, logs:
  - Accuracy, Balanced Accuracy, Precision, Recall, F1, ROC-AUC, EER
  - Full Confusion Matrix visualization saved to `reports/confusion_matrix.png`
  - Generator/Sub-dataset Performance Breakdown
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
import matplotlib.pyplot as plt
import numpy as np
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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    MANIFEST_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    SAMPLE_RATE,
)
from src.features import extract_features_from_file


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) where FPR == FNR."""
    if len(set(y_true)) < 2:
        return 0.0, 0.50
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        diffs = np.abs(fpr - fnr)
        if np.all(np.isnan(diffs)):
            return 0.0, 0.50
        idx = int(np.nanargmin(diffs))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
        return eer, eer_thresh
    except Exception:
        return 0.0, 0.50


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    output_path: str = "reports/confusion_matrix.png",
    class_names: List[str] = ["Human Voice", "Synthetic Spoof"],
) -> None:
    """
    Renders and saves a publication-quality confusion matrix heatmap.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 5), dpi=300)

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("VoiceShield Confusion Matrix (Held-Out Test Set)", fontsize=12, fontweight="bold", pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, fontsize=10)
    plt.yticks(tick_marks, class_names, fontsize=10)

    # Text annotations inside cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center", color=color, fontsize=14, fontweight="bold")

    plt.ylabel("Actual True Label", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Risk Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f" • Confusion Matrix Plot Saved : {output_path}")


def evaluate_test_set(
    manifest_path: str = MANIFEST_PATH,
    model_path: str = "models/voice_detector.joblib",
    metadata_path: str = MODEL_METADATA_PATH,
    split_name: str = "test",
    threshold: Optional[float] = None,
    output_plot: str = "reports/confusion_matrix.png",
    output_report: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs rigorous diagnostic evaluation on held-out test audio samples."""
    # Find active model file
    if not os.path.exists(model_path):
        alt_path = os.path.splitext(model_path)[0] + ".pkl"
        if os.path.exists(alt_path):
            model_path = alt_path
        elif os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    feature_mode = metadata.get("feature_mode", "step1")
    if threshold is None:
        threshold = float(metadata.get("optimal_decision_threshold", metadata.get("selected_threshold", 0.50)))

    # Parse Test Manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    test_rows = [r for r in rows if r.get("split", "").lower() == split_name.lower()]
    if not test_rows:
        print(f"{Colors.YELLOW}[!] Warning: No samples with split='{split_name}' found. Evaluating on full manifest.{Colors.RESET}")
        test_rows = rows

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  VOICESHIELD STEP 3: HELD-OUT TEST SPLIT EVALUATION REPORT{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f" • Evaluated Model File: {model_path}")
    print(f" • Test Split Samples  : {len(test_rows)}")
    print(f" • Decision Threshold  : {threshold:.3f}")
    print(f" • Feature Mode        : {feature_mode}\n")

    X_test = []
    y_test = []
    generators = []

    for r in test_rows:
        rel_path = r["file_path"]
        abs_path = os.path.join(ROOT_DIR, rel_path) if not os.path.isabs(rel_path) else rel_path

        if not os.path.exists(abs_path):
            continue

        try:
            feat = extract_features_from_file(abs_path, target_sr=SAMPLE_RATE, mode=feature_mode)
            X_test.append(feat)
            y_test.append(int(r["label"]))
            generators.append(r.get("generator_type", "Unknown"))
        except Exception as e:
            print(f"Warning: Failed to extract test sample '{abs_path}': {e}")

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)
    generators = np.array(generators)

    # Model Inference
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    # Compute Metrics
    acc = float(accuracy_score(y_test, preds))
    b_acc = float(balanced_accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    auc = float(roc_auc_score(y_test, probs)) if len(set(y_test)) > 1 else 0.5
    eer, eer_thresh = compute_eer(y_test, probs)
    cm = confusion_matrix(y_test, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Generator-Level Breakdown
    gen_breakdown = {}
    for gen in set(generators):
        mask = (generators == gen)
        if np.sum(mask) > 0:
            gen_acc = float(accuracy_score(y_test[mask], preds[mask]))
            gen_breakdown[gen] = {
                "sample_count": int(np.sum(mask)),
                "accuracy": gen_acc,
                "mean_spoof_prob": float(np.mean(probs[mask])),
            }

    print(f"{Colors.BOLD}{Colors.GREEN}► BENCHMARK PERFORMANCE METRICS:{Colors.RESET}")
    print(f" • Accuracy                 : {Colors.BOLD}{acc*100.0:.2f}%{Colors.RESET}")
    print(f" • Balanced Accuracy        : {b_acc*100.0:.2f}%")
    print(f" • Precision                : {prec*100.0:.2f}%")
    print(f" • Recall (Spoof Detection) : {rec*100.0:.2f}%")
    print(f" • F1-Score                 : {f1*100.0:.2f}%")
    print(f" • ROC-AUC                  : {auc:.4f}")
    print(f" • Equal Error Rate (EER)   : {Colors.BOLD}{eer*100.0:.2f}%{Colors.RESET} (EER Threshold: {eer_thresh:.3f})")

    print(f"\n{Colors.BOLD}► CONFUSION MATRIX:{Colors.RESET}")
    print(f"               Predicted Human   Predicted Spoof")
    print(f" Actual Human  :     {tn:<15}     {fp} (False Positives)")
    print(f" Actual Spoof  :     {fn:<15}     {tp} (True Positives)")

    print(f"\n{Colors.BOLD}► SYNTHESIZER / GENERATOR BREAKDOWN:{Colors.RESET}")
    for gen, stats in sorted(gen_breakdown.items(), key=lambda x: x[0]):
        print(f" • {gen:<18}: {stats['accuracy']*100.0:>5.1f}% accuracy ({stats['sample_count']} samples, Avg Prob: {stats['mean_spoof_prob']:.3f})")

    # Plot & Save Confusion Matrix
    plot_and_save_confusion_matrix(cm, output_path=output_plot)

    # Save JSON Report
    if output_report is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        output_report = os.path.join(REPORTS_DIR, f"eval_report_{split_name}.json")

    report = {
        "split": split_name,
        "total_samples": len(X_test),
        "threshold": threshold,
        "accuracy": acc,
        "balanced_accuracy": b_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "equal_error_rate": eer,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "generator_breakdown": gen_breakdown,
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f" • JSON Diagnostic Report Saved : {output_report}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}\n")
    return report


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Step 3: Model Evaluation Tool")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--model", type=str, default="models/voice_detector.joblib", help="Path to model .joblib/.pkl")
    parser.add_argument("--metadata", type=str, default=MODEL_METADATA_PATH, help="Path to metadata JSON")
    parser.add_argument("--split", type=str, default="test", help="Split name ('test', 'validation', 'train')")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold (optional)")
    parser.add_argument("--output-plot", type=str, default="reports/confusion_matrix.png", help="Path for confusion matrix PNG")
    parser.add_argument("--output-report", type=str, default=None, help="Path for evaluation JSON report")

    args = parser.parse_args()
    evaluate_test_set(
        manifest_path=args.manifest,
        model_path=args.model,
        metadata_path=args.metadata,
        split_name=args.split,
        threshold=args.threshold,
        output_plot=args.output_plot,
        output_report=args.output_report,
    )


if __name__ == "__main__":
    main()
