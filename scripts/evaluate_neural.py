"""
VoiceShield Neural Step 3: Standalone Deep Learning Evaluation & Benchmark Tool.
Loads models/voiceshield_neural_best.pt, evaluates across held-out splits, computes
EER, ROC-AUC, per-synthesizer metrics, and renders Confusion Matrix / ROC curves.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MANIFEST_PATH
from src.dataset_loader import TARGET_DURATION_SEC, TARGET_SAMPLE_RATE, VoiceShieldDataset, parse_manifest
from src.neural_model import VoiceShieldNeuralDetector


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """Computes Equal Error Rate (EER) and decision threshold where FPR == FNR."""
    if len(set(y_true)) < 2:
        return 0.0, 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.5
    return eer, eer_thresh


def evaluate_neural_checkpoint(
    checkpoint_path: str = "models/voiceshield_neural_best.pt",
    manifest_path: str = MANIFEST_PATH,
    output_report_json: str = "reports/neural_evaluation.json",
    output_plot_png: str = "reports/neural_confusion_matrix.png",
    device_name: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluates saved neural model checkpoint on test/validation samples.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Neural checkpoint not found at: {checkpoint_path}")

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    print(f"\n[*] Loading Neural Checkpoint: {checkpoint_path} on {device}...")

    # 1. Instantiate & Load Weights with auto-detected backbone
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    backbone_name = "lightweight" if any("conv1" in k for k in state.keys()) else "facebook/wav2vec2-base"
    model = VoiceShieldNeuralDetector(backbone_name=backbone_name, device=device)
    model.load_state_dict(state)
    model.eval()

    # 2. Ingest Manifest & Filter Test Partition
    records = parse_manifest(manifest_path)
    test_records = [r for r in records if r.get("split") in ["test", "val", "validation"]]
    if len(test_records) == 0:
        test_records = records  # Full benchmark fallback

    print(f"[*] Evaluating on {len(test_records)} samples...")

    test_dataset = VoiceShieldDataset(
        records=test_records,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_sec=TARGET_DURATION_SEC,
        is_train=False,
        augmenter=None,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    # 3. Batch Inference
    y_true_list = []
    y_prob_list = []
    file_results = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for audio, targets, meta in test_loader:
            audio = audio.to(device)
            logits = model(audio)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            targets_np = targets.numpy().flatten()

            y_true_list.extend(targets_np)
            y_prob_list.extend(probs)

            for i in range(len(probs)):
                file_results.append({
                    "file_path": meta["file_path"][i],
                    "speaker_id": meta["speaker_id"][i],
                    "generator_type": meta["generator_type"][i],
                    "ground_truth": int(targets_np[i]),
                    "spoof_prob": float(probs[i]),
                })

    eval_duration = time.perf_counter() - start_time
    y_true = np.array(y_true_list, dtype=int)
    y_prob = np.array(y_prob_list, dtype=float)

    # 4. Global Metrics Computation
    eer, opt_thresh = compute_eer(y_true, y_prob)
    y_pred = (y_prob >= opt_thresh).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else acc
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # 5. Per-Generator Breakdown
    gen_breakdown: Dict[str, Dict[str, Any]] = {}
    df = pd.DataFrame(file_results)
    for gen, group in df.groupby("generator_type"):
        g_true = group["ground_truth"].values
        g_prob = group["spoof_prob"].values
        g_pred = (g_prob >= opt_thresh).astype(int)
        g_acc = float(accuracy_score(g_true, g_pred))
        gen_breakdown[str(gen)] = {
            "sample_count": len(group),
            "accuracy": round(g_acc * 100, 2),
            "mean_spoof_prob": round(float(np.mean(g_prob)), 4),
        }

    # 6. Render Confusion Matrix & ROC Curve Visualization
    os.makedirs(os.path.dirname(output_plot_png), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Confusion Matrix Heatmap
    cax = ax1.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="xx-large", weight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=11)
    ax1.set_ylabel("True Label", fontsize=11)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Bona Fide (0)", "Spoof (1)"])
    ax1.set_yticklabels(["Bona Fide (0)", "Spoof (1)"])
    ax1.set_title(f"Confusion Matrix (Thresh: {opt_thresh:.2f})", fontsize=12, weight="bold")
    fig.colorbar(cax, ax=ax1)

    # ROC Curve
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax2.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc:.4f})")
        ax2.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
        ax2.scatter([eer], [1 - eer], color="#dc2626", zorder=5, label=f"EER Point ({eer*100:.1f}%)")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_title("ROC-AUC Curve", fontsize=12, weight="bold")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot_png, dpi=200)
    plt.close()

    # 7. Print Terminal Summary
    print("\n" + "=" * 70)
    print("        VOICESHIELD NEURAL MODEL BENCHMARK RESULTS")
    print("=" * 70)
    print(f" • Accuracy              : {acc*100.0:.2f}%")
    print(f" • Balanced Accuracy     : {bal_acc*100.0:.2f}%")
    print(f" • ROC-AUC               : {auc:.4f}")
    print(f" • Equal Error Rate (EER): {eer*100.0:.2f}% (Threshold: {opt_thresh:.3f})")
    print(f" • Precision             : {prec*100.0:.1f}%")
    print(f" • Recall                : {rec*100.0:.1f}%")
    print(f" • Macro F1-Score        : {f1:.4f}")
    print(f" • Confusion Matrix      : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("\nPer-Generator Accuracy Breakdown:")
    for gen, res in gen_breakdown.items():
        print(f"  - {gen:<20}: {res['accuracy']:>6.1f}% ({res['sample_count']:02d} samples, Mean Prob: {res['mean_spoof_prob']:.3f})")
    print(f"\n[OK] Plot saved to  : {output_plot_png}")

    # 8. Persist JSON Report
    report = {
        "checkpoint": checkpoint_path,
        "sample_count": len(test_records),
        "evaluation_time_sec": round(eval_duration, 3),
        "overall_metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "roc_auc": round(auc, 4),
            "eer": round(eer, 4),
            "optimal_threshold": round(opt_thresh, 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "per_generator_breakdown": gen_breakdown,
        "per_file_results": file_results,
    }

    os.makedirs(os.path.dirname(output_report_json), exist_ok=True)
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Report saved to: {output_report_json}")
    return report


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Neural Model Benchmark CLI")
    parser.add_argument("--checkpoint", type=str, default="models/voiceshield_neural_best.pt", help="Path to .pt weights")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.csv")
    parser.add_argument("--report", type=str, default="reports/neural_evaluation.json", help="Output JSON report")
    parser.add_argument("--plot", type=str, default="reports/neural_confusion_matrix.png", help="Output PNG plot")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()
    evaluate_neural_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_report_json=args.report,
        output_plot_png=args.plot,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
