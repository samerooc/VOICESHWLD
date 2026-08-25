"""
VoiceShield Phase 8 — Forensic CLI Inspection & Accuracy Benchmark Harness.

Features:
  1. Single-File Deep Forensic Inspection (--file <path>):
     - Prints structured terminal report with signal diagnostics, biomechanical jitter/HNR,
       LPC residual physics, neural consensus probabilities, risk band, and latency.
  2. Batch Dataset Evaluation & Accuracy Benchmark (--dir <path>):
     - Traverses folders containing real/fake audio samples.
     - Computes Accuracy, Precision, Recall, F1-Score, and Equal Error Rate (EER).
     - Exports comprehensive forensic JSON and CSV audit reports.

Usage:
    # Single file inspection
    python scripts/benchmark_accuracy.py --file data/demo/synthetic_sample.wav

    # Batch directory evaluation
    python scripts/benchmark_accuracy.py --dir data/test_samples/ --output-json reports/benchmark_results.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import SAMPLE_RATE, decode_and_sanitize_audio
from src.neural_engine import ProductionNeuralDetector

# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BLUE = "\033[94m"
CLR_GRAY = "\033[90m"


def compute_binary_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.50,
) -> Dict[str, float]:
    """
    Calculate classification metrics and Equal Error Rate (EER).
    y_true: 1 = spoof / AI clone, 0 = authentic human
    """
    if len(y_true) == 0:
        return {}

    y_t = np.array(y_true)
    y_s = np.array(y_scores)
    y_pred = (y_s >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_t == 1)))
    fp = int(np.sum((y_pred == 1) & (y_t == 0)))
    tn = int(np.sum((y_pred == 0) & (y_t == 0)))
    fn = int(np.sum((y_pred == 0) & (y_t == 1)))

    acc = float((tp + tn) / max(1, len(y_t)))
    prec = float(tp / max(1, tp + fp))
    rec = float(tp / max(1, tp + fn))
    f1 = float(2 * prec * rec / max(1e-9, prec + rec))

    # Calculate Equal Error Rate (EER)
    # FPR and FNR across sweep thresholds
    thresholds = np.linspace(0.0, 1.0, 200)
    fpr_list: List[float] = []
    fnr_list: List[float] = []

    for th in thresholds:
        pred_th = (y_s >= th).astype(int)
        th_fp = np.sum((pred_th == 1) & (y_t == 0))
        th_fn = np.sum((pred_th == 0) & (y_t == 1))
        n_neg = np.sum(y_t == 0)
        n_pos = np.sum(y_t == 1)
        fpr = float(th_fp / max(1, n_neg))
        fnr = float(th_fn / max(1, n_pos))
        fpr_list.append(fpr)
        fnr_list.append(fnr)

    fpr_arr = np.array(fpr_list)
    fnr_arr = np.array(fnr_list)
    eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr)))
    eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "equal_error_rate": round(eer, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


# ---------------------------------------------------------------------------
# Single-File Inspection Mode
# ---------------------------------------------------------------------------

def inspect_single_file(file_path: str, detector: ProductionNeuralDetector) -> Dict[str, Any]:
    """Execute deep forensic analysis and print CLI summary table."""
    if not os.path.exists(file_path):
        print(f"{CLR_RED}Error: File not found: {file_path}{CLR_RESET}")
        sys.exit(1)

    print(f"\n{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}      VOICESHIELD DEEP AUDIO FORENSIC INSPECTION REPORT              {CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"Target File: {CLR_BOLD}{os.path.abspath(file_path)}{CLR_RESET}")

    with open(file_path, "rb") as fh:
        raw_bytes = fh.read()

    t0 = time.perf_counter()
    pred = detector.predict(raw_bytes)
    latency_ms = (time.perf_counter() - t0) * 1000

    diag = pred.get("diagnostics", {})
    fb = pred.get("forensic_breakdown", {})
    score = pred.get("risk_score", 50)
    prob = pred.get("spoof_probability", 0.50)

    # Colorize based on score
    if score <= 25:
        tag_color = CLR_GREEN
        tag_label = "● AUTHENTIC HUMAN VOICE (LOW RISK)"
    elif score <= 60:
        tag_color = CLR_YELLOW
        tag_label = "▲ SUSPICIOUS / REVIEW REQUIRED"
    else:
        tag_color = CLR_RED
        tag_label = "■ HIGH RISK (LIKELY AI / CLONED VOICE)"

    print("-" * 71)
    print(f"Verdict: {tag_color}{CLR_BOLD}{tag_label}{CLR_RESET}")
    print(f"Risk Score: {tag_color}{CLR_BOLD}{score}/100{CLR_RESET} | Spoof Probability: {tag_color}{prob*100:.1f}%{CLR_RESET}")
    print(f"Risk Description: {pred.get('risk_description', '')}")
    print("-" * 71)

    print(f"{CLR_BOLD}1. Signal Quality & Acoustics:{CLR_RESET}")
    print(f"   • Duration: {diag.get('duration_sec', 0.0):.2f}s (Voiced: {diag.get('voiced_sec', 0.0):.2f}s, Ratio: {diag.get('voiced_ratio', 0.0)*100:.1f}%)")
    print(f"   • SNR: {diag.get('snr_db', 0.0):.1f} dB | Mode: {fb.get('snr_weight_mode', 'clean').upper()} | Clipped: {diag.get('is_clipped', False)}")

    print(f"\n{CLR_BOLD}2. Biomechanical Vocal Tract Diagnostics:{CLR_RESET}")
    print(f"   • Local Jitter (PPQ): {fb.get('jitter_local', 0.0)*100:.4f}%  (Baseline human: 0.5% - 1.5%)")
    print(f"   • Local Shimmer (APQ): {fb.get('shimmer_local', 0.0)*100:.4f}%")
    print(f"   • Harmonics-to-Noise (HNR): {fb.get('hnr_db', 0.0):.1f} dB")

    print(f"\n{CLR_BOLD}3. Physical Vocoder & LPC Artifacts:{CLR_RESET}")
    print(f"   • LPC Residual Kurtosis: {fb.get('lpc_kurtosis', 3.0):.2f} (Gaussian baseline: 3.00)")
    print(f"   • Phase Entropy (>4kHz): {fb.get('phase_entropy', 0.5):.4f}")
    print(f"   • LFCC Anomaly Score: {fb.get('lfcc_spoof_prob', 0.5):.4f} (Variance: {fb.get('lfcc_variance', 0.0):.2f})")

    print(f"\n{CLR_BOLD}4. Neural Deep Learning Assessment:{CLR_RESET}")
    print(f"   • Transformer Spoof Prob: {fb.get('transformer_spoof_prob', 0.5)*100:.1f}%")
    print(f"   • Active Model Backbone: {fb.get('active_model_id', 'Native Backbone')}")
    print(f"   • Execution Latency: {latency_ms:.1f} ms | Real-Time Compliant: {latency_ms < 250.0}")
    print("=" * 71 + "\n")

    return pred


# ---------------------------------------------------------------------------
# Batch Evaluation & Benchmark Mode
# ---------------------------------------------------------------------------

def run_batch_benchmark(
    directory_path: str,
    detector: ProductionNeuralDetector,
    ground_truth_label: Optional[str] = None,
    output_json_path: Optional[str] = None,
    output_csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate all audio samples in a directory and calculate accuracy metrics."""
    if not os.path.exists(directory_path):
        print(f"{CLR_RED}Error: Directory not found: {directory_path}{CLR_RESET}")
        sys.exit(1)

    # Collect audio files
    extensions = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.webm")
    audio_files: List[str] = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(directory_path, "**", ext), recursive=True))

    if len(audio_files) == 0:
        print(f"{CLR_YELLOW}No audio files found in: {directory_path}{CLR_RESET}")
        return {}

    print(f"\n{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}         VOICESHIELD BATCH DATASET FORENSIC BENCHMARK               {CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"Evaluation Target: {os.path.abspath(directory_path)}")
    print(f"Total Samples Found: {len(audio_files)}")
    print("-" * 71)

    y_true: List[int] = []
    y_scores: List[float] = []
    latencies: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    for idx, fpath in enumerate(audio_files, start=1):
        fname = os.path.basename(fpath).lower()

        # Determine ground truth if available
        gt = None
        if ground_truth_label:
            gt = 1 if ground_truth_label.lower() in ("fake", "spoof", "ai", "synth") else 0
        elif any(k in fname for k in ("fake", "spoof", "synth", "clone", "tts")):
            gt = 1
        elif any(k in fname for k in ("real", "human", "bonafide", "authentic", "original")):
            gt = 0

        try:
            with open(fpath, "rb") as fh:
                raw_bytes = fh.read()

            t0 = time.perf_counter()
            pred = detector.predict(raw_bytes)
            lat = (time.perf_counter() - t0) * 1000
            latencies.append(lat)

            score = pred.get("risk_score", 50)
            prob = pred.get("spoof_probability", 0.50)

            record = {
                "file": fpath,
                "filename": os.path.basename(fpath),
                "risk_score": score,
                "spoof_probability": round(prob, 4),
                "risk_band": pred.get("risk_band", "Review Required"),
                "ground_truth": gt,
                "latency_ms": round(lat, 2),
            }
            sample_records.append(record)

            if gt is not None:
                y_true.append(gt)
                y_scores.append(prob)

            verdict_tag = f"{CLR_GREEN}HUMAN{CLR_RESET}" if score <= 25 else (f"{CLR_YELLOW}REVIEW{CLR_RESET}" if score <= 60 else f"{CLR_RED}CLONE{CLR_RESET}")
            print(f"  [{idx:>3}/{len(audio_files)}] {os.path.basename(fpath):<32} -> Score: {score:>3}/100 [{verdict_tag}] ({lat:.1f}ms)")

        except Exception as exc:
            print(f"  [{idx:>3}/{len(audio_files)}] {os.path.basename(fpath):<32} -> {CLR_RED}ERROR: {exc}{CLR_RESET}")

    # Compute aggregate metrics
    metrics = {}
    if len(y_true) > 0:
        metrics = compute_binary_metrics(y_true, y_scores)

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    print("\n" + "=" * 71)
    print(f"{CLR_BOLD}BENCHMARK AGGREGATE SUMMARY:{CLR_RESET}")
    print(f"Total Evaluated: {len(sample_records)} | Mean Latency: {mean_lat:.1f}ms | P95 Latency: {p95_lat:.1f}ms")

    if metrics:
        print(f"Accuracy:  {CLR_GREEN}{metrics['accuracy']*100:.2f}%{CLR_RESET}")
        print(f"Precision: {metrics['precision']*100:.2f}% | Recall: {metrics['recall']*100:.2f}% | F1: {metrics['f1_score']:.4f}")
        print(f"Equal Error Rate (EER): {CLR_CYAN}{metrics['equal_error_rate']*100:.2f}%{CLR_RESET}")
        print(f"Confusion Matrix: TP={metrics['true_positives']}, FP={metrics['false_positives']}, TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
    print("=" * 71 + "\n")

    summary_payload = {
        "timestamp": time.time(),
        "total_samples": len(sample_records),
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "metrics": metrics,
        "samples": sample_records,
    }

    # Optional JSON export
    if output_json_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as fh:
            json.dump(summary_payload, fh, indent=2)
        print(f"Exported JSON benchmark to: {output_json_path}")

    # Optional CSV export
    if output_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "filename", "risk_score", "spoof_probability", "risk_band", "ground_truth", "latency_ms"])
            writer.writeheader()
            writer.writerows(sample_records)
        print(f"Exported CSV benchmark to: {output_csv_path}")

    return summary_payload


# ---------------------------------------------------------------------------
# CLI Argument Parser Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield Forensic CLI Inspection & Accuracy Benchmark Harness"
    )
    parser.add_argument("--file", type=str, default=None, help="Path to single audio file for deep inspection")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory for batch evaluation")
    parser.add_argument("--ground-truth", type=str, default=None, choices=["real", "fake", "human", "spoof"], help="Ground truth label")
    parser.add_argument("--output-json", type=str, default=None, help="Path to export JSON benchmark results")
    parser.add_argument("--output-csv", type=str, default=None, help="Path to export CSV benchmark results")

    args = parser.parse_args()

    if not args.file and not args.dir:
        # Default self-test with synthesized sample
        print(f"{CLR_YELLOW}No --file or --dir argument passed. Generating in-memory test audio for demonstration...{CLR_RESET}")
        detector = ProductionNeuralDetector(load_hf=False)
        t = np.linspace(0, 1.5, 24000, endpoint=False)
        synthetic_audio = (0.35 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, synthetic_audio, 16000, format="WAV", subtype="PCM_16")
        demo_file = "demo_synthetic_sample.wav"
        with open(demo_file, "wb") as f:
            f.write(buf.getvalue())

        inspect_single_file(demo_file, detector)
        if os.path.exists(demo_file):
            os.remove(demo_file)
        return

    detector = ProductionNeuralDetector(load_hf=False)

    if args.file:
        inspect_single_file(args.file, detector)
    elif args.dir:
        run_batch_benchmark(
            args.dir,
            detector,
            ground_truth_label=args.ground_truth,
            output_json_path=args.output_json,
            output_csv_path=args.output_csv,
        )


if __name__ == "__main__":
    main()
