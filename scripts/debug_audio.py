#!/usr/bin/env python
"""
VoiceShield Standalone Audio ML Debugging & Forensic Diagnostic Tool.
Investigates root causes for false positives (human misclassified as AI)
and false negatives (AI misclassified as human).

Usage:
    python scripts/debug_audio.py --file data/human/human_01.wav --true-label human
    python scripts/debug_audio.py --file data/ai_voice/elevenlabs_01.wav --true-label ai
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import joblib
import librosa
import numpy as np
import soundfile as sf

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.audio_io import load_audio_from_file
from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SAMPLE_RATE,
)
from src.features import extract_features_from_audio, get_feature_names
from src.preprocessing import preprocess_audio


import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =============================================================================
# Helper: Terminal Formatting
# =============================================================================
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_banner(title: str, char: str = "=") -> None:
    print(f"\n{Colors.BOLD}{Colors.CYAN}{char * 75}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{char * 75}{Colors.RESET}")


def print_section(title: str) -> None:
    print(f"\n{Colors.BOLD}{Colors.YELLOW}[>] {title}{Colors.RESET}")
    print(f"{Colors.DIM}{'-' * 60}{Colors.RESET}")


# =============================================================================
# 1. Audio Pipeline & DSP Health Checks
# =============================================================================
def inspect_audio_health(
    file_path: str,
) -> Tuple[np.ndarray, int, Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Loads raw audio, computes raw signal diagnostics, applies VoiceShield
    preprocessing, and evaluates if DSP transforms alter the signal drastically.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Read raw audio properties via soundfile
    info = sf.info(file_path)
    raw_audio, raw_sr = sf.read(file_path, dtype="float32")
    if raw_audio.ndim > 1:
        raw_audio_mono = np.mean(raw_audio, axis=1)
    else:
        raw_audio_mono = raw_audio

    raw_dur = float(len(raw_audio_mono) / raw_sr)
    raw_peak = float(np.max(np.abs(raw_audio_mono))) if len(raw_audio_mono) > 0 else 0.0
    raw_rms = float(np.sqrt(np.mean(raw_audio_mono**2))) if len(raw_audio_mono) > 0 else 0.0
    raw_clipped_pct = (
        float(np.sum(np.abs(raw_audio_mono) >= 0.999) / len(raw_audio_mono) * 100.0)
        if len(raw_audio_mono) > 0
        else 0.0
    )

    # Raw SNR estimation
    raw_snr = compute_rough_snr(raw_audio_mono, raw_sr)

    # Frame-level silence ratio (raw)
    frame_length = int(raw_sr * 0.032)
    hop_length = int(raw_sr * 0.016)
    if len(raw_audio_mono) >= frame_length:
        rms_frames = librosa.feature.rms(
            y=raw_audio_mono, frame_length=frame_length, hop_length=hop_length
        )[0]
        silence_thresh = max(1e-4, float(np.mean(rms_frames) * 0.1))
        raw_silence_ratio = float(np.sum(rms_frames < silence_thresh) / max(1, len(rms_frames)))
    else:
        raw_silence_ratio = 0.0

    raw_diag = {
        "channels": info.channels,
        "sample_rate": raw_sr,
        "duration_sec": raw_dur,
        "format": info.format,
        "subtype": info.subtype,
        "peak_amplitude": raw_peak,
        "rms_energy": raw_rms,
        "clipping_pct": raw_clipped_pct,
        "snr_db": raw_snr,
        "silence_ratio": raw_silence_ratio,
    }

    # Run standard VoiceShield Preprocessing
    clean_audio, effective_sr, proc_diag = preprocess_audio(
        raw_audio_mono, sample_rate=raw_sr, target_sr=SAMPLE_RATE
    )

    proc_dur = float(len(clean_audio) / effective_sr) if len(clean_audio) > 0 else 0.0
    proc_peak = float(np.max(np.abs(clean_audio))) if len(clean_audio) > 0 else 0.0
    proc_rms = float(np.sqrt(np.mean(clean_audio**2))) if len(clean_audio) > 0 else 0.0
    proc_clipped_pct = (
        float(np.sum(np.abs(clean_audio) >= 0.999) / len(clean_audio) * 100.0)
        if len(clean_audio) > 0
        else 0.0
    )
    proc_snr = compute_rough_snr(clean_audio, effective_sr)

    proc_summary = {
        "sample_rate": effective_sr,
        "duration_sec": proc_dur,
        "peak_amplitude": proc_peak,
        "rms_energy": proc_rms,
        "clipping_pct": proc_clipped_pct,
        "snr_db": proc_snr,
        "duration_delta_pct": float((proc_dur - raw_dur) / max(1e-5, raw_dur) * 100.0),
        "rms_delta_pct": float((proc_rms - raw_rms) / max(1e-5, raw_rms) * 100.0),
    }

    return raw_audio_mono, raw_sr, raw_diag, clean_audio, proc_summary


def compute_rough_snr(audio: np.ndarray, sr: int) -> float:
    """Computes estimated Signal-to-Noise Ratio (dB) via energy percentile floor."""
    if len(audio) < int(sr * 0.1) or not np.any(audio):
        return 0.0
    frame_len = min(512, len(audio))
    hop_len = max(64, frame_len // 2)
    try:
        rms_frames = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]
        if len(rms_frames) == 0:
            return 0.0
        signal_pwr = np.percentile(rms_frames, 90) ** 2 + 1e-9
        noise_pwr = np.percentile(rms_frames, 10) ** 2 + 1e-9
        snr = float(10.0 * np.log10(signal_pwr / noise_pwr))
        return float(np.clip(snr, -10.0, 60.0))
    except Exception:
        return 0.0


# =============================================================================
# 2. Out-Of-Distribution (OOD) & Feature Anomaly Inspection
# =============================================================================
def get_feature_names_for_dim(dim: int) -> List[str]:
    """Dynamically returns feature names matching vector dimension."""
    if dim == 42:
        return (
            [f"mfcc_mean_{i+1}" for i in range(20)]
            + [f"mfcc_std_{i+1}" for i in range(20)]
            + ["rms_energy", "zero_crossing_rate"]
        )
    elif dim == 55:
        return get_feature_names()
    else:
        return [f"feature_{i+1}" for i in range(dim)]


def compute_feature_anomalies(
    features: np.ndarray,
    model_pipeline: Any,
    metadata: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Calculates Z-scores against training distribution from StandardScaler/RobustScaler
    or metadata statistics, flagging the top outlier features.
    """
    dim = len(features)
    feature_names = get_feature_names_for_dim(dim)

    train_mean = None
    train_std = None

    # 1. Try to extract from pipeline scaler
    scaler = (
        model_pipeline.named_steps.get("scaler")
        if hasattr(model_pipeline, "named_steps")
        else None
    )

    if scaler is not None:
        if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
            train_mean = np.array(scaler.mean_, dtype=np.float32)
            train_std = np.array(scaler.scale_, dtype=np.float32)
        elif hasattr(scaler, "center_") and hasattr(scaler, "scale_"):
            train_mean = np.array(scaler.center_, dtype=np.float32)
            train_std = np.array(scaler.scale_, dtype=np.float32)

    # 2. Fallback to metadata
    if (train_mean is None or len(train_mean) != dim) and metadata:
        if "train_feature_mean" in metadata and "train_feature_std" in metadata:
            m = np.array(metadata["train_feature_mean"], dtype=np.float32)
            s = np.array(metadata["train_feature_std"], dtype=np.float32)
            if len(m) == dim and len(s) == dim:
                train_mean = m
                train_std = s

    # 3. Default fallback if nothing exists
    if train_mean is None or len(train_mean) != dim:
        train_mean = np.zeros(dim, dtype=np.float32)
        train_std = np.ones(dim, dtype=np.float32)

    train_std = np.where(train_std < 1e-6, 1.0, train_std)
    z_scores = (features - train_mean) / train_std

    # Build detailed outlier feature list
    anomalies = []
    for i in range(dim):
        anomalies.append({
            "index": i,
            "name": feature_names[i],
            "value": float(features[i]),
            "train_mean": float(train_mean[i]),
            "train_std": float(train_std[i]),
            "z_score": float(z_scores[i]),
            "abs_z": float(np.abs(z_scores[i])),
        })

    # Sort descending by absolute Z-score
    anomalies.sort(key=lambda x: x["abs_z"], reverse=True)
    return z_scores, anomalies


# =============================================================================
# 3. Model Inference, Tree Consensus & Feature Group Breakdown
# =============================================================================
def evaluate_model_decisions(
    features: np.ndarray,
    model_pipeline: Any,
    threshold: float,
) -> Dict[str, Any]:
    """
    Evaluates model probabilities, tree-level consensus, and feature group contributions.
    """
    x_input = features.reshape(1, -1)

    # 1. Overall Probability
    try:
        raw_probs = model_pipeline.predict_proba(x_input)[0]
        p_human = float(raw_probs[0])
        p_spoof = float(raw_probs[1])
    except Exception as e:
        p_human, p_spoof = 0.5, 0.5

    pred_label = "ai" if p_spoof >= threshold else "human"

    # 2. Inspect Ensemble / Tree Voting Consensus
    classifier = (
        model_pipeline.named_steps.get("classifier")
        if hasattr(model_pipeline, "named_steps")
        else model_pipeline
    )
    scaler = (
        model_pipeline.named_steps.get("scaler")
        if hasattr(model_pipeline, "named_steps")
        else None
    )

    x_scaled = scaler.transform(x_input) if scaler is not None else x_input

    tree_consensus = None
    clf_type = type(classifier).__name__

    # Case A: RandomForest / ExtraTrees
    if hasattr(classifier, "estimators_"):
        trees = classifier.estimators_
        total_trees = len(trees)
        spoof_votes = 0
        human_votes = 0
        for t in trees:
            t_pred = t.predict(x_scaled)[0]
            if t_pred == 1:
                spoof_votes += 1
            else:
                human_votes += 1

        tree_consensus = {
            "type": clf_type,
            "total_estimators": total_trees,
            "spoof_votes": spoof_votes,
            "human_votes": human_votes,
            "spoof_vote_pct": round(spoof_votes / total_trees * 100.0, 1),
            "human_vote_pct": round(human_votes / total_trees * 100.0, 1),
        }

    # Case B: XGBoost / LightGBM
    elif hasattr(classifier, "feature_importances_"):
        tree_consensus = {
            "type": clf_type,
            "total_estimators": getattr(classifier, "n_estimators", "N/A"),
            "max_depth": getattr(classifier, "max_depth", "N/A"),
            "learning_rate": getattr(classifier, "learning_rate", "N/A"),
        }

    # 3. Feature Group Contribution Breakdown
    feature_names = get_feature_names_for_dim(len(features))
    groups = {
        "MFCC Means (Timbre / Vocal Tract)": [
            i for i, n in enumerate(feature_names) if "mfcc_mean" in n
        ],
        "MFCC Stds (Phonetic Modulation)": [
            i for i, n in enumerate(feature_names) if "mfcc_std" in n
        ],
        "Pitch & Harmonicity (F0 / Jitter / HNR)": [
            i
            for i, n in enumerate(feature_names)
            if any(k in n for k in ["f0", "pitch", "jitter", "shimmer", "hnr"])
        ],
        "Spectral Dynamics (Flatness / Centroid / Flux)": [
            i
            for i, n in enumerate(feature_names)
            if any(k in n for k in ["spectral", "rolloff", "bandwidth", "contrast"])
        ],
        "Energy & Voice Activity (RMS / ZCR / Voicing)": [
            i
            for i, n in enumerate(feature_names)
            if any(k in n for k in ["rms", "zcr", "voiced", "pause"])
        ],
    }

    importances = getattr(classifier, "feature_importances_", None)
    if importances is None or len(importances) != len(features):
        importances = np.ones(len(features), dtype=np.float32) / len(features)

    group_analysis = []
    for g_name, indices in groups.items():
        if not indices:
            continue
        g_imp = float(np.sum(importances[indices]))
        g_vals = features[indices]
        g_names = [feature_names[i] for i in indices]
        group_analysis.append({
            "group": g_name,
            "feature_count": len(indices),
            "total_importance": g_imp,
            "mean_val": float(np.mean(g_vals)),
            "std_val": float(np.std(g_vals)),
        })

    group_analysis.sort(key=lambda x: x["total_importance"], reverse=True)

    return {
        "p_human": p_human,
        "p_spoof": p_spoof,
        "pred_label": pred_label,
        "threshold": threshold,
        "classifier_type": clf_type,
        "tree_consensus": tree_consensus,
        "group_analysis": group_analysis,
    }


# =============================================================================
# 4. Root Cause Generator
# =============================================================================
def generate_root_cause_diagnosis(
    true_label: str,
    pred_label: str,
    raw_diag: Dict[str, Any],
    proc_summary: Dict[str, Any],
    anomalies: List[Dict[str, Any]],
    model_eval: Dict[str, Any],
) -> List[str]:
    """Generates precise diagnostic conclusions and actionable debugging insights."""
    conclusions = []
    is_misclassified = true_label.lower() != pred_label.lower()

    if is_misclassified:
        error_type = (
            "FALSE POSITIVE (Human Voice flagged as AI)"
            if true_label == "human"
            else "FALSE NEGATIVE (AI Voice flagged as Human)"
        )
        conclusions.append(f"{Colors.RED}{Colors.BOLD}MISCLASSIFICATION DETECTED: {error_type}{Colors.RESET}")
    else:
        conclusions.append(f"{Colors.GREEN}{Colors.BOLD}CORRECT CLASSIFICATION: Audio accurately predicted as {pred_label.upper()}{Colors.RESET}")

    # Check 1: Audio duration
    if raw_diag["duration_sec"] < 1.0:
        conclusions.append(
            f"⚠️ Audio duration is very short ({raw_diag['duration_sec']:.2f}s). "
            f"Models typically require >= 1.5s for stable spectral variance estimation."
        )

    # Check 2: SNR and Background Noise
    if raw_diag["snr_db"] < 12.0:
        conclusions.append(
            f"⚠️ High background noise floor (Estimated SNR: {raw_diag['snr_db']:.1f} dB < 12 dB). "
            f"Noise corrupts high-frequency MFCC coefficients, mimicking synthetic vocoder artifacts."
        )

    # Check 3: Digital Clipping / Saturation
    if raw_diag["clipping_pct"] > 0.5:
        conclusions.append(
            f"⚠️ Significant digital clipping detected ({raw_diag['clipping_pct']:.2f}% saturated samples). "
            f"Clipping introduces harmonic distortion mimicking vocoder synthesis."
        )

    # Check 4: Silence Ratio
    if raw_diag["silence_ratio"] > 0.40:
        conclusions.append(
            f"⚠️ High silence ratio ({raw_diag['silence_ratio']*100.0:.1f}%). "
            f"Extensive dead air skews energy distribution."
        )

    # Check 5: Non-standard Sample Rate
    if raw_diag["sample_rate"] != 16000:
        conclusions.append(
            f"ℹ️ Original sample rate was {raw_diag['sample_rate']} Hz (resampled to 16,000 Hz). "
            f"If original audio was narrowband (e.g. 8kHz telephony), missing frequencies above 4kHz can distort MFCCs."
        )

    # Check 6: Severe Outlier Features (OOD)
    extreme_outliers = [a for a in anomalies if a["abs_z"] >= 3.0]
    if extreme_outliers:
        outlier_names = ", ".join([f"{a['name']} (Z={a['z_score']:+.1f})" for a in extreme_outliers[:3]])
        conclusions.append(
            f"⚠️ Out-Of-Distribution (OOD) anomaly: {len(extreme_outliers)} features deviate by > 3.0σ from training distribution! "
            f"Top outliers: [{outlier_names}]."
        )

    # Check 7: Decision Boundary Proximity
    margin = abs(model_eval["p_spoof"] - model_eval["threshold"])
    if margin < 0.12:
        conclusions.append(
            f"ℹ️ Near decision boundary: P(spoof)={model_eval['p_spoof']:.3f} is close to threshold={model_eval['threshold']:.3f} (Margin: {margin:.3f}). "
            f"Threshold tuning or temperature calibration can resolve borderline cases."
        )

    return conclusions


# =============================================================================
# Main Debug Runner
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield Audio ML Diagnostic & Debugging Tool"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Path to audio file to debug (WAV, MP3, M4A, FLAC, OGG)",
    )
    parser.add_argument(
        "--true-label",
        "-l",
        type=str,
        default="unknown",
        choices=["human", "ai", "bona_fide", "spoof", "unknown"],
        help="Ground truth label: 'human' / 'bona_fide' vs 'ai' / 'spoof'",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=MODEL_PATH,
        help=f"Path to trained model pipeline (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=MODEL_METADATA_PATH,
        help=f"Path to model metadata JSON (default: {MODEL_METADATA_PATH})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=None,
        help="Custom decision threshold for spoof detection (0.0 to 1.0)",
    )

    args = parser.parse_args()

    # Normalize true label
    true_label = args.true_label.lower()
    if true_label in ["bona_fide", "real"]:
        true_label = "human"
    elif true_label in ["spoof", "synthetic", "cloned"]:
        true_label = "ai"

    print_banner(f"VOICESHIELD AUDIO ML FORENSIC DIAGNOSTIC SUITE")
    print(f"{Colors.BOLD}Target Audio File  :{Colors.RESET} {args.file}")
    print(f"{Colors.BOLD}Ground Truth Label :{Colors.RESET} {true_label.upper()}")
    print(f"{Colors.BOLD}Model Path         :{Colors.RESET} {args.model}")

    # 1. Load Model & Metadata
    if not os.path.exists(args.model):
        print(f"{Colors.RED}Error: Model file '{args.model}' does not exist.{Colors.RESET}")
        sys.exit(1)

    try:
        model_pipeline = joblib.load(args.model)
    except Exception as e:
        print(f"{Colors.RED}Failed to load model pipeline: {e}{Colors.RESET}")
        sys.exit(1)

    metadata = None
    if os.path.exists(args.metadata):
        try:
            with open(args.metadata, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Threshold selection
    if args.threshold is not None:
        threshold = args.threshold
    elif metadata and "selected_threshold" in metadata:
        threshold = float(metadata["selected_threshold"])
    elif metadata and "optimal_decision_threshold" in metadata:
        threshold = float(metadata["optimal_decision_threshold"])
    else:
        threshold = DEFAULT_DECISION_THRESHOLD

    # 2. Inspect Audio Health & Preprocessing Transformations
    print_section("1. AUDIO PIPELINE & SIGNAL INTEGRITY HEALTH CHECK")
    try:
        raw_audio, raw_sr, raw_diag, clean_audio, proc_summary = inspect_audio_health(
            args.file
        )
    except Exception as e:
        print(f"{Colors.RED}Audio inspection failed: {e}{Colors.RESET}")
        sys.exit(1)

    print(
        f" • Original Format       : {raw_diag['format']} ({raw_diag['subtype']}), {raw_diag['channels']} Ch, {raw_diag['sample_rate']} Hz"
    )
    print(
        f" • Raw Audio Duration    : {raw_diag['duration_sec']:.2f} s | Preprocessed: {proc_summary['duration_sec']:.2f} s"
    )
    print(
        f" • Estimated Signal SNR  : {raw_diag['snr_db']:.1f} dB ({'Excellent' if raw_diag['snr_db'] >= 25 else 'Moderate' if raw_diag['snr_db'] >= 15 else 'Poor / Noisy'})"
    )
    print(
        f" • Digital Clipping      : {raw_diag['clipping_pct']:.2f}% samples near +/-1.0 ({'Clean' if raw_diag['clipping_pct'] < 0.1 else 'Severe Distortion'})"
    )
    print(
        f" • Silence Ratio         : {raw_diag['silence_ratio']*100.0:.1f}% frames silent"
    )
    print(
        f" • Peak Amplitude        : {raw_diag['peak_amplitude']:.3f} -> Normalised: {proc_summary['peak_amplitude']:.3f}"
    )
    print(
        f" • RMS Energy            : {raw_diag['rms_energy']:.4f} -> Delta: {proc_summary['rms_delta_pct']:+.1f}%"
    )

    # 3. Extract Features
    print_section("2. FEATURE EXTRACTION & OUT-OF-DISTRIBUTION (OOD) ANALYSIS")
    expected_dim = getattr(
        getattr(model_pipeline, "named_steps", {}).get("scaler"),
        "n_features_in_",
        metadata.get("feature_dimension", 229) if metadata else 229,
    )
    if expected_dim == 229:
        mode = "advanced"
    elif expected_dim == 55:
        mode = "step1"
    else:
        mode = "legacy"
    features = extract_features_from_audio(clean_audio, SAMPLE_RATE, mode=mode)

    print(f" • Feature Vector Dim    : {len(features)} elements (Expected: {expected_dim})")
    print(f" • NaN / Inf Sanitization: {'Passed (0 NaN/Inf)' if np.all(np.isfinite(features)) else 'Failed'}")

    z_scores, anomalies = compute_feature_anomalies(features, model_pipeline, metadata)
    max_z = float(np.max(np.abs(z_scores)))
    ood_flag = max_z > 3.0

    print(
        f" • OOD Status            : {Colors.RED if ood_flag else Colors.GREEN}{'OUT-OF-DISTRIBUTION ANOMALY' if ood_flag else 'IN-DISTRIBUTION NORMAL'}{Colors.RESET} (Max |Z| = {max_z:.2f}σ)"
    )

    print(f"\n {Colors.BOLD}Top 5 Most Deviated / Outlier Features (vs Training Mean):{Colors.RESET}")
    print(f" {'Idx':<4} | {'Feature Name':<28} | {'Observed':<10} | {'Train Mean ± Std':<20} | {'Z-Score':<10}")
    print(f" {'-'*4}-|-{'-'*28}-|-{'-'*10}-|-{'-'*20}-|-{'-'*10}")
    for a in anomalies[:5]:
        z_color = Colors.RED if a["abs_z"] >= 3.0 else (Colors.YELLOW if a["abs_z"] >= 2.0 else Colors.RESET)
        print(
            f" {a['index']:<4} | {a['name']:<28} | {a['value']:<10.4f} | {a['train_mean']:>8.3f} ± {a['train_std']:<8.3f} | {z_color}{a['z_score']:>+8.2f}σ{Colors.RESET}"
        )

    # 4. Model Prediction & Probability Breakdown
    print_section("3. MODEL PREDICTION & ENSEMBLE CONSENSUS")
    model_eval = evaluate_model_decisions(features, model_pipeline, threshold)

    pred_color = Colors.RED if model_eval["pred_label"] == "ai" else Colors.GREEN
    print(f" • Model Classification  : {pred_color}{Colors.BOLD}{model_eval['pred_label'].upper()}{Colors.RESET}")
    print(f" • Decision Threshold    : {model_eval['threshold']:.3f}")
    print(f" • P(Human / Bona Fide)  : {model_eval['p_human']*100.0:.2f}%")
    print(f" • P(AI / Spoof Clone)   : {model_eval['p_spoof']*100.0:.2f}%")

    if model_eval["tree_consensus"]:
        tc = model_eval["tree_consensus"]
        print(f" • Classifier Architecture: {tc['type']}")
        if "total_estimators" in tc and "spoof_votes" in tc:
            print(
                f" • Tree Voting Consensus : {tc['spoof_votes']}/{tc['total_estimators']} trees voted SPOOF ({tc['spoof_vote_pct']}%) | {tc['human_votes']}/{tc['total_estimators']} voted HUMAN ({tc['human_vote_pct']}%)"
            )

    print(f"\n {Colors.BOLD}Feature Group Importance Breakdown:{Colors.RESET}")
    for g in model_eval["group_analysis"]:
        bar_len = int(g["total_importance"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f" • {g['group']:<45} : {bar} {g['total_importance']*100.0:>5.1f}%")

    # 5. Root Cause Conclusions
    print_section("4. ROOT CAUSE ANALYSIS & DIAGNOSTIC CONCLUSIONS")
    conclusions = generate_root_cause_diagnosis(
        true_label,
        model_eval["pred_label"],
        raw_diag,
        proc_summary,
        anomalies,
        model_eval,
    )
    for c in conclusions:
        print(f" {c}")

    # 6. Manual Action Items
    print_section("5. 🛠️ MANUAL ACTION ITEMS FOR USER")
    print(
        f"""
 1. Threshold Retuning:
    If this audio is a legitimate human voice misclassified as AI (False Positive),
    try adjusting the decision threshold:
      python scripts/debug_audio.py --file "{args.file}" --true-label {true_label} --threshold 0.35

 2. Dataset Noise / Codec Augmentation:
    If high background noise (SNR < 15dB) or telephony compression threw off the MFCCs,
    run the dataset augmentation script to retrain with simulated telephony/babble noise:
      python scripts/augment_data.py --input-dir data/human --output-dir data/human_augmented

 3. Retrain Baseline Pipeline with Balanced Robust Scaler:
    To retrain and calibrate the model against new outliers:
      python scripts/train_model.py

 4. Run Full Comprehensive Diagnostic Check:
    Inspect all models and dataset splits systematically:
      python scripts/diagnose_model.py
"""
    )
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
