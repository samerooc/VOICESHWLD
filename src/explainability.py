"""
VoiceShield Explainability & Signal Diagnostics Module.
Provides non-causal acoustic explanations, prosodic signal diagnostics,
uncertainty evaluation, out-of-distribution detection, and feature importance analysis.
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple
import librosa
import numpy as np
import pandas as pd

from src.config import N_MFCC, REPORTS_DIR, SAMPLE_RATE, TOTAL_FEATURES

EXPLAINABILITY_DISCLAIMER: str = (
    "This explanation describes model signals, not proof of identity. "
    "Acoustic feature contributions indicate correlation with synthetic training patterns "
    "and must not be interpreted as definitive causal proof."
)


def compute_signal_diagnostics(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Computes per-file signal diagnostics: duration, sample rate, silence ratio,
    clipping ratio, pitch variation, energy variation, and spectral summary.
    """
    if audio is None or len(audio) == 0:
        return {
            "duration": 0.0,
            "duration_seconds": 0.0,
            "sample_rate": sample_rate,
            "silence_ratio": 1.0,
            "clipping_ratio": 0.0,
            "pitch_mean_hz": None,
            "pitch_std_hz": None,
            "pitch_variation": "Missing / Unvoiced",
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "energy_variation": "Flat / Zero Energy",
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_variation": "Low",
            "spectral_summary": "Empty or zero-amplitude audio",
            "audio_quality": "Empty / Invalid",
        }

    duration = float(len(audio) / sample_rate)

    # 1. Clipping Ratio (Proportion of saturated samples near +/- 1.0)
    clipping_samples = np.sum(np.abs(audio) >= 0.999)
    clipping_ratio = float(clipping_samples / len(audio)) if len(audio) > 0 else 0.0

    # 2. Frame-level RMS Energy & Silence Ratio
    frame_length = 512
    hop_length = 256
    rms_frames = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    energy_mean = float(np.mean(rms_frames)) if len(rms_frames) > 0 else 0.0
    energy_std = float(np.std(rms_frames)) if len(rms_frames) > 0 else 0.0

    # Silence threshold: frames with RMS < 10% of mean energy or < 0.001
    silence_thresh = max(0.001, energy_mean * 0.1)
    silent_frames = np.sum(rms_frames < silence_thresh)
    silence_ratio = float(silent_frames / len(rms_frames)) if len(rms_frames) > 0 else 0.0

    if energy_std < 0.005:
        energy_var_desc = "Low Dynamic Range (Compressed / Flat Volume)"
    elif energy_std > 0.05:
        energy_var_desc = "High Dynamic Range (Natural Accentuation)"
    else:
        energy_var_desc = "Moderate Energy Variance"

    # 3. Pitch (Fundamental Frequency F0) Estimation using librosa.yin
    pitch_mean = None
    pitch_std = None
    pitch_variation_desc = "Missing / Unvoiced"

    zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, hop_length=hop_length)))

    if zcr_val < 0.30:
        try:
            f0 = librosa.yin(
                y=audio,
                fmin=65,
                fmax=400,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            )
            voiced_f0 = f0[~np.isnan(f0)]
            voiced_f0 = voiced_f0[(voiced_f0 >= 65) & (voiced_f0 <= 400)]

            if len(voiced_f0) > 5 and len(voiced_f0) / len(f0) > 0.20:
                pitch_mean = float(np.mean(voiced_f0))
                pitch_std = float(np.std(voiced_f0))
                if pitch_std < 10.0:
                    pitch_variation_desc = "Low Variance (Flat / Robotic Pitch)"
                elif pitch_std > 50.0:
                    pitch_variation_desc = "High Dynamic Expressiveness"
                else:
                    pitch_variation_desc = "Natural Conversational Modulation"
            else:
                pitch_variation_desc = "Missing / Unvoiced"
        except Exception:
            pitch_variation_desc = "Calculation Skipped / Aperiodic"
    else:
        pitch_variation_desc = "Missing / Unvoiced"

    # 4. Spectral Centroid & Dynamics
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate, hop_length=hop_length)[0]
    spec_mean = float(np.mean(spec_cent)) if len(spec_cent) > 0 else 0.0
    spec_std = float(np.std(spec_cent)) if len(spec_cent) > 0 else 0.0

    if spec_std > 800.0:
        spec_var_desc = "High Spectral Diversity"
    elif spec_std < 200.0:
        spec_var_desc = "Constrained Spectral Range (Narrowband / Filtered)"
    else:
        spec_var_desc = "Moderate Spectral Dynamics"

    spectral_summary = (
        f"Mean Centroid {spec_mean:.0f} Hz (Std: {spec_std:.0f} Hz) - {spec_var_desc}"
    )

    # 5. Overall Audio Quality Indicator
    if energy_mean < 1e-4 or silence_ratio > 0.70:
        audio_quality = "Degraded (Faint / Heavy Silence)"
    elif clipping_ratio > 0.05:
        audio_quality = "Distorted (Significant Clipping Saturated)"
    elif duration < 1.0:
        audio_quality = "Sub-Optimal (Short Sample < 1.0s)"
    elif spec_mean < 1000.0:
        audio_quality = "Bandwidth-Limited (Telephony 8kHz Narrowband)"
    else:
        audio_quality = "Standard Broadcast / Clear Audio"

    return {
        "duration": round(duration, 2),
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "silence_ratio": round(silence_ratio, 3),
        "clipping_ratio": round(clipping_ratio, 4),
        "pitch_mean_hz": round(pitch_mean, 1) if pitch_mean is not None else None,
        "pitch_std_hz": round(pitch_std, 1) if pitch_std is not None else None,
        "pitch_variation": pitch_variation_desc,
        "energy_mean": round(energy_mean, 5),
        "energy_std": round(energy_std, 5),
        "energy_variation": energy_var_desc,
        "spectral_centroid_mean": round(spec_mean, 1),
        "spectral_centroid_std": round(spec_std, 1),
        "spectral_variation": spec_var_desc,
        "spectral_summary": spectral_summary,
        "audio_quality": audio_quality,
    }


def check_out_of_distribution(
    features: np.ndarray,
    train_mean: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> Tuple[bool, float, str]:
    """
    Evaluates whether input features fall significantly outside the training distribution.
    Uses standardized Z-score distance across 42 features.
    """
    if train_mean is None or train_std is None or len(train_mean) != len(features):
        # Default heuristic baseline if training stats are unavailable
        norm_val = float(np.linalg.norm(features))
        if norm_val > 1500.0 or norm_val < 5.0:
            return True, norm_val, "Input signal energy/spectral norm deviates from standard voice domain."
        return False, norm_val, "Within anticipated voice distribution range."

    std_safe = np.where(train_std < 1e-5, 1.0, train_std)
    z_scores = np.abs((features - train_mean) / std_safe)
    max_z = float(np.max(z_scores))
    mean_z = float(np.mean(z_scores))

    if max_z > 4.5 or mean_z > 2.5:
        return (
            True,
            round(max_z, 2),
            f"OUT-OF-DISTRIBUTION WARNING: Max feature deviation ({max_z:.1f}σ) exceeds expected bounds. "
            "Model confidence is uncalibrated on this anomalous acoustic profile.",
        )

    return False, round(max_z, 2), "In-Distribution: Features align with standard baseline acoustic distribution."


def get_feature_summary_table(features: np.ndarray) -> List[Dict[str, Any]]:
    """
    Constructs a structured feature summary table mapping raw acoustic features
    into 5 canonical categories (spectral, energy, pitch, timing, quality).
    """
    if features is None or len(features) < TOTAL_FEATURES:
        return []

    # Features: [MFCC_1..20_mean, MFCC_1..20_std, RMS, ZCR]
    mfcc_means = features[:20]
    mfcc_stds = features[20:40]
    rms = float(features[40])
    zcr = float(features[41])

    summary_rows = [
        {
            "category": "spectral",
            "feature_group": "Low & Mid Spectral Envelope (MFCC 1-12 Mean)",
            "value": f"{np.mean(mfcc_means[:12]):.2f}",
            "reference_range": "-150.0 to 50.0",
            "interpretation": "Vocal tract formant resonance and low-frequency spectral envelope.",
        },
        {
            "category": "spectral",
            "feature_group": "High-Frequency Harmonic Cues (MFCC 13-20 Mean)",
            "value": f"{np.mean(mfcc_means[12:20]):.2f}",
            "reference_range": "-25.0 to 25.0",
            "interpretation": "Upper formant friction, breathiness, and vocoder synthesis artifacts.",
        },
        {
            "category": "energy",
            "feature_group": "Root Mean Square Energy (RMS)",
            "value": f"{rms:.5f}",
            "reference_range": "0.010 to 0.350",
            "interpretation": "Overall signal loudness and frame power distribution.",
        },
        {
            "category": "timing",
            "feature_group": "Macro Prosody & Pause Dynamics (MFCC 1-10 Std)",
            "value": f"{np.mean(mfcc_stds[:10]):.2f}",
            "reference_range": "5.0 to 45.0",
            "interpretation": "Natural conversational cadence, pause variation, and syllable timing.",
        },
        {
            "category": "pitch",
            "feature_group": "Micro-Temporal Jitter & Phase Dynamics (MFCC 11-20 Std)",
            "value": f"{np.mean(mfcc_stds[10:]):.2f}",
            "reference_range": "3.0 to 25.0",
            "interpretation": "Micro-variations in pitch period consistency and vocal tract dynamics.",
        },
        {
            "category": "quality",
            "feature_group": "Zero Crossing Rate (ZCR / Noise Floor)",
            "value": f"{zcr:.5f}",
            "reference_range": "0.020 to 0.200",
            "interpretation": "High-frequency transitions and unvoiced fricative proportion.",
        },
    ]
    return summary_rows


def get_global_feature_importance(model: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts Random Forest feature importances from a trained pipeline
    and aggregates them into 5 canonical categories: spectral, energy, pitch, timing, quality.
    """
    rf = None
    if hasattr(model, "named_steps") and "classifier" in model.named_steps:
        rf = model.named_steps["classifier"]
    elif hasattr(model, "feature_importances_"):
        rf = model

    if rf is None or not hasattr(rf, "feature_importances_"):
        raw_df = pd.DataFrame({
            "feature_name": [f"Feature_{i+1}" for i in range(TOTAL_FEATURES)],
            "importance": [1.0 / TOTAL_FEATURES] * TOTAL_FEATURES,
        })
        group_df = pd.DataFrame({
            "category": ["spectral", "timing", "pitch", "energy", "quality"],
            "feature_group": ["Spectral Envelope (MFCC 1-12)", "Macro Timing / Prosody (MFCC 1-10 Std)", "Pitch / Jitter Micro-Dynamics (MFCC 11-20 Std)", "Signal Energy (RMS)", "Acoustic Quality / ZCR"],
            "importance_share": [0.45, 0.25, 0.15, 0.10, 0.05],
        })
        return raw_df, group_df
    
    importances = np.array(rf.feature_importances_, dtype=np.float32)
    n_feats = len(importances)

    from src.features import get_feature_names
    if n_feats == 187:
        names = get_feature_names("step1")
    elif n_feats == 229:
        names = get_feature_names("advanced")
    elif n_feats == 77:
        names = get_feature_names("extended")
    elif n_feats == 42:
        names = get_feature_names("legacy")
    else:
        names = [f"Feature_{i+1}" for i in range(n_feats)]

    if len(names) != n_feats:
        names = [f"Feature_{i+1}" for i in range(n_feats)]

    raw_df = pd.DataFrame({
        "feature_name": names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Aggregate into 5 Canonical Groups: spectral, energy, pitch, timing, quality
    total_imp = float(np.sum(importances)) + 1e-9
    if n_feats >= 42:
        imp_spectral = float(np.sum(importances[:20])) / total_imp
        imp_timing = float(np.sum(importances[20:30])) / total_imp
        imp_pitch = float(np.sum(importances[30:40])) / total_imp
        imp_energy = float(importances[40]) / total_imp if n_feats > 40 else 0.1
        imp_quality = float(importances[41]) / total_imp if n_feats > 41 else 0.1
    else:
        imp_spectral = 0.45
        imp_timing = 0.25
        imp_pitch = 0.15
        imp_energy = 0.10
        imp_quality = 0.05

    group_data = [
        {"category": "spectral", "feature_group": "Spectral Formants & Harmonics", "importance_share": round(imp_spectral, 4)},
        {"category": "timing", "feature_group": "Macro Timing & Prosody Dynamics", "importance_share": round(imp_timing, 4)},
        {"category": "pitch", "feature_group": "Pitch Micro-Jitter & Phase Modulation", "importance_share": round(imp_pitch, 4)},
        {"category": "energy", "feature_group": "Signal Energy Distribution", "importance_share": round(imp_energy, 4)},
        {"category": "quality", "feature_group": "Acoustic Noise Floor & Transition Quality", "importance_share": round(imp_quality, 4)},
    ]
    group_df = pd.DataFrame(group_data).sort_values(by="importance_share", ascending=False).reset_index(drop=True)

    # Save to reports/feature_importance.csv
    os.makedirs(REPORTS_DIR, exist_ok=True)
    csv_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    raw_df.to_csv(csv_path, index=False)

    return raw_df, group_df


def build_explainability_report(
    model: Any,
    audio: np.ndarray,
    sample_rate: int,
    prediction_result: Dict[str, Any],
    train_mean: Optional[np.ndarray] = None,
    train_std: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Builds comprehensive per-file explainability package combining signal diagnostics,
    uncertainty evaluation, OOD detection, feature tables, and canonical feature groups.
    """
    spoof_prob = prediction_result.get("spoof_probability", 0.50)
    threshold = prediction_result.get("decision_threshold_used", 0.50)
    features = prediction_result.get("features", np.zeros(TOTAL_FEATURES, dtype=np.float32))

    # 1. Uncertainty Band: 0.40 <= P(spoof) <= 0.60
    is_uncertain = (0.40 <= spoof_prob <= 0.60)
    uncertainty_banner = (
        "UNCERTAIN — MANUAL REVIEW REQUIRED"
        if is_uncertain
        else ("CONFIDENT DETECTED SIGNAL" if (spoof_prob >= 0.75 or spoof_prob <= 0.20) else "MODERATE CONFIDENCE SIGNAL")
    )

    # 2. Distance from Decision Threshold
    threshold_distance = round(float(spoof_prob - threshold), 4)

    # 3. Signal Diagnostics
    diagnostics = compute_signal_diagnostics(audio, sample_rate)

    # 4. Out-of-Distribution Check
    is_ood, ood_score, ood_msg = check_out_of_distribution(features, train_mean, train_std)

    # 5. Feature Summary Table
    feature_table = get_feature_summary_table(features)

    # 6. Global Feature Importance
    _, top_groups_df = get_global_feature_importance(model)

    confidence_status = (
        "Confidence is not calibrated. Decision support only — not proof of identity."
    )

    return {
        # Per-file explanation essentials
        "duration": diagnostics["duration"],
        "sample_rate": diagnostics["sample_rate"],
        "silence_ratio": diagnostics["silence_ratio"],
        "clipping_ratio": diagnostics["clipping_ratio"],
        "pitch_variation": diagnostics["pitch_variation"],
        "energy_variation": diagnostics["energy_variation"],
        "spectral_summary": diagnostics["spectral_summary"],
        "spoof_probability": round(float(spoof_prob), 4),
        "distance_from_threshold": threshold_distance,
        "confidence_status": confidence_status,
        # Extended explainability state
        "is_uncertain": is_uncertain,
        "uncertainty_banner": uncertainty_banner,
        "decision_threshold": threshold,
        "threshold_distance": threshold_distance,
        "calibration_status": confidence_status,
        "is_out_of_distribution": is_ood,
        "ood_score": ood_score,
        "ood_message": ood_msg,
        "signal_diagnostics": diagnostics,
        "feature_summary_table": feature_table,
        "top_feature_groups": top_groups_df.to_dict(orient="records"),
        "disclaimer": EXPLAINABILITY_DISCLAIMER,
    }
