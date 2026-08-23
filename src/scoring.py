"""
VoiceShield Risk Scoring & Advisory Engine (Phase 9 & 13).
Calibrates spoof probabilities into 0-100 risk scores, assigns 5-state risk bands,
and generates non-blocking manual verification guidance.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from src.config import (
    DEFAULT_DECISION_THRESHOLD,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
)
from src.features import extract_features_from_audio, extract_segmented_features
from src.model_contract import (
    CLASS_NAMES,
    HUMAN_READABLE_LABELS,
    LABEL_BONA_FIDE,
    LABEL_SPOOF,
    validate_model_probabilities,
)
from src.preprocessing import preprocess_audio
from src.calibration import compute_risk_state, calibrate_probability


def calculate_risk_score(spoof_probability: float) -> int:
    """Converts 0.0-1.0 spoof probability into integer 0-100 risk score."""
    return int(np.clip(round(spoof_probability * 100), 0, 100))


def get_risk_band(risk_score: int, spoof_prob: Optional[float] = None) -> Any:
    """Legacy helper preserving return signature."""
    p = (risk_score / 100.0) if spoof_prob is None else spoof_prob
    score, band_str, badge, details = compute_risk_state(p, quality_flag="acceptable")
    
    # Map band_str to UI-friendly display name
    display_bands = {
        "low": "Low",
        "review": "Review required",
        "high": "High risk",
        "uncertain": "Review required",
        "low_quality": "Review required",
    }
    return (
        details["description"],
        display_bands.get(band_str, "Review required"),
        badge,
        details["recommendations"],
    )


def predict_and_score(
    model: Any,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    decision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extracts acoustic features, runs multi-segment inference, validates probabilities,
    calculates calibrated risk score, and compiles advisory report adhering to strict model contracts.
    """
    if model is None:
        raise ValueError("Model is None. Cannot run prediction.")

    # 1. Standard Preprocessing & Diagnostic Metadata
    clean_audio, effective_sr, audio_diag = preprocess_audio(audio, sample_rate=sample_rate, target_sr=SAMPLE_RATE)

    # 2. Extract global feature vector (42 features)
    features = extract_features_from_audio(clean_audio, effective_sr)
    x_global = features.reshape(1, -1)

    # 3. Global probability extraction
    global_probs = model.predict_proba(x_global)[0]
    p_global_human, p_global_spoof = validate_model_probabilities(global_probs)

    # 4. Multi-segment temporal ensemble (Sliding window voting)
    seg_features = extract_segmented_features(clean_audio, effective_sr, window_duration=2.5, hop_duration=1.0)
    valid_window_count = len(seg_features)

    if valid_window_count > 1:
        seg_x = np.array(seg_features, dtype=np.float32)
        seg_probs = model.predict_proba(seg_x)
        seg_spoof_probs = seg_probs[:, 1]
        seg_median_spoof = float(np.median(seg_spoof_probs))
        
        raw_spoof_prob = float(0.50 * p_global_spoof + 0.50 * seg_median_spoof)
        raw_human_prob = float(1.0 - raw_spoof_prob)
        human_prob, spoof_prob = validate_model_probabilities(np.array([raw_human_prob, raw_spoof_prob]))
    else:
        human_prob, spoof_prob = p_global_human, p_global_spoof

    # 5. Thresholding & class prediction
    threshold = decision_threshold if decision_threshold is not None else DEFAULT_DECISION_THRESHOLD
    pred_class = LABEL_SPOOF if spoof_prob >= threshold else LABEL_BONA_FIDE
    pred_label = HUMAN_READABLE_LABELS.get(pred_class, "Unknown")

    # 6. Risk scoring & 5-State Calibration
    quality_status = audio_diag.get("quality_status", "acceptable")
    risk_score, risk_band_name, badge_type, state_details = compute_risk_state(
        spoof_prob,
        quality_flag=quality_status,
        valid_window_count=valid_window_count,
    )

    is_uncertain = state_details.get("is_uncertain", False)

    # UI display band mapping
    ui_band_display = {
        "low": "Low",
        "review": "Review required",
        "high": "High risk",
        "uncertain": "Review required",
        "low_quality": "Review required",
    }.get(risk_band_name, "Review required")

    return {
        "prediction_class": pred_class,
        "prediction_label": pred_label,
        "human_probability": round(human_prob, 4),
        "bona_fide_probability": round(human_prob, 4),
        "spoof_probability": round(spoof_prob, 4),
        "raw_model_score": round(p_global_spoof, 4),
        "calibrated_probability": round(spoof_prob, 4),
        "risk_score": risk_score,
        "risk_band": ui_band_display,
        "risk_state": risk_band_name,
        "risk_description": state_details["description"],
        "risk_badge_type": badge_type,
        "is_uncertain": is_uncertain,
        "uncertainty": is_uncertain,
        "valid_window_count": valid_window_count,
        "total_window_count": valid_window_count,
        "quality_flags": audio_diag.get("quality_flags", []),
        "audio_diagnostics": audio_diag,
        "recommendations": state_details["recommendations"],
        "decision_threshold_used": threshold,
        "features": features,
        "audio_saved": False,
        "disclaimer": STATUTORY_DISCLAIMER,
    }
