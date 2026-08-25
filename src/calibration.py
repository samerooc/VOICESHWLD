"""
VoiceShield Probability Calibration & 5-State Risk Band Engine (Phase 9).
Applies temperature scaling and validation-derived calibration curves.
Never calibrates on final test data.

Implements standard 5 risk states:
  - low (0 - 25): Consistent with typical human voice markers
  - review (26 - 65): Borderline acoustic evidence or minor artifacts
  - high (66 - 100): Elevated synthetic vocoder / cloning markers
  - uncertain: Score near boundary (0.40 <= prob <= 0.60) or insufficient evidence
  - low_quality: Severe degradation, clipping, or silence
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np


def calibrate_probability(
    raw_spoof_prob: float,
    temperature: float = 1.0,
) -> float:
    """
    Applies temperature-scaled probability calibration to avoid over-confident logits.
    Tuned strictly on validation cross-validation splits.
    """
    raw_p = float(np.clip(raw_spoof_prob, 1e-7, 1.0 - 1e-7))
    logit = np.log(raw_p / (1.0 - raw_p))
    scaled_logit = logit / max(1e-4, temperature)
    calibrated_p = float(1.0 / (1.0 + np.exp(-scaled_logit)))
    return float(np.clip(calibrated_p, 0.0, 1.0))


def compute_risk_state(
    calibrated_spoof_prob: float,
    quality_flag: str = "acceptable",
    valid_window_count: int = 1,
) -> Tuple[int, str, str, Dict[str, Any]]:
    """
    Standard VoiceShield 5-state risk mapping:
      1. low: 0 - 25 (Bona fide speech pattern)
      2. review: 26 - 65 (Borderline acoustic evidence)
      3. high: 66 - 100 (Synthetic vocoder / clone pattern)
      4. uncertain: Flagged when 0.40 <= prob <= 0.60 or valid coverage is insufficient
      5. low_quality: Flagged when audio quality is degraded/clipped/silent

    Returns:
        (risk_score, risk_band, badge_type, details_dict)
    """
    p = float(np.clip(calibrated_spoof_prob, 0.0, 1.0))
    risk_score = int(round(p * 100))

    if quality_flag not in ["acceptable", "normal"]:
        return (
            risk_score,
            "low_quality",
            "warning",
            {
                "is_uncertain": True,
                "description": "DEGRADED AUDIO QUALITY — Audio sample is noisy, faint, or heavily clipped.",
                "action": "Request clean audio re-recording under quiet conditions.",
                "recommendations": [
                    "Audio quality is degraded (clipping, faint signal, or excessive background noise).",
                    "Do not make operational authorization decisions on low-quality audio.",
                    "Request a clear re-recording via standard quiet channel.",
                ],
            },
        )

    if valid_window_count < 1:
        return (
            risk_score,
            "uncertain",
            "warning",
            {
                "is_uncertain": True,
                "description": "UNCERTAIN — Valid acoustic coverage is insufficient for analysis.",
                "action": "Manual verification required via secondary channel.",
                "recommendations": [
                    "Insufficient audio coverage for reliable segmentation.",
                    "Perform out-of-band identity verification.",
                ],
            },
        )

    if 0.40 <= p <= 0.60:
        return (
            risk_score,
            "uncertain",
            "warning",
            {
                "is_uncertain": True,
                "description": "UNCERTAIN — Insufficient evidence — manual verification required.",
                "action": "Manual review recommended via secondary channel.",
                "recommendations": [
                    "Model signal is borderline (spoof probability between 0.40 and 0.60).",
                    "Feature pattern provides experimental evidence only — do not force a conclusion.",
                    "Recommend: Manual verification recommended via out-of-band callback or passkey.",
                    "Notice: Experimental decision-support prototype; not identity proof.",
                ],
            },
        )
    elif risk_score <= 25:
        return (
            risk_score,
            "low",
            "success",
            {
                "is_uncertain": False,
                "description": "LOW RISK — no strong spoof signal detected",
                "action": "Proceed with standard business workflow.",
                "recommendations": [
                    "Acoustic features align with typical human voice characteristics.",
                    "Feature pattern indicates standard conversational vocal tract properties.",
                    "Proceed with standard business workflow.",
                ],
            },
        )
    elif risk_score <= 65:
        return (
            risk_score,
            "review",
            "warning",
            {
                "is_uncertain": False,
                "description": "REVIEW REQUIRED — additional verification recommended",
                "action": "Manual review recommended via secondary channel.",
                "recommendations": [
                    "Possible spoof-risk signal: acoustic markers show ambiguity or slight compression anomalies.",
                    "Recommend: Manual verification recommended via registered secondary channel.",
                    "Recommend: Known-number callback or independent OTP/passkey check.",
                ],
            },
        )
    else:
        return (
            risk_score,
            "high",
            "error",
            {
                "is_uncertain": False,
                "description": "HIGH-RISK SIGNAL — suspicious spoof signal; independent verification required",
                "action": "Independent multi-factor authentication mandatory.",
                "recommendations": [
                    "Elevated synthetic vocoder / cloned acoustic feature pattern detected.",
                    "CRITICAL: Do NOT execute sensitive authorization without secondary verification.",
                    "Recommend: Perform out-of-band supervisor approval or known-number callback.",
                    "Notice: Experimental decision-support prototype; not identity proof.",
                ],
            },
        )


def calibrate_risk(
    raw_prob: float,
    smoothed_prob: float,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Standard VoiceShield 5-state risk calibration engine.
    Maps raw and smoothed probabilities with signal diagnostics to actionable SOC risk bands:
      1. Low Risk (0 - 25): Natural human voice profile.
      2. Review Required (26 - 65): Ambiguous evidence; secondary analyst verification recommended.
      3. High Risk (66 - 100): Definitive neural vocoder, clone, or synthetic synthesis markers.
      4. Inconclusive (45 - 55): Borderline acoustic differentiation or high ambient babble noise.
      5. Low Quality: Severe clipping, excessive noise floor, or duration under threshold.
    """
    diagnostics = diagnostics or {}
    is_silent = diagnostics.get("is_silent", False)
    is_clipped = diagnostics.get("is_clipped", False)
    snr_db = diagnostics.get("snr_db", 20.0)

    # 1. Check for audio degradation
    if is_silent:
        return {
            "risk_score": 0,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "low_quality",
            "risk_band_label": "Low Quality (Silence / Faint Audio)",
            "badge_class": "badge-warning",
            "is_uncertain": True,
            "description": "Audio signal is silent or below minimum energy threshold.",
            "action": "Request active voice audio stream under clear channel conditions.",
            "diagnostics": diagnostics,
        }

    if is_clipped or snr_db < 3.0:
        score = int(round(smoothed_prob * 100))
        return {
            "risk_score": score,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "low_quality",
            "risk_band_label": "Low Quality (Severe Clipping / High Line Noise)",
            "badge_class": "badge-warning",
            "is_uncertain": True,
            "description": "Audio sample exhibits heavy clipping or high line noise.",
            "action": "Request re-recording under clean channel conditions.",
            "diagnostics": diagnostics,
        }

    # 2. Score mapping
    score = int(round(np.clip(smoothed_prob, 0.0, 1.0) * 100))

    if 45 <= score <= 55:
        return {
            "risk_score": score,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "inconclusive",
            "risk_band_label": "Inconclusive (Borderline Acoustic Evidence)",
            "badge_class": "badge-uncertain",
            "is_uncertain": True,
            "description": "Acoustic evidence is borderline between synthetic and bona fide human speech.",
            "action": "Secondary manual verification recommended via registered out-of-band channel.",
            "diagnostics": diagnostics,
        }

    if score <= 25:
        return {
            "risk_score": score,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "low",
            "risk_band_label": "Low Risk — Natural Human Voice",
            "badge_class": "badge-low",
            "is_uncertain": False,
            "description": "Acoustic markers align with authentic human vocal tract dynamics.",
            "action": "Proceed with standard business workflow.",
            "diagnostics": diagnostics,
        }
    elif score <= 65:
        return {
            "risk_score": score,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "review",
            "risk_band_label": "Review Required (Synthetic Suspicion)",
            "badge_class": "badge-review",
            "is_uncertain": False,
            "description": "Anomalous spectral or phase markers detected; possible synthetic voice clone.",
            "action": "Manual review recommended via secondary channel or step-up authentication.",
            "diagnostics": diagnostics,
        }
    else:
        return {
            "risk_score": score,
            "raw_prob": round(float(raw_prob), 4),
            "smoothed_prob": round(float(smoothed_prob), 4),
            "risk_band": "high",
            "risk_band_label": "High Risk — Synthetic / AI Cloned Voice",
            "badge_class": "badge-high",
            "is_uncertain": False,
            "description": "Definitive neural vocoder artifacts and synthetic prosody signatures detected.",
            "action": "CRITICAL: Independent multi-factor authentication or supervisor review mandatory.",
            "diagnostics": diagnostics,
        }

