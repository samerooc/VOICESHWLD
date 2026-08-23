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
