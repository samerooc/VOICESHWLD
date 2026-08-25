"""
Unit tests for VoiceShield Explainability, Signal Diagnostics & Uncertainty Engine.
"""

import numpy as np
import pytest

from src.config import MODEL_PATH, SAMPLE_RATE, TOTAL_FEATURES
from src.explainability import (
    EXPLAINABILITY_DISCLAIMER,
    build_explainability_report,
    check_out_of_distribution,
    compute_signal_diagnostics,
    get_feature_summary_table,
    get_global_feature_importance,
)
from src.model import load_model


def test_signal_diagnostics_on_clean_and_low_quality_audio():
    """Verify signal diagnostics on normal voice and low-quality/silent audio."""
    # Clean sine wave signal
    t = np.linspace(0, 1.5, int(SAMPLE_RATE * 1.5), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    diag = compute_signal_diagnostics(audio, SAMPLE_RATE)

    assert diag["duration_seconds"] == 1.5
    assert "silence_ratio" in diag
    assert "energy_mean" in diag
    assert "spectral_centroid_mean" in diag

    # Low-quality / faint audio
    faint_audio = (1e-5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    diag_faint = compute_signal_diagnostics(faint_audio, SAMPLE_RATE)
    assert "Degraded" in diag_faint["audio_quality"] or "Faint" in diag_faint["audio_quality"]


def test_signal_diagnostics_missing_pitch_and_empty():
    """Verify diagnostics gracefully handle aperiodic noise, missing pitch, and empty signals."""
    # Pure white noise (unvoiced / no clear pitch)
    noise = (0.1 * np.random.randn(int(SAMPLE_RATE * 1.0))).astype(np.float32)
    diag_noise = compute_signal_diagnostics(noise, SAMPLE_RATE)
    assert diag_noise["duration_seconds"] == 1.0
    assert diag_noise["pitch_variation"] in ["Missing / Unvoiced", "Calculation Skipped / Aperiodic", "Low Variance (Flat / Robotic Pitch)"]

    # Empty audio array
    empty_diag = compute_signal_diagnostics(np.array([], dtype=np.float32), SAMPLE_RATE)
    assert empty_diag["duration_seconds"] == 0.0
    assert empty_diag["audio_quality"] == "Empty / Invalid"


def test_uncertainty_band_and_threshold_distance():
    """Verify uncertainty band triggers when 0.40 <= P(spoof) <= 0.60."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Simulated uncertain result
    fake_pred_uncertain = {
        "prediction_class": 0,
        "prediction_label": "Ambiguous / Review Required",
        "human_probability": 0.52,
        "spoof_probability": 0.48,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }

    report_unc = build_explainability_report(model, audio, SAMPLE_RATE, fake_pred_uncertain)
    assert report_unc["is_uncertain"] is True
    assert "UNCERTAIN" in report_unc["uncertainty_banner"]
    assert report_unc["threshold_distance"] == 0.08  # 0.48 - 0.40
    assert "Confidence is not calibrated" in report_unc["calibration_status"]

    # Simulated confident result
    fake_pred_confident = {
        "prediction_class": 1,
        "prediction_label": "Likely Spoof / AI Voice",
        "human_probability": 0.10,
        "spoof_probability": 0.90,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }
    report_conf = build_explainability_report(model, audio, SAMPLE_RATE, fake_pred_confident)
    assert report_conf["is_uncertain"] is False


def test_out_of_distribution_detection():
    """Verify OOD detection on anomalous feature vectors."""
    train_mean = np.zeros(TOTAL_FEATURES, dtype=np.float32)
    train_std = np.ones(TOTAL_FEATURES, dtype=np.float32)

    # In-distribution vector
    normal_feat = np.random.normal(0, 1, TOTAL_FEATURES).astype(np.float32)
    is_ood, score, msg = check_out_of_distribution(normal_feat, train_mean, train_std)
    assert is_ood is False

    # Extreme anomalous vector (10 sigma outlier)
    anomalous_feat = np.ones(TOTAL_FEATURES, dtype=np.float32) * 10.0
    is_ood_out, score_out, msg_out = check_out_of_distribution(anomalous_feat, train_mean, train_std)
    assert is_ood_out is True
    assert "OUT-OF-DISTRIBUTION" in msg_out


def test_feature_summary_table_and_empty_handling():
    """Verify feature summary table structure and empty input handling."""
    valid_feat = np.ones(TOTAL_FEATURES, dtype=np.float32) * 2.5
    rows = get_feature_summary_table(valid_feat)

    assert len(rows) == 6
    categories = {r["category"] for r in rows}
    assert {"spectral", "energy", "pitch", "timing", "quality"}.issubset(categories)
    for r in rows:
        assert "category" in r
        assert "feature_group" in r
        assert "value" in r
        assert "reference_range" in r
        assert "interpretation" in r

    # Empty features
    empty_rows = get_feature_summary_table(np.array([], dtype=np.float32))
    assert empty_rows == []


def test_per_file_explainability_fields():
    """Verify that build_explainability_report includes all required per-file fields."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    fake_pred = {
        "prediction_class": 0,
        "prediction_label": "Likely Human Voice",
        "human_probability": 0.85,
        "spoof_probability": 0.15,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }

    report = build_explainability_report(model, audio, SAMPLE_RATE, fake_pred)
    required_keys = [
        "duration",
        "sample_rate",
        "silence_ratio",
        "clipping_ratio",
        "pitch_variation",
        "energy_variation",
        "spectral_summary",
        "spoof_probability",
        "distance_from_threshold",
        "confidence_status",
    ]
    for k in required_keys:
        assert k in report, f"Missing key in per-file explanation: {k}"


def test_global_feature_importance_shape_and_groups():
    """Verify global feature importance shapes and 5 canonical group aggregation."""
    model = load_model(MODEL_PATH)
    if model is not None:
        raw_df, group_df = get_global_feature_importance(model)

        assert len(raw_df) in [TOTAL_FEATURES, 178]
        assert "feature_name" in raw_df.columns
        assert "importance" in raw_df.columns
        assert np.all(raw_df["importance"] >= 0.0)

        assert len(group_df) == 5
        assert "category" in group_df.columns
        assert "feature_group" in group_df.columns
        assert "importance_share" in group_df.columns
        categories = set(group_df["category"])
        assert {"spectral", "energy", "pitch", "timing", "quality"}.issubset(categories)
        assert np.isclose(group_df["importance_share"].sum(), 1.0, atol=0.05)


def test_explainability_disclaimer_presence():
    """Verify explicit statutory disclaimer is attached."""
    assert "not proof of identity" in EXPLAINABILITY_DISCLAIMER


def test_explainability_probability_boundaries():
    """Verify boundary conditions at exact thresholds 0.40 and 0.60."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    # Boundary 0.40
    pred_40 = {
        "prediction_class": 1,
        "prediction_label": "Ambiguous / Review Required",
        "human_probability": 0.60,
        "spoof_probability": 0.40,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }
    rep_40 = build_explainability_report(model, audio, SAMPLE_RATE, pred_40)
    assert rep_40["is_uncertain"] is True
    assert "UNCERTAIN" in rep_40["uncertainty_banner"]

    # Boundary 0.60
    pred_60 = {
        "prediction_class": 1,
        "prediction_label": "Ambiguous / Review Required",
        "human_probability": 0.40,
        "spoof_probability": 0.60,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }
    rep_60 = build_explainability_report(model, audio, SAMPLE_RATE, pred_60)
    assert rep_60["is_uncertain"] is True


def test_explainability_missing_metadata_and_low_quality():
    """Verify report generation handles None metadata and low-quality audio without crashing."""
    model = load_model(MODEL_PATH)
    # Low quality audio (very faint)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    faint_audio = (1e-6 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    pred = {
        "prediction_class": 0,
        "prediction_label": "Likely Human Voice",
        "human_probability": 0.85,
        "spoof_probability": 0.15,
        "decision_threshold_used": 0.40,
        "features": np.zeros(TOTAL_FEATURES, dtype=np.float32),
    }

    # Pass train_mean=None and train_std=None
    rep = build_explainability_report(model, faint_audio, SAMPLE_RATE, pred, train_mean=None, train_std=None)
    assert rep is not None
    assert "audio_quality" in rep["signal_diagnostics"]
    assert "Degraded" in rep["signal_diagnostics"]["audio_quality"] or "Faint" in rep["signal_diagnostics"]["audio_quality"]

