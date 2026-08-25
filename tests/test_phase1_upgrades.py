"""
Unit & Integration Tests for VoiceShield Phase 1 Upgrades:
- 77-dimensional acoustic feature extraction (F0, Jitter, Shimmer, HNR, Spectral, Prosody)
- Multi-condition data augmentations (telephony 8kHz, noise injection, reverb)
- Continuous retraining logging & Feature Drift Detector (KS test, PSI)
"""

import numpy as np
import pytest

from src.config import EXTENDED_TOTAL_FEATURES, LEGACY_TOTAL_FEATURES, SAMPLE_RATE
from src.features import (
    extract_extended_features,
    extract_features_from_audio,
    extract_high_frequency_artifacts,
    extract_mfcc_features,
    extract_pitch_and_jitter,
    extract_prosody_timing,
    extract_shimmer_and_hnr,
    extract_spectral_dynamics,
    get_feature_names,
)
from src.model import build_pipeline, get_available_classifiers
from src.retraining import (
    FeatureDriftDetector,
    InferenceAuditLogger,
    calculate_psi,
)
from scripts.augment_data import (
    apply_gain_jitter,
    apply_room_reverberation,
    apply_telephony_8khz,
    inject_noise,
)


@pytest.fixture
def synthetic_speech():
    """Generates 2.0 seconds of synthetic speech-like tone with harmonics and pauses."""
    t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False)
    # Fundamental 220Hz + harmonics + modulation
    f0 = 220.0
    sig = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Add a 300ms pause in the middle
    pause_start = int(0.8 * SAMPLE_RATE)
    pause_end = int(1.1 * SAMPLE_RATE)
    sig[pause_start:pause_end] = 0.0
    return sig.astype(np.float32)


def test_extended_features_dimension_and_validity(synthetic_speech):
    """Verifies that extract_extended_features returns a 77-dimensional vector without NaNs or Infs."""
    feats = extract_extended_features(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert feats.shape == (EXTENDED_TOTAL_FEATURES,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()
    assert feats.dtype == np.float32


def test_feature_extraction_modes(synthetic_speech):
    """Verifies dual legacy (42-dim) and extended (77-dim) extraction modes."""
    legacy_feats = extract_features_from_audio(synthetic_speech, sample_rate=SAMPLE_RATE, mode="legacy")
    assert legacy_feats.shape == (LEGACY_TOTAL_FEATURES,)

    ext_feats = extract_features_from_audio(synthetic_speech, sample_rate=SAMPLE_RATE, mode="extended")
    assert ext_feats.shape == (EXTENDED_TOTAL_FEATURES,)


def test_pitch_jitter_shimmer_hnr_extraction(synthetic_speech):
    """Verifies F0 pitch, Jitter, Shimmer, and HNR descriptors."""
    pj = extract_pitch_and_jitter(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert "f0_mean" in pj
    assert "jitter_local" in pj
    assert pj["f0_mean"] > 100.0  # Synthetic signal is ~220Hz

    sh = extract_shimmer_and_hnr(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert "shimmer_local" in sh
    assert "hnr_mean" in sh


def test_spectral_dynamics_and_prosody(synthetic_speech):
    """Verifies spectral dynamics and prosodic pause extraction."""
    sd = extract_spectral_dynamics(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert "spectral_flatness_mean" in sd
    assert "spectral_flux_mean" in sd
    assert "spectral_rolloff_85_mean" in sd

    pt = extract_prosody_timing(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert "pause_count" in pt
    assert pt["pause_count"] >= 1.0  # We inserted a 300ms pause


def test_augmentations(synthetic_speech):
    """Verifies acoustic data augmentations execute without error and preserve shape."""
    tele = apply_telephony_8khz(synthetic_speech, sample_rate=SAMPLE_RATE)
    assert len(tele) == len(synthetic_speech)
    assert not np.isnan(tele).any()

    noisy = inject_noise(synthetic_speech, noise_type="pink", snr_db=10.0, sample_rate=SAMPLE_RATE)
    assert len(noisy) == len(synthetic_speech)

    reverb = apply_room_reverberation(synthetic_speech, rt60=0.3, sample_rate=SAMPLE_RATE)
    assert len(reverb) == len(synthetic_speech)


def test_feature_drift_detector_and_psi():
    """Verifies PSI calculation and Drift Detector alert states."""
    baseline = np.random.normal(0.0, 1.0, 500)
    stable_live = np.random.normal(0.05, 1.02, 500)
    drifted_live = np.random.normal(3.0, 2.5, 500)

    psi_stable = calculate_psi(baseline, stable_live)
    psi_drifted = calculate_psi(baseline, drifted_live)

    assert psi_stable < 0.15
    assert psi_drifted > 0.25

    detector = FeatureDriftDetector()
    sample_matrix = np.random.normal(0, 1, (20, EXTENDED_TOTAL_FEATURES)).astype(np.float32)
    report = detector.evaluate_batch_drift(sample_matrix)
    assert "overall_drift_status" in report
    assert report["num_features_evaluated"] == EXTENDED_TOTAL_FEATURES


def test_inference_audit_logger(tmp_path):
    """Verifies privacy-safe audit logging."""
    logger = InferenceAuditLogger(log_dir=str(tmp_path))
    dummy_feat = np.zeros(EXTENDED_TOTAL_FEATURES, dtype=np.float32)
    rec = logger.log_prediction(
        features=dummy_feat,
        spoof_probability=0.85,
        risk_score=85,
        risk_band="High",
        is_uncertain=False,
    )
    assert rec["risk_score"] == 85
    assert rec["feature_dim"] == EXTENDED_TOTAL_FEATURES
