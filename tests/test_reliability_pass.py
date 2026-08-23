"""
Comprehensive Reliability, Quality, and Label Consistency Tests (Phase 9 Reliability Pass).
"""

import io
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from api import app
from src.audio_io import estimate_snr_db, get_audio_metadata, load_audio_from_bytes
from src.config import (
    CLASS_MAPPING,
    FEATURE_SCHEMA_VERSION,
    LABEL_MAP,
    MODEL_PATH,
    SAMPLE_RATE,
    TOTAL_FEATURES,
)
from src.features import extract_features_from_audio
from src.model import load_model
from src.scoring import calculate_risk_score, get_risk_band, predict_and_score

client = TestClient(app)


def test_label_mapping_integrity():
    """Verify explicit class mapping: 0 = bona_fide, 1 = spoof."""
    assert LABEL_MAP[0] == "bona_fide"
    assert LABEL_MAP[1] == "spoof"
    assert CLASS_MAPPING["0"] == "bona_fide"
    assert CLASS_MAPPING["1"] == "spoof"


def test_probability_sum_and_bounds():
    """Verify that predict_and_score guarantees P(human) + P(spoof) == 1.0 within tolerance."""
    model = load_model(MODEL_PATH)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    res = predict_and_score(model, audio, sample_rate=SAMPLE_RATE)
    assert 0.0 <= res["bona_fide_probability"] <= 1.0
    assert 0.0 <= res["spoof_probability"] <= 1.0
    assert np.isclose(res["bona_fide_probability"] + res["spoof_probability"], 1.0, atol=1e-3)
    assert res["raw_model_score"] == res["spoof_probability"]


def test_stereo_to_mono_deterministic_conversion():
    """Verify stereo audio is converted to mono via deterministic channel averaging."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    left = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    right = (0.6 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    stereo = np.column_stack([left, right])

    bio = io.BytesIO()
    sf.write(bio, stereo, SAMPLE_RATE, format="WAV")

    audio_mono, sr = load_audio_from_bytes(bio.getvalue(), target_sr=SAMPLE_RATE)
    assert audio_mono.ndim == 1
    assert len(audio_mono) == SAMPLE_RATE
    # Expected mean amplitude is roughly average of left and right
    assert np.allclose(audio_mono, (left + right) / 2.0, atol=1e-2)


def test_feature_shape_and_nan_inf_safety():
    """Verify feature extractor produces exact 42 features and safely handles NaNs."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    clean_audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    feats = extract_features_from_audio(clean_audio, sample_rate=SAMPLE_RATE)
    assert feats.shape == (TOTAL_FEATURES,)
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # NaN input array safety
    nan_audio = np.full(SAMPLE_RATE, np.nan, dtype=np.float32)
    nan_feats = extract_features_from_audio(nan_audio, sample_rate=SAMPLE_RATE)
    assert nan_feats.shape == (TOTAL_FEATURES,)
    assert not np.isnan(nan_feats).any()


def test_audio_quality_diagnostics_and_clipping():
    """Verify clipping and quality flag calculations."""
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    clean = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    meta_clean = get_audio_metadata(clean, SAMPLE_RATE)
    assert meta_clean["quality_flag"] == "acceptable"
    assert meta_clean["clipping_ratio"] < 0.05

    clipped = np.clip((5.0 * np.sin(2 * np.pi * 440.0 * t)), -1.0, 1.0).astype(np.float32)
    meta_clipped = get_audio_metadata(clipped, SAMPLE_RATE)
    assert meta_clipped["clipping_ratio"] > 0.50
    assert meta_clipped["quality_flag"] == "heavily_clipped"


def test_uncertainty_band_triggers():
    """Verify uncertainty band triggers on 0.40 <= P <= 0.60."""
    desc, band, _, _ = get_risk_band(48, spoof_prob=0.48)
    assert "UNCERTAIN" in desc or band == "Review required"

    desc_50, band_50, _, _ = get_risk_band(50, spoof_prob=0.50)
    assert "Insufficient evidence" in desc_50


def test_api_metadata_includes_class_mapping():
    """Verify GET /metadata includes explicit class mapping and schema versions."""
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert data["class_mapping"] == {"0": "bona_fide", "1": "spoof"}
    assert data["audio_saved"] is False
