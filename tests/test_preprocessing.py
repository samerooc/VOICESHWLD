"""
VoiceShield Shared Preprocessing Unit Tests (Phase 4).
Validates normalization, float32 conversion, resampling, and format safety.
"""

import pytest
import numpy as np
from src.preprocessing import (
    preprocess_audio,
    load_preprocessing_config,
    detect_audio_container_and_codec,
)


def test_preprocessing_config_loaded():
    cfg = load_preprocessing_config()
    assert "audio" in cfg or "target_sample_rate" in cfg
    assert cfg.get("audio", {}).get("target_sample_rate", cfg.get("target_sample_rate")) == 16000


def test_preprocessing_mono_and_float32():
    # 2 channels 48kHz
    sr = 48000
    stereo = np.random.uniform(-0.8, 0.8, (sr * 2, 2)).astype(np.float64)

    clean, effective_sr, diag = preprocess_audio(stereo, sample_rate=sr)

    assert clean.ndim == 1
    assert clean.dtype == np.float32
    assert effective_sr == 16000
    assert np.all(np.isfinite(clean))
    assert np.max(np.abs(clean)) <= 1.0
    assert "quality_flags" in diag


def test_preprocessing_silence_detection():
    sr = 16000
    silence = np.zeros(sr * 2, dtype=np.float32)
    clean, effective_sr, diag = preprocess_audio(silence, sample_rate=sr)
    assert "silent_or_faint" in diag["quality_flags"] or diag["quality_status"] == "low_quality"
