"""
Unit & Integration Test Suite for Phase 1 Audio Engineering & Feature Extraction.
Tests:
1. In-memory zero-persistence audio processor & diagnostics.
2. Silero VAD & Energy fallback voice activity filtering.
3. 178-dimensional LFCC, MFCC, Praat Glottal Jitter/Shimmer, and Spectral feature extraction.
"""

import io
import os
import sys
import numpy as np
import pytest
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import compute_snr_db, load_audio_from_bytes
from src.config import SAMPLE_RATE
from src.features import (
    TOTAL_DIMENSIONS,
    extract_features,
    get_feature_names,
)
from src.vad import VoiceActivityDetector


def make_synthetic_wav_bytes(duration: float = 1.5, freq: float = 440.0, sr: int = SAMPLE_RATE) -> bytes:
    """Generates synthetic PCM 16-bit WAV bytes in memory."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.6 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


# -----------------------------------------------------------------------------
# 1. In-Memory Audio Processor Tests
# -----------------------------------------------------------------------------
def test_load_audio_from_bytes_valid_wav():
    """Verify in-memory decoding, standardizing to 16kHz float32, and diagnostics."""
    wav_bytes = make_synthetic_wav_bytes(duration=2.0, freq=440.0)
    audio, diag = load_audio_from_bytes(wav_bytes, target_sr=SAMPLE_RATE)

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert len(audio) == 32000
    assert np.all(audio >= -1.0) and np.all(audio <= 1.0)
    assert not np.any(np.isnan(audio))

    assert diag["duration_sec"] == 2.0
    assert diag["sample_rate"] == SAMPLE_RATE
    assert diag["is_silent"] is False
    assert diag["snr_db"] > 0.0


def test_load_audio_from_bytes_corrupt_payload():
    """Verify rejection of malformed / corrupted byte streams."""
    corrupt_bytes = b"RIFF1234CORRUPT_BYTES_HEADER_DATA\x00\xff"
    with pytest.raises(ValueError) as exc:
        load_audio_from_bytes(corrupt_bytes)
    assert "failed" in str(exc.value).lower()


def test_load_audio_from_bytes_empty_payload():
    """Verify rejection of 0-byte payload."""
    with pytest.raises(ValueError) as exc:
        load_audio_from_bytes(b"")
    assert "empty" in str(exc.value).lower()


# -----------------------------------------------------------------------------
# 2. Voice Activity Detection (VAD) Tests
# -----------------------------------------------------------------------------
def test_filter_voiced_audio_strips_silence():
    """Verify VAD removes leading/trailing silence from speech/sine signal."""
    vad = VoiceActivityDetector(sample_rate=SAMPLE_RATE)
    
    # 1.0s silence + 1.0s sine wave + 1.0s silence = 3.0s (48000 samples)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    combined = np.concatenate([silence, sine, silence])

    voiced = vad.filter_voiced_audio(combined, sr=SAMPLE_RATE)
    assert len(voiced) < len(combined)
    assert len(voiced) > 0
    assert not np.any(np.isnan(voiced))


# -----------------------------------------------------------------------------
# 3. High-Dimensional Feature Extraction (178 Dimensions) Tests
# -----------------------------------------------------------------------------
def test_feature_names_count_and_uniqueness():
    """Verify exactly 178 unique feature names."""
    names = get_feature_names()
    assert len(names) == 178
    assert len(set(names)) == 178
    assert "lfcc_01_mean" in names
    assert "mfcc_01_mean" in names
    assert "glottal_jitter_local" in names
    assert "spectral_centroid_mean" in names
    assert "rms_energy_mean" in names


def test_extract_features_random_noise():
    """Verify 178-dim extraction on 3.0s gaussian noise."""
    audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32) * 0.3
    feats = extract_features(audio, sr=SAMPLE_RATE)

    assert isinstance(feats, np.ndarray)
    assert feats.dtype == np.float32
    assert feats.shape == (178,)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))


def test_extract_features_sine_tone():
    """Verify 178-dim extraction on 440Hz periodic sine tone."""
    t = np.linspace(0, 2.0, SAMPLE_RATE * 2, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    feats = extract_features(audio, sr=SAMPLE_RATE)

    assert feats.shape == (178,)
    assert not np.any(np.isnan(feats))


def test_extract_features_silence():
    """Verify zero/silent audio does not produce NaNs or Infs."""
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    feats = extract_features(audio, sr=SAMPLE_RATE)

    assert feats.shape == (178,)
    assert not np.any(np.isnan(feats))
    assert not np.any(np.isinf(feats))
