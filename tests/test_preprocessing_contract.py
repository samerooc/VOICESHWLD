"""
VoiceShield Preprocessing Contract Tests.
Verifies mono downmixing, resampling, DC removal, peak normalization, and diagnostics.
"""

import pytest
import numpy as np
from src.preprocessing_contract import preprocess_audio_signal


def test_preprocessing_mono_conversion():
    stereo = np.random.uniform(-0.5, 0.5, (32000, 2)).astype(np.float32)
    mono, sr, _ = preprocess_audio_signal(stereo, sample_rate=16000)
    assert mono.ndim == 1
    assert len(mono) == 32000
    assert sr == 16000


def test_preprocessing_resampling():
    audio_44k = np.random.uniform(-0.5, 0.5, 44100).astype(np.float32)
    resampled, sr, _ = preprocess_audio_signal(audio_44k, sample_rate=44100, target_sr=16000)
    assert sr == 16000
    assert pytest.approx(len(resampled), abs=50) == 16000


def test_preprocessing_dc_offset_removal():
    sig = np.sin(np.linspace(0, 10, 16000)) + 0.5  # Heavy DC offset
    clean, _, _ = preprocess_audio_signal(sig, sample_rate=16000)
    assert pytest.approx(float(np.mean(clean)), abs=1e-5) == 0.0


def test_preprocessing_peak_normalization():
    sig = np.sin(np.linspace(0, 10, 16000)) * 0.25
    normalized, _, diag = preprocess_audio_signal(sig, sample_rate=16000, normalize_amplitude=True)
    assert pytest.approx(diag["peak_amplitude"], abs=1e-3) == 1.0


def test_preprocessing_empty_input_rejection():
    with pytest.raises(ValueError, match="empty or None"):
        preprocess_audio_signal(np.array([]))
