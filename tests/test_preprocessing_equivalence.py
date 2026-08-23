"""
VoiceShield Preprocessing Equivalence Tests (Section D).
Verifies that preprocessing is mathematically identical across training and inference.
"""

import pytest
import numpy as np
from src.preprocessing_contract import preprocess_audio_signal
from src.preprocessing import preprocess_audio


def test_preprocessing_equivalence_on_synthetic_signal():
    np.random.seed(42)
    sig = np.random.uniform(-0.8, 0.8, 48000).astype(np.float32)

    clean_1, sr_1, _ = preprocess_audio_signal(sig, sample_rate=48000, target_sr=16000)
    clean_2, sr_2, _ = preprocess_audio(sig, sample_rate=48000, target_sr=16000)

    assert sr_1 == sr_2 == 16000
    np.testing.assert_allclose(clean_1, clean_2, atol=1e-4)
