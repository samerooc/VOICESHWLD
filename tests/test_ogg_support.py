"""
VoiceShield OGG, OPUS, and Compressed Audio Decoding & Analysis Tests.
Verifies seamless in-memory ingestion, feature extraction, and risk scoring for OGG files.
"""

import io
import pytest
import numpy as np
import soundfile as sf
from src.audio_io import load_audio_from_bytes, get_audio_metadata
from src.features import extract_features_from_audio
from src.scoring import predict_and_score
from src.model_registry import verify_and_load_model


def test_ogg_vorbis_in_memory_decoding_and_analysis():
    # 1. Synthesize audio waveform
    y = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, y, 16000, format="OGG", subtype="VORBIS")
    ogg_bytes = bio.getvalue()

    # 2. Decode from bytes
    audio, sr = load_audio_from_bytes(ogg_bytes, target_sr=16000, file_ext=".ogg")
    assert audio is not None
    assert len(audio) > 0
    assert sr == 16000

    # 3. Extract features
    features = extract_features_from_audio(audio, sample_rate=sr)
    assert features.shape in [(42,), (178,)]
    assert np.all(np.isfinite(features))

    # 4. Predict and score
    model, _ = verify_and_load_model("models/voice_detector.pkl", "models/model_metadata.json")
    res = predict_and_score(model, audio, sample_rate=sr)
    assert "prediction_label" in res
    assert "spoof_probability" in res
    assert "risk_band" in res
    assert res["bona_fide_probability"] + res["spoof_probability"] == pytest.approx(1.0, 1e-4)


def test_ogg_stereo_to_mono_and_resampling():
    stereo = np.random.uniform(-0.5, 0.5, (48000, 2)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, stereo, 48000, format="OGG", subtype="VORBIS")
    ogg_bytes = bio.getvalue()

    audio, sr = load_audio_from_bytes(ogg_bytes, target_sr=16000, file_ext=".ogg")
    assert audio.ndim == 1
    assert sr == 16000
    assert pytest.approx(len(audio), abs=100) == 16000
