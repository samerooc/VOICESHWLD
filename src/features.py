"""
VoiceShield Feature Extraction Module (Phase 1).
Extracts 42 MFCC, RMS Energy, and Zero Crossing Rate features with fixed vector shape.
"""

from typing import Optional
import librosa
import numpy as np

from src.audio_io import load_audio_from_file
from src.config import N_MFCC, SAMPLE_RATE, TOTAL_FEATURES


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extracts 42-element acoustic feature vector from normalized audio:
      - 20 MFCC means
      - 20 MFCC standard deviations
      - 1 RMS Energy mean
      - 1 Zero Crossing Rate mean

    Args:
        audio: 1D numpy array of audio signal.
        sample_rate: Sampling rate in Hz (default: 16000).

    Returns:
        1D float32 numpy array of shape (42,).

    Raises:
        ValueError: If audio signal is empty or invalid.
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Cannot extract features from empty audio signal.")

    # Remove DC offset and normalize amplitude
    audio_clean = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    audio_clean = audio_clean - np.mean(audio_clean)
    max_amp = np.max(np.abs(audio_clean))
    if max_amp > 1e-6:
        audio_clean = audio_clean / (max_amp + 1e-9)

    # Trim non-speech leading/trailing silence
    try:
        trimmed, _ = librosa.effects.trim(audio_clean, top_db=30)
        if len(trimmed) >= int(sample_rate * 0.4):
            audio_clean = trimmed
    except Exception:
        pass

    if not np.any(audio_clean):
        # Return zero feature vector if signal is entirely silent or NaN
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    # 1. MFCC Features (20 coefficients)
    mfccs = librosa.feature.mfcc(
        y=audio_clean,
        sr=sample_rate,
        n_mfcc=N_MFCC,
    )
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)

    # 2. RMS Energy
    rms_arr = librosa.feature.rms(y=audio_clean)
    rms = float(np.mean(rms_arr))

    # 3. Zero Crossing Rate
    zcr_arr = librosa.feature.zero_crossing_rate(y=audio_clean)
    zcr = float(np.mean(zcr_arr))

    # 4. Concatenate into fixed 42-dimensional vector
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [rms, zcr],
    ])

    # 5. Sanitize output vector against NaNs
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if len(features) != TOTAL_FEATURES:
        raise ValueError(f"Feature dimension mismatch: expected {TOTAL_FEATURES}, got {len(features)}.")

    return features


def extract_segmented_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_duration: float = 2.5,
    hop_duration: float = 1.0,
) -> list:
    """
    Extracts 42-dimensional acoustic feature vectors across sliding temporal windows.
    Enables highly robust multi-segment ensemble analysis on uploaded audio files.
    """
    if audio is None or len(audio) == 0:
        return []

    window_samples = int(sample_rate * window_duration)
    hop_samples = int(sample_rate * hop_duration)

    if len(audio) <= window_samples:
        return [extract_features_from_audio(audio, sample_rate)]

    feature_list = []
    for start in range(0, len(audio) - window_samples + 1, hop_samples):
        chunk = audio[start : start + window_samples]
        if len(chunk) == window_samples and np.max(np.abs(chunk)) > 1e-4:
            feat = extract_features_from_audio(chunk, sample_rate)
            feature_list.append(feat)

    if not feature_list:
        feature_list.append(extract_features_from_audio(audio, sample_rate))

    return feature_list


def extract_features_from_file(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Convenience function: loads WAV file from disk and extracts 42 features.
    """
    audio, sr = load_audio_from_file(file_path, target_sr=sample_rate)
    return extract_features_from_audio(audio, sample_rate=sr)
