"""
VoiceShield Phase 1: High-Dimensional Acoustic Feature Extractor (178 Dimensions).
Extracts Linear Frequency Cepstral Coefficients (LFCCs + Deltas + Double Deltas),
MFCC dynamics, Praat Glottal Jitter/Shimmer/HNR, and Spectral/Energy coherence markers.
"""

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
from scipy.fftpack import dct

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_io import load_audio_from_file
from src.config import N_MFCC, SAMPLE_RATE

TOTAL_DIMENSIONS: int = 178


def compute_lfcc_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_lfcc: int = 20,
    n_filterbanks: int = 30,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Computes 20 Linear Frequency Cepstral Coefficients (LFCCs), deltas, and delta-deltas
    across linear frequency filterbanks (120 dimensions: 60 means + 60 standard deviations).
    """
    if len(audio) < win_length:
        pad_len = win_length - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    # 1. Linear Magnitude Spectrogram
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window="hann")) ** 2
    num_freq_bins = stft.shape[0]

    # 2. Construct Linear Filterbank Matrix
    freq_bins = np.linspace(0, sr / 2, num_freq_bins)
    filter_centers = np.linspace(0, sr / 2, n_filterbanks + 2)
    filterbank = np.zeros((n_filterbanks, num_freq_bins), dtype=np.float32)

    for i in range(n_filterbanks):
        f_left, f_center, f_right = filter_centers[i], filter_centers[i + 1], filter_centers[i + 2]
        left_slope = (freq_bins - f_left) / (f_center - f_left + 1e-8)
        right_slope = (f_right - freq_bins) / (f_right - f_center + 1e-8)
        filterbank[i] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    # 3. Energy Aggregation & Discrete Cosine Transform (DCT-II)
    bank_energies = np.dot(filterbank, stft)
    log_energies = np.log(np.maximum(bank_energies, 1e-10))
    lfcc_raw = dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc]

    # 4. Deltas and Double-Deltas
    delta_1 = librosa.feature.delta(lfcc_raw, order=1)
    delta_2 = librosa.feature.delta(lfcc_raw, order=2)

    lfcc_matrix = np.vstack([lfcc_raw, delta_1, delta_2])  # (60, n_frames)

    means = np.mean(lfcc_matrix, axis=1)
    stds = np.std(lfcc_matrix, axis=1)

    return np.concatenate([means, stds]).astype(np.float32)  # 120 dims


def compute_mfcc_dynamics(audio: np.ndarray, sr: int = SAMPLE_RATE, n_mfcc: int = 20) -> np.ndarray:
    """
    Computes 20 MFCCs (means and standard deviations across frames = 40 dimensions).
    """
    if len(audio) < 400:
        pad_len = 400 - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    means = np.mean(mfccs, axis=1)
    stds = np.std(mfccs, axis=1)

    return np.concatenate([means, stds]).astype(np.float32)  # 40 dims


def compute_glottal_praat_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extracts Glottal Micro-Jitter, Shimmer, Pitch (F0), and Harmonics-to-Noise Ratio (HNR)
    via Praat / DSP biomechanical analysis (6 dimensions).
    """
    if len(audio) == 0 or np.all(audio == 0):
        return np.zeros(6, dtype=np.float32)

    try:
        from src.forensic_dsp import extract_praat_glottal_metrics
        bio = extract_praat_glottal_metrics(audio, sr=sr)
        f0_mean = float(bio.get("f0_mean", 150.0))
        f0_std = float(bio.get("f0_std", 20.0))
        f0_range = float(f0_std * 2.5)
        jitter_val = float(bio.get("jitter_local", 0.015))
        shimmer_val = float(bio.get("shimmer_local", 0.035))
        hnr_val = float(bio.get("hnr_db", 12.0))

        return np.array([f0_mean, f0_std, f0_range, jitter_val, shimmer_val, hnr_val], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)


def compute_spectral_coherence_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Computes Spectral Centroid, Flatness, Roll-off (85%), and Contrast (8 dimensions).
    """
    if len(audio) < 400:
        pad_len = 400 - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    # Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=160)
    cent_mean, cent_std = float(np.mean(cent)), float(np.std(cent))

    # Spectral Flatness
    flat = librosa.feature.spectral_flatness(y=audio, n_fft=512, hop_length=160)
    flat_mean, flat_std = float(np.mean(flat)), float(np.std(flat))

    # Spectral Roll-off (85% energy)
    roll = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85, n_fft=512, hop_length=160)
    roll_mean, roll_std = float(np.mean(roll)), float(np.std(roll))

    # Spectral Contrast
    cont = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=512, hop_length=160)
    cont_mean, cont_std = float(np.mean(cont)), float(np.std(cont))

    return np.array([
        cent_mean, cent_std,
        flat_mean, flat_std,
        roll_mean, roll_std,
        cont_mean, cont_std,
    ], dtype=np.float32)  # 8 dims


def compute_energy_dynamics_features(audio: np.ndarray) -> np.ndarray:
    """
    Computes RMS Energy (mean, std) and Zero Crossing Rate (mean, std) (4 dimensions).
    """
    if len(audio) < 400:
        pad_len = 400 - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")

    rms = librosa.feature.rms(y=audio, frame_length=400, hop_length=160)
    rms_mean, rms_std = float(np.mean(rms)), float(np.std(rms))

    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=400, hop_length=160)
    zcr_mean, zcr_std = float(np.mean(zcr)), float(np.std(zcr))

    return np.array([rms_mean, rms_std, zcr_mean, zcr_std], dtype=np.float32)  # 4 dims


def extract_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    sample_rate: Optional[int] = None,
) -> np.ndarray:
    """
    Extracts the full 178-dimensional acoustic representation vector from raw audio:
      - 120 LFCCs (Static, Delta, Delta-Delta Mean & Std)
      - 40 MFCCs (Mean & Std)
      - 6 Praat Glottal Micro-Jitter / Shimmer / HNR
      - 8 Spectral Coherence & Envelope Markers
      - 4 Energy & Temporal Dynamics Markers
    """
    effective_sr = sample_rate or sr or SAMPLE_RATE
    if audio is None or len(audio) == 0:
        return np.zeros(TOTAL_DIMENSIONS, dtype=np.float32)

    # 1. LFCC (120 dims)
    lfcc_feats = compute_lfcc_features(audio, sr=effective_sr, n_lfcc=20)

    # 2. MFCC (40 dims)
    mfcc_feats = compute_mfcc_dynamics(audio, sr=effective_sr, n_mfcc=20)

    # 3. Praat Glottal Jitter / Shimmer (6 dims)
    praat_feats = compute_glottal_praat_features(audio, sr=effective_sr)

    # 4. Spectral Coherence (8 dims)
    spec_feats = compute_spectral_coherence_features(audio, sr=effective_sr)

    # 5. Energy Dynamics (4 dims)
    energy_feats = compute_energy_dynamics_features(audio)

    # Concatenate to 178 dimensions
    feature_vector = np.concatenate([
        lfcc_feats,
        mfcc_feats,
        praat_feats,
        spec_feats,
        energy_feats,
    ]).astype(np.float32)

    # Final sanitization
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return feature_vector


def get_feature_names(mode: Optional[str] = None) -> List[str]:
    """
    Returns ordered, descriptive feature names for all dimensions (178 by default).
    Supports mode='legacy' (42), mode='step1' (187), or default (178).
    """
    if mode == "legacy":
        names = []
        for i in range(1, 21):
            names.append(f"mfcc_{i:02d}_mean")
        for i in range(1, 21):
            names.append(f"mfcc_{i:02d}_std")
        names.extend(["rms_energy_mean", "zero_crossing_rate_mean"])
        return names

    names = []

    # 1. LFCC Means (20 static + 20 delta + 20 delta2 = 60)
    for i in range(1, 21):
        names.append(f"lfcc_{i:02d}_mean")
    for i in range(1, 21):
        names.append(f"lfcc_delta_{i:02d}_mean")
    for i in range(1, 21):
        names.append(f"lfcc_delta2_{i:02d}_mean")

    # 2. LFCC Stds (20 static + 20 delta + 20 delta2 = 60)
    for i in range(1, 21):
        names.append(f"lfcc_{i:02d}_std")
    for i in range(1, 21):
        names.append(f"lfcc_delta_{i:02d}_std")
    for i in range(1, 21):
        names.append(f"lfcc_delta2_{i:02d}_std")

    # 3. MFCC Means & Stds (40)
    for i in range(1, 21):
        names.append(f"mfcc_{i:02d}_mean")
    for i in range(1, 21):
        names.append(f"mfcc_{i:02d}_std")

    # 4. Glottal Praat Dynamics (6)
    names.extend([
        "glottal_f0_mean",
        "glottal_f0_std",
        "glottal_f0_pitch_range",
        "glottal_jitter_local",
        "glottal_shimmer_local",
        "glottal_hnr_mean",
    ])

    # 5. Spectral Coherence (8)
    names.extend([
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_flatness_mean",
        "spectral_flatness_std",
        "spectral_rolloff_85_mean",
        "spectral_rolloff_85_std",
        "spectral_contrast_mean",
        "spectral_contrast_std",
    ])

    # 6. Energy & Temporal Dynamics (4)
    names.extend([
        "rms_energy_mean",
        "rms_energy_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ])

    return names


def extract_features_from_audio(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    mode: str = "full",
) -> np.ndarray:
    """
    Extracts acoustic feature vector from audio array.
    Supports mode='legacy' (42 dims) or mode='full'/'advanced' (178 dims).
    """
    if audio is None or len(audio) == 0:
        if mode == "legacy":
            return np.zeros(42, dtype=np.float32)
        elif mode in ("extended", "phase1", "77"):
            return np.zeros(77, dtype=np.float32)
        return np.zeros(TOTAL_DIMENSIONS, dtype=np.float32)

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    if mode == "legacy":
        # 20 MFCC means + 20 MFCC stds + 1 RMS + 1 ZCR = 42 dims
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20, n_fft=512, hop_length=160)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        rms = float(np.mean(librosa.feature.rms(y=audio, frame_length=400, hop_length=160)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio, frame_length=400, hop_length=160)))
        return np.concatenate([mfcc_means, mfcc_stds, [rms, zcr]]).astype(np.float32)

    if mode in ("extended", "phase1", "77"):
        return extract_extended_features(audio, sample_rate=sample_rate)

    return extract_features(audio, sr=sample_rate)


def extract_features_from_file(
    file_path: str,
    target_sr: int = SAMPLE_RATE,
    mode: str = "full",
) -> np.ndarray:
    """Loads audio from file and extracts acoustic features."""
    audio, sr = load_audio_from_file(file_path, target_sr=target_sr)
    return extract_features_from_audio(audio, sample_rate=sr, mode=mode)


def extract_segmented_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_duration: float = 2.5,
    hop_duration: float = 1.0,
    mode: str = "full",
) -> List[np.ndarray]:
    """
    Slices audio into sliding windows and extracts feature vectors for each window.
    """
    if len(audio) == 0:
        return []

    win_samples = int(window_duration * sample_rate)
    hop_samples = int(hop_duration * sample_rate)

    if len(audio) <= win_samples:
        return [extract_features_from_audio(audio, sample_rate=sample_rate, mode=mode)]

    segments = []
    start = 0
    while start + win_samples <= len(audio):
        chunk = audio[start : start + win_samples]
        feats = extract_features_from_audio(chunk, sample_rate=sample_rate, mode=mode)
        segments.append(feats)
        start += hop_samples

    return segments if segments else [extract_features_from_audio(audio, sample_rate=sample_rate, mode=mode)]


def extract_extended_features(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> np.ndarray:
    """Extracts 77-dimensional acoustic feature vector."""
    target_sr = sample_rate or sr or SAMPLE_RATE
    raw_feats = extract_features(audio, sr=target_sr)
    if len(raw_feats) >= 77:
        return raw_feats[:77].astype(np.float32)
    padded = np.zeros(77, dtype=np.float32)
    padded[: len(raw_feats)] = raw_feats
    return padded


def extract_high_frequency_artifacts(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> np.ndarray:
    """Extracts high-frequency spectral artifacts (rolloff, flatness, high-band energy ratio)."""
    target_sr = sample_rate or sr or SAMPLE_RATE
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=target_sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    stft = np.abs(librosa.stft(audio, n_fft=512)) ** 2
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=512)
    hf_mask = freqs >= 5500
    hf_energy = np.sum(stft[hf_mask, :])
    tot_energy = np.sum(stft) + 1e-10
    hf_ratio = float(hf_energy / tot_energy)
    return np.array([rolloff, flatness, hf_ratio], dtype=np.float32)


def extract_mfcc_features(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
    n_mfcc: int = 20,
) -> np.ndarray:
    target_sr = sample_rate or sr or SAMPLE_RATE
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    mfccs = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    return np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)]).astype(np.float32)


def extract_pitch_and_jitter(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> Dict[str, float]:
    target_sr = sample_rate or sr or SAMPLE_RATE
    from src.forensic_dsp import extract_praat_glottal_metrics
    bio = extract_praat_glottal_metrics(audio, sr=target_sr)
    return {
        "f0_mean": float(bio.get("f0_mean", 220.0)),
        "f0_std": float(bio.get("f0_std", 15.0)),
        "jitter_local": float(bio.get("jitter_local", 0.015)),
    }


def extract_shimmer_and_hnr(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> Dict[str, float]:
    target_sr = sample_rate or sr or SAMPLE_RATE
    from src.forensic_dsp import extract_praat_glottal_metrics
    bio = extract_praat_glottal_metrics(audio, sr=target_sr)
    return {
        "shimmer_local": float(bio.get("shimmer_local", 0.03)),
        "hnr_mean": float(bio.get("hnr_db", 14.0)),
    }


def extract_spectral_dynamics(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> Dict[str, float]:
    target_sr = sample_rate or sr or SAMPLE_RATE
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=target_sr, roll_percent=0.85)))
    spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
    flux = float(np.mean(np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0)))) if spec.shape[1] > 1 else 0.05
    return {
        "spectral_flatness_mean": flatness,
        "spectral_flux_mean": flux,
        "spectral_rolloff_85_mean": rolloff,
    }


def extract_prosody_timing(
    audio: np.ndarray,
    sample_rate: Optional[int] = None,
    sr: Optional[int] = None,
) -> Dict[str, float]:
    target_sr = sample_rate or sr or SAMPLE_RATE
    if len(audio) < 400:
        audio = np.pad(audio, (0, 400 - len(audio)))
    hop = int(target_sr * 0.02)
    win = int(target_sr * 0.04)
    frames = [audio[i : i + win] for i in range(0, len(audio) - win, hop)]
    energies = [np.sqrt(np.mean(f ** 2)) for f in frames] if frames else [0.1]
    is_silent = [1 if e < 0.01 else 0 for e in energies]
    pause_count = float(sum(1 for i in range(1, len(is_silent)) if is_silent[i] == 1 and is_silent[i - 1] == 0))
    return {
        "pause_count": max(1.0, pause_count),
        "speech_rate": float(len(audio) / target_sr),
    }
