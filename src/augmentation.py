"""
VoiceShield Production Acoustic & Telephony Data Augmentation Engine (Step 3).
Simulates real-world acoustic degradations, transmission channels, and codec distortions:
  1. Telephony Simulation: G.711 A-law / mu-law compression, 8kHz downsample/upsample, 300Hz-3400Hz bandpass
  2. Environmental Noise: Additive Gaussian white and 1/f pink noise (10dB to 30dB SNR)
  3. Room Impulse Response (RIR): Synthetic reverberation with exponential decay
  4. Codec & Non-linear Distortion: Subband quantization, mild peak clipping, and gain variation (+/- 3dB)
"""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import librosa
import numpy as np
from scipy.signal import butter, sosfilt


# =============================================================================
# 1. Telephony & Codec Channel Simulation
# =============================================================================
def apply_bandpass_filter(
    audio: np.ndarray,
    sr: int = 16000,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 5,
) -> np.ndarray:
    """
    Applies a standard Butterworth bandpass filter simulating PSTN / Telephony bandwidth (300Hz - 3400Hz).
    """
    if len(audio) < order * 4 or not np.any(audio):
        return audio

    nyq = 0.5 * sr
    low = max(20.0, lowcut) / nyq
    high = min(nyq - 10.0, highcut) / nyq

    if low >= high:
        return audio

    try:
        sos = butter(order, [low, high], btype="bandpass", output="sos")
        filtered = sosfilt(sos, audio).astype(np.float32)
        return np.nan_to_num(filtered, nan=0.0)
    except Exception:
        return audio


def apply_g711_compression(
    audio: np.ndarray,
    mode: str = "mulaw",
    mu: int = 255,
) -> np.ndarray:
    """
    Simulates ITU-T G.711 mu-law or A-law dynamic range companding / 8-bit quantization.
    """
    if not np.any(audio):
        return audio

    max_val = np.max(np.abs(audio)) + 1e-9
    norm_audio = audio / max_val

    if mode == "alaw":
        A = 87.6
        abs_x = np.abs(norm_audio)
        sign_x = np.sign(norm_audio)
        compressed = np.where(
            abs_x < (1.0 / A),
            (A * abs_x) / (1.0 + np.log(A)),
            (1.0 + np.log(A * abs_x)) / (1.0 + np.log(A)),
        ) * sign_x
        quantized = np.round((compressed + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
        abs_q = np.abs(quantized)
        expanded = np.where(
            abs_q < (1.0 / (1.0 + np.log(A))),
            (abs_q * (1.0 + np.log(A))) / A,
            np.exp(abs_q * (1.0 + np.log(A)) - 1.0) / A,
        ) * np.sign(quantized)
        return (expanded * max_val).astype(np.float32)

    # Default mu-law
    mu_float = float(mu)
    compressed = np.sign(norm_audio) * np.log1p(mu_float * np.abs(norm_audio)) / np.log1p(mu_float)
    quantized = np.round((compressed + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0
    expanded = np.sign(quantized) * (1.0 / mu_float) * ((1.0 + mu_float) ** np.abs(quantized) - 1.0)
    return (expanded * max_val).astype(np.float32)


def apply_8k_telephony_resampling(
    audio: np.ndarray,
    orig_sr: int = 16000,
) -> np.ndarray:
    """
    Downsamples to 8,000 Hz telephony standard and upsamples back to 16,000 Hz.
    """
    if len(audio) < 128 or not np.any(audio):
        return audio
    try:
        downsampled = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=8000)
        upsampled = librosa.resample(y=downsampled, orig_sr=8000, target_sr=orig_sr)
        return upsampled.astype(np.float32)
    except Exception:
        return audio


def simulate_telephony(
    audio: np.ndarray,
    sr: int = 16000,
) -> np.ndarray:
    """
    Complete PSTN / Telephony simulation:
      1. 8kHz downsampling/upsampling
      2. 300Hz - 3400Hz bandpass filtering
      3. G.711 mu-law compression
    """
    resampled = apply_8k_telephony_resampling(audio, orig_sr=sr)
    filtered = apply_bandpass_filter(resampled, sr=sr, lowcut=300.0, highcut=3400.0)
    compressed = apply_g711_compression(filtered, mode="mulaw")
    return compressed


# =============================================================================
# 2. Environmental Noise & Babble Injection
# =============================================================================
def add_gaussian_noise(
    audio: np.ndarray,
    target_snr_db: Optional[float] = None,
) -> np.ndarray:
    """
    Injects additive white Gaussian noise at a randomized or specified SNR (10dB to 30dB).
    """
    if not np.any(audio):
        return audio

    if target_snr_db is None:
        target_snr_db = random.uniform(10.0, 30.0)

    signal_power = np.mean(audio**2) + 1e-12
    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))

    noise = np.random.normal(0, np.sqrt(target_noise_power), len(audio)).astype(np.float32)
    augmented = audio + noise
    return np.nan_to_num(augmented).astype(np.float32)


def add_pink_noise(
    audio: np.ndarray,
    target_snr_db: Optional[float] = None,
) -> np.ndarray:
    """
    Injects 1/f Pink noise simulating natural ambient room background.
    """
    if not np.any(audio):
        return audio

    if target_snr_db is None:
        target_snr_db = random.uniform(12.0, 28.0)

    n_samples = len(audio)
    white = np.random.randn(n_samples)
    fft_white = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n_samples) + 1e-6
    fft_pink = fft_white / np.sqrt(frequencies)
    pink_noise = np.fft.irfft(fft_pink, n=n_samples).astype(np.float32)

    signal_power = np.mean(audio**2) + 1e-12
    noise_power = np.mean(pink_noise**2) + 1e-12
    scale = np.sqrt(signal_power / (noise_power * (10.0 ** (target_snr_db / 10.0))))

    augmented = audio + (pink_noise * scale)
    return np.nan_to_num(augmented).astype(np.float32)


# =============================================================================
# 3. Room Impulse Response (RIR) & Synthetic Reverberation
# =============================================================================
def apply_synthetic_reverb(
    audio: np.ndarray,
    sr: int = 16000,
    decay_time: float = 0.3,
    wet_ratio: float = 0.25,
) -> np.ndarray:
    """
    Simulates room acoustics using an exponentially decaying stochastic impulse response.
    """
    if len(audio) < 256 or not np.any(audio):
        return audio

    impulse_len = int(sr * decay_time)
    if impulse_len <= 0:
        return audio

    time_vec = np.linspace(0, decay_time, impulse_len)
    decay = np.exp(-3.0 * time_vec / max(1e-4, decay_time))
    rir = np.random.randn(impulse_len) * decay
    rir[0] = 1.0
    rir = rir / (np.linalg.norm(rir) + 1e-9)

    reverbed = np.convolve(audio, rir, mode="same").astype(np.float32)
    mixed = (1.0 - wet_ratio) * audio + wet_ratio * reverbed
    return np.nan_to_num(mixed).astype(np.float32)


# =============================================================================
# 4. Gain Scaling & Peak Saturation
# =============================================================================
def apply_gain_variation(
    audio: np.ndarray,
    gain_db: Optional[float] = None,
) -> np.ndarray:
    """
    Applies random amplitude gain scaling within +/- 3 dB.
    """
    if not np.any(audio):
        return audio
    if gain_db is None:
        gain_db = random.uniform(-3.0, 3.0)

    scale_factor = 10.0 ** (gain_db / 20.0)
    scaled = audio * scale_factor
    return np.nan_to_num(scaled).astype(np.float32)


def apply_mild_clipping(
    audio: np.ndarray,
    clip_percentile: float = 98.0,
) -> np.ndarray:
    """
    Simulates microphone pre-amp saturation and soft clipping.
    """
    if not np.any(audio):
        return audio

    threshold = float(np.percentile(np.abs(audio), clip_percentile))
    if threshold < 1e-4:
        return audio

    clipped = np.clip(audio, -threshold, threshold)
    return clipped.astype(np.float32)


# =============================================================================
# 5. High-Level Composable Augmentation Pipeline
# =============================================================================
class AcousticAugmenter:
    """
    Production acoustic augmentation pipeline with randomized degradation selection.
    """

    def __init__(
        self,
        p_telephony: float = 0.35,
        p_noise: float = 0.40,
        p_reverb: float = 0.30,
        p_clipping: float = 0.20,
        p_gain: float = 0.30,
        sr: int = 16000,
    ):
        self.p_telephony = p_telephony
        self.p_noise = p_noise
        self.p_reverb = p_reverb
        self.p_clipping = p_clipping
        self.p_gain = p_gain
        self.sr = sr

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Applies a randomized sequence of acoustic degradations."""
        if audio is None or len(audio) == 0:
            return audio

        aug = audio.copy().astype(np.float32)

        # 1. Gain scaling
        if random.random() < self.p_gain:
            aug = apply_gain_variation(aug)

        # 2. Telephony simulation
        if random.random() < self.p_telephony:
            aug = simulate_telephony(aug, sr=self.sr)

        # 3. Room reverberation
        if random.random() < self.p_reverb:
            decay = random.uniform(0.15, 0.40)
            wet = random.uniform(0.15, 0.35)
            aug = apply_synthetic_reverb(aug, sr=self.sr, decay_time=decay, wet_ratio=wet)

        # 4. Ambient noise
        if random.random() < self.p_noise:
            snr = random.uniform(10.0, 30.0)
            if random.random() < 0.5:
                aug = add_gaussian_noise(aug, target_snr_db=snr)
            else:
                aug = add_pink_noise(aug, target_snr_db=snr)

        # 5. Microphone peak saturation
        if random.random() < self.p_clipping:
            pct = random.uniform(95.0, 99.0)
            aug = apply_mild_clipping(aug, clip_percentile=pct)

        # Normalization
        max_val = np.max(np.abs(aug))
        if max_val > 1e-5:
            aug = aug / (max_val + 1e-9)

        return np.nan_to_num(aug, nan=0.0).astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sr: int = 16000,
    augmentation_type: str = "random",
) -> np.ndarray:
    """
    Unified entrypoint wrapper for acoustic data augmentation.
    """
    if augmentation_type == "telephony":
        return simulate_telephony(audio, sr=sr)
    elif augmentation_type == "noise":
        return add_gaussian_noise(audio)
    elif augmentation_type == "pinknoise":
        return add_pink_noise(audio)
    elif augmentation_type == "reverb":
        return apply_synthetic_reverb(audio, sr=sr)
    elif augmentation_type == "clipping":
        return apply_mild_clipping(audio)
    elif augmentation_type == "gain":
        return apply_gain_variation(audio)
    else:
        augmenter = AcousticAugmenter(sr=sr)
        return augmenter.augment(audio)
