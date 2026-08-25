"""
VoiceShield Tri-Modal Branch B: Higher-Order DSP & Bispectral Physics Module.
Extracts Quadratic Phase Coupling (Bispectrum), Cepstral Peak Prominence (CPP),
glottal opening quotient anomalies, and phase dispersion markers characteristic of
neural vocoder synthesis artifacts.
"""

import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.nn.functional as F

PHYSICS_RAW_DIM: int = 32
PHYSICS_EMBED_DIM: int = 64


def compute_bispectrum_coupling_metric(audio: np.ndarray, n_fft: int = 256) -> float:
    """
    Estimates Quadratic Phase Coupling (QPC) via the normalized bispectrum diagonal slice.
    Synthetic speech exhibits phase coherence breakdown across harmonically coupled poles.
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    # Compute STFT
    _, _, zxx = signal.stft(audio, nperseg=n_fft, noverlap=n_fft // 2, window="hann")
    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    # Simplified bispectrum proxy: Third-order moment expectation
    # B(f1, f2) = E[X(f1) * X(f2) * X*(f1 + f2)]
    # We compute the average diagonal coherence
    n_freqs = magnitude.shape[0]
    mid = n_freqs // 4
    if mid < 2:
        return 0.0

    x_f1 = zxx[1:mid, :]
    x_sum = zxx[2 : 2 * mid : 2, :]  # step 2 to match (mid - 1) shape

    bispec = np.abs(np.mean(x_f1 * x_f1 * np.conj(x_sum), axis=1))
    norm = np.mean((magnitude[1:mid, :] ** 2) * magnitude[2 : 2 * mid : 2, :], axis=1) + 1e-8
    bicoherence = float(np.mean(bispec / norm))
    return float(np.clip(bicoherence, 0.0, 1.0))


def compute_cepstral_peak_prominence(audio: np.ndarray, sr: int = 16000) -> float:
    """
    Computes Cepstral Peak Prominence (CPP).
    Measures the degree of harmonic organization relative to the background noise floor.
    """
    if len(audio) < 400:
        return 0.0

    frame = audio[:400] * np.hanning(400)
    spectrum = np.abs(np.fft.rfft(frame, n=512)) + 1e-8
    log_spec = np.log(spectrum)
    ceps = np.fft.irfft(log_spec)

    # Search for quefrency peak corresponding to f0 range (70Hz - 400Hz)
    min_quef = int(sr / 400.0)
    max_quef = int(sr / 70.0)

    if max_quef >= len(ceps):
        max_quef = len(ceps) - 1

    if min_quef >= max_quef:
        return 0.0

    search_region = ceps[min_quef:max_quef]
    peak_val = float(np.max(search_region))
    baseline_val = float(np.mean(ceps[min_quef:max_quef]))
    cpp = max(0.0, peak_val - baseline_val)
    return float(np.clip(cpp * 10.0, 0.0, 1.0))


def extract_raw_physics_vector(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts a 32-dimensional physics & phase feature vector from raw 1D audio.
    """
    if len(audio) == 0:
        return np.zeros(PHYSICS_RAW_DIM, dtype=np.float32)

    audio = np.nan_to_num(audio.flatten(), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    feats = []

    # 1. Bispectrum Phase Coupling Metric
    qpc = compute_bispectrum_coupling_metric(audio)
    feats.append(qpc)

    # 2. Cepstral Peak Prominence
    cpp = compute_cepstral_peak_prominence(audio, sr=sr)
    feats.append(cpp)

    # 3. Higher-order statistical moments of analytic signal envelope (Hilbert Transform)
    try:
        analytic = signal.hilbert(audio[:4000])
        env = np.abs(analytic)
        env_mean = float(np.mean(env))
        env_std = float(np.std(env))
        env_skew = float(np.mean((env - env_mean) ** 3) / (env_std ** 3 + 1e-6))
        env_kurt = float(np.mean((env - env_mean) ** 4) / (env_std ** 4 + 1e-6) - 3.0)
        feats.extend([env_mean, env_std, np.clip(env_skew, -5.0, 5.0), np.clip(env_kurt, -5.0, 10.0)])
    except Exception:
        feats.extend([0.0, 0.0, 0.0, 0.0])

    # 4. Phase slope index across subbands (group delay variations)
    fft_vals = np.fft.rfft(audio[:2048] if len(audio) >= 2048 else np.pad(audio, (0, 2048 - len(audio))))
    phase = np.unwrap(np.angle(fft_vals))
    group_delay = -np.diff(phase)
    gd_mean = float(np.mean(group_delay))
    gd_std = float(np.std(group_delay))
    feats.extend([gd_mean, gd_std])

    # 5. Glottal open quotient proxy: ratio of low-frequency spectral energy to total energy
    spec_mag = np.abs(fft_vals) ** 2
    f0_energy = float(np.sum(spec_mag[: int(len(spec_mag) * 0.1)]))
    total_energy = float(np.sum(spec_mag)) + 1e-8
    open_quotient_proxy = f0_energy / total_energy
    feats.append(float(np.clip(open_quotient_proxy, 0.0, 1.0)))

    # Pad or truncate to exact PHYSICS_RAW_DIM (32 dimensions)
    while len(feats) < PHYSICS_RAW_DIM:
        feats.append(0.0)

    return np.array(feats[:PHYSICS_RAW_DIM], dtype=np.float32)


class PhysicsEmbeddingProjector(nn.Module):
    """
    Projects higher-order DSP physics & phase vectors into a dense 64-dimensional embedding space.
    """

    def __init__(self, in_dim: int = PHYSICS_RAW_DIM, embed_dim: int = PHYSICS_EMBED_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 32)
        return self.net(x)
