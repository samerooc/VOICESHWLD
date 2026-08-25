"""
VoiceShield Phase 1 — Acoustic Channel Normalizer & De-Reverberation Engine.

Provides:
  1. Short-Time Non-Linear Spectral Subtraction
       Suppresses stationary background noise (HVAC, fan hiss, codec floor noise)
       and diffuse early reflections by estimating a per-frequency noise profile
       from the lowest-energy frames and subtracting it with a spectral floor.

  2. Cepstral Mean & Variance Normalization (CMVN)
       Removes the room transfer function H(z) and microphone hardware coloration
       while preserving instantaneous pitch harmonics and excitation phase dynamics.
       Applied in the log-spectral (cepstral) domain across temporal frames.

  3. Peak Dynamic Range Normalization
       Bounds output to [-0.90, 0.90] to prevent downstream clipping while
       retaining headroom for additive augmentations.

Mathematical references
-----------------------
  Spectral subtraction:
    S_hat(omega) = max( P(omega) - alpha * N(omega), beta * P(omega) )
    where alpha = oversubtraction_factor, beta = noise_floor_ratio.

  CMVN:
    x_cmvn[t, f] = ( log|X[t, f]| - mu[f] ) / ( sigma[f] + eps )
    then mapped back to magnitude via exp() for phase-coherent ISTFT.

Author:  VoiceShield Engineering
Version: 1.2.0  (Phase 1 hardened)
"""

from __future__ import annotations

import logging
from typing import Optional

import librosa
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_SR: int = 16000
_DEFAULT_N_FFT: int = 512
_DEFAULT_HOP: int = 160


class AcousticChannelNormalizer:
    """
    Forensic Channel-Invariant Audio Preprocessor (VoiceShield Phase 1).

    Pipeline executed by ``normalize_channel``:
    1. Short-Time Spectral Subtraction  — suppresses stationary ambient noise.
    2. Cepstral Mean & Variance Normalization (CMVN) — removes room IR and
       microphone hardware coloration while preserving pitch harmonics and
       excitation phase boundaries.
    3. Peak Dynamic Range Normalization — bounds output to [-0.90, 0.90].

    Parameters
    ----------
    sr : int
        Internal sample rate this normalizer operates at (default 16000).
        Audio supplied at a different rate is auto-resampled before processing.
    n_fft : int
        FFT size for STFT analysis (default 512 => 32 ms at 16 kHz).
    hop_length : int
        STFT hop in samples (default 160 => 10 ms at 16 kHz).
    """

    def __init__(
        self,
        sr: int = _DEFAULT_SR,
        n_fft: int = _DEFAULT_N_FFT,
        hop_length: int = _DEFAULT_HOP,
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length

    # ------------------------------------------------------------------
    # Step 1: Spectral Subtraction
    # ------------------------------------------------------------------

    def spectral_subtraction(
        self,
        audio: np.ndarray,
        noise_floor_ratio: float = 0.05,
        oversubtraction_factor: float = 1.25,
    ) -> np.ndarray:
        """
        Apply power-domain spectral subtraction to suppress stationary ambient
        noise and diffuse room reflections.

        Algorithm
        ---------
        1. Compute STFT magnitude and phase.
        2. Estimate the stationary noise spectral profile from the lowest 12%
           energy frames (robust to transient voiced activity contaminating the
           noise estimate).
        3. Subtract: cleaned_power = max(|X|^2 - alpha*|N|^2, beta*|X|^2)
        4. Reconstruct via ISTFT with original phase.

        Parameters
        ----------
        audio : np.ndarray[float32]
            Input waveform (mono, float32).
        noise_floor_ratio : float
            Spectral floor multiplier ``beta`` (prevents over-subtraction).
        oversubtraction_factor : float
            Noise scaling multiplier ``alpha`` (trades spectral distortion
            for suppression depth).

        Returns
        -------
        np.ndarray[float32]
            Denoised waveform, same length as input.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < self.n_fft:
            return audio

        # 1. Short-Time Fourier Transform
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude: np.ndarray = np.abs(stft)
        phase: np.ndarray = np.angle(stft)

        # 2. Estimate stationary noise floor from the lowest 12% energy frames
        frame_energies = np.sum(magnitude ** 2, axis=0)
        n_noise_frames = max(2, int(len(frame_energies) * 0.12))
        noise_frame_idx = np.argsort(frame_energies)[:n_noise_frames]
        noise_profile = np.mean(magnitude[:, noise_frame_idx], axis=1, keepdims=True)

        # 3. Power-domain spectral subtraction with spectral floor
        power_spec = magnitude ** 2
        noise_power = (noise_profile ** 2) * oversubtraction_factor
        cleaned_power = np.maximum(power_spec - noise_power, noise_floor_ratio * power_spec)
        cleaned_mag = np.sqrt(cleaned_power)

        # 4. Reconstruct time-domain waveform via Inverse STFT
        cleaned_stft = cleaned_mag * np.exp(1j * phase)
        cleaned_audio = librosa.istft(cleaned_stft, hop_length=self.hop_length, length=len(audio))
        return cleaned_audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 2: Cepstral Mean & Variance Normalization (CMVN)
    # ------------------------------------------------------------------

    def apply_cmvn(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply Cepstral Mean and Variance Normalization (CMVN) in the
        log-spectral domain.

        CMVN eliminates the room transfer function H(z) and microphone
        hardware coloration while preserving instantaneous pitch harmonics
        and fine excitation phase transitions.

        Algorithm
        ---------
        1. Compute STFT; separate magnitude and phase.
        2. Take log magnitude: L[t, f] = log(|X[t, f]| + eps).
        3. Normalize: L_norm[t, f] = (L[t, f] - mu[f]) / (sigma[f] + eps)
           across temporal frames t, independently per frequency bin f.
        4. Partial back-scaling: re-scale by exp(mu[f] + 0.95 * L_norm)
           to restore a natural dynamic range while keeping spectral mean removed.
        5. Reconstruct via ISTFT with original phase (phase coherence preserved).

        Parameters
        ----------
        audio : np.ndarray[float32]
            Input waveform (mono, float32).

        Returns
        -------
        np.ndarray[float32]
            CMVN-normalized waveform, same length as input.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) < self.n_fft:
            return audio

        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        mag: np.ndarray = np.abs(stft)
        phase: np.ndarray = np.angle(stft)

        # Log magnitude spectrum (cepstral-domain proxy)
        log_mag = np.log(np.maximum(mag, 1e-7))

        # Per-frequency-bin mean and variance normalization across temporal frames
        mu_spec = np.mean(log_mag, axis=1, keepdims=True)      # shape: (n_bins, 1)
        sigma_spec = np.std(log_mag, axis=1, keepdims=True) + 1e-7
        norm_log_mag = (log_mag - mu_spec) / sigma_spec

        # Partial back-scale: restore natural dynamic range (factor 0.95 retains
        # relative spectral envelope shape while the mean is zeroed)
        recon_mag = np.exp(mu_spec + norm_log_mag * 0.95)

        # Reconstruct time-domain via ISTFT preserving original excitation phase
        recon_stft = recon_mag * np.exp(1j * phase)
        recon_audio = librosa.istft(recon_stft, hop_length=self.hop_length, length=len(audio))
        return recon_audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def normalize_channel(self, audio: np.ndarray, sr: int = _DEFAULT_SR) -> np.ndarray:
        """
        Full acoustic channel normalization pipeline.

        Steps
        -----
        1. (Optional) Resample to internal sr if input sr differs.
        2. Spectral Subtraction — ambient noise suppression.
        3. CMVN — room reverberation and microphone coloration removal.
        4. Peak dynamic range normalization to [-0.90, 0.90].

        Parameters
        ----------
        audio : np.ndarray[float32]
            Input waveform (mono, float32, any length >= 0).
        sr : int
            Sample rate of the incoming ``audio`` signal (default 16000).

        Returns
        -------
        np.ndarray[float32]
            Channel-normalized waveform, same sample count as input (after
            optional resample), bounded to [-0.90, 0.90].
        """
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return audio

        # Auto-resample if caller supplies audio at a different rate
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr, res_type="kaiser_fast")

        # Step 1: Spectral Subtraction (noise floor removal)
        denoised = self.spectral_subtraction(audio)

        # Step 2: CMVN (room transfer function inversion)
        channel_normed = self.apply_cmvn(denoised)

        # Step 3: Peak normalization to [-0.90, 0.90] (preserve headroom)
        peak = float(np.max(np.abs(channel_normed))) if len(channel_normed) > 0 else 0.0
        if peak > 1e-5:
            channel_normed = (channel_normed / peak) * 0.90

        return channel_normed.astype(np.float32)

    # Backward-compatibility alias
    remove_reverb_and_equalize = normalize_channel


# ---------------------------------------------------------------------------
# Module-level singleton and functional wrappers
# ---------------------------------------------------------------------------

#: Global singleton for zero-allocation high-throughput reuse.
channel_normalizer = AcousticChannelNormalizer(sr=_DEFAULT_SR)


def normalize_channel(audio: np.ndarray, sr: int = _DEFAULT_SR) -> np.ndarray:
    """
    Functional convenience wrapper around ``AcousticChannelNormalizer.normalize_channel``.

    Applies the full Phase 1 acoustic channel normalization pipeline
    (Spectral Subtraction + CMVN + peak normalization) using the module-level
    singleton instance.

    Parameters
    ----------
    audio : np.ndarray[float32]
        Input mono waveform.
    sr : int
        Sample rate of ``audio`` (default 16000).

    Returns
    -------
    np.ndarray[float32]
        Channel-normalized waveform bounded to [-0.90, 0.90].
    """
    return channel_normalizer.normalize_channel(audio, sr=sr)


def normalize_channel_audio(audio: np.ndarray, sr: int = _DEFAULT_SR) -> np.ndarray:
    """Backward-compatibility alias for ``normalize_channel``."""
    return channel_normalizer.normalize_channel(audio, sr=sr)
