"""
VoiceShield Phase 2 — Biomechanical Glottal Dynamics & LFCC Filterbank Analyzer.

Extracts three independent forensic channels from voiced speech to distinguish
authentic human biomechanics from neural-vocoder synthesis artefacts:

  1. Voiced-Only Glottal Perturbation  (Praat Parselmouth)
       Tracks F₀ strictly between 75 – 500 Hz on voiced frames (F₀ > 0).
       Measures Local Jitter, Local Shimmer, and HNR.
       Natural human speech: jitter ∈ [0.006, 0.022] from random vocal-fold
       biomechanical perturbations.
       Synthetic speech: robotic mathematical regularity (jitter < 0.0035).
       Zero-Division Guard: if voiced frames < 10, returns safe baseline metrics.

  2. ASVspoof Linear Frequency Cepstral Coefficients (LFCC)
       30 linearly-spaced triangular filterbanks across 0 – 8000 Hz.
       12 LFCCs via DCT-II; inter-frame cepstral variance measures temporal
       spectral dynamics.  Neural vocoders produce hyper-regular or severely
       distorted cepstral trajectories.

  3. High-Frequency Vocoder Brickwall Cutoff Ratio
       Energy ratio above 5.5 kHz relative to total band.
       Neural vocoders (HiFi-GAN, WaveGlow, FastSpeech2) impose a hard brickwall
       above their Nyquist limit, producing near-zero energy above 5.5 kHz.

Output contract (exact Phase 2 spec keys):
  jitter_local    – float   (local period perturbation quotient)
  shimmer_local   – float   (local amplitude perturbation quotient)
  hnr_db          – float   (harmonics-to-noise ratio in dB)
  lfcc_variance   – float   (mean inter-frame LFCC variance across 12 coefficients)
  hf_cutoff_ratio – float   (energy ratio above 5.5 kHz, 0–1)
  glottal_risk    – float ∈ [0.0, 1.0]
  lfcc_risk       – float ∈ [0.0, 1.0]

Author:  VoiceShield Engineering
Version: 2.0.0  (Phase 2)
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
from scipy.fftpack import dct

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases (all declared at module top)
# ---------------------------------------------------------------------------
_LFCCMatrix  = np.ndarray   # shape (n_lfcc, T)
_FilterBank  = np.ndarray   # shape (n_filters, n_fft//2 + 1)


# ---------------------------------------------------------------------------
# Praat / Parselmouth lazy import
# ---------------------------------------------------------------------------

def _get_praat() -> Tuple[Any, Any]:
    """Lazy-import Praat Parselmouth.  Returns (parselmouth, praat_call) or (None, None)."""
    try:
        pm = importlib.import_module("parselmouth")
        praat_call = getattr(importlib.import_module("parselmouth.praat"), "call")
        return pm, praat_call
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Linear Filterbank Builder
# ---------------------------------------------------------------------------

def linear_filterbank(
    n_filters: int = 30,
    n_fft: int = 512,
    sr: int = 16000,
    low_freq: float = 0.0,
    high_freq: float = 8000.0,
) -> _FilterBank:
    """
    Build an ASVspoof-standard linear-frequency triangular filterbank.

    Each of the ``n_filters`` triangular filters has its centre frequency
    linearly spaced across [``low_freq``, ``high_freq``].  This is distinct
    from the mel-scale filterbank used in MFCCs.

    Parameters
    ----------
    n_filters : int
        Number of triangular filters (default 30, matching ASVspoof baseline).
    n_fft : int
        FFT size.
    sr : int
        Sample rate in Hz.
    low_freq : float
        Lowest filterbank edge in Hz.
    high_freq : float
        Highest filterbank edge in Hz.

    Returns
    -------
    np.ndarray
        Filterbank matrix of shape ``(n_filters, n_fft//2 + 1)``.
    """
    linear_points = np.linspace(low_freq, high_freq, n_filters + 2)
    fft_bin_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fbank = np.zeros((n_filters, len(fft_bin_freqs)), dtype=np.float32)

    for m in range(1, n_filters + 1):
        f_minus = linear_points[m - 1]
        f_centre = linear_points[m]
        f_plus = linear_points[m + 1]

        left_slope  = (fft_bin_freqs - f_minus)  / max(1e-6, f_centre - f_minus)
        right_slope = (f_plus - fft_bin_freqs)   / max(1e-6, f_plus - f_centre)
        fbank[m - 1] = np.maximum(0.0, np.minimum(left_slope, right_slope))

    return fbank


# ---------------------------------------------------------------------------
# LFCC Feature Extractor
# ---------------------------------------------------------------------------

def extract_lfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 12,
    n_filters: int = 30,
    n_fft: int = 512,
    hop_length: int = 160,
) -> _LFCCMatrix:
    """
    Extract 12 Linear Frequency Cepstral Coefficients (ASVspoof standard).

    Processing chain:
      1. Short-time power spectrum via STFT.
      2. 30 linearly-spaced triangular filterbanks (0 – Nyquist Hz).
      3. Natural-log compression of filter energies.
      4. DCT-II orthonormal transform → 12 LFCC coefficients.

    Parameters
    ----------
    audio : np.ndarray
        1-D float voiced speech signal.
    sr : int
        Sample rate in Hz.
    n_lfcc : int
        Number of cepstral coefficients to return (default 12).
    n_filters : int
        Number of triangular filterbanks (default 30).
    n_fft : int
        FFT window size.
    hop_length : int
        STFT hop length in samples.

    Returns
    -------
    np.ndarray
        LFCC matrix of shape ``(n_lfcc, T)``.
    """
    # Guard: zero-length input → return zero LFCC matrix (variance = 0)
    if len(audio) == 0:
        return np.zeros((n_lfcc, 1), dtype=np.float32)

    # Pad short segments to at least n_fft samples (reflect requires len ≥ 1)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)), mode="constant", constant_values=0.0)

    # Power spectrum  (magnitude squared)
    stft_mag = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=400))
    stft_power = stft_mag ** 2  # shape: (n_fft//2 + 1, T)

    fbank = linear_filterbank(
        n_filters=n_filters, n_fft=n_fft, sr=sr,
        low_freq=0.0, high_freq=float(sr) / 2.0,
    )
    # Apply filterbank
    filter_energies = np.dot(fbank, stft_power)   # shape: (n_filters, T)

    # Floor to avoid log(0)
    filter_energies = np.where(filter_energies == 0, np.finfo(float).eps, filter_energies)

    # Natural-log compression (matches ASVspoof LFCC reference implementation)
    log_energies = np.log(filter_energies)

    # DCT-II orthonormal transform along filterbank axis
    lfcc_coeffs = dct(log_energies, type=2, axis=0, norm="ortho")[:n_lfcc, :]
    return lfcc_coeffs


# ---------------------------------------------------------------------------
# Praat Glottal Metrics  (Voiced-Only)
# ---------------------------------------------------------------------------

def extract_praat_glottal_metrics(voiced_audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract voiced-frame glottal perturbation metrics via Praat Parselmouth.

    Tracks F₀ strictly between 75 – 500 Hz on voiced frames (F₀ > 0).
    Local Jitter and Local Shimmer are computed only on voiced frames.

    Zero-Division Guard
    -------------------
    If fewer than 10 voiced F₀ frames are detected, the function returns
    calibrated safe baseline values without raising any exceptions.

    Parameters
    ----------
    voiced_audio : np.ndarray
        1-D float voiced speech segment at ``sr`` Hz.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict
        jitter_local, shimmer_local, hnr_db, f0_mean, f0_std, formant_dispersion.
    """
    # Safe baseline — returned on Parselmouth import failure, short input,
    # or insufficient voiced frames (< 10).
    safe_baseline: Dict[str, float] = {
        "jitter_local":       0.015,
        "shimmer_local":      0.035,
        "hnr_db":             12.0,
        "f0_mean":            150.0,
        "f0_std":             20.0,
        "formant_dispersion": 1050.0,
    }

    pm, praat_call = _get_praat()
    if pm is None or praat_call is None:
        log.warning("Praat Parselmouth not available; returning baseline glottal metrics.")
        return safe_baseline

    if len(voiced_audio) < int(sr * 0.15):
        log.debug("Praat: voiced segment too short; returning baseline.")
        return safe_baseline

    try:
        snd = pm.Sound(voiced_audio, sampling_frequency=float(sr))

        # Pitch tracking (voiced frame detection)
        pitch = snd.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=500.0)
        f0_values = pitch.selected_array["frequency"]
        voiced_f0 = f0_values[f0_values > 0]

        # ZERO-DIVISION GUARD: require at least 10 voiced frames
        if len(voiced_f0) < 10:
            log.debug(
                "Praat: only %d voiced frames detected (< 10); returning baseline.",
                len(voiced_f0),
            )
            return safe_baseline

        f0_mean = float(np.mean(voiced_f0))
        f0_std  = float(np.std(voiced_f0))

        # PointProcess from voiced frames only
        point_process = praat_call(snd, "To PointProcess (periodic, cc)", 75.0, 500.0)

        # Local Jitter (period perturbation quotient)
        local_jitter = praat_call(
            point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
        )
        # Local Shimmer (amplitude perturbation quotient)
        local_shimmer = praat_call(
            [snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )
        # Harmonics-to-Noise Ratio
        harmonicity = snd.to_harmonicity(time_step=0.01, minimum_pitch=75.0)
        hnr = praat_call(harmonicity, "Get mean", 0.0, 0.0)

        # Formant Dispersion (F2 - F1, Burg method)
        formant = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500.0)
        f1 = praat_call(formant, "Get mean", 1, 0.0, 0.0, "Hertz")
        f2 = praat_call(formant, "Get mean", 2, 0.0, 0.0, "Hertz")
        dispersion = (
            float(f2 - f1)
            if (not np.isnan(f1) and not np.isnan(f2) and f2 > f1)
            else 1050.0
        )

        jl = float(np.nan_to_num(local_jitter,  nan=0.015))
        sl = float(np.nan_to_num(local_shimmer, nan=0.035))
        return {
            "jitter_local":       jl,
            "local_jitter":       jl,
            "shimmer_local":      sl,
            "local_shimmer":      sl,
            "hnr_db":             float(np.nan_to_num(hnr, nan=12.0)),
            "f0_mean":            round(f0_mean, 2),
            "f0_std":             round(f0_std,  2),
            "formant_dispersion": round(dispersion, 2),
        }

    except Exception as exc:  # noqa: BLE001
        log.warning("Praat glottal extraction failed (%s); returning baseline.", exc)
        return safe_baseline


# ---------------------------------------------------------------------------
# High-Frequency Vocoder Brickwall Cutoff Ratio
# ---------------------------------------------------------------------------

def extract_vocoder_cutoff_ratio(
    audio: np.ndarray,
    sr: int = 16000,
    cutoff_hz: float = 5500.0,
) -> float:
    """
    Compute energy ratio above ``cutoff_hz`` relative to total spectral energy.

    Neural vocoders that operate at 22 kHz internal rate but output 16 kHz
    impose a hard brickwall above their Nyquist limit (~5.5 kHz), resulting in
    near-zero high-frequency energy.  Authentic broadband human voice contains
    significant energy above 5.5 kHz from fricatives and aspiration noise.

    Parameters
    ----------
    audio : np.ndarray
        1-D float audio signal.
    sr : int
        Sample rate in Hz.
    cutoff_hz : float
        Frequency threshold for the brickwall test (default 5500 Hz).

    Returns
    -------
    float
        Energy ratio ∈ [0, 1].  Near 0 → vocoder brickwall suspected.
    """
    if len(audio) < 256:
        return 0.0

    stft_power = np.abs(librosa.stft(audio, n_fft=512, hop_length=160)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)

    total_energy = float(np.sum(stft_power)) + 1e-9
    hf_energy    = float(np.sum(stft_power[freqs >= cutoff_hz, :]))
    return float(np.clip(hf_energy / total_energy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main Forensic DSP Analyzer class
# ---------------------------------------------------------------------------

class ForensicDSPAnalyzer:
    """
    Multi-Feature Forensic DSP & Biomechanical Acoustic Extractor (Phase 2).

    Combines three independent forensic channels into a unified risk surface:
      • Glottal perturbation biomechanics (Praat Parselmouth)
      • ASVspoof LFCC cepstral dynamics (linear filterbank)
      • Neural vocoder brickwall cutoff ratio (HF spectral energy)

    Usage
    -----
    >>> analyzer = ForensicDSPAnalyzer(sr=16000)
    >>> metrics = analyzer.extract_dsp_metrics(voiced_audio)
    >>> print(metrics["glottal_risk"], metrics["lfcc_risk"])
    """

    def __init__(self, sr: int = 16000) -> None:
        self.sr = sr

    def extract_dsp_metrics(
        self,
        voiced_audio: np.ndarray,
        sr: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract all Phase 2 forensic DSP metrics from a voiced speech segment.

        Parameters
        ----------
        voiced_audio : np.ndarray
            1-D float array of voiced speech at ``sr`` Hz.
        sr : int, optional
            Sample rate.  Defaults to the value set in ``__init__``.

        Returns
        -------
        dict with Phase 2 spec keys:
            jitter_local    – Local period perturbation quotient
            shimmer_local   – Local amplitude perturbation quotient
            hnr_db          – Harmonics-to-noise ratio (dB)
            lfcc_variance   – Mean inter-frame LFCC variance across 12 coefficients
            hf_cutoff_ratio – Spectral energy ratio above 5.5 kHz
            glottal_risk    – ∈ [0.0, 1.0]  glottal biomechanical synthetic risk
            lfcc_risk       – ∈ [0.0, 1.0]  LFCC cepstral synthetic risk

        Additional keys (not in minimal spec, preserved for downstream fusion):
            spectral_risk       – ∈ [0.0, 1.0]
            combined_dsp_risk   – ∈ [0.0, 1.0]  (weighted fusion)
            f0_mean, f0_std, formant_dispersion, is_human_glottal
        """
        sample_rate = sr if sr is not None else self.sr

        # ----------------------------------------------------------------
        # 1. Praat Glottal Biomechanics (voiced-frame only, guard ≥ 10 frames)
        # ----------------------------------------------------------------
        bio     = extract_praat_glottal_metrics(voiced_audio, sr=sample_rate)
        jitter  = bio["jitter_local"]
        shimmer = bio["shimmer_local"]
        hnr     = bio["hnr_db"]

        # ----------------------------------------------------------------
        # 2. ASVspoof LFCC (30 linear filterbanks, 12 DCT-II coefficients)
        # ----------------------------------------------------------------
        lfcc_mat = extract_lfcc(voiced_audio, sr=sample_rate, n_lfcc=12, n_filters=30)
        # Inter-frame cepstral variance: mean variance across time for each coeff
        lfcc_var = float(np.mean(np.var(lfcc_mat, axis=1)))
        # Guard against non-finite LFCC variance
        if not np.isfinite(lfcc_var):
            lfcc_var = 0.0

        # ----------------------------------------------------------------
        # 3. Glottal Risk Calibration (Hardened for Live Microphones & Soft Speech)
        # ----------------------------------------------------------------
        # Natural human vocal fold physiology: jitter in [0.006, 0.040] with shimmer >= 0.010.
        # AI Voice Clones (direct file): robotic rigidity jitter < 0.0030.
        # AI Voice played through speaker -> mic: room reflections inflate jitter to 0.003–0.025
        # creating 'fake' human-like perturbation. We use a tighter human band and
        # weight HNR + shimmer as secondary discriminators.
        if 0.006 <= jitter <= 0.040 and shimmer >= 0.010 and 6.0 <= hnr <= 25.0:
            # Strong triple-verified authentic human signature
            is_human_glottal = True
            glottal_risk = 0.06   # High-confidence authentic human vocal perturbation
        elif 0.006 <= jitter <= 0.040 and shimmer >= 0.008:
            # Consistent jitter & shimmer, but HNR may be slightly elevated
            is_human_glottal = True
            glottal_risk = 0.12   # Plausible human, minor uncertainty
        elif 0.040 < jitter <= 0.065:
            is_human_glottal = True
            glottal_risk = 0.18   # Elevated room reflections or soft speaking
        elif jitter < 0.0020:
            is_human_glottal = False
            glottal_risk = 0.92   # Pathological vocoder pitch rigidity (<0.20% jitter)
        elif jitter < 0.006 and hnr > 25.0:
            # Low jitter + very high HNR: classic vocoder signature even with mic reverb
            is_human_glottal = False
            glottal_risk = 0.78   # Likely AI voice replayed through speaker
        elif jitter > 0.075 and lfcc_var >= 10.0:
            is_human_glottal = False
            glottal_risk = 0.85   # Severe neural vocoder glitching / phase distortion
        else:
            is_human_glottal = False
            glottal_risk = 0.20   # Neutral / borderline boundary zone

        # ----------------------------------------------------------------
        # 4. LFCC Risk Calibration
        # ----------------------------------------------------------------
        if lfcc_var > 10.5:
            lfcc_risk = 0.85   # Severe neural vocoder filterbank distortion
        elif lfcc_var < 0.30:
            lfcc_risk = 0.85   # Pathological cepstral uniformity (very regular AI speech)
        elif lfcc_var < 0.60:
            lfcc_risk = 0.65   # Low variance: borderline AI suspect
        elif 0.8 <= lfcc_var <= 9.5:
            lfcc_risk = 0.06   # Authentic human dynamic range
        else:
            lfcc_risk = 0.20   # Borderline / transitional zone

        # ----------------------------------------------------------------
        # 5. High-Frequency Vocoder Brickwall Cutoff Risk
        # ----------------------------------------------------------------
        hf_cutoff_ratio = extract_vocoder_cutoff_ratio(
            voiced_audio, sr=sample_rate, cutoff_hz=5500.0
        )
        if 0.00001 <= hf_cutoff_ratio < 0.0004:
            spectral_risk = 0.82   # Brickwall filter detected (vocoder fingerprint)
        elif hf_cutoff_ratio < 0.0010:
            spectral_risk = 0.55   # Possible vocoder with slight HF leakage from mic
        elif hf_cutoff_ratio >= 0.0010:
            spectral_risk = 0.12   # Authentic broadband voice energy
        else:
            spectral_risk = 0.25   # Neutral narrowband baseline

        # ----------------------------------------------------------------
        # 6. Combined DSP Risk  (weighted fusion)
        # Increased spectral (HF brickwall) weight — it is the most reliable
        # discriminator for speaker-replayed AI audio since rooms cannot ADD
        # HF energy that the vocoder removed.
        # ----------------------------------------------------------------
        combined_dsp_risk = float(np.clip(
            0.40 * spectral_risk + 0.35 * glottal_risk + 0.25 * lfcc_risk,
            0.01, 0.99,
        ))

        return {
            # Phase 2 spec keys (required)
            "jitter_local":       round(jitter,          5),
            "shimmer_local":      round(shimmer,         5),
            "hnr_db":             round(hnr,             2),
            "lfcc_variance":      round(lfcc_var,        4),
            "hf_cutoff_ratio":    round(hf_cutoff_ratio, 6),
            "glottal_risk":       round(glottal_risk,    4),
            "lfcc_risk":          round(lfcc_risk,       4),
            # Additional forensic context (preserved for downstream fusion)
            "spectral_risk":      round(spectral_risk,   4),
            "combined_dsp_risk":  round(combined_dsp_risk, 4),
            "f0_mean":            bio["f0_mean"],
            "f0_std":             bio["f0_std"],
            "formant_dispersion": bio["formant_dispersion"],
            "is_human_glottal":   is_human_glottal,
        }

    extract = extract_dsp_metrics
    analyze = extract_dsp_metrics


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (functional API)
# ---------------------------------------------------------------------------

def extract_dsp_metrics(voiced_audio: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Functional convenience wrapper around ``ForensicDSPAnalyzer.extract_dsp_metrics``.

    Parameters
    ----------
    voiced_audio : np.ndarray
        1-D float voiced speech segment.
    sr : int
        Sample rate in Hz (default 16000).

    Returns
    -------
    dict
        All Phase 2 spec keys plus additional forensic context fields.
    """
    return ForensicDSPAnalyzer(sr=sr).extract_dsp_metrics(voiced_audio, sr=sr)

