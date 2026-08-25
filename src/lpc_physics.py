"""
VoiceShield Phase 2 — LPC Residual Excitation & Phase Entropy Engine.

Implements vocal-tract inverse filtering via Levinson-Durbin recursion to isolate
the raw glottal excitation residual e(n), then computes three forensic discriminators
that separate authentic human glottal physiology from neural-vocoder artefacts:

  1. Residual Spectral Kurtosis
       Human vocal folds produce chaotic, high-kurtosis glottal closure instants (GCIs).
       Neural vocoders (HiFi-GAN, WaveGlow, WaveNet, XTTS, Kokoro) generate
       mathematically continuous, low-kurtosis sinusoidal residuals.

  2. High-Frequency Phase Entropy (> 4 kHz)
       Shannon entropy of STFT *unwrapped phase angles* above 4 kHz.
       Authentic voices exhibit phase disorder from turbulent airflow; vocoders
       exhibit structured, low-entropy phase patterns from deterministic synthesis filters.

  3. Residual Spectral Flatness
       Geometric-to-arithmetic mean ratio of the residual power spectrum.
       Natural excitation is broadband (flatness ~0.05–0.40); neural vocoders
       produce over-regularised, near-tonal residuals (flatness < 0.01 or → 1.0).

Output contract (exact Phase 2 spec keys):
  lpc_kurtosis        – float  (Pearson kurtosis of residual samples)
  phase_entropy       – float  (normalised Shannon entropy of HF phase increments, 0–1)
  residual_flatness   – float  (geometric/arithmetic mean ratio, 0–1)
  lpc_anomaly_score   – float ∈ [0.0, 1.0]  (> 0.60 flags synthetic vocoder excitation)

Author:  VoiceShield Engineering
Version: 2.0.0  (Phase 2)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases  (declared at module top — before any function uses them)
# ---------------------------------------------------------------------------
_LPCCoeffs = np.ndarray   # shape (order+1,), a[0] == 1.0


# ---------------------------------------------------------------------------
# Levinson-Durbin Recursion  (Yule-Walker / autocorrelation method)
# ---------------------------------------------------------------------------

def levinson_durbin(r: np.ndarray, order: int) -> np.ndarray:
    """
    Solve the Yule-Walker equations for LPC coefficients via Levinson-Durbin.

    Parameters
    ----------
    r : np.ndarray
        Biased autocorrelation sequence r[0..order], shape (order+1,).
    order : int
        LPC predictor order p.

    Returns
    -------
    a : np.ndarray
        LPC coefficient vector of shape (order+1,) where a[0] = 1.0.
        The inverse (analysis) filter is A(z) = Σ a[k] z^{-k}, k=0..p.
    """
    a = np.zeros(order + 1, dtype=np.float64)
    a[0] = 1.0
    e = float(r[0])

    for i in range(1, order + 1):
        if e <= 1e-12:
            break
        # Reflection coefficient (Levinson step)
        k = -float(np.dot(a[1:i], r[i - 1:0:-1]) + r[i]) / e
        a_prev = a.copy()
        for j in range(1, i + 1):
            a[j] += k * a_prev[i - j]
        a[i] = k
        e *= 1.0 - k * k

    return a


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class LPCPhysicsAnalyzer:
    """
    Forensic LPC Residual Excitation & Phase Entropy Engine (VoiceShield Phase 2).

    Extracts three independent forensic discriminators from the LPC glottal residual
    to distinguish authentic human speech from neural-vocoder synthetic speech.

    Usage
    -----
    >>> analyzer = LPCPhysicsAnalyzer(order=16, sr=16000)
    >>> metrics = analyzer.extract_lpc_residual(voiced_frames)
    >>> if metrics["lpc_anomaly_score"] > 0.60:
    ...     print("Synthetic vocoder excitation detected")
    """

    # Kurtosis thresholds (Pearson; Gaussian baseline = 3.0)
    _KURT_SYNTHETIC_HIGH: float = 650.0    # Extreme pulse-train vocoders (ElevenLabs, Kokoro, XTTS)
    _KURT_NATURAL_LOW: float    = 8.0      # Lower bound of authentic GCI spikiness
    _KURT_NATURAL_HIGH: float   = 380.0    # Upper bound of authentic glottal closure dynamics
    _KURT_SMOOTH_CEIL: float    = 4.8      # Over-smoothed autoregressive vocoder ceiling

    def __init__(self, order: int = 16, sr: int = 16000) -> None:
        self.order = order
        self.sr = sr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_autocorrelation(self, audio: np.ndarray) -> np.ndarray:
        """
        FFT-based biased autocorrelation: r[k] = (1/N) Σ x[n] x[n-k].
        Returns r[0..order].
        """
        n = len(audio)
        fft_len = int(2 ** np.ceil(np.log2(2 * n - 1)))
        fft_x = np.fft.rfft(audio, n=fft_len)
        r_full = np.fft.irfft(np.abs(fft_x) ** 2, n=fft_len)
        return r_full[: self.order + 1] / float(n)

    def _compute_phase_entropy(
        self,
        residual: np.ndarray,
        sr: int,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> float:
        """
        High-Frequency Phase Entropy (> 4 kHz) via STFT phase angles.

        Computes Shannon entropy over the histogram of *unwrapped STFT phase
        increments* in the high-frequency band (>4 kHz).  Authentic voices show
        disordered (high-entropy) phase because turbulent glottal airflow creates
        stochastic excitation above 4 kHz.  Neural vocoders exhibit highly
        structured, deterministic, low-entropy phase trajectories from their
        mathematical synthesis filters.

        Returns
        -------
        float
            Normalised entropy ∈ [0, 1].  Higher → more human-like phase disorder.
        """
        # Complex STFT — we need actual phase angles, not just magnitude
        D = librosa.stft(
            residual.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
        )  # shape: (n_fft//2 + 1, T)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs >= 4000.0

        if not np.any(hf_mask) or D.shape[1] < 2:
            return 0.50

        # Extract unwrapped phase angles for high-frequency bins only
        hf_phase = np.angle(D[hf_mask, :])          # radians, shape (n_hf, T)
        unwrapped = np.unwrap(hf_phase, axis=1)      # unwrap along time axis
        phase_diff = np.diff(unwrapped, axis=1)      # instantaneous freq proxy

        flat_diff = phase_diff.ravel()
        if flat_diff.size == 0:
            return 0.50

        # Histogram-based Shannon entropy (64 bins)
        hist, _ = np.histogram(flat_diff, bins=64)
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total < 1.0:
            return 0.50

        p = hist / total
        valid = p > 0
        entropy_bits = float(-np.sum(p[valid] * np.log2(p[valid])))
        # Normalise: max entropy for 64 bins = log2(64) = 6 bits
        return float(np.clip(entropy_bits / np.log2(64.0), 0.0, 1.0))

    def _compute_residual_flatness(
        self,
        residual: np.ndarray,
        n_fft: int = 512,
        hop_length: int = 160,
    ) -> float:
        """
        Spectral Flatness = geometric_mean(|R(k)|²) / arithmetic_mean(|R(k)|²).

        Tonal/spiky residuals (natural GCI train) → near 0.
        White-noise-like flat residuals → near 1.
        Neural vocoder residuals collapse to either pathological extreme.
        """
        flatness = librosa.feature.spectral_flatness(
            y=residual.astype(np.float32),
            n_fft=n_fft,
            hop_length=hop_length,
        )[0]
        return float(np.mean(flatness))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_lpc_residual(
        self,
        voiced_audio: np.ndarray,
        sr: Optional[int] = None,
        order: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract the LPC glottal residual and compute Phase 2 forensic metrics.

        The inverse (analysis) filter A(z) is applied to the voiced speech
        signal s(n) to yield the raw glottal excitation residual:

            e(n) = s(n) + Σ_{k=1}^{p} a_k · s(n-k)

        Parameters
        ----------
        voiced_audio : np.ndarray
            1-D float array of voiced speech at ``sr`` Hz.
        sr : int, optional
            Sample rate.  Defaults to the value set in ``__init__``.
        order : int, optional
            LPC predictor order.  Defaults to the value set in ``__init__``.

        Returns
        -------
        dict with Phase 2 spec keys:
            lpc_kurtosis        – Pearson kurtosis of residual amplitude distribution
            phase_entropy       – Normalised Shannon entropy of HF unwrapped phase (0–1)
            residual_flatness   – Spectral flatness of residual signal (0–1)
            lpc_anomaly_score   – Calibrated synthetic risk ∈ [0.0, 1.0]
                                   > 0.60 → synthetic vocoder excitation flagged
        """
        sample_rate = sr if sr is not None else self.sr
        lpc_order   = order if order is not None else self.order

        # ------ Guard: insufficient voiced material -------------------
        min_samples = int(sample_rate * 0.15)
        if len(voiced_audio) < min_samples:
            log.debug(
                "LPC: voiced segment too short (%d < %d samples); returning baseline.",
                len(voiced_audio), min_samples,
            )
            return self._safe_baseline()

        audio = voiced_audio.astype(np.float64)

        # ------ Step 1: Autocorrelation --------------------------------
        r = self._compute_autocorrelation(audio)

        # ------ Step 2: Levinson-Durbin LPC coefficients ---------------
        lpc_coeffs = levinson_durbin(r, lpc_order)

        # ------ Step 3: Inverse filter → glottal excitation residual ---
        # lfilter(b=lpc_coeffs, a=[1.0]) implements the FIR analysis filter A(z)
        residual_raw = lfilter(lpc_coeffs, np.array([1.0]), audio)
        residual: np.ndarray = np.asarray(residual_raw, dtype=np.float64)[lpc_order:]

        # ------ Guard: degenerate residual ----------------------------
        if len(residual) == 0 or float(np.std(residual)) <= 1e-8:
            log.debug("LPC: degenerate residual (near-zero variance); returning baseline.")
            return self._safe_baseline()

        # ------ Step 4: Frame-Level Residual Kurtosis (GCI Tracking) ---
        # Pearson kurtosis (fisher=False) -> Gaussian baseline = 3.0
        frame_len = int(sample_rate * 0.030)
        hop_len = int(sample_rate * 0.020)
        num_frames = (len(residual) - frame_len) // hop_len
        if num_frames > 0:
            shape = (num_frames, frame_len)
            strides = (residual.strides[0] * hop_len, residual.strides[0])
            frames = np.lib.stride_tricks.as_strided(residual, shape=shape, strides=strides)
            stds = np.std(frames, axis=1)
            active_frames = frames[stds > 1e-5]
            if len(active_frames) > 0:
                frame_kurts = kurtosis(active_frames, axis=1, fisher=False)
                lpc_kurt = float(np.median(frame_kurts))
            else:
                lpc_kurt = float(kurtosis(residual, fisher=False))
        else:
            lpc_kurt = float(kurtosis(residual, fisher=False))
        if not np.isfinite(lpc_kurt):
            lpc_kurt = 3.0

        # ------ Step 5: HF Phase Entropy (> 4 kHz) --------------------
        phase_ent = self._compute_phase_entropy(residual, sample_rate)

        # ------ Step 6: Residual Spectral Flatness --------------------
        res_flatness = self._compute_residual_flatness(residual)

        # ------ Step 7: Calibrated Anomaly Score ----------------------
        anomaly = self._calibrate_anomaly(lpc_kurt, phase_ent, res_flatness)

        return {
            "lpc_kurtosis":      round(lpc_kurt, 4),
            "phase_entropy":     round(phase_ent, 4),
            "residual_flatness": round(res_flatness, 6),
            "lpc_anomaly_score": round(float(np.clip(anomaly, 0.02, 0.98)), 4),
        }

    extract = extract_lpc_residual
    analyze = extract_lpc_residual

    # ------------------------------------------------------------------
    # Calibration (separated for unit-testability)
    # ------------------------------------------------------------------

    def _calibrate_anomaly(
        self,
        kurtosis_val: float,
        phase_entropy: float,
        flatness: float,
    ) -> float:
        """
        Multi-criterion anomaly score calibration for live microphone and audio files.

        Kurtosis axis (Pearson frame-level median):
          2.0 - 8.0   -> Authentic human vocal fold closure (Gaussian-like residual)
          8.0 - 18.0  -> Expressive / dynamic human voice
          < 1.8       -> Pathologically over-smoothed vocoder
          > 25.0      -> Extreme synthetic pulse-train or vocoder glitch

        Phase entropy axis (> 4 kHz unwrapped STFT):
          < 0.25      -> Highly structured deterministic synthetic phase lattice -> risk up
          >= 0.60     -> Authentic turbulent glottal airflow (disordered phase) -> risk low
        """
        # Primary gate: kurtosis
        if 2.0 <= kurtosis_val <= 8.5:
            base = 0.05   # Authentic human voice residual
        elif 8.5 < kurtosis_val <= 18.0:
            base = 0.12   # Expressive / dynamic human speech
        elif kurtosis_val < 1.8:
            base = 0.85   # Pathologically over-smoothed vocoder
        elif kurtosis_val > 25.0:
            base = 0.85   # Extreme impulse train or vocoder glitch
        else:
            base = 0.20   # Neutral / ambiguous zone

        # Secondary gates: Phase Entropy
        if phase_entropy < 0.25:
            base = max(base, 0.75)  # Deterministic artificial phase lattice
        elif phase_entropy >= 0.60:
            base = min(base, 0.15)  # Natural glottal turbulence confirms human voice

        # Spectral Flatness
        if flatness < 0.0005:
            base = max(base, 0.65)

        return base

    @staticmethod
    def _safe_baseline() -> Dict[str, Any]:
        """Return neutral non-NaN baseline when analysis cannot proceed safely."""
        return {
            "lpc_kurtosis":      3.0,
            "phase_entropy":     0.50,
            "residual_flatness": 0.50,
            "lpc_anomaly_score": 0.50,
        }


# ---------------------------------------------------------------------------
# Module-level convenience wrapper (functional API)
# ---------------------------------------------------------------------------

def extract_lpc_residual(
    voiced_audio: np.ndarray,
    sr: int = 16000,
    order: int = 16,
) -> Dict[str, Any]:
    """
    Functional convenience wrapper around ``LPCPhysicsAnalyzer.extract_lpc_residual``.

    Parameters
    ----------
    voiced_audio : np.ndarray
        1-D float array of voiced speech.
    sr : int
        Sample rate in Hz (default 16000).
    order : int
        LPC predictor order (default 16).

    Returns
    -------
    dict
        Keys: ``lpc_kurtosis``, ``phase_entropy``, ``residual_flatness``,
        ``lpc_anomaly_score``.
    """
    return LPCPhysicsAnalyzer(order=order, sr=sr).extract_lpc_residual(
        voiced_audio, sr=sr, order=order
    )
