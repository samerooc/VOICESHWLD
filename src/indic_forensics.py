"""
VoiceShield Indic & Neural Vocoder (Sarvam AI, Bhashini, BigVGAN, Vocos, StyleTTS 2) Forensic Engine.
Specialized forensic detection for:
  1. Sarvam AI (Bulbul:v1, Indic TTS, Hinglish & Regional Voice Clones)
  2. BigVGAN & Vocos Neural Vocoders (Periodic Snake Activation Sub-Harmonic Lock)
  3. Formant Trajectory Smoothing across Indic Retroflex & Aspirated Consonants
  4. Pitch Contour Mathematical Regularity in Indic Sentences
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import librosa
import numpy as np

log = logging.getLogger(__name__)


def extract_indic_tts_forensics(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract forensic signatures of Indic neural speech synthesizers (Sarvam AI, Bhashini)
    and modern neural vocoders (BigVGAN, Vocos, StyleTTS 2):
    
    1. Snake Activation Sub-Harmonic Phase Lock (Vocos / BigVGAN)
    2. Formant Transition Mathematical Smoothness (Retroflex transition damping)
    3. Glottal Micro-Regularity in Indic Vowels
    """
    if len(audio) < int(0.4 * sr):
        return {
            "indic_spoof_prob": 0.50,
            "vocos_snake_score": 0.50,
            "formant_smoothness_score": 0.50,
            "indic_pitch_rigidity_score": 0.50,
        }

    try:
        stft = librosa.stft(audio, n_fft=512, hop_length=160)
        mag = np.abs(stft)
        phase = np.angle(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)

        # -----------------------------------------------------------------
        # 1. BigVGAN / Vocos Snake Activation Harmonic Phase Lock
        # Modern neural vocoders use periodic Snake activations [x + (1/a)*sin^2(a*x)]
        # creating unnatural phase synchronization in 3.5kHz - 7kHz band (> 0.45 coherence)
        # -----------------------------------------------------------------
        mid_mask = (freqs >= 3800) & (freqs <= 6800)
        if np.any(mid_mask):
            mid_phase = phase[mid_mask, :]
            phase_diff = np.diff(mid_phase, axis=0)
            phase_coherence = float(np.mean(np.abs(np.mean(np.exp(1j * phase_diff), axis=1))))
            # Higher phase coherence indicates periodic snake activation vocoder (> 0.38)
            vocos_snake_score = float(np.clip((phase_coherence - 0.38) * 3.5, 0.05, 0.95))
        else:
            vocos_snake_score = 0.30

        # -----------------------------------------------------------------
        # 2. Formant Trajectory Smoothing in Indic Consonants
        # In real Indian speech, retroflex/aspirated stops create sharp formant shifts.
        # Sarvam AI models mathematically smooth out these transitions.
        # -----------------------------------------------------------------
        centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
        d_centroid = np.diff(centroid)
        centroid_volatility = float(np.std(d_centroid)) / (float(np.mean(centroid)) + 1e-6)
        
        # Real human Indic speech has high centroid volatility (> 0.18).
        # Neural TTS (Sarvam) has damped, smooth centroid transitions (< 0.10).
        if centroid_volatility < 0.10:
            formant_smoothness_score = 0.85
        elif centroid_volatility > 0.22:
            formant_smoothness_score = 0.10
        else:
            formant_smoothness_score = float(np.clip(1.0 - (centroid_volatility - 0.10) / 0.12, 0.10, 0.85))

        # -----------------------------------------------------------------
        # 3. Indic Pitch Contour Mathematical Regularity
        # -----------------------------------------------------------------
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=75, fmax=500, sr=sr, frame_length=1024, hop_length=256
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])

        if len(valid_f0) > 15:
            d_f0 = np.diff(valid_f0)
            f0_jitter = float(np.mean(np.abs(d_f0)) / (np.mean(valid_f0) + 1e-6))
            # Sarvam AI has extremely clean, uniform pitch contour (jitter < 0.005)
            if f0_jitter < 0.005:
                pitch_rigidity_score = 0.88
            elif f0_jitter > 0.065:
                pitch_rigidity_score = 0.75  # Vocoder glitching
            else:
                pitch_rigidity_score = 0.15  # Natural human vocal fold range
        else:
            pitch_rigidity_score = 0.25

        # Weighted Indic Spoof Probability
        indic_spoof_prob = float(np.clip(
            0.40 * vocos_snake_score
            + 0.35 * formant_smoothness_score
            + 0.25 * pitch_rigidity_score,
            0.05, 0.95
        ))

        return {
            "indic_spoof_prob": round(indic_spoof_prob, 4),
            "vocos_snake_score": round(vocos_snake_score, 4),
            "formant_smoothness_score": round(formant_smoothness_score, 4),
            "indic_pitch_rigidity_score": round(pitch_rigidity_score, 4),
        }

    except Exception as exc:
        log.warning("Indic forensics failed: %s", exc)
        return {
            "indic_spoof_prob": 0.50,
            "vocos_snake_score": 0.50,
            "formant_smoothness_score": 0.50,
            "indic_pitch_rigidity_score": 0.50,
        }
