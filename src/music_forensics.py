"""
VoiceShield Enterprise AI Music, Song & Singing Voice Forensic Engine (v3.2.0).
Implements academic SOTA forensic methods for detecting:
  1. Suno AI, Udio, Stable Audio, MusicLM, Mubert (Generative Diffusion / Flow-Matching Music)
  2. RVC (Retrieval-based Voice Conversion), So-VITS-SVC, DiffSinger (AI Singing Voice Clones)
  3. Neural Audio Codecs (EnCodec, SoundStream, DAC, Descript Audio Codec)
  4. Transposed Convolution 2D-FFT Checkerboard Artifacts & High-Frequency "Digital Haze"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple
import librosa
import numpy as np

log = logging.getLogger(__name__)


def is_music_track(audio: np.ndarray, sr: int = 16000) -> Tuple[bool, float]:
    """
    Robust music & singing voice activity detector.
    A track is only classified as music if it contains active polyphonic chord accompaniment
    or strong rhythmic percussion.
    """
    if len(audio) < int(0.6 * sr):
        return False, 0.0

    try:
        sample = audio[: int(6.0 * sr)]
        y_harm, y_perc = librosa.effects.hpss(sample)
        perc_energy = float(np.mean(y_perc ** 2))
        total_energy = float(np.mean(sample ** 2)) + 1e-9
        perc_ratio = perc_energy / total_energy

        # Chroma pitch class distribution
        chroma = librosa.feature.chroma_stft(y=sample, sr=sr, n_fft=1024)
        chroma_mean = np.mean(chroma, axis=1)
        active_pitches = int(np.sum(chroma_mean > 0.45))
        chroma_entropy = float(-np.sum(chroma_mean * np.log(chroma_mean + 1e-6)))

        # Music requires rhythmic percussion (> 0.035) AND polyphonic chords
        is_music = bool(perc_ratio > 0.035 and active_pitches >= 3 and chroma_entropy > 1.8)
        music_conf = float(np.clip(perc_ratio * 4.0, 0.0, 1.0))
        return is_music, music_conf
    except Exception as exc:
        log.debug("Music detection error: %s", exc)
        return False, 0.0


def extract_music_diffusion_artifacts(audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract multi-modal forensic signatures of generative music (Suno, Udio)
    and singing voice conversion (RVC, So-VITS):
    
    1. 2D-FFT Spectrogram Checkerboard Peaks (Transposed Convolution deconvolution artifacts)
    2. High-Frequency "Digital Haze" (Uniform diffusion noise floor in 4.5kHz - 8kHz)
    3. Neural Codec (EnCodec / DAC) Subband Ripple (RVQ codebook quantization)
    4. Instantaneous Phase Dispersion (Watery overlap-add reconstruction smearing)
    5. Singing Pitch-Grid Autotune Snap (RVC / DiffSinger vocal step-discontinuities)
    """
    if len(audio) < int(0.5 * sr):
        return {
            "music_spoof_prob": 0.50,
            "checkerboard_score": 0.50,
            "digital_haze_score": 0.50,
            "neural_codec_score": 0.50,
            "phase_dispersion_score": 0.50,
            "singing_vibrato_score": 0.50,
        }

    try:
        stft = librosa.stft(audio, n_fft=512, hop_length=160)
        mag = np.abs(stft)
        phase = np.angle(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=512)

        # -----------------------------------------------------------------
        # 1. 2D-FFT Checkerboard Artifacts in Spectrogram
        # -----------------------------------------------------------------
        log_mag = np.log1p(mag)
        fft2d = np.fft.fft2(log_mag)
        fft2d_shift = np.fft.fftshift(fft2d)
        fft2d_norm = np.abs(fft2d_shift)

        # Measure peak-to-average power ratio in 2D frequency spectrum
        papr = float(np.percentile(fft2d_norm, 99.5) / (np.median(fft2d_norm) + 1e-8))
        checkerboard_score = float(np.clip((papr - 28.0) / 35.0, 0.05, 0.95))

        # -----------------------------------------------------------------
        # 2. High-Frequency "Digital Haze" (Diffusion Noise Residual)
        # -----------------------------------------------------------------
        hf_mask = freqs >= 4500
        if np.any(hf_mask):
            hf_mag = mag[hf_mask, :]
            hf_flatness = float(np.mean(librosa.feature.spectral_flatness(S=hf_mag)))
            digital_haze_score = float(np.clip((hf_flatness - 0.18) * 3.5, 0.05, 0.95))
        else:
            digital_haze_score = 0.30

        # -----------------------------------------------------------------
        # 3. Neural Codec (EnCodec / DAC) Subband Ripple
        # -----------------------------------------------------------------
        if np.any(hf_mask):
            total_hf = float(np.mean(hf_mag)) + 1e-6
            norm_hf = hf_mag / total_hf
            subband_var = float(np.mean(np.std(norm_hf, axis=0)))
            neural_codec_score = float(np.clip((subband_var - 0.92) * 1.8, 0.05, 0.95))
        else:
            neural_codec_score = 0.30

        # -----------------------------------------------------------------
        # 4. Instantaneous Phase Dispersion
        # -----------------------------------------------------------------
        phase_diff = np.diff(phase, axis=0)
        phase_coherence = float(np.mean(np.abs(np.mean(np.exp(1j * phase_diff), axis=1))))
        phase_dispersion_score = float(np.clip((phase_coherence - 0.22) * 3.0, 0.05, 0.95))

        # -----------------------------------------------------------------
        # 5. Singing Voice Vibrato & Pitch Contour Analysis
        # -----------------------------------------------------------------
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=80, fmax=600, sr=sr, frame_length=1024, hop_length=256
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
        
        if len(valid_f0) > 20:
            d_f0 = np.diff(valid_f0)
            f0_jitter_var = float(np.var(d_f0))
            if f0_jitter_var < 0.25:
                singing_vibrato_score = 0.85
            elif f0_jitter_var > 600.0:
                singing_vibrato_score = 0.80
            else:
                singing_vibrato_score = 0.10
        else:
            singing_vibrato_score = 0.30

        # Unified AI Music / Song Spoof Consensus
        music_spoof_prob = float(np.clip(
            0.30 * checkerboard_score
            + 0.25 * digital_haze_score
            + 0.20 * neural_codec_score
            + 0.15 * phase_dispersion_score
            + 0.10 * singing_vibrato_score,
            0.05, 0.95
        ))
        if checkerboard_score >= 0.70 or (digital_haze_score >= 0.65 and neural_codec_score >= 0.60):
            music_spoof_prob = max(music_spoof_prob, 0.76)

        return {
            "music_spoof_prob": round(music_spoof_prob, 4),
            "checkerboard_score": round(checkerboard_score, 4),
            "digital_haze_score": round(digital_haze_score, 4),
            "neural_codec_score": round(neural_codec_score, 4),
            "phase_dispersion_score": round(phase_dispersion_score, 4),
            "singing_vibrato_score": round(singing_vibrato_score, 4),
        }

    except Exception as exc:
        log.warning("Music forensics analysis failed: %s", exc)
        return {
            "music_spoof_prob": 0.50,
            "checkerboard_score": 0.50,
            "digital_haze_score": 0.50,
            "neural_codec_score": 0.50,
            "phase_dispersion_score": 0.50,
            "singing_vibrato_score": 0.50,
        }


def isolate_singing_vocals(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Separate singing vocal harmonics from backing music tracks using
    Harmonic-Percussive Source Separation (HPSS).
    Allows the Wav2Vec2 transformer to evaluate the isolated vocal timbre directly.
    """
    if len(audio) < int(0.5 * sr):
        return audio

    try:
        y_harmonic, _ = librosa.effects.hpss(audio, margin=(1.2, 3.0))
        peak = np.max(np.abs(y_harmonic))
        if peak > 1e-4:
            y_harmonic = (y_harmonic / peak) * 0.85
        return y_harmonic.astype(np.float32)
    except Exception:
        return audio
