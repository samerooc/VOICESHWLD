"""
VoiceShield Preprocessing Contract Module.
Ensures single identical audio preprocessing pipeline between Training and Inference.
"""

from typing import Dict, Optional, Tuple
import librosa
import numpy as np

from src.model_contract import EXPECTED_SAMPLE_RATE, MIN_AUDIO_DURATION_SEC, MAX_AUDIO_DURATION_SEC


def preprocess_audio_signal(
    audio: np.ndarray,
    sample_rate: int = EXPECTED_SAMPLE_RATE,
    target_sr: int = EXPECTED_SAMPLE_RATE,
    trim_silence: bool = True,
    top_db: float = 30.0,
    normalize_amplitude: bool = True,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    """
    Standard Preprocessing Pipeline (Identical for Training & Inference):
      1. Mono downmix (if multi-channel).
      2. Deterministic float32 conversion & NaN/Inf sanitization.
      3. Resampling to target_sr (16,000 Hz).
      4. DC offset removal (mean subtraction).
      5. Peak amplitude normalization to [-1.0, 1.0].
      6. Non-speech leading/trailing silence trimming (if speech duration >= 0.4s).
      7. Diagnostic metadata calculation.

    Returns:
        (preprocessed_audio, sample_rate, diagnostics_dict)
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Audio array is empty or None.")

    # 1. Multi-channel to Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # 2. Float32 conversion & Sanitization
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # 3. Resample if necessary
    if sample_rate != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sample_rate, target_sr=target_sr)
        effective_sr = target_sr
    else:
        effective_sr = sample_rate

    # 4. DC Offset Subtraction
    audio = audio - np.mean(audio)

    # 5. Amplitude Peak Normalization
    max_amp = float(np.max(np.abs(audio)))
    if normalize_amplitude and max_amp > 1e-6:
        audio = audio / (max_amp + 1e-9)

    # 6. Silence Trimming
    trimmed_duration = len(audio) / effective_sr
    if trim_silence and trimmed_duration >= MIN_AUDIO_DURATION_SEC:
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            if len(trimmed) >= int(effective_sr * 0.40):
                audio = trimmed
        except Exception:
            pass

    final_duration = float(len(audio) / effective_sr)
    rms_energy = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    diagnostics = {
        "duration_seconds": final_duration,
        "sample_rate": effective_sr,
        "rms_energy": rms_energy,
        "peak_amplitude": float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0,
    }

    return audio, effective_sr, diagnostics
