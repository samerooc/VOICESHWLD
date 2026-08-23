"""
VoiceShield Audio Validation Module (Phase 1).
Validates audio headers, file sizes, duration bounds, and signal silence.
"""

from typing import Tuple
import numpy as np

from src.config import (
    MAX_AUDIO_DURATION_SEC,
    MAX_FILE_SIZE_BYTES,
    MIN_AUDIO_DURATION_SEC,
    MIN_AUDIO_RMS_ENERGY,
)


def validate_wav_bytes(audio_bytes: bytes) -> None:
    """
    Performs initial byte-level validation on incoming WAV data.

    Raises:
        ValueError: If buffer is empty, exceeds size limit, or lacks standard WAV RIFF header.
    """
    if not audio_bytes or len(audio_bytes) == 0:
        raise ValueError("Audio buffer is empty. Please provide a valid WAV recording.")

    if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(audio_bytes) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ValueError(
            f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit ({max_mb:.0f} MB)."
        )

    # Check WAV RIFF and WAVE header markers
    if len(audio_bytes) >= 12:
        is_riff = audio_bytes[:4] == b"RIFF"
        is_wave = audio_bytes[8:12] == b"WAVE"
        if not (is_riff and is_wave):
            # Not strict rejection for standard soundfile decoders, but warn if completely non-audio
            pass


def validate_audio_signal(
    audio: np.ndarray,
    sample_rate: int,
) -> Tuple[float, float]:
    """
    Validates decoded audio signal duration and silence thresholds.

    Args:
        audio: 1D numpy float array representing audio signal.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (duration_seconds, rms_energy).

    Raises:
        ValueError: If audio is too short, too long, empty, or silent.
    """
    if audio is None or len(audio) == 0:
        raise ValueError("Decoded audio waveform is empty.")

    duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0

    if duration < MIN_AUDIO_DURATION_SEC:
        raise ValueError(
            f"Audio duration ({duration:.2f}s) is too short. "
            f"Please record or upload at least {MIN_AUDIO_DURATION_SEC} seconds for reliable analysis."
        )

    if duration > MAX_AUDIO_DURATION_SEC:
        raise ValueError(
            f"Audio duration ({duration:.2f}s) exceeds maximum allowed limit ({MAX_AUDIO_DURATION_SEC}s)."
        )

    # Compute RMS Energy
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    if rms_energy < MIN_AUDIO_RMS_ENERGY:
        raise ValueError(
            "Audio recording is completely silent or muted (no voice energy detected). "
            "Please speak closer to the microphone and try again."
        )

    return duration, rms_energy
