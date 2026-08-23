"""
VoiceShield Audio I/O Module (Phase 1 & Phase 9 Reliability Pass).
Reads, normalizes, and validates WAV audio in memory using soundfile and librosa.
Provides robust stereo-to-mono, bit-depth normalization, resampling, and clipping/SNR diagnostics.
"""

import io
import os
import tempfile
from typing import Dict, Optional, Tuple, Union
import librosa
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE
from src.privacy import safe_delete_file
from src.validation import validate_audio_signal, validate_wav_bytes


def estimate_snr_db(audio: np.ndarray) -> float:
    """
    Estimates Signal-to-Noise Ratio (SNR) in dB using 10th percentile energy as noise floor estimate.
    """
    if len(audio) == 0:
        return 0.0
    frame_len = 512
    hop_len = 256
    frames = librosa.util.frame(audio, frame_length=frame_len, hop_length=hop_len)
    frame_energy = np.mean(frames ** 2, axis=0) + 1e-12
    signal_power = np.mean(frame_energy)
    noise_power = np.percentile(frame_energy, 10) + 1e-12
    snr = 10.0 * np.log10(signal_power / noise_power)
    return float(np.clip(snr, 0.0, 60.0))


def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
    sample_rate: Optional[int] = None,
    file_ext: str = ".wav",
    *args,
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """
    Loads audio from in-memory bytes across multiple formats (.wav, .mp3, .mp4, .m4a, .ogg, .flac),
    converts to mono, resamples to target_sr, and runs signal validation checks.
    """
    if sample_rate is not None:
        target_sr = sample_rate
    elif "sr" in kwargs and kwargs["sr"] is not None:
        target_sr = kwargs["sr"]

    if "ext" in kwargs and kwargs["ext"]:
        file_ext = kwargs["ext"]

    validate_wav_bytes(audio_bytes)

    audio: Optional[np.ndarray] = None
    native_sr: int = target_sr

    # Attempt 1: Direct in-memory decoding via soundfile (Fastest, zero-disk touch for WAV, OGG, FLAC)
    try:
        with io.BytesIO(audio_bytes) as bio:
            audio_raw, native_sr = sf.read(bio, dtype="float32")
            if audio_raw.ndim > 1:
                audio_raw = np.mean(audio_raw, axis=1)

            audio_raw = np.nan_to_num(audio_raw, nan=0.0, posinf=1.0, neginf=-1.0)
            if np.max(np.abs(audio_raw)) > 1.0:
                audio_raw = audio_raw / (np.max(np.abs(audio_raw)) + 1e-9)

            if native_sr != target_sr:
                audio = librosa.resample(y=audio_raw, orig_sr=native_sr, target_sr=target_sr)
            else:
                audio = audio_raw
    except Exception:
        pass

    # Attempt 2: In-memory decoding via pydub (Handles OGG Opus, Vorbis, WebM, AAC, M4A without disk writes)
    if audio is None:
        try:
            from pydub import AudioSegment
            fmt = file_ext.lstrip(".").lower()
            if fmt == "opus":
                fmt = "ogg"
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt if fmt else None)
            seg = seg.set_channels(1)
            raw_data = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (seg.sample_width * 8 - 1)) if seg.sample_width > 0 else 32768.0
            audio_raw = np.nan_to_num(raw_data / max_val, nan=0.0, posinf=1.0, neginf=-1.0)
            native_sr = seg.frame_rate

            if native_sr != target_sr:
                audio = librosa.resample(y=audio_raw, orig_sr=native_sr, target_sr=target_sr)
            else:
                audio = audio_raw
        except Exception:
            pass

    # Attempt 3: Ephemeral tempfile with guaranteed cleanup via librosa.load
    if audio is None:
        temp_path: Optional[str] = None
        suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            audio, native_sr = librosa.load(temp_path, sr=target_sr, mono=True)
        except Exception as err:
            raise ValueError(f"Unable to decode audio format ({suffix}): {err}")
        finally:
            safe_delete_file(temp_path)

    if audio is None or len(audio) == 0:
        raise ValueError("Decoded audio waveform is empty.")

    # Validate signal duration and silence
    validate_audio_signal(audio, target_sr)
    return audio.astype(np.float32), target_sr


def load_audio_from_file(
    file_path: str,
    target_sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """
    Loads a local audio file from disk (.wav, .mp3, .mp4, .m4a, .ogg, .flac).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: '{file_path}'")

    ext = os.path.splitext(file_path)[1].lower()
    from src.config import SUPPORTED_AUDIO_EXTENSIONS
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: '{ext}'. Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}"
        )

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    return load_audio_from_bytes(audio_bytes, target_sr=target_sr, file_ext=ext)


def get_audio_metadata(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Dict[str, Any]:
    """
    Computes comprehensive acoustic quality diagnostics (duration, sample rate, RMS energy,
    peak amplitude, clipping ratio, silence ratio, estimated SNR).
    """
    if len(audio) == 0:
        return {
            "duration_seconds": 0.0,
            "sample_rate": sample_rate,
            "rms_energy": 0.0,
            "peak_amplitude": 0.0,
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "snr_db": 0.0,
            "zero_crossing_rate": 0.0,
            "num_samples": 0,
            "quality_flag": "empty",
        }

    duration = float(len(audio) / sample_rate) if sample_rate > 0 else 0.0
    rms_energy = float(np.sqrt(np.mean(audio ** 2)))
    peak_amplitude = float(np.max(np.abs(audio)))
    clipping_ratio = float(np.mean(np.abs(audio) >= 0.99))
    silence_ratio = float(np.mean(np.abs(audio) < 1e-4))
    snr_db = estimate_snr_db(audio)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))

    # Quality flag
    if rms_energy < 1e-4 or silence_ratio > 0.85:
        quality_flag = "silent_or_faint"
    elif clipping_ratio > 0.10:
        quality_flag = "heavily_clipped"
    elif duration < 1.0:
        quality_flag = "very_short"
    else:
        quality_flag = "acceptable"

    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sample_rate,
        "rms_energy": round(rms_energy, 6),
        "peak_amplitude": round(peak_amplitude, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "snr_db": round(snr_db, 2),
        "zero_crossing_rate": round(zcr, 6),
        "num_samples": len(audio),
        "quality_flag": quality_flag,
    }
