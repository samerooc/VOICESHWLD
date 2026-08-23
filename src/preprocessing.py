"""
VoiceShield Production Shared Preprocessing Engine (Phase 4).
Adheres strictly to configs/preprocessing.yaml and the immutable preprocessing contract.
Shared identically across training, inference, and batch evaluation.
"""

import io
import os
import tempfile
from typing import Any, Dict, Optional, Tuple, Union
import yaml
import librosa
import numpy as np
import soundfile as sf

from src.privacy import safe_delete_file

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "preprocessing.yaml")


def load_preprocessing_config(config_path: str = CONFIG_PATH) -> Dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "audio": {
            "target_sample_rate": 16000,
            "channels": 1,
            "dtype": "float32",
            "resampling_method": "soxr_hq",
        },
        "limits": {
            "min_duration_seconds": 0.50,
            "max_duration_seconds": 300.00,
            "min_rms_energy": 1e-5,
            "clipping_threshold": 0.99,
            "silence_threshold_db": 30.0,
        },
        "transformations": {
            "dc_offset_removal": True,
            "peak_normalization": True,
            "amplitude_scaling": True,
            "silence_trimming": {
                "enabled": True,
                "top_db": 30.0,
                "min_speech_duration_seconds": 0.40,
            },
        },
    }


def detect_audio_container_and_codec(audio_bytes: bytes, file_ext: str = ".wav") -> Tuple[str, str]:
    """
    Detects container format and underlying codec from bytes header.
    """
    if len(audio_bytes) < 12:
        return "unknown", "unknown"

    header_hex = audio_bytes[:12].hex()

    if audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "wav", "pcm_s16le"
    elif audio_bytes[:4] == b"fLaC":
        return "flac", "flac"
    elif audio_bytes[:4] == b"OggS":
        if b"OpusHead" in audio_bytes[:36]:
            return "ogg", "opus"
        return "ogg", "vorbis"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] in [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"]:
        return "mp3", "mp3"
    elif audio_bytes[4:8] in [b"ftyp", b"moov"] or "m4a" in file_ext.lower():
        return "m4a", "aac"

    ext_clean = file_ext.lstrip(".").lower()
    return ext_clean or "wav", "unknown"


def decode_audio_bytes_safely(audio_bytes: bytes, file_ext: str = ".wav") -> Tuple[np.ndarray, int, str, str]:
    """
    Safely decodes audio bytes into float32 array and native sample rate in-memory.
    Never persists audio permanently.
    """
    container, codec = detect_audio_container_and_codec(audio_bytes, file_ext=file_ext)
    audio_raw: Optional[np.ndarray] = None
    native_sr = 16000

    # Attempt 1: soundfile in-memory
    try:
        with io.BytesIO(audio_bytes) as bio:
            audio_raw, native_sr = sf.read(bio, dtype="float32")
    except Exception:
        pass

    # Attempt 2: pydub
    if audio_raw is None:
        try:
            from pydub import AudioSegment
            fmt = container if container != "unknown" else file_ext.lstrip(".").lower()
            if fmt == "opus":
                fmt = "ogg"
            seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt if fmt else None)
            seg = seg.set_channels(1)
            raw_data = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (seg.sample_width * 8 - 1)) if seg.sample_width > 0 else 32768.0
            audio_raw = raw_data / max_val
            native_sr = seg.frame_rate
        except Exception:
            pass

    # Attempt 3: temporary file with guaranteed immediate deletion
    if audio_raw is None:
        temp_path: Optional[str] = None
        suffix = file_ext if file_ext.startswith(".") else f".{file_ext}"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name
            audio_raw, native_sr = librosa.load(temp_path, sr=None, mono=False)
        finally:
            if temp_path:
                safe_delete_file(temp_path)

    if audio_raw is None or len(audio_raw) == 0:
        raise ValueError(f"Decoding failed for container '{container}' / codec '{codec}'.")

    return audio_raw, native_sr, container, codec


def preprocess_audio(
    audio: Union[np.ndarray, bytes],
    sample_rate: int = 16000,
    target_sr: int = 16000,
    file_ext: str = ".wav",
    config: Optional[Dict] = None,
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """
    Standard shared preprocessing execution matching Phase 4 contract specifications.
    Accepts raw audio array or raw bytes, normalizes, resamples, cleans, and extracts quality flags.
    """
    cfg = config or load_preprocessing_config()
    audio_cfg = cfg.get("audio", {})
    target_sample_rate = audio_cfg.get("target_sample_rate", target_sr)

    container = "wav"
    codec = "pcm"

    if isinstance(audio, (bytes, bytearray)):
        audio_arr, native_sr, container, codec = decode_audio_bytes_safely(bytes(audio), file_ext=file_ext)
        sample_rate = native_sr
    else:
        audio_arr = np.asarray(audio, dtype=np.float32)

    if audio_arr is None or len(audio_arr) == 0:
        raise ValueError("Audio array is empty or None.")

    # 1. Multi-channel to Mono downmix
    if audio_arr.ndim > 1:
        audio_arr = np.mean(audio_arr, axis=1 if audio_arr.shape[1] < audio_arr.shape[0] else 0)

    # 2. Float32 conversion & Finite Values Sanitization
    audio_arr = np.nan_to_num(audio_arr, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # 3. High quality resampling
    if sample_rate != target_sample_rate:
        audio_arr = librosa.resample(y=audio_arr, orig_sr=sample_rate, target_sr=target_sample_rate)
        effective_sr = target_sample_rate
    else:
        effective_sr = sample_rate

    # 4. DC Offset Subtraction
    if cfg.get("transformations", {}).get("dc_offset_removal", True):
        audio_arr = audio_arr - np.mean(audio_arr)

    # 5. Amplitude Peak Normalization
    max_amp = float(np.max(np.abs(audio_arr)))
    if cfg.get("transformations", {}).get("peak_normalization", True) and max_amp > 1e-6:
        audio_arr = audio_arr / (max_amp + 1e-9)

    # 6. Silence Trimming
    trim_cfg = cfg.get("transformations", {}).get("silence_trimming", {})
    if trim_cfg.get("enabled", True):
        top_db = trim_cfg.get("top_db", 30.0)
        min_speech = trim_cfg.get("min_speech_duration_seconds", 0.40)
        try:
            trimmed, _ = librosa.effects.trim(audio_arr, top_db=top_db)
            if len(trimmed) >= int(effective_sr * min_speech):
                audio_arr = trimmed
        except Exception:
            pass

    # 7. Quality Flags and Diagnostics
    duration = float(len(audio_arr) / effective_sr)
    rms_energy = float(np.sqrt(np.mean(audio_arr ** 2))) if len(audio_arr) > 0 else 0.0
    clipping_ratio = float(np.mean(np.abs(audio_arr) >= 0.99))
    silence_ratio = float(np.mean(np.abs(audio_arr) < 1e-4))

    quality_flags = []
    if rms_energy < 1e-4 or silence_ratio > 0.85:
        quality_flags.append("silent_or_faint")
    if clipping_ratio > 0.10:
        quality_flags.append("heavily_clipped")
    if duration < 1.0:
        quality_flags.append("very_short")
    if not quality_flags:
        quality_flags.append("acceptable")

    diagnostics = {
        "duration_seconds": round(duration, 3),
        "sample_rate": effective_sr,
        "rms_energy": round(rms_energy, 6),
        "peak_amplitude": round(float(np.max(np.abs(audio_arr))) if len(audio_arr) > 0 else 0.0, 4),
        "clipping_ratio": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 4),
        "quality_flags": quality_flags,
        "quality_status": "acceptable" if "acceptable" in quality_flags else "low_quality",
        "container": container,
        "codec": codec,
    }

    return audio_arr.astype(np.float32), effective_sr, diagnostics
