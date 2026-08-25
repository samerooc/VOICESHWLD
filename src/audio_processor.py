"""
VoiceShield Phase 1 — Universal In-Memory Audio Processor.

Provides zero-disk-write audio ingestion, decoding, resampling, per-utterance
zero-mean unit-variance normalization, and voiced-segment VAD isolation via
Short-Time Energy (STE) + Zero-Crossing Rate (ZCR) with a 30 ms window / 10 ms hop.

Supported input formats (in-memory bytes):
  WAV · MP3 · M4A · FLAC · OGG · WebM/Opus · AAC · 8 kHz raw PCM · μ-law (G.711)

Fallback decode chain:
  1. soundfile  (fastest; handles WAV, FLAC, OGG natively)
  2. pydub / ffmpeg  (MP3, M4A, AAC, WebM/Opus, any container ffmpeg can read)
  3. audioop / audioop-lts  (G.711 μ-law headerless telephony; Python ≥ 3.13 uses audioop-lts)
  4. Raw headerless 8 kHz 16-bit PCM (last resort)

Author:  VoiceShield Engineering
Version: 1.4.0  (Phase 1 hardened)
"""

from __future__ import annotations

import io
import importlib
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000           # Target sample rate (Hz)
VAD_FRAME_LEN: int = 480           # 30 ms at 16 kHz
VAD_HOP_LEN: int = 160             # 10 ms at 16 kHz
MIN_VOICED_DURATION_SEC: float = 0.40
MIN_AUDIO_RMS_ENERGY: float = 0.0015
MIN_AUDIO_SNR_DB: float = 3.0

# Header magic bytes for container-format detection
_CONTAINER_MAGIC = (
    b"RIFF",          # WAV
    b"ID3",           # MP3 with ID3 tag
    b"\xff\xfb",      # MP3 sync word
    b"\xff\xf3",      # MP3 sync word (MPEG-2 Layer 3)
    b"\xff\xf2",      # MP3 sync word
    b"OggS",          # OGG container
    b"fLaC",          # FLAC
    b"\x1aE\xdf\xa3", # Matroska / WebM
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_known_container(audio_bytes: bytes) -> bool:
    """Return True if the byte stream begins with a known container header."""
    for magic in _CONTAINER_MAGIC:
        if audio_bytes[: len(magic)] == magic:
            return True
    # M4A / AAC / MP4: 'ftyp' box within first 12 bytes
    if b"ftyp" in audio_bytes[:12]:
        return True
    return False


def _try_audioop_ulaw(audio_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
    """
    Attempt G.711 mu-law decoding using stdlib ``audioop`` (Python <= 3.12) or
    the drop-in ``audioop-lts`` package (Python >= 3.13).

    Returns ``(float32_samples, 8000)`` on success or ``None`` on failure.
    """
    audioop_mod = None
    for mod_name in ("audioop", "audioop_lts"):
        try:
            audioop_mod = importlib.import_module(mod_name)
            break
        except ImportError:
            continue

    if audioop_mod is None:
        return None

    try:
        pcm16_bytes = audioop_mod.ulaw2lin(audio_bytes, 2)  # 2 = 16-bit output
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, 8000
    except Exception as exc:
        log.debug("audioop mu-law decode failed: %s", exc)
        return None


def _try_raw_pcm_8k(audio_bytes: bytes) -> Optional[Tuple[np.ndarray, int]]:
    """
    Treat the byte stream as headerless 8 kHz 16-bit little-endian signed PCM.
    Only attempted when the byte count is a valid integer number of int16 samples.
    """
    if len(audio_bytes) % 2 != 0 or len(audio_bytes) < 320:  # need >= 20 ms
        return None
    try:
        samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
        return samples, 8000
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core DSP functions
# ---------------------------------------------------------------------------

def compute_snr_db(
    audio: np.ndarray,
    frame_length: int = VAD_FRAME_LEN,
    hop_length: int = VAD_HOP_LEN,
) -> float:
    """
    Estimate Signal-to-Noise Ratio (dB) from short-time RMS energy frames.

    Signal power  -- 95th percentile of per-frame energy.
    Noise floor   -- 10th percentile of per-frame energy (stationary noise proxy).

    If the signal is uniformly high-energy (ratio < 2x) a synthetic noise floor
    is derived at -40 dB below signal level so clean tones report realistic SNR.

    Returns:
        SNR in dB, clipped to [0, 80].
    """
    if len(audio) == 0:
        return 0.0

    fl = min(frame_length, len(audio))
    hl = max(1, fl // 3)

    energy = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0] ** 2
    if len(energy) == 0 or float(np.max(energy)) == 0.0:
        return 0.0

    signal_p = float(np.percentile(energy, 95))
    noise_p = float(np.percentile(energy, 10))

    # Synthetic noise floor for continuous high-energy signals (e.g. pure tones)
    if signal_p > 1e-6 and signal_p / max(1e-12, noise_p) < 2.0:
        noise_p = signal_p * 1e-4

    noise_p = max(noise_p, 1e-12)
    snr = 10.0 * float(np.log10(max(1e-10, signal_p / noise_p)))
    return float(np.clip(snr, 0.0, 80.0))


def extract_voiced_segments(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_length: int = VAD_FRAME_LEN,
    hop_length: int = VAD_HOP_LEN,
    energy_threshold_percentile: float = 20.0,
) -> Tuple[np.ndarray, float]:
    """
    Strict Voice Activity Detection (VAD) using Short-Time Energy (STE) and
    Zero-Crossing Rate (ZCR).

    Frame parameters follow the Phase 1 specification:
      Window: 30 ms  ->  480 samples at 16 kHz
      Hop:    10 ms  ->  160 samples at 16 kHz

    A frame is classified as *voiced* when:
      RMS > adaptive_threshold  AND  ZCR < 0.45

    The adaptive threshold is:
      min_rms + 0.15 * (max_rms - min_rms)

    where ``min_rms`` is the ``energy_threshold_percentile``-th percentile of
    per-frame RMS values (default 20th), providing robustness to slowly varying
    noise floors without fixed numerical thresholds.

    Fallback: if no voiced frames are found, ``librosa.effects.trim`` is applied
    at -25 dBFS before returning to prevent zero-length output.

    Returns:
        (voiced_audio: np.ndarray[float32], voiced_ratio: float in [0, 1])
    """
    if len(audio) == 0:
        return np.array([], dtype=np.float32), 0.0

    fl = min(frame_length, len(audio))
    hl = max(1, fl // 3)

    rms = librosa.feature.rms(y=audio, frame_length=fl, hop_length=hl)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=fl, hop_length=hl)[0]

    n_frames = min(len(rms), len(zcr))
    rms, zcr = rms[:n_frames], zcr[:n_frames]

    min_rms = float(np.percentile(rms, energy_threshold_percentile)) if n_frames > 0 else 0.0
    max_rms = float(np.max(rms)) if n_frames > 0 else 0.0
    thresh = min_rms + 0.15 * (max_rms - min_rms)

    voiced_mask_frames = (rms > thresh) & (zcr < 0.45)
    voiced_indices = np.where(voiced_mask_frames)[0]

    if len(voiced_indices) == 0:
        # Energy-based fallback: trim leading/trailing silence at -25 dBFS
        try:
            trimmed, _ = librosa.effects.trim(audio, top_db=25)
            voiced_audio = trimmed if len(trimmed) >= int(0.10 * sr) else audio
        except Exception:
            voiced_audio = audio
    else:
        # Build sample-level voiced mask by OR-ing frame intervals
        sample_mask = np.zeros(len(audio), dtype=bool)
        for idx in voiced_indices:
            s = int(idx) * hl
            e = min(len(audio), s + fl)
            sample_mask[s:e] = True
        voiced_audio = audio[sample_mask]

        # Safety: if VAD leaves less than 200 ms, fall back to full audio
        if len(voiced_audio) < int(0.20 * sr):
            voiced_audio = audio

    voiced_ratio = float(len(voiced_audio)) / max(1, float(len(audio)))
    return voiced_audio.astype(np.float32), float(np.clip(voiced_ratio, 0.0, 1.0))


def normalize_audio_standard(audio: np.ndarray) -> np.ndarray:
    r"""
    Per-utterance zero-mean unit-variance normalization:

        x_hat = (x - mu) / (sigma + 1e-7)

    Applied independently per utterance so that quiet and loud recordings
    share the same dynamic range entering the classifier.

    Returns:
        float32 array with mean ~= 0 and std ~= 1.
    """
    if len(audio) == 0:
        return audio.astype(np.float32)
    audio = audio.astype(np.float32)
    mu = float(np.mean(audio))
    sigma = float(np.std(audio))
    return ((audio - mu) / (sigma + 1e-7)).astype(np.float32)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def decode_and_sanitize_audio(
    audio_input: Union[bytes, np.ndarray, str],
    target_sr: int = SAMPLE_RATE,
    orig_sr: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Universal zero-persistence in-memory audio decoder.

    Supported container / codec inputs
    -----------------------------------
    - bytes / bytearray  -- any audio container (see fallback chain below)
    - np.ndarray          -- pre-decoded float32 PCM (resampled if orig_sr given)
    - str                 -- filesystem path (read once, treated as bytes)

    Fallback decode chain for byte streams
    ---------------------------------------
    1. soundfile      -- WAV, FLAC, OGG/Vorbis, OGG/OPUS
    2. pydub / ffmpeg -- MP3, M4A, AAC, WebM/Opus, WAV (fallback)
    3. G.711 mu-law   -- headerless telephony (audioop / audioop-lts)
    4. Raw PCM 8 kHz  -- headerless 16-bit little-endian PCM (last resort)

    Post-decode pipeline
    --------------------
    1. Flatten to mono (mean of channels if stereo/multi-channel).
    2. Resample to target_sr via librosa kaiser_fast.
    3. Sanitise NaN / Inf; peak-limit to [-1, 1] if clipping detected.
    4. Run 30 ms / 10 ms STE + ZCR VAD -> voiced segment slice.
    5. Compute diagnostics.

    Returns
    -------
    full_audio : np.ndarray[float32]
        Full waveform at target_sr, range [-1, 1].
    voiced_audio : np.ndarray[float32]
        Voiced-only waveform (VAD-sliced) at target_sr.
    diagnostics : dict
        Keys: original_sr, duration_sec, voiced_sec, snr_db, is_clipped, is_silent,
        rms_energy, voiced_ratio, sample_rate, num_samples.
    """
    # ------------------------------------------------------------------
    # Input normalisation: str path -> bytes
    # ------------------------------------------------------------------
    if isinstance(audio_input, str):
        if not os.path.exists(audio_input):
            raise FileNotFoundError(f"Audio file not found: {audio_input!r}")
        with open(audio_input, "rb") as fh:
            audio_input = fh.read()

    # ------------------------------------------------------------------
    # Branch A: numpy array (already decoded PCM)
    # ------------------------------------------------------------------
    if isinstance(audio_input, np.ndarray):
        raw_data = np.nan_to_num(audio_input.flatten(), nan=0.0).astype(np.float32)
        native_sr: int = orig_sr or target_sr
        if native_sr != target_sr:
            raw_data = librosa.resample(
                raw_data, orig_sr=native_sr, target_sr=target_sr, res_type="kaiser_fast"
            )

    # ------------------------------------------------------------------
    # Branch B: raw byte stream
    # ------------------------------------------------------------------
    else:
        if not audio_input:
            raise ValueError("Audio byte payload is empty.")

        raw_data = None
        native_sr = target_sr  # will be overwritten by successful decoder

        # Attempt 1: soundfile (WAV, FLAC, OGG/Vorbis, OGG/OPUS)
        try:
            stream = io.BytesIO(bytes(audio_input))
            data_sf, native_sr_sf = sf.read(stream, dtype="float32", always_2d=False)
            if data_sf.ndim > 1:
                data_sf = np.mean(data_sf, axis=1)
            raw_data = data_sf
            native_sr = int(native_sr_sf)
            log.debug("soundfile decode OK: sr=%d frames=%d", native_sr, len(raw_data))
        except Exception as exc:
            log.debug("soundfile decode failed (%s); trying pydub.", exc)

        # Attempt 2: pydub / ffmpeg (MP3, M4A, AAC, WebM/Opus, etc.)
        if raw_data is None:
            try:
                stream = io.BytesIO(bytes(audio_input))
                seg = AudioSegment.from_file(stream)
                samples = np.array(seg.get_array_of_samples(), dtype=np.int32)
                if seg.channels > 1:
                    samples = samples.reshape(-1, seg.channels).mean(axis=1)
                scale = float(1 << (8 * seg.sample_width - 1))
                raw_data = samples.astype(np.float32) / scale
                native_sr = int(seg.frame_rate)
                log.debug("pydub decode OK: sr=%d samples=%d", native_sr, len(raw_data))
            except Exception as exc:
                log.debug("pydub decode failed (%s); trying mu-law.", exc)

        # Fail fast for recognised containers that both decoders rejected
        if raw_data is None and _is_known_container(bytes(audio_input)):
            raise ValueError(
                "Failed to decode audio in-memory: recognised container header but "
                "both soundfile and pydub/ffmpeg failed. The file may be corrupted "
                "or ffmpeg may not be installed."
            )

        # Attempt 3: G.711 mu-law telephony (audioop / audioop-lts)
        if raw_data is None:
            result = _try_audioop_ulaw(bytes(audio_input))
            if result is not None:
                raw_data, native_sr = result
                log.debug("mu-law decode OK: %d samples @ 8kHz", len(raw_data))

        # Attempt 4: headerless raw PCM 8 kHz 16-bit (last resort)
        if raw_data is None:
            result = _try_raw_pcm_8k(bytes(audio_input))
            if result is not None:
                raw_data, native_sr = result
                log.debug("raw PCM 8kHz fallback OK: %d samples", len(raw_data))

        if raw_data is None or len(raw_data) == 0:
            raise ValueError(
                "Failed to decode audio payload in-memory: "
                "unsupported or corrupt audio stream. "
                "Ensure ffmpeg is installed for MP3/M4A/AAC/WebM support."
            )

        # Resample to target_sr if needed
        if native_sr != target_sr:
            raw_data = librosa.resample(
                raw_data, orig_sr=native_sr, target_sr=target_sr, res_type="kaiser_fast"
            )

    # ------------------------------------------------------------------
    # Sanitise and bound amplitude to [-1.0, 1.0]
    # ------------------------------------------------------------------
    raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    peak = float(np.max(np.abs(raw_data))) if len(raw_data) > 0 else 0.0
    if peak > 1.0:
        raw_data = (raw_data / peak).astype(np.float32)

    # ------------------------------------------------------------------
    # VAD: voiced segment extraction (30 ms window, 10 ms hop)
    # ------------------------------------------------------------------
    voiced_data, voiced_ratio = extract_voiced_segments(
        raw_data, sr=target_sr, frame_length=VAD_FRAME_LEN, hop_length=VAD_HOP_LEN
    )

    # ------------------------------------------------------------------
    # Diagnostics computation
    # ------------------------------------------------------------------
    duration_sec = float(len(raw_data) / target_sr)
    voiced_sec = float(len(voiced_data) / target_sr)
    snr_db = compute_snr_db(raw_data, frame_length=VAD_FRAME_LEN, hop_length=VAD_HOP_LEN)
    rms_energy = float(np.sqrt(np.mean(raw_data ** 2))) if len(raw_data) > 0 else 0.0

    # is_silent: triggered when voiced duration < 0.4 s  OR  SNR < 3 dB
    # per Phase 1 specification
    is_silent = bool(
        rms_energy < MIN_AUDIO_RMS_ENERGY
        or voiced_sec < MIN_VOICED_DURATION_SEC
        or snr_db < MIN_AUDIO_SNR_DB
    )
    # is_clipped: peak at or above full-scale before normalisation
    is_clipped = bool(peak >= 0.999)

    diagnostics: Dict[str, Any] = {
        "original_sr":        int(native_sr),
        "duration_sec":       round(duration_sec, 3),
        "voiced_sec":         round(voiced_sec, 3),
        "voiced_duration_sec": round(voiced_sec, 3),   # alias for backward compat
        "voiced_ratio":       round(voiced_ratio, 3),
        "snr_db":             round(max(0.0, snr_db), 2),
        "rms_energy":         round(rms_energy, 6),
        "is_silent":          is_silent,
        "is_clipped":         is_clipped,
        "sample_rate":        target_sr,
        "num_samples":        int(len(raw_data)),
    }

    return raw_data, voiced_data, diagnostics


# ---------------------------------------------------------------------------
# Public API — Class interface
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    Universal In-Memory Audio Ingestion, Resampling, Channel Normalization,
    and Voiced-Segment VAD Processor (VoiceShield Phase 1).

    Usage::

        processor = AudioProcessor(target_sr=16000)
        full_audio, voiced_audio, diag = processor.load_audio_from_bytes(audio_bytes)

    Parameters
    ----------
    target_sr : int
        Output sample rate in Hz (default 16000).
    """

    def __init__(self, target_sr: int = SAMPLE_RATE) -> None:
        self.target_sr = target_sr

    def load_audio_from_bytes(
        self,
        audio_bytes: bytes,
        target_sr: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Decode in-memory audio bytes, resample to target_sr, and extract
        voiced segments via 30 ms / 10 ms STE + ZCR VAD.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio container bytes (WAV/MP3/M4A/FLAC/OGG/WebM/AAC/mu-law/PCM).
        target_sr : int, optional
            Override the instance-level target sample rate.

        Returns
        -------
        full_audio : np.ndarray[float32]
            Complete waveform at target_sr, bounded to [-1, 1].
        voiced_audio : np.ndarray[float32]
            VAD-extracted voiced speech segments.
        diagnostics : dict
            Keys: original_sr, duration_sec, voiced_sec, snr_db,
                  is_clipped, is_silent, rms_energy, sample_rate, num_samples.
        """
        sr = target_sr if target_sr is not None else self.target_sr
        return decode_and_sanitize_audio(audio_bytes, target_sr=sr)


# ---------------------------------------------------------------------------
# Functional backward-compatibility helpers
# ---------------------------------------------------------------------------

def load_audio_from_bytes(
    audio_bytes: bytes,
    target_sr: int = SAMPLE_RATE,
    file_ext: Optional[str] = None,          # retained for API surface compatibility
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Functional helper — backward-compatible wrapper returning (raw_audio, diagnostics).
    The voiced_audio tensor is discarded; use ``decode_and_sanitize_audio`` for full output.
    """
    raw_audio, _voiced, diag = decode_and_sanitize_audio(audio_bytes, target_sr=target_sr)
    return raw_audio, diag
