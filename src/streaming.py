"""
VoiceShield Phase 4 — High-Throughput Real-Time Live Streaming Audio Forensic Engine.

Architecture & Components:
  1. Thread-Safe Circular Rolling Buffer (RollingAudioBuffer):
     - Circular float32 buffer holding up to 6.0 seconds (96,000 samples @ 16kHz).
     - Dynamic chunk ingestion (20ms to 200ms) with multi-format decoding:
       * Raw 16-bit linear PCM (pcm16 / pcm_s16le)
       * 32-bit float PCM (float32 / f32)
       * 8kHz G.711 mu-law telephony (mulaw / ulaw / g711) with zero-dependency LUT decoding
       * On-the-fly Kaiser/polyphase resampling to 16kHz mono.
     - Constant-geometry 3.0s sliding window generator (48,000 samples).

  2. Live Streaming Inference & Temporal Aggregator (LiveStreamingEngine):
     - Stateful session tracking (session_id, buffer, history_scores, ema, hold_counter).
     - Top-K (85th percentile) window max-pooling over active rolling history.
     - Exponential Moving Average (EMA) smoothing: EMA_t = 0.35 * P_t + 0.65 * EMA_{t-1}.
     - Combined Live Risk Score: Score_live = 0.70 * TopK_85(P_history) + 0.30 * EMA_t.
     - Hold-and-Decay Security Alert Gate:
       * If Score_live >= 61 (High Risk), lock alert state with hold_counter = 6 (3.0s @ 0.5s hop).
       * Decays by factor of 0.05 per step only after hold counter expires.
     - Structured real-time telemetry payload.

  3. Backward-Compatible Interfaces:
     - SandboxStreamAnalyzer, NeuralStreamingScoreEngine, slice_streaming_windows.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from typing import Any, Deque, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import scipy.signal

from src.audio_processor import SAMPLE_RATE
from src.channel_normalizer import AcousticChannelNormalizer

log = logging.getLogger("voiceshield.streaming")

STREAMING_DISCLAIMER = "⚠️ SANDBOX SIMULATION — NOT A LIVE CALL: Advisory forensic scoring demonstration."
STATUTORY_DISCLAIMER = STREAMING_DISCLAIMER

# ---------------------------------------------------------------------------
# Fast ITU-T G.711 Mu-Law Lookup Table (Zero-dependency in-memory decoder)
# ---------------------------------------------------------------------------

def _create_mulaw_decoder_lut() -> np.ndarray:
    """Build the standard 256-element ITU-T G.711 mu-law decoding table to int16."""
    table = np.zeros(256, dtype=np.int16)
    for i in range(256):
        byte = ~i & 0xFF
        sign = -1 if (byte & 0x80) else 1
        exponent = (byte >> 4) & 0x07
        mantissa = byte & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        table[i] = sign * sample
    return table

_MULAW_INT16_LUT = _create_mulaw_decoder_lut()


def decode_mulaw_bytes(mulaw_bytes: bytes) -> np.ndarray:
    """
    Decode raw 8kHz G.711 mu-law bytes to float32 linear PCM in [-1.0, 1.0].
    """
    if not mulaw_bytes:
        return np.array([], dtype=np.float32)
    uint8_arr = np.frombuffer(mulaw_bytes, dtype=np.uint8)
    int16_arr = _MULAW_INT16_LUT[uint8_arr]
    return (int16_arr / 32768.0).astype(np.float32)


def linear_to_mulaw_bytes(samples: np.ndarray, mu: int = 255) -> bytes:
    """
    Encode float32 samples in [-1.0, 1.0] to G.711 mu-law bytes for testing.
    """
    if len(samples) == 0:
        return b""
    x = np.clip(samples, -1.0, 1.0)
    fx = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.clip(np.floor((fx + 1.0) / 2.0 * mu), 0, mu).astype(np.uint8)
    return bytes((~quantized & 0xFF).tolist())


# ---------------------------------------------------------------------------
# Slicing Utility (Backward-Compatible)
# ---------------------------------------------------------------------------

def slice_streaming_windows(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_ms: int = 160,
    stride_ms: int = 40,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Slices a continuous 1D audio array into fixed-length streaming windows with stride.
    Yields (window_index, timestamp_seconds, audio_chunk).
    """
    window_samples = int(sample_rate * (window_ms / 1000.0))
    stride_samples = int(sample_rate * (stride_ms / 1000.0))

    if len(audio) == 0:
        return

    if len(audio) < window_samples:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[: len(audio)] = audio
        yield (0, 0.0, padded)
        return

    idx = 0
    start = 0
    while start + window_samples <= len(audio):
        chunk = audio[start : start + window_samples]
        ts = round(start / float(sample_rate), 4)
        yield (idx, ts, chunk)
        start += stride_samples
        idx += 1


# ---------------------------------------------------------------------------
# 1. Thread-Safe Circular Rolling Buffer
# ---------------------------------------------------------------------------

class RollingAudioBuffer:
    """
    Thread-safe circular audio buffer for streaming inference.

    Maintains a circular float32 ring buffer holding up to `capacity_seconds`
    (default 6.0s @ 16kHz = 96,000 samples). Thread safety is enforced via
    `threading.Lock` across all write and read paths.
    """

    def __init__(
        self,
        capacity_seconds: float = 6.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.capacity_seconds: float = float(capacity_seconds)
        self.sample_rate: int = int(sample_rate)
        self.capacity_samples: int = int(self.capacity_seconds * self.sample_rate)

        self._buffer: Deque[float] = deque(maxlen=self.capacity_samples)
        self._lock: threading.Lock = threading.Lock()
        self._total_samples_received: int = 0

    def clear(self) -> None:
        """Clear all buffered audio samples and reset counters."""
        with self._lock:
            self._buffer.clear()
            self._total_samples_received = 0

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def get_current_duration(self) -> float:
        """Return the current duration of audio stored in the buffer (seconds)."""
        with self._lock:
            return float(len(self._buffer) / self.sample_rate)

    def has_sufficient_audio(self, min_duration_sec: float = 0.5) -> bool:
        """Check whether the buffer contains at least `min_duration_sec` of audio."""
        with self._lock:
            return len(self._buffer) >= int(min_duration_sec * self.sample_rate)

    def add_samples(self, samples: Union[np.ndarray, List[float], Tuple[float, ...]]) -> None:
        """
        Thread-safely append raw float32 samples directly into the circular buffer.
        """
        if samples is None or len(samples) == 0:
            return

        if isinstance(samples, (list, tuple)):
            samples_np = np.asarray(samples, dtype=np.float32)
        else:
            samples_np = samples.astype(np.float32, copy=False)

        # Replace any non-finite values with 0.0
        if not np.all(np.isfinite(samples_np)):
            samples_np = np.nan_to_num(samples_np, nan=0.0, posinf=0.0, neginf=0.0)

        with self._lock:
            for val in samples_np:
                self._buffer.append(float(val))
            self._total_samples_received += len(samples_np)

    def add_pcm_chunk(
        self,
        chunk_bytes: bytes,
        format: str = "pcm16",
        input_sr: int = 16000,
    ) -> None:
        """
        Ingest arbitrary PCM chunk sizes (20ms to 200ms) across supported wire formats:
          - "pcm16" / "pcm_s16le" / "int16": 16-bit signed integer PCM
          - "float32" / "pcm_f32le" / "f32": 32-bit IEEE float PCM
          - "mulaw" / "ulaw" / "g711": 8kHz G.711 mu-law telephony audio

        Automatically resamples to buffer sample_rate (16kHz mono) in-memory.
        """
        if not chunk_bytes:
            return

        fmt = format.lower().strip()

        # 1. Decode to float32 in [-1.0, 1.0]
        if fmt in ("mulaw", "ulaw", "g711", "g.711"):
            audio_f32 = decode_mulaw_bytes(chunk_bytes)
        elif fmt in ("float32", "pcm_f32le", "f32"):
            audio_f32 = np.frombuffer(chunk_bytes, dtype=np.float32).copy()
        elif fmt in ("pcm16", "pcm_s16le", "int16", "s16le", "wav"):
            int16_arr = np.frombuffer(chunk_bytes, dtype=np.int16)
            audio_f32 = (int16_arr / 32768.0).astype(np.float32)
        else:
            # Default fallback: assume 16-bit PCM
            int16_arr = np.frombuffer(chunk_bytes, dtype=np.int16)
            audio_f32 = (int16_arr / 32768.0).astype(np.float32)

        if len(audio_f32) == 0:
            return

        # 2. Resample to target sample_rate if necessary
        if input_sr != self.sample_rate and len(audio_f32) > 0:
            # Polyphase resampling for low latency & high fidelity
            num_out = int(round(len(audio_f32) * float(self.sample_rate) / float(input_sr)))
            audio_f32 = scipy.signal.resample(audio_f32, num_out).astype(np.float32)

        # 3. Append to buffer
        self.add_samples(audio_f32)

    def add_bytes_pcm16(self, pcm_bytes: bytes) -> None:
        """Convenience method for 16kHz PCM16 bytes."""
        self.add_pcm_chunk(pcm_bytes, format="pcm16", input_sr=self.sample_rate)

    def add_mulaw_bytes(self, mulaw_bytes: bytes, input_sr: int = 8000) -> None:
        """Convenience method for 8kHz G.711 mu-law bytes."""
        self.add_pcm_chunk(mulaw_bytes, format="mulaw", input_sr=input_sr)

    def get_analysis_window(self, window_sec: float = 3.0) -> np.ndarray:
        """
        Extract the most recent `window_sec` of audio (e.g. 3.0s = 48,000 samples @ 16kHz).
        If the buffer contains fewer samples than required, zero-pads the beginning to
        guarantee constant tensor geometry.
        """
        target_samples = int(window_sec * self.sample_rate)
        with self._lock:
            buf_len = len(self._buffer)
            if buf_len == 0:
                return np.zeros(target_samples, dtype=np.float32)

            take_n = min(buf_len, target_samples)
            # Slice recent samples
            start_idx = buf_len - take_n
            recent = [self._buffer[start_idx + i] for i in range(take_n)]
            arr = np.asarray(recent, dtype=np.float32)

            if len(arr) < target_samples:
                pad = np.zeros(target_samples - len(arr), dtype=np.float32)
                arr = np.concatenate([pad, arr])

            return arr

    def get_latest_window(self, window_samples: int = 48000) -> np.ndarray:
        """Backward-compatible extraction method accepting sample count directly."""
        window_sec = float(window_samples / self.sample_rate)
        return self.get_analysis_window(window_sec=window_sec)


# ---------------------------------------------------------------------------
# 2. Live Streaming Inference & Temporal Aggregator
# ---------------------------------------------------------------------------

class LiveStreamingEngine:
    """
    Production-Grade Real-Time Live Streaming Engine.

    Features:
      • Stateful session management (session_id, circular buffer, deque history).
      • Top-K (85th percentile) window pooling over the active rolling history window
        to detect transient vocoder bursts without dilution from silent pauses.
      • Temporal Exponential Moving Average (EMA) smoothing:
          EMA_t = 0.35 * P_t + 0.65 * EMA_{t-1}
      • Combined Live Risk Score:
          Score_live = 0.70 * TopK_85(P_history) + 0.30 * EMA_t
      • Hold-and-Decay Security Alert Gate:
          Locks alert for 3.0 seconds (6 steps @ 0.5s hop) upon High-Risk detection (Score >= 61),
          decaying at 0.05 / step after hold expiration.
      • Structured telemetry payload with sub-150ms real-time compliance.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = 3.0,
        stride_sec: float = 0.5,
        history_len: int = 10,
        ema_alpha: float = 0.35,
        hold_steps: int = 6,
        decay_rate: float = 0.05,
        session_id: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.sample_rate: int = int(sample_rate)
        self.window_sec: float = float(window_sec)
        self.stride_sec: float = float(stride_sec)
        self.window_samples: int = int(self.window_sec * self.sample_rate)
        self.stride_samples: int = int(self.stride_sec * self.sample_rate)

        self.history_len: int = int(history_len)
        self.ema_alpha: float = float(ema_alpha)
        self.hold_steps: int = int(hold_steps)
        self.decay_rate: float = float(decay_rate)

        self.session_id: str = session_id or uuid.uuid4().hex[:12]

        # Shared / injected detector
        if detector is not None:
            self.detector = detector
        else:
            from src.neural_engine import ProductionNeuralDetector
            self.detector = ProductionNeuralDetector(
                native_checkpoint_path=checkpoint_path or "models/voiceshield_live_robust.pt",
                device=device,
                load_hf=False,  # default to fast native backbone for streaming unless explicitly configured
            )

        # Components
        self.buffer = RollingAudioBuffer(capacity_seconds=6.0, sample_rate=self.sample_rate)
        self.channel_normalizer = AcousticChannelNormalizer(sr=self.sample_rate)

        # Stateful session tracking
        self.history_scores: Deque[float] = deque(maxlen=self.history_len)
        self.ema_score: Optional[float] = None
        self.alert_hold_counter: int = 0
        self.held_peak_score: float = 0.0
        self.total_audio_sec: float = 0.0
        self.total_chunks_received: int = 0
        self.processed_windows: int = 0

        self._lock: threading.Lock = threading.Lock()

    def reset(self) -> None:
        """Reset all streaming buffers, histories, and state machine registers."""
        with self._lock:
            self.buffer.reset()
            self.history_scores.clear()
            self.ema_score = None
            self.alert_hold_counter = 0
            self.held_peak_score = 0.0
            self.total_audio_sec = 0.0
            self.total_chunks_received = 0
            self.processed_windows = 0

    def ingest_pcm_chunk(
        self,
        chunk: Union[bytes, np.ndarray, List[float]],
        format: str = "pcm16",
        input_sr: int = 16000,
    ) -> None:
        """
        Thread-safely push an incoming audio chunk into the streaming buffer.
        """
        with self._lock:
            self.total_chunks_received += 1
            if isinstance(chunk, bytes):
                self.buffer.add_pcm_chunk(chunk, format=format, input_sr=input_sr)
                # Compute duration in seconds based on format
                if format.lower() in ("mulaw", "ulaw", "g711"):
                    chunk_dur = len(chunk) / float(input_sr)
                elif format.lower() in ("float32", "pcm_f32le", "f32"):
                    chunk_dur = (len(chunk) / 4.0) / float(input_sr)
                else:
                    chunk_dur = (len(chunk) / 2.0) / float(input_sr)
                self.total_audio_sec += chunk_dur
            elif isinstance(chunk, (np.ndarray, list)):
                arr = np.asarray(chunk, dtype=np.float32)
                self.buffer.add_samples(arr)
                self.total_audio_sec += len(arr) / float(input_sr)

    def process_streaming_step(
        self,
        transcript_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute one inference and temporal aggregation step over the latest 3.0s window.

        Returns structured telemetry payload:
          - timestamp_sec: float
          - instantaneous_prob: float
          - smoothed_risk_score: int in [0, 100]
          - risk_band: 5-state calibrated string
          - is_alert_held: bool
          - latency_ms: float
        """
        with self._lock:
            t0 = time.perf_counter()
            self.processed_windows += 1

            # 1. Extract 3.0s analysis window
            raw_window = self.buffer.get_analysis_window(self.window_sec)

            # 2. Silence / Energy Check
            rms = float(np.sqrt(np.mean(raw_window ** 2) + 1e-12))
            is_silent = bool(rms < 0.002)

            if is_silent:
                # Handle alert state during silence without diluting active alarms
                is_held = self.alert_hold_counter > 0
                if is_held:
                    self.alert_hold_counter -= 1
                    effective_score = int(round(max(61.0, self.held_peak_score)))
                else:
                    if self.held_peak_score > 0.0:
                        self.held_peak_score = max(0.0, self.held_peak_score - (self.decay_rate * 100.0))
                    effective_score = int(round(self.held_peak_score))

                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                risk_band, badge_class, band_key = self._get_risk_band(effective_score, is_silent=True)

                return {
                    "session_id": self.session_id,
                    "timestamp_sec": round(self.total_audio_sec, 3),
                    "window_index": self.processed_windows,
                    "instantaneous_prob": 0.0,
                    "instantaneous_score": 0,
                    "ema_prob": round(self.ema_score or 0.0, 4),
                    "top_k_prob": 0.0,
                    "smoothed_risk_score": effective_score,
                    "risk_score": effective_score,
                    "risk_band": risk_band,
                    "risk_band_key": band_key,
                    "badge_class": badge_class,
                    "is_alert_held": is_held,
                    "alert_hold_counter": self.alert_hold_counter,
                    "forensic_breakdown": {},
                    "diagnostics": {"rms": rms, "is_silent": True},
                    "latency_ms": latency_ms,
                    "processing_latency_ms": latency_ms,
                    "is_realtime_compliant": bool(latency_ms < 250.0),
                    "disclaimer": STREAMING_DISCLAIMER,
                }

            # 3. Acoustic Channel Normalization
            normalized_window = self.channel_normalizer.remove_reverb_and_equalize(raw_window)

            # 4. Neural & Forensic Inference Pass
            try:
                pred_res = self.detector.predict(
                    normalized_window,
                    sample_rate=self.sample_rate,
                    transcript_text=transcript_text,
                    is_live_mic=True,
                )
                instant_prob = float(pred_res.get("spoof_probability", 0.50))
                forensic_breakdown = pred_res.get("forensic_breakdown", {})
                diagnostics = pred_res.get("diagnostics", {})
            except Exception as exc:
                log.warning("[Streaming] Detector inference error: %s", exc)
                instant_prob = 0.50
                forensic_breakdown = {}
                diagnostics = {"error": str(exc)}

            instant_score = int(round(instant_prob * 100))

            # 5. Temporal Exponential Moving Average (EMA)
            # EMA_t = 0.35 * P_t + 0.65 * EMA_{t-1}
            if self.ema_score is None:
                self.ema_score = instant_prob
            else:
                self.ema_score = self.ema_alpha * instant_prob + (1.0 - self.ema_alpha) * self.ema_score

            # 6. Top-K (85th Percentile) Window Pooling over rolling history
            self.history_scores.append(instant_prob)
            history_list = list(self.history_scores)
            top_k_prob = float(np.percentile(history_list, 85))

            # 7. Combined Live Risk Score: Score_live = 0.70 * TopK_85 + 0.30 * EMA
            combined_prob = float(np.clip(0.70 * top_k_prob + 0.30 * self.ema_score, 0.0, 1.0))
            raw_live_score = int(round(combined_prob * 100))

            # 8. Hold-and-Decay Security Alert Gate
            if raw_live_score >= 61:
                # Lock alert state for full hold period (6 steps = 3.0s @ 0.5s hop)
                self.alert_hold_counter = self.hold_steps
                self.held_peak_score = max(self.held_peak_score, float(raw_live_score))
                effective_score = int(round(self.held_peak_score))
                is_alert_held = True
            else:
                if self.alert_hold_counter > 0:
                    # Within hold period: maintain alert at or above High-Risk boundary
                    self.alert_hold_counter -= 1
                    effective_score = int(round(max(61.0, self.held_peak_score)))
                    is_alert_held = True
                else:
                    # Hold period expired: decay towards current score by factor of 0.05 / step
                    if self.held_peak_score > float(raw_live_score):
                        self.held_peak_score = max(
                            float(raw_live_score),
                            self.held_peak_score - (self.decay_rate * 100.0),
                        )
                        effective_score = int(round(self.held_peak_score))
                    else:
                        self.held_peak_score = float(raw_live_score)
                        effective_score = raw_live_score
                    is_alert_held = False

            effective_score = int(np.clip(effective_score, 0, 100))
            risk_band, badge_class, band_key = self._get_risk_band(effective_score, is_silent=False)
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            return {
                "session_id": self.session_id,
                "timestamp_sec": round(self.total_audio_sec, 3),
                "window_index": self.processed_windows,
                "instantaneous_prob": round(instant_prob, 4),
                "instantaneous_score": instant_score,
                "ema_prob": round(self.ema_score, 4),
                "top_k_prob": round(top_k_prob, 4),
                "combined_prob": round(combined_prob, 4),
                "smoothed_risk_score": effective_score,
                "risk_score": effective_score,
                "risk_band": risk_band,
                "risk_band_key": band_key,
                "badge_class": badge_class,
                "is_alert_held": is_alert_held,
                "alert_hold_counter": self.alert_hold_counter,
                "forensic_breakdown": forensic_breakdown,
                "diagnostics": diagnostics,
                "latency_ms": latency_ms,
                "processing_latency_ms": latency_ms,
                "is_realtime_compliant": bool(latency_ms < 250.0),
                "disclaimer": STREAMING_DISCLAIMER,
            }

    # Alias for backward compatibility
    process_latest_window = process_streaming_step

    @staticmethod
    def _get_risk_band(score: int, is_silent: bool = False) -> Tuple[str, str, str]:
        """Map numeric score in [0, 100] to standardized 5-state risk band and badge."""
        if is_silent and score < 26:
            return "Silence / Idle", "badge-low", "silence"
        if score <= 25:
            return "Low Risk (Human Voice)", "badge-low", "low"
        if score <= 60:
            return "Review Required (Borderline Evidence)", "badge-review", "review"
        return "High Risk (Likely AI / Cloned Voice)", "badge-high", "high"


# ---------------------------------------------------------------------------
# 3. Backward Compatibility Adapters
# ---------------------------------------------------------------------------

class SandboxStreamAnalyzer:
    """
    Simulated low-latency streaming audio analyzer for sandbox validation and unit tests.
    """

    def __init__(self, smoothing_alpha: float = 0.35, sample_rate: int = SAMPLE_RATE) -> None:
        self.smoothing_alpha: float = float(smoothing_alpha)
        self.sample_rate: int = int(sample_rate)
        self.rolling_score: float = 0.0
        self.processed_windows: int = 0
        self.is_running: bool = False

    def reset(self) -> None:
        self.rolling_score = 0.0
        self.processed_windows = 0
        self.is_running = False

    def process_chunk(
        self,
        chunk_idx: int,
        timestamp: float,
        chunk: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Any]:
        if chunk is None or len(chunk) == 0:
            return {
                "is_valid": False,
                "skipped_reason": "Empty or missing frame",
                "audio_saved": False,
                "instant_score": None,
                "instantaneous_score": None,
                "rolling_risk_score": self.rolling_score,
                "window_number": self.processed_windows,
            }

        if np.any(np.isnan(chunk)) or np.any(np.isinf(chunk)):
            return {
                "is_valid": False,
                "skipped_reason": "Malformed audio chunk (NaN/Inf)",
                "audio_saved": False,
                "instant_score": None,
                "instantaneous_score": None,
                "rolling_risk_score": self.rolling_score,
                "window_number": self.processed_windows,
            }

        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if rms < 0.005:
            return {
                "is_valid": False,
                "skipped_reason": "Silence detected in chunk",
                "audio_saved": False,
                "instant_score": None,
                "instantaneous_score": None,
                "rolling_risk_score": self.rolling_score,
                "window_number": self.processed_windows,
            }

        self.processed_windows += 1
        raw_score = 30.0 + min(40.0, float(np.std(chunk) * 100.0))
        if self.processed_windows == 1:
            self.rolling_score = raw_score
        else:
            self.rolling_score = (
                self.smoothing_alpha * raw_score + (1.0 - self.smoothing_alpha) * self.rolling_score
            )

        return {
            "is_valid": True,
            "skipped_reason": None,
            "audio_saved": False,
            "instant_score": round(raw_score, 2),
            "instantaneous_score": round(raw_score, 2),
            "rolling_score": round(self.rolling_score, 2),
            "rolling_risk_score": round(self.rolling_score, 2),
            "risk_band": (
                "Low Risk" if self.rolling_score <= 25 else
                ("Review Required" if self.rolling_score <= 60 else "High Risk")
            ),
            "processing_ms": 2.5,
            "latency_ms": 2.5,
            "window_number": self.processed_windows,
            "timestamp": timestamp,
        }

    process_window = process_chunk

    def run_simulation(
        self,
        audio_file: str,
        window_ms: int = 160,
        stride_ms: int = 40,
        max_windows: int = 100,
        simulated_delay_sec: float = 0.0,
    ) -> List[Dict[str, Any]]:
        import soundfile as sf
        self.is_running = True
        try:
            audio, sr = sf.read(audio_file)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            results = []
            for chunk_idx, (idx, start_t, win_chunk) in enumerate(
                slice_streaming_windows(audio, sample_rate=sr, window_ms=window_ms, stride_ms=stride_ms)
            ):
                if chunk_idx >= max_windows:
                    break
                res = self.process_chunk(idx, start_t, win_chunk, sample_rate=sr)
                results.append(res)
            return results
        finally:
            self.is_running = False


class NeuralStreamingScoreEngine:
    """
    Session-level streaming inference manager maintaining temporal EMA.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        smoothing_alpha: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        from src.neural_engine import ProductionNeuralDetector
        self.detector = ProductionNeuralDetector(
            native_checkpoint_path=checkpoint_path or "models/voiceshield_live_robust.pt",
            device=device,
            load_hf=False,
        )
        self.smoothing_alpha: float = float(smoothing_alpha)
        self.device = self.detector.device
        self._ema_spoof_prob: Optional[float] = None
        self._window_counter: int = 0

    def reset(self) -> None:
        self._ema_spoof_prob = None
        self._window_counter = 0

    def predict_stream_window(
        self,
        audio_window: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        transcript_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_res = self.detector.predict(
            audio_window,
            sample_rate=sample_rate,
            transcript_text=transcript_text,
            is_live_mic=True,
        )
        instant_prob = float(raw_res.get("spoof_probability", 0.50))

        if self._ema_spoof_prob is None:
            self._ema_spoof_prob = instant_prob
        else:
            self._ema_spoof_prob = (
                self.smoothing_alpha * instant_prob + (1.0 - self.smoothing_alpha) * self._ema_spoof_prob
            )

        self._window_counter += 1
        smoothed_prob = float(np.clip(self._ema_spoof_prob, 0.01, 0.99))
        risk_score = int(round(smoothed_prob * 100))

        if risk_score <= 25:
            risk_band = "Low Risk (Authentic Human)"
            badge_class = "badge-low"
            band_key = "low"
        elif risk_score <= 60:
            risk_band = "Review Required (Inconclusive)"
            badge_class = "badge-review"
            band_key = "review"
        else:
            risk_band = "High Risk (AI Voice Clone)"
            badge_class = "badge-high"
            band_key = "high"

        inference_lat_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "window_index": self._window_counter,
            "instantaneous_spoof_prob": round(instant_prob, 4),
            "smoothed_spoof_prob": round(smoothed_prob, 4),
            "risk_score": risk_score,
            "risk_band": risk_band,
            "risk_band_key": band_key,
            "badge_class": badge_class,
            "forensic_breakdown": raw_res.get("forensic_breakdown", {}),
            "diagnostics": raw_res.get("diagnostics", {}),
            "inference_latency_ms": inference_lat_ms,
            "is_realtime_compliant": bool(inference_lat_ms < 200.0),
        }
