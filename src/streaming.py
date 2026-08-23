"""
VoiceShield Sandbox Audio Streaming Module (Phase 7).
Implements in-memory rolling buffer audio windowing, silence detection,
and stateful Exponential Moving Average (EMA) risk aggregation.

===============================================================================
                SANDBOX SIMULATION — NOT A LIVE CALL
This simulation demonstrates the processing flow only.
It is not a telecom integration or production latency benchmark.
===============================================================================
"""

import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np

from src.config import (
    MIN_AUDIO_RMS_ENERGY,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
)
from src.features import extract_features_from_audio
from src.model import load_metadata, load_model
from src.scoring import calculate_risk_score, get_risk_band

STREAMING_DISCLAIMER: str = (
    "SANDBOX SIMULATION — NOT A LIVE CALL. "
    "This simulation demonstrates the processing flow only. "
    "It is not a telecom integration or production latency benchmark."
)


def slice_streaming_windows(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    window_ms: int = 160,
    stride_ms: int = 40,
) -> Generator[Tuple[int, float, np.ndarray], None, None]:
    """
    Slices a 1D audio array into overlapping fixed-duration analysis windows.

    Args:
        audio: 1D normalized float32 array.
        sample_rate: Sampling frequency in Hz.
        window_ms: Window size in milliseconds (default: 160 ms = 2560 samples).
        stride_ms: Stride step size in milliseconds (default: 40 ms = 640 samples).

    Yields:
        Tuple of (window_index, timestamp_seconds, window_audio_array).
    """
    window_samples = int(sample_rate * (window_ms / 1000.0))
    stride_samples = int(sample_rate * (stride_ms / 1000.0))

    if audio is None or len(audio) == 0:
        return

    if len(audio) < window_samples:
        # Safely pad if audio is shorter than one analysis window
        padded = np.pad(audio, (0, window_samples - len(audio)))
        yield (0, 0.0, padded)
        return

    num_windows = max(1, (len(audio) - window_samples) // stride_samples + 1)
    for idx in range(num_windows):
        start = idx * stride_samples
        end = start + window_samples
        chunk = audio[start:end]
        timestamp = round(start / sample_rate, 3)
        yield (idx, timestamp, chunk)


class SandboxStreamAnalyzer:
    """
    Stateful sandbox stream analyzer with rolling risk score computation.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        metadata_path: str = MODEL_METADATA_PATH,
        smoothing_alpha: float = 0.20,
        energy_threshold: float = MIN_AUDIO_RMS_ENERGY * 10,
    ):
        self.model = load_model(model_path)
        self.metadata = load_metadata(metadata_path)
        self.decision_threshold = (
            self.metadata.get("optimal_decision_threshold", 0.40)
            if self.metadata
            else 0.40
        )
        self.alpha = smoothing_alpha
        self.energy_threshold = energy_threshold
        self.rolling_score = 0.0
        self.processed_windows = 0
        self.skipped_windows = 0
        self.is_running = False
        self.is_stopped = False

    def reset(self) -> None:
        self.rolling_score = 0.0
        self.processed_windows = 0
        self.skipped_windows = 0
        self.is_running = False
        self.is_stopped = False

    def process_chunk(
        self,
        window_idx: int,
        timestamp_sec: float,
        chunk: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Any]:
        """
        Analyzes an individual 160 ms audio window in-memory.
        """
        start_time = time.perf_counter()
        temp_features: Optional[np.ndarray] = None

        try:
            # 1. Missing / Empty chunk validation
            if chunk is None or len(chunk) == 0:
                self.skipped_windows += 1
                return {
                    "window_number": window_idx + 1,
                    "window_index": window_idx,
                    "timestamp_sec": timestamp_sec,
                    "is_valid": False,
                    "skipped_reason": "Empty / Missing Audio Frame",
                    "instantaneous_score": None,
                    "instantaneous_spoof_prob": 0.0,
                    "rolling_score": round(self.rolling_score, 1),
                    "rolling_risk_score": round(self.rolling_score, 1),
                    "risk_band": "Review required",
                    "processing_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "audio_saved": False,
                }

            # 2. Silence / Low energy frame detection
            rms_val = float(np.sqrt(np.mean(chunk ** 2)))
            if rms_val < self.energy_threshold:
                self.skipped_windows += 1
                return {
                    "window_number": window_idx + 1,
                    "window_index": window_idx,
                    "timestamp_sec": timestamp_sec,
                    "is_valid": False,
                    "skipped_reason": "Silence / Low Energy Frame",
                    "instantaneous_score": None,
                    "instantaneous_spoof_prob": 0.0,
                    "rolling_score": round(self.rolling_score, 1),
                    "rolling_risk_score": round(self.rolling_score, 1),
                    "risk_band": "Review required",
                    "processing_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "audio_saved": False,
                }

            # 3. Model inference using existing features & pipeline
            if self.model is None:
                raise ValueError("Model pipeline is not loaded.")

            temp_features = extract_features_from_audio(chunk, sample_rate=sample_rate)
            probs = self.model.predict_proba([temp_features])[0]
            spoof_prob = float(probs[1])
            instant_score = spoof_prob * 100.0

            # 4. Update Rolling Exponential Moving Average (EMA)
            if self.processed_windows == 0 and self.rolling_score == 0.0:
                self.rolling_score = instant_score
            else:
                self.rolling_score = self.alpha * instant_score + (1.0 - self.alpha) * self.rolling_score

            self.processed_windows += 1
            _, risk_band_name, _, _ = get_risk_band(int(round(self.rolling_score)), spoof_prob=self.rolling_score / 100.0)

            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return {
                "window_number": window_idx + 1,
                "window_index": window_idx,
                "timestamp_sec": timestamp_sec,
                "is_valid": True,
                "skipped_reason": None,
                "instantaneous_score": round(instant_score, 1),
                "instantaneous_spoof_prob": round(spoof_prob, 4),
                "rolling_score": round(self.rolling_score, 1),
                "rolling_risk_score": round(self.rolling_score, 1),
                "risk_band": risk_band_name,
                "processing_ms": elapsed_ms,
                "audio_saved": False,
            }

        except Exception as ex:
            self.skipped_windows += 1
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "window_number": window_idx + 1,
                "window_index": window_idx,
                "timestamp_sec": timestamp_sec,
                "is_valid": False,
                "skipped_reason": f"Signal Error: {ex}",
                "instantaneous_score": None,
                "instantaneous_spoof_prob": 0.0,
                "rolling_score": round(self.rolling_score, 1),
                "rolling_risk_score": round(self.rolling_score, 1),
                "risk_band": "Review required",
                "processing_ms": elapsed_ms,
                "audio_saved": False,
            }

        finally:
            if temp_features is not None:
                del temp_features

    # Alias for backward compatibility
    process_window = process_chunk

    def run_simulation(
        self,
        audio_file: str,
        window_ms: int = 160,
        stride_ms: int = 40,
        max_windows: int = 25,
        simulated_delay_sec: float = 0.01,
    ) -> List[Dict[str, Any]]:
        """
        Runs streaming simulation over a prerecorded audio file.
        """
        from src.audio_io import load_audio_from_file
        audio, sr = load_audio_from_file(audio_file, target_sr=SAMPLE_RATE)
        generator = slice_streaming_windows(audio, sample_rate=sr, window_ms=window_ms, stride_ms=stride_ms)

        self.reset()
        results: List[Dict[str, Any]] = []

        for idx, timestamp, chunk in generator:
            if max_windows and idx >= max_windows:
                break
            res = self.process_chunk(idx, timestamp, chunk, sample_rate=sr)
            results.append(res)
            if simulated_delay_sec > 0:
                time.sleep(simulated_delay_sec)

        return results
