"""
VoiceShield Neural Step 4: Real-Time Neural Inference & Streaming Score Engine.
Consumes raw audio directly from in-memory buffers, executes torch-accelerated
forward passes, applies Exponential Moving Average (EMA) smoothing, and maps to
5 calibrated risk bands under strict sub-200ms latency SLAs.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MIN_AUDIO_RMS_ENERGY, SAMPLE_RATE
from src.neural_model import VoiceShieldNeuralDetector

TARGET_WINDOW_SAMPLES: int = 48000  # 3.0s @ 16kHz
DEFAULT_NEURAL_CHECKPOINT: str = "models/voiceshield_neural_best.pt"


class RollingAudioBuffer:
    """
    High-performance in-memory circular audio buffer for real-time telephony stream aggregation.
    """

    def __init__(self, capacity_seconds: float = 6.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.capacity_samples = int(capacity_seconds * sample_rate)
        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self.write_ptr = 0
        self.total_samples_written = 0

    def add_samples(self, chunk: np.ndarray) -> None:
        """Appends new 1D audio samples into the circular buffer."""
        if chunk is None or len(chunk) == 0:
            return

        chunk = chunk.flatten().astype(np.float32)
        n = len(chunk)

        if n >= self.capacity_samples:
            self.buffer[:] = chunk[-self.capacity_samples :]
            self.write_ptr = 0
            self.total_samples_written += n
            return

        end_ptr = self.write_ptr + n
        if end_ptr <= self.capacity_samples:
            self.buffer[self.write_ptr : end_ptr] = chunk
            self.write_ptr = end_ptr % self.capacity_samples
        else:
            first_part = self.capacity_samples - self.write_ptr
            self.buffer[self.write_ptr :] = chunk[:first_part]
            self.buffer[: n - first_part] = chunk[first_part:]
            self.write_ptr = n - first_part

        self.total_samples_written += n

    def get_latest_window(self, window_samples: int = TARGET_WINDOW_SAMPLES) -> np.ndarray:
        """Retrieves the most recent consecutive window of audio."""
        if self.total_samples_written == 0:
            return np.zeros(window_samples, dtype=np.float32)

        available = min(self.total_samples_written, self.capacity_samples)
        if available < self.capacity_samples:
            ordered = self.buffer[:available]
        else:
            ordered = np.concatenate([self.buffer[self.write_ptr :], self.buffer[: self.write_ptr]])

        if len(ordered) < window_samples:
            # Pad with zeros symmetrically if not enough samples yet
            pad_total = window_samples - len(ordered)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(ordered, (pad_left, pad_right), mode="constant")

        return ordered[-window_samples:]

    def clear(self) -> None:
        self.buffer.fill(0.0)
        self.write_ptr = 0
        self.total_samples_written = 0


def prepare_waveform_tensor(
    audio: np.ndarray,
    target_samples: int = TARGET_WINDOW_SAMPLES,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Converts raw in-memory numpy waveform into standardized (1, target_samples) PyTorch tensor
    without intermediate disk writes.
    """
    if audio is None or len(audio) == 0:
        return torch.zeros((1, target_samples), dtype=torch.float32, device=device)

    arr = np.nan_to_num(audio.flatten(), nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    # DC offset removal
    arr = arr - np.mean(arr)
    max_amp = np.max(np.abs(arr))
    if max_amp > 1e-6:
        arr = arr / (max_amp + 1e-8)

    num_samples = len(arr)
    if num_samples == target_samples:
        tensor = torch.from_numpy(arr)
    elif num_samples > target_samples:
        # Deterministic center crop
        start = (num_samples - target_samples) // 2
        tensor = torch.from_numpy(arr[start : start + target_samples])
    else:
        # Symmetrical zero-padding
        pad_total = target_samples - num_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = np.pad(arr, (pad_left, pad_right), mode="constant")
        tensor = torch.from_numpy(padded)

    tensor = tensor.unsqueeze(0).float()
    if device is not None:
        tensor = tensor.to(device)

    return tensor


class NeuralStreamingScoreEngine:
    """
    Low-latency real-time inference engine consuming in-memory audio buffers,
    executing GPU/CPU forward passes, and maintaining smoothed EMA risk scores.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_NEURAL_CHECKPOINT,
        smoothing_alpha: float = 0.35,
        energy_threshold: float = MIN_AUDIO_RMS_ENERGY * 5,
        device: Optional[str] = None,
    ) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.energy_threshold = energy_threshold

        # Device selection & thread optimization
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.type == "cpu":
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

        self.use_amp = self.device.type == "cuda"
        self.checkpoint_path = checkpoint_path
        self.model: Optional[VoiceShieldNeuralDetector] = None

        # Load weights during startup
        self._load_model()

        # Streaming state variables
        self.rolling_ema_prob: Optional[float] = None
        self.rolling_risk_score: int = 0
        self.processed_frames: int = 0
        self.skipped_frames: int = 0

    def _load_model(self) -> None:
        """Loads neural detector weights once during initialization."""
        if not os.path.exists(self.checkpoint_path):
            print(f"[!] Warning: Checkpoint not found at {self.checkpoint_path}. Initializing base model.")
            self.model = VoiceShieldNeuralDetector(backbone_name="lightweight", device=self.device)
            self.model.eval()
            return

        try:
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            backbone_name = "lightweight" if any("conv1" in k for k in state.keys()) else "facebook/wav2vec2-base"
            self.model = VoiceShieldNeuralDetector(backbone_name=backbone_name, device=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            print(f"[OK] NeuralStreamingScoreEngine loaded: {self.checkpoint_path} ({backbone_name}) on {self.device}")
        except Exception as e:
            print(f"[!] Error loading checkpoint ({e}). Using native lightweight fallback.")
            self.model = VoiceShieldNeuralDetector(backbone_name="lightweight", device=self.device)
            self.model.eval()

    def reset(self) -> None:
        """Resets stateful EMA buffers and frame counters on new call session."""
        self.rolling_ema_prob = None
        self.rolling_risk_score = 0
        self.processed_frames = 0
        self.skipped_frames = 0

    def compute_risk_band(self, prob: float, is_silent: bool = False) -> Tuple[str, str, str]:
        """
        Maps probability and signal diagnostics to 5 calibrated risk states:
          1. Low Risk (0-25): Natural human voice markers
          2. Review Required (26-65): Borderline synthetic evidence
          3. High Risk (66-100): Elevated synthetic vocoder / cloning artifacts
          4. Inconclusive (45-55): Low confidence margin
          5. Low Quality: Degraded, silent, or faint signal
        """
        if is_silent:
            return "low_quality", "Low Quality (Silence / Faint Audio)", "badge-warning"

        score = int(round(prob * 100))

        if 45 <= score <= 55:
            return "inconclusive", "Inconclusive (Borderline Acoustic Signal)", "badge-uncertain"

        if score <= 25:
            return "low", "Low Risk — Natural Human Voice", "badge-low"
        elif score >= 66:
            return "high", "High Risk — Synthetic / AI Cloned Voice", "badge-high"
        else:
            return "review", "Review Required (Synthetic Suspicion)", "badge-review"

    @torch.inference_mode()
    def predict_stream_window(
        self,
        audio_window: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Any]:
        """
        Executes real-time forward pass over a 3.0s window in-memory with sub-200ms latency profiling.

        Args:
            audio_window: 1D float32 audio segment (e.g. from RollingAudioBuffer).
            sample_rate: Sample rate in Hz (default: 16000).

        Returns:
            Dictionary with instantaneous and EMA smoothed predictions, risk states, and telemetry.
        """
        start_time = time.perf_counter()

        # 1. Basic energy and silence check
        rms = float(np.sqrt(np.mean(audio_window ** 2))) if len(audio_window) > 0 else 0.0
        is_silent = rms < self.energy_threshold

        if is_silent or len(audio_window) < 512:
            self.skipped_frames += 1
            band_key, band_desc, badge = self.compute_risk_band(0.0, is_silent=True)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "frame_id": self.processed_frames + self.skipped_frames,
                "is_valid": False,
                "is_silent": True,
                "rms_energy": round(rms, 6),
                "instantaneous_spoof_prob": 0.0,
                "ema_spoof_prob": round(self.rolling_ema_prob or 0.0, 4),
                "risk_score": self.rolling_risk_score,
                "risk_band": band_desc,
                "risk_band_key": band_key,
                "badge_class": badge,
                "inference_latency_ms": latency_ms,
                "is_realtime_compliant": latency_ms < 200.0,
            }

        # 2. In-Memory Tensor Adapter
        x_tensor = prepare_waveform_tensor(audio_window, target_samples=TARGET_WINDOW_SAMPLES, device=self.device)

        # 3. Model Forward Pass
        if self.use_amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logit = self.model(x_tensor)
        else:
            logit = self.model(x_tensor)

        prob_inst = float(torch.sigmoid(logit).item())
        prob_inst = float(np.clip(prob_inst, 0.0, 1.0))

        # 4. Exponential Moving Average (EMA) Smoothing
        if self.rolling_ema_prob is None:
            self.rolling_ema_prob = prob_inst
        else:
            self.rolling_ema_prob = float(
                self.smoothing_alpha * prob_inst + (1.0 - self.smoothing_alpha) * self.rolling_ema_prob
            )

        self.processed_frames += 1
        self.rolling_risk_score = int(round(self.rolling_ema_prob * 100))

        # 5. Risk Band Mapping & Latency Profiling
        band_key, band_desc, badge = self.compute_risk_band(self.rolling_ema_prob, is_silent=False)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "frame_id": self.processed_frames + self.skipped_frames,
            "is_valid": True,
            "is_silent": False,
            "rms_energy": round(rms, 6),
            "instantaneous_spoof_prob": round(prob_inst, 4),
            "ema_spoof_prob": round(self.rolling_ema_prob, 4),
            "risk_score": self.rolling_risk_score,
            "risk_band": band_desc,
            "risk_band_key": band_key,
            "badge_class": badge,
            "inference_latency_ms": latency_ms,
            "is_realtime_compliant": latency_ms < 200.0,
        }
