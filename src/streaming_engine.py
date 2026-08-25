"""
VoiceShield Tri-Modal Real-Time Streaming Engine.
Consumes continuous raw audio frames from thread-safe RollingAudioBuffers,
runs low-latency Tri-Modal forward passes (Acoustic + Physics + NLP Intent),
and emits unified real-time risk payloads under sub-150ms latency SLAs.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MIN_AUDIO_RMS_ENERGY, SAMPLE_RATE
from src.streaming import RollingAudioBuffer
from src.tri_modal_model import VoiceShieldTriModalEngine


class TriModalStreamingEngine:
    """
    Stateful real-time Tri-Modal streaming evaluator.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        smoothing_alpha: float = 0.35,
        energy_threshold: float = MIN_AUDIO_RMS_ENERGY * 5,
    ) -> None:
        self.smoothing_alpha = smoothing_alpha
        self.energy_threshold = energy_threshold

        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VoiceShieldTriModalEngine(backbone_name="lightweight", device=dev)
        self.model.eval()

        self.rolling_ema_score: Optional[float] = None
        self.processed_windows = 0

    def reset(self) -> None:
        self.rolling_ema_score = None
        self.processed_windows = 0

    def process_stream_window(
        self,
        audio_window: np.ndarray,
        transcription: str = "",
        timestamp: float = 0.0,
        sample_rate: int = SAMPLE_RATE,
    ) -> Dict[str, Any]:
        """
        Evaluates a sliding window and returns unified Tri-Modal assessment payload.
        """
        start_t = time.perf_counter()

        rms = float(np.sqrt(np.mean(audio_window ** 2))) if len(audio_window) > 0 else 0.0
        is_silent = rms < self.energy_threshold

        if is_silent or len(audio_window) < 512:
            return {
                "timestamp": round(timestamp, 2),
                "combined_risk_score": 0,
                "risk_band": "Low Quality (Silence / Faint Audio)",
                "risk_band_key": "low_quality",
                "badge_class": "badge-warning",
                "breakdown": {
                    "neural_acoustic_score": 0,
                    "dsp_phase_anomaly": 0,
                    "nlp_intent_threat": 0,
                },
                "transcription": transcription,
                "latency_ms": round((time.perf_counter() - start_t) * 1000, 2),
                "is_silent": True,
            }

        # Run Tri-Modal Inference
        pred = self.model.predict_tri_modal(audio_window, transcription=transcription, sr=sample_rate)

        raw_combined = float(pred["combined_risk_score"])
        if self.rolling_ema_score is None:
            self.rolling_ema_score = raw_combined
        else:
            self.rolling_ema_score = (
                self.smoothing_alpha * raw_combined + (1.0 - self.smoothing_alpha) * self.rolling_ema_score
            )

        self.processed_windows += 1
        smoothed_int = int(round(self.rolling_ema_score))
        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

        return {
            "timestamp": round(timestamp, 2),
            "combined_risk_score": smoothed_int,
            "risk_band": pred["risk_band"],
            "risk_band_key": pred["risk_band_key"],
            "badge_class": pred["badge_class"],
            "breakdown": pred["breakdown"],
            "nlp_threat_categories": pred["nlp_threat_categories"],
            "transcription": transcription,
            "modality_weights": pred["modality_weights"],
            "latency_ms": latency_ms,
            "is_silent": False,
        }
