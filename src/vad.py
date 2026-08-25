"""
VoiceShield Phase 1: Voice Activity Detection (Silero VAD & Energy Fallback).
Strips leading/trailing silence and unvoiced noise segments to maximize feature SNR
prior to acoustic classification.
"""

import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MIN_AUDIO_RMS_ENERGY, SAMPLE_RATE


class VoiceActivityDetector:
    """
    Voice Activity Detection Gate utilizing Silero VAD with seamless offline energy fallback.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, threshold: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = None
        self.utils = None
        self._init_vad_model()

    def _init_vad_model(self) -> None:
        """Attempts to load Silero VAD model via torch.hub with offline safety."""
        try:
            # Load Silero VAD model
            torch.set_num_threads(1)
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                trust_repo=True,
            )
            self.model = model
            self.get_speech_timestamps, _, _, _, _ = utils
            self.model.eval()
        except Exception as e:
            # Fallback to local energy-based gating if offline or torch.hub download fails
            self.model = None
            self.utils = None

    def filter_voiced_audio(
        self,
        audio: np.ndarray,
        sr: Optional[int] = None,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> np.ndarray:
        """
        Extracts and concatenates all voiced speech segments from the audio array.

        Args:
            audio: 1D float32 audio waveform.
            sr: Sample rate (default: self.sample_rate).
            min_speech_duration_ms: Minimum speech chunk duration in milliseconds.
            min_silence_duration_ms: Minimum silence duration to trigger boundary.

        Returns:
            np.ndarray: Concatenated voiced speech audio (float32).
        """
        if audio is None or len(audio) == 0:
            return np.zeros(0, dtype=np.float32)

        sample_rate = sr or self.sample_rate

        # 1. Silero VAD neural gating
        if self.model is not None and sample_rate in [8000, 16000]:
            try:
                wav_tensor = torch.from_numpy(audio.flatten()).float()
                speech_timestamps = self.get_speech_timestamps(
                    wav_tensor,
                    self.model,
                    sampling_rate=sample_rate,
                    threshold=self.threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                )
                if speech_timestamps:
                    voiced_chunks = [audio[ts["start"] : ts["end"]] for ts in speech_timestamps]
                    return np.concatenate(voiced_chunks).astype(np.float32)
            except Exception:
                pass

        # 2. Automated Energy-Based Fallback (Short-Time Energy Thresholding)
        return self._energy_based_filter(audio, sample_rate)

    def _energy_based_filter(
        self,
        audio: np.ndarray,
        sr: int,
        frame_ms: int = 30,
        hop_ms: int = 10,
        energy_threshold: float = MIN_AUDIO_RMS_ENERGY * 2,
    ) -> np.ndarray:
        """
        Energy-based voice activity filtering using frame RMS and adaptive noise floor.
        """
        frame_len = int(sr * (frame_ms / 1000.0))
        hop_len = int(sr * (hop_ms / 1000.0))

        if len(audio) < frame_len:
            return audio.astype(np.float32)

        # Frame audio
        num_frames = 1 + (len(audio) - frame_len) // hop_len
        voiced_mask = np.zeros(len(audio), dtype=bool)

        for i in range(num_frames):
            start = i * hop_len
            end = start + frame_len
            frame = audio[start:end]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms >= energy_threshold:
                voiced_mask[start:end] = True

        voiced_audio = audio[voiced_mask]
        
        # If all filtered out (e.g. faint whisper), return original to avoid zero output
        if len(voiced_audio) < int(sr * 0.1):
            return audio.astype(np.float32)

        return voiced_audio.astype(np.float32)
