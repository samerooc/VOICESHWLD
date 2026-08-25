"""
VoiceShield Neural Step 2: Dynamic On-the-Fly Waveform Augmentation Suite.
Operates directly on 1D / 2D raw 16kHz audio tensors with randomized execution probabilities:
  1. Telephony & Bandwidth Restriction (16kHz -> 8kHz -> 16kHz resampling + 300Hz-3400Hz bandpass).
  2. Additive Noise Injection (White/Pink noise with randomized 10dB to 30dB SNR).
  3. Amplitude Gain Scaling (+-4 dB) & Mild Threshold Clipping.
  4. Temporal Perturbation (Random Time-Masking: 50ms - 200ms zeroed chunks).
"""

import math
import random
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


class WaveformAugmenter(nn.Module):
    """
    Applies modular, randomized acoustic waveform augmentations directly to PyTorch tensors.
    Configured for 16,000 Hz raw waveforms.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        p_telephony: float = 0.35,
        p_noise: float = 0.40,
        p_gain_and_clip: float = 0.40,
        p_time_mask: float = 0.35,
        snr_range_db: Tuple[float, float] = (10.0, 30.0),
        gain_range_db: Tuple[float, float] = (-4.0, 4.0),
        mask_duration_ms: Tuple[float, float] = (50.0, 200.0),
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.p_telephony = p_telephony
        self.p_noise = p_noise
        self.p_gain_and_clip = p_gain_and_clip
        self.p_time_mask = p_time_mask
        self.snr_range_db = snr_range_db
        self.gain_range_db = gain_range_db
        self.mask_duration_ms = mask_duration_ms

        # Fixed Resamplers for telephony downsampling/upsampling
        self.downsample_to_8k = T.Resample(orig_freq=sample_rate, new_freq=8000)
        self.upsample_to_16k = T.Resample(orig_freq=8000, new_freq=sample_rate)

    @torch.no_grad()
    def apply_telephony_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Simulates PSTN telephony codec by downsampling to 8kHz, applying 300Hz-3400Hz
        bandpass filter, and resampling back to 16kHz.
        """
        orig_len = waveform.shape[-1]
        
        # 1. Resample to 8kHz and back to 16kHz
        down = self.downsample_to_8k(waveform)
        up = self.upsample_to_16k(down)

        # Match exact sample count in case of resampling rounding
        if up.shape[-1] > orig_len:
            up = up[..., :orig_len]
        elif up.shape[-1] < orig_len:
            up = torch.nn.functional.pad(up, (0, orig_len - up.shape[-1]))

        # 2. Apply 300Hz - 3400Hz Bandpass filter
        try:
            center_freq = (300.0 + 3400.0) / 2.0  # 1850 Hz
            q_factor = center_freq / (3400.0 - 300.0)  # ~0.596
            filtered = F.bandpass_biquad(
                waveform=up,
                sample_rate=self.sample_rate,
                central_freq=center_freq,
                Q=q_factor,
            )
            return filtered
        except Exception:
            return up

    @torch.no_grad()
    def apply_additive_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Injects Gaussian white or pink noise at randomized SNR between 10dB and 30dB.
        """
        signal_power = torch.mean(waveform ** 2, dim=-1, keepdim=True) + 1e-9
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)

        # 50% chance of 1st-order pink/low-frequency colored noise filtering
        if random.random() < 0.5:
            # Simple 1-pole filter to create 1/f spectral tilt
            noise = torch.cumsum(noise, dim=-1)
            noise_power_actual = torch.mean(noise ** 2, dim=-1, keepdim=True) + 1e-9
            noise = noise * torch.sqrt(noise_power / noise_power_actual)

        return waveform + noise

    @torch.no_grad()
    def apply_gain_and_clipping(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Applies random amplitude gain scaling and optional mild clipping.
        """
        gain_db = random.uniform(self.gain_range_db[0], self.gain_range_db[1])
        gain_linear = 10.0 ** (gain_db / 20.0)
        scaled = waveform * gain_linear

        # 40% chance of mild distortion clipping
        if random.random() < 0.40:
            clip_thresh = random.uniform(0.75, 0.98)
            scaled = torch.clamp(scaled, -clip_thresh, clip_thresh)

        return scaled

    @torch.no_grad()
    def apply_time_masking(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Zeros out a random continuous slice (50ms - 200ms) to force robust representation learning.
        """
        total_samples = waveform.shape[-1]
        mask_ms = random.uniform(self.mask_duration_ms[0], self.mask_duration_ms[1])
        mask_samples = int((mask_ms / 1000.0) * self.sample_rate)

        if mask_samples >= total_samples:
            return waveform

        start_idx = random.randint(0, total_samples - mask_samples)
        masked = waveform.clone()
        masked[..., start_idx : start_idx + mask_samples] = 0.0
        return masked

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Executes randomized augmentation pipeline sequentially.

        Args:
            waveform: Float tensor of shape (N,) or (1, N) or (B, N).

        Returns:
            Augmented tensor of identical shape, sanitized and bounded.
        """
        out = waveform.clone()

        # 1. Telephony filter
        if random.random() < self.p_telephony:
            out = self.apply_telephony_filter(out)

        # 2. Additive noise
        if random.random() < self.p_noise:
            out = self.apply_additive_noise(out)

        # 3. Gain and clipping
        if random.random() < self.p_gain_and_clip:
            out = self.apply_gain_and_clipping(out)

        # 4. Time masking
        if random.random() < self.p_time_mask:
            out = self.apply_time_masking(out)

        # Clean NaN/Infs and bound output to [-1.0, 1.0]
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        max_val = torch.max(torch.abs(out))
        if max_val > 1.0:
            out = out / (max_val + 1e-8)

        return out
