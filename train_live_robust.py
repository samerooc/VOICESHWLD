"""
VoiceShield Production Live-Robust Deepfake Neural Model Fine-Tuning Pipeline.
Trains foundation backbones (Wav2Vec2 / XLSR / Lightweight) with dynamic acoustic channel augmentations:
  - Room Impulse Response (RIR) simulation
  - Lossy Opus/MP3 & Telephony Codec simulation
  - Additive ambient babble/noise (10-25 dB SNR)
  - Randomized microphone EQ curves (low-pass, high-boost, spectral tilt)
  - Binary Focal Loss (gamma=2.0, alpha=0.25) with differential learning rates.
"""

import os
import sys
import time
import random
import argparse
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, lfilter

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import decode_and_sanitize_audio
from src.neural_model import VoiceShieldNeuralDetector


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss focusing on hard, ambiguous room-acoustic voice clone samples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction="none")
        p_t = torch.exp(-bce)
        focal_loss = self.alpha * ((1.0 - p_t) ** self.gamma) * bce
        return focal_loss.mean()


class RobustAcousticAugmentor:
    """
    Simulates real-world acoustic channel distortions:
    1. Room Impulse Response (RIR) reflections.
    2. Additive ambient babble & room hiss (10-25 dB SNR).
    3. Microphone frequency coloration EQ (tilt, low-pass, high-boost).
    4. Codec bit-depth & lossy compression quantization.
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr

    def apply_rir_reflection(self, audio: np.ndarray) -> np.ndarray:
        """Simulates multi-path early reflections from room surfaces."""
        num_reflections = random.randint(2, 5)
        augmented = audio.copy()
        for _ in range(num_reflections):
            delay_samples = int(self.sr * random.uniform(0.015, 0.060))
            attenuation = random.uniform(0.05, 0.22)
            if len(audio) > delay_samples:
                augmented[delay_samples:] += audio[:-delay_samples] * attenuation
        return augmented

    def apply_mic_eq(self, audio: np.ndarray) -> np.ndarray:
        """Applies randomized microphone frequency response coloration."""
        filter_choice = random.choice(["lowpass", "highboost", "notch", "none"])
        if filter_choice == "lowpass":
            cutoff = random.uniform(3000, 7000)
            b, a = butter(2, cutoff / (self.sr / 2), btype="low")
            audio = lfilter(b, a, audio)
        elif filter_choice == "highboost":
            cutoff = random.uniform(2000, 4000)
            b, a = butter(1, cutoff / (self.sr / 2), btype="high")
            high_band = lfilter(b, a, audio)
            audio = audio + high_band * random.uniform(0.2, 0.5)
        return audio.astype(np.float32)

    def apply_ambient_noise(self, audio: np.ndarray, target_snr_db: float) -> np.ndarray:
        """Injects stationary and pink ambient room noise at specified SNR."""
        sig_power = np.mean(audio ** 2) + 1e-9
        noise_power = sig_power / (10 ** (target_snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
        return (audio + noise).astype(np.float32)

    def apply_lossy_codec_quantization(self, audio: np.ndarray) -> np.ndarray:
        """Simulates lossy WhatsApp Opus / MP3 compression bit-depth quantization."""
        levels = random.choice([64, 128, 256, 512])
        quantized = np.round(audio * levels) / levels
        return quantized.astype(np.float32)

    def augment(self, audio: np.ndarray) -> np.ndarray:
        # 1. Room Reflections / RIR
        if random.random() < 0.65:
            audio = self.apply_rir_reflection(audio)

        # 2. Ambient Noise (SNR 10dB to 25dB)
        if random.random() < 0.70:
            snr = random.uniform(10.0, 25.0)
            audio = self.apply_ambient_noise(audio, snr)

        # 3. Mic Coloration EQ (lowpass, highboost, muffling)
        if random.random() < 0.60:
            audio = self.apply_mic_eq(audio)

        # 4. WhatsApp / Telephony Lossy Codec Simulation
        if random.random() < 0.40:
            audio = self.apply_lossy_codec_quantization(audio)

        # 5. Soft / Muffled Speech Simulation (Volume scaling 0.08 to 1.25)
        gain = random.uniform(0.08, 1.25)
        audio = audio * gain

        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = (audio / peak) * random.uniform(0.30, 0.90)

        return audio.astype(np.float32)


class LiveRobustAudioDataset(Dataset):
    def __init__(self, file_label_pairs: List[Tuple[str, int]], sr: int = 16000, max_len_sec: float = 3.0, is_train: bool = True):
        self.pairs = file_label_pairs
        self.sr = sr
        self.max_samples = int(max_len_sec * sr)
        self.is_train = is_train
        self.augmentor = RobustAcousticAugmentor(sr=sr)
        self._cache = {}

        # Pre-cache raw audio in memory for ultra-fast multi-epoch training
        for path, _ in self.pairs:
            try:
                raw, voiced, _ = decode_and_sanitize_audio(path, target_sr=self.sr)
                self._cache[path] = voiced if len(voiced) >= int(0.5 * self.sr) else raw
            except Exception:
                self._cache[path] = np.zeros(self.max_samples, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        audio = self._cache.get(path, np.zeros(self.max_samples, dtype=np.float32)).copy()

        if self.is_train:
            audio = self.augmentor.augment(audio)

        # Fixed length padding / trimming
        if len(audio) < self.max_samples:
            pad = np.zeros(self.max_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        else:
            audio = audio[: self.max_samples]

        return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.float32)


def collect_all_data_pairs() -> List[Tuple[str, int]]:
    valid_exts = {".wav", ".mp3", ".mpeg", ".ogg", ".flac", ".m4a", ".aac", ".webm"}
    human = []
    ai = []
    for root, _, fnames in os.walk("data"):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                p = os.path.join(root, f)
                low = p.lower()
                if "human" in low or "real" in low or "bonafide" in low:
                    human.append(p)
                elif "ai" in low or "spoof" in low or "clone" in low or "fake" in low:
                    ai.append(p)

    pairs = [(p, 0) for p in set(human)] + [(p, 1) for p in set(ai)]
    random.shuffle(pairs)
    return pairs


def train_live_robust_model(
    backbone_name: str = "lightweight",
    epochs: int = 8,
    batch_size: int = 32,
    output_path: str = "models/voiceshield_live_robust.pt",
):
    print("=" * 80, flush=True)
    print("VOICESHIELD LIVE-ROBUST ACOUSTIC DEEPFAKE FINE-TUNING", flush=True)
    print(f"Backbone: {backbone_name} | Epochs: {epochs} | Output: {output_path}", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: [{device}]", flush=True)

    all_pairs = collect_all_data_pairs()
    print(f"[*] Total dataset audio records found: {len(all_pairs)}", flush=True)

    split_idx = int(len(all_pairs) * 0.85)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"[*] Train set: {len(train_pairs)} samples | Val set: {len(val_pairs)} samples", flush=True)

    print("[*] Pre-caching in-memory waveforms for accelerated training...", flush=True)
    train_ds = LiveRobustAudioDataset(train_pairs, is_train=True)
    val_ds = LiveRobustAudioDataset(val_pairs, is_train=False)
    print("[+] Waveforms pre-cached in RAM successfully!", flush=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = VoiceShieldNeuralDetector(backbone_name=backbone_name, dropout=0.25, device=device).to(device)

    criterion = BinaryFocalLoss(alpha=0.25, gamma=2.0)

    # Differential Learning Rates: Backbone (1e-4) & Attention Classification Head (5e-4)
    backbone_params = [p for n, p in model.named_parameters() if "backbone" in n]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
            {"params": head_params, "lr": 5e-4, "weight_decay": 1e-4},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\n[*] Starting {epochs} Epochs of Acoustic Channel Robust Fine-Tuning...\n", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for audios, labels in train_loader:
            audios, labels = audios.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(audios)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            probs = torch.sigmoid(logits)
            train_loss += loss.item() * len(labels)
            preds = (probs >= 0.50).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        train_loss /= max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for audios, labels in val_loader:
                audios, labels = audios.to(device), labels.to(device)
                logits = model(audios)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                val_loss += loss.item() * len(labels)
                preds = (probs >= 0.50).float()
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_loss /= max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d}/{epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "backbone_name": backbone_name,
                },
                output_path,
            )

    print(f"\n[+] Fine-Tuning Complete! Live-Robust Model Saved to: {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceShield Live-Robust Fine-Tuning Pipeline")
    parser.add_argument("--backbone", type=str, default="lightweight", help="Backbone architecture")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output", type=str, default="models/voiceshield_live_robust.pt", help="Output path")
    args = parser.parse_args()

    train_live_robust_model(
        backbone_name=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=args.output,
    )
