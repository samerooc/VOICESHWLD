"""
VoiceShield Native Acoustic Neural Model Training Pipeline.
Trains LightweightSpeechBackbone + MultiHeadAttentionPooling across 1,100+ audio files
with Focal Loss, Cosine Annealing, and robust room-acoustic simulation.
"""

import os
import sys
import json
import time
import random
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import decode_and_sanitize_audio
from src.neural_model import VoiceShieldNeuralDetector

# Deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction="none")
        pt = torch.exp(-bce)
        focal_loss = self.alpha * ((1.0 - pt) ** self.gamma) * bce
        return focal_loss.mean()


class AugmentedAudioDataset(Dataset):
    def __init__(self, file_label_pairs, sample_rate=16000, max_len_sec=3.0, is_train=True):
        self.pairs = file_label_pairs
        self.sr = sample_rate
        self.max_samples = int(max_len_sec * sample_rate)
        self.is_train = is_train

    def __len__(self):
        return len(self.pairs)

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        # 1. Random Gain (0.7 to 1.15)
        gain = np.random.uniform(0.7, 1.15)
        audio = audio * gain

        # 2. Add mild Gaussian noise (SNR 20-35 dB) with 60% probability
        if np.random.rand() < 0.60:
            noise_amp = np.random.uniform(0.001, 0.008)
            audio = audio + np.random.normal(0, noise_amp, len(audio)).astype(np.float32)

        # 3. Simple simulated room reflection with 40% probability
        if np.random.rand() < 0.40:
            delay = int(self.sr * np.random.uniform(0.015, 0.040))
            if len(audio) > delay:
                audio[delay:] += audio[:-delay] * np.random.uniform(0.08, 0.20)

        # Peak normalize
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = (audio / peak) * 0.88
        return audio.astype(np.float32)

    def __getitem__(self, idx):
        file_path, label = self.pairs[idx]
        try:
            raw, voiced, _ = decode_and_sanitize_audio(file_path, target_sr=self.sr)
            audio = voiced if len(voiced) >= int(0.5 * self.sr) else raw
        except Exception:
            audio = np.zeros(self.max_samples, dtype=np.float32)

        if self.is_train:
            audio = self._augment(audio)

        # Pad or trim to fixed length
        if len(audio) < self.max_samples:
            pad_len = self.max_samples - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")
        else:
            audio = audio[:self.max_samples]

        return torch.from_numpy(audio).float(), torch.tensor(label, dtype=torch.float32)


def collect_all_dataset_pairs():
    human_files = []
    ai_files = []

    for root, _, fnames in os.walk("data"):
        for f in fnames:
            if f.endswith(".wav") or f.endswith(".mp3"):
                p = os.path.join(root, f)
                low = p.lower()
                if "human" in low or "real" in low or "bonafide" in low:
                    human_files.append(p)
                elif "ai" in low or "spoof" in low or "clone" in low or "fake" in low:
                    ai_files.append(p)

    pairs = [(p, 0) for p in set(human_files)] + [(p, 1) for p in set(ai_files)]
    random.shuffle(pairs)
    return pairs


def main():
    print("=" * 80)
    print("VOICESHIELD SOTA NATIVE ACOUSTIC NEURAL CLASSIFIER TRAINING")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: [{device}]")

    all_pairs = collect_all_dataset_pairs()
    print(f"[*] Total unique dataset audio records: {len(all_pairs)}")

    split_idx = int(len(all_pairs) * 0.85)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    print(f"[*] Train set: {len(train_pairs)} samples | Val set: {len(val_pairs)} samples")

    train_ds = AugmentedAudioDataset(train_pairs, is_train=True)
    val_ds = AugmentedAudioDataset(val_pairs, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)

    model = VoiceShieldNeuralDetector(backbone_name="lightweight", dropout=0.25, device=device).to(device)

    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    best_ckpt_path = "models/voiceshield_neural_best.pt"

    print("\n[*] Training 8 Epochs with Multi-Head Attention & Focal Loss...\n")
    for epoch in range(1, 9):
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
            f"Epoch {epoch:02d}/08 ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} (Acc: {train_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} (Acc: {val_acc:.1f}%)"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                best_ckpt_path,
            )

    print(f"\n[+] Fine-Tuning Complete! Best High-Performance Checkpoint Saved to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
