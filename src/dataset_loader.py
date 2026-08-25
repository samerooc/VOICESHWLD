"""
VoiceShield Neural Step 2: PyTorch Dataset Loader with Dynamic Waveform Augmentations.
Loads raw audio from data/manifest.csv, standardizes to 16kHz mono, and delivers
fixed 3.0s (48,000 samples) tensors for deep learning architectures.
"""

import csv
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio

from src.config import MANIFEST_PATH, SAMPLE_RATE
from src.waveform_augmentations import WaveformAugmenter

TARGET_SAMPLE_RATE: int = 16000
TARGET_DURATION_SEC: float = 3.0
TARGET_SAMPLES: int = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)  # 48,000


class VoiceShieldDataset(Dataset):
    """
    PyTorch Dataset for raw audio waveforms with speaker attribution and on-the-fly augmentations.
    Delivers (48000,) float32 waveforms and (1,) float32 binary labels.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        sample_rate: int = TARGET_SAMPLE_RATE,
        duration_sec: float = TARGET_DURATION_SEC,
        is_train: bool = True,
        augmenter: Optional[WaveformAugmenter] = None,
    ) -> None:
        self.records = records
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * duration_sec)
        self.is_train = is_train
        self.augmenter = augmenter if (augmenter is not None and is_train) else (WaveformAugmenter() if is_train else None)

    def __len__(self) -> int:
        return len(self.records)

    def _load_and_resample(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Loads raw audio file into a 1D float32 PyTorch tensor at 16,000 Hz.
        """
        # Resolve path relative to project root if needed
        if not os.path.isabs(file_path) and not os.path.exists(file_path):
            proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(proj_root, file_path)
            if os.path.exists(candidate):
                file_path = candidate

        try:
            waveform, orig_sr = torchaudio.load(file_path)
            # Convert multi-channel to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # Resample if needed
            if orig_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
                waveform = resampler(waveform)
            audio = waveform.squeeze(0).float()
        except Exception:
            # Fallback to soundfile / numpy
            data, orig_sr = sf.read(file_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if orig_sr != self.sample_rate:
                import librosa
                data = librosa.resample(data, orig_sr=orig_sr, target_sr=self.sample_rate)
            audio = torch.from_numpy(data).float()

        # Remove DC offset & peak normalize
        audio = audio - torch.mean(audio)
        max_amp = torch.max(torch.abs(audio))
        if max_amp > 1e-6:
            audio = audio / (max_amp + 1e-8)

        return audio, self.sample_rate

    def _slice_or_pad(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Standardizes audio sequence length to exactly target_samples (48,000 samples).
        """
        num_samples = audio.shape[-1]

        if num_samples == self.target_samples:
            return audio

        if num_samples > self.target_samples:
            if self.is_train:
                # Random 3.0s slice
                max_start = num_samples - self.target_samples
                start = random.randint(0, max_start)
                return audio[start : start + self.target_samples]
            else:
                # Deterministic center crop
                start = (num_samples - self.target_samples) // 2
                return audio[start : start + self.target_samples]

        # Audio is shorter than target length: pad to 48,000
        if self.is_train:
            # Repeat-pad then slice
            repeat_factor = (self.target_samples // max(1, num_samples)) + 1
            repeated = audio.repeat(repeat_factor)
            return repeated[: self.target_samples]
        else:
            # Deterministic center-zero-pad
            pad_total = self.target_samples - num_samples
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return torch.nn.functional.pad(audio, (pad_left, pad_right), value=0.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.records[idx]
        file_path = record["file_path"]
        raw_label = record["label"]

        # Parse binary label (0.0 = bona_fide, 1.0 = spoof)
        if isinstance(raw_label, str):
            label_val = 1.0 if raw_label.strip() in ["1", "spoof", "spoofed", "synthetic"] else 0.0
        else:
            label_val = float(raw_label)

        # 1. Load audio tensor
        audio, _ = self._load_and_resample(file_path)

        # 2. Slice or Pad to exact 48,000 samples (3.0 seconds)
        audio_fixed = self._slice_or_pad(audio)

        # 3. Dynamic Waveform Augmentation (if enabled)
        if self.is_train and self.augmenter is not None:
            audio_fixed = self.augmenter(audio_fixed)

        # Ensure strict 1D shape (48000,) and label shape (1,)
        audio_tensor = audio_fixed.view(-1).float()
        if audio_tensor.shape[0] != self.target_samples:
            if audio_tensor.shape[0] > self.target_samples:
                audio_tensor = audio_tensor[: self.target_samples]
            else:
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.target_samples - audio_tensor.shape[0]))

        label_tensor = torch.tensor([label_val], dtype=torch.float32)

        meta = {
            "file_path": file_path,
            "speaker_id": record.get("speaker_id", "unknown"),
            "generator_type": record.get("generator_type", "none"),
        }

        return audio_tensor, label_tensor, meta


def parse_manifest(manifest_path: str = MANIFEST_PATH) -> List[Dict[str, Any]]:
    """
    Parses manifest.csv into a list of verified record dictionaries.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Map alternative column names if present
            f_path = row.get("file_path") or row.get("path")
            if not f_path:
                continue
            records.append({
                "file_path": f_path,
                "label": row.get("label", "0"),
                "speaker_id": row.get("speaker_id", "unknown"),
                "generator_type": row.get("generator_type", row.get("source", "none")),
                "split": row.get("split", "train"),
            })

    if not records:
        raise ValueError(f"No valid records found in {manifest_path}")

    return records


def create_dataloaders(
    manifest_path: str = MANIFEST_PATH,
    batch_size: int = 16,
    num_workers: int = 0,
    val_split: float = 0.20,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory function that loads manifest.csv, performs group-aware speaker stratification,
    and returns PyTorch (train_loader, val_loader) instances.

    Args:
        manifest_path: Path to dataset manifest CSV.
        batch_size: Batch size for training and validation.
        num_workers: Number of subprocess workers for data loading.
        val_split: Proportion of data to allocate for validation.
        seed: Random seed for reproducible partitioning.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    records = parse_manifest(manifest_path)
    rng = random.Random(seed)

    # 1. Group records by speaker_id for leakage-free validation partitioning
    speaker_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        spk = r.get("speaker_id", f"spk_{len(speaker_groups)}")
        if spk not in speaker_groups:
            speaker_groups[spk] = []
        speaker_groups[spk].append(r)

    unique_speakers = list(speaker_groups.keys())
    rng.shuffle(unique_speakers)

    train_records = []
    val_records = []

    # Check if manifest already has explicit split columns
    has_predefined_splits = any(r.get("split") in ["val", "validation", "test"] for r in records)
    if has_predefined_splits and len(records) > 20:
        for r in records:
            if r.get("split") in ["val", "validation", "test"]:
                val_records.append(r)
            else:
                train_records.append(r)
    else:
        # Perform speaker-stratified partition
        target_val_count = int(len(records) * val_split)
        curr_val_count = 0

        for spk in unique_speakers:
            spk_recs = speaker_groups[spk]
            if curr_val_count < target_val_count and len(val_records) == 0 or (curr_val_count + len(spk_recs) <= target_val_count * 1.3):
                val_records.extend(spk_recs)
                curr_val_count += len(spk_recs)
            else:
                train_records.extend(spk_recs)

    # Fallback sanity check in case all fell into one split
    if len(val_records) == 0:
        val_records = train_records[-max(1, int(len(train_records) * val_split)) :]
        train_records = train_records[: -len(val_records)]

    # 2. Instantiate Datasets
    train_dataset = VoiceShieldDataset(
        records=train_records,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_sec=TARGET_DURATION_SEC,
        is_train=True,
        augmenter=WaveformAugmenter(sample_rate=TARGET_SAMPLE_RATE),
    )

    val_dataset = VoiceShieldDataset(
        records=val_records,
        sample_rate=TARGET_SAMPLE_RATE,
        duration_sec=TARGET_DURATION_SEC,
        is_train=False,
        augmenter=None,
    )

    # 3. Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
