"""
Unit & Benchmark Tests for VoiceShield Neural Step 2 Dataset Loader & Waveform Augmenter.
Validates:
  - Strict output tensor shape: (batch_size, 48000) float32
  - Label tensor shape: (batch_size, 1) float32
  - Deterministic validation cropping vs. dynamic training augmentations
  - Data loading and dynamic augmentation throughput (samples / second)
"""

import os
import time
import pytest
import torch

from src.dataset_loader import (
    TARGET_SAMPLES,
    VoiceShieldDataset,
    create_dataloaders,
    parse_manifest,
)
from src.waveform_augmentations import WaveformAugmenter


@pytest.fixture
def manifest_path():
    p = "data/manifest.csv"
    if not os.path.exists(p):
        pytest.skip(f"Dataset manifest not found at {p}")
    return p


def test_waveform_augmenter_shapes_and_finiteness():
    """Verifies that WaveformAugmenter preserves exact shape and returns clean float32 tensors."""
    augmenter = WaveformAugmenter(sample_rate=16000)
    waveform = torch.sin(2 * 3.14159 * 440.0 * torch.linspace(0, 3.0, 48000)).float()

    # 1. Test individual sub-augmentations
    telephony = augmenter.apply_telephony_filter(waveform)
    assert telephony.shape == (48000,)
    assert not torch.isnan(telephony).any()

    noisy = augmenter.apply_additive_noise(waveform)
    assert noisy.shape == (48000,)
    assert not torch.isnan(noisy).any()

    gained = augmenter.apply_gain_and_clipping(waveform)
    assert gained.shape == (48000,)
    assert not torch.isnan(gained).any()

    masked = augmenter.apply_time_masking(waveform)
    assert masked.shape == (48000,)
    assert not torch.isnan(masked).any()

    # 2. Test full forward pass
    aug_out = augmenter(waveform)
    assert aug_out.shape == (48000,)
    assert aug_out.dtype == torch.float32
    assert not torch.isnan(aug_out).any()
    assert not torch.isinf(aug_out).any()


def test_dataloader_batch_shapes_and_types(manifest_path):
    """Loads a batch of 4 audio samples through create_dataloaders and asserts strict contracts."""
    batch_size = 4
    train_loader, val_loader = create_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
        val_split=0.20,
    )

    assert len(train_loader) > 0
    assert len(val_loader) > 0

    # Fetch first training batch
    audio_batch, label_batch, meta_batch = next(iter(train_loader))

    # Assert shape (batch_size, 48000)
    assert isinstance(audio_batch, torch.Tensor)
    assert audio_batch.shape == (batch_size, TARGET_SAMPLES)
    assert audio_batch.dtype == torch.float32
    assert not torch.isnan(audio_batch).any()

    # Assert labels (batch_size, 1)
    assert isinstance(label_batch, torch.Tensor)
    assert label_batch.shape == (batch_size, 1)
    assert label_batch.dtype == torch.float32
    assert ((label_batch == 0.0) | (label_batch == 1.0)).all()

    # Verify metadata fields
    assert "file_path" in meta_batch
    assert "speaker_id" in meta_batch
    assert len(meta_batch["file_path"]) == batch_size


def test_val_loader_deterministic_crop(manifest_path):
    """Verifies that validation dataset applies deterministic cropping and does not mutate signals."""
    records = parse_manifest(manifest_path)
    val_dataset = VoiceShieldDataset(
        records=records[:4],
        is_train=False,
        augmenter=None,
    )

    audio1, label1, _ = val_dataset[0]
    audio2, label2, _ = val_dataset[0]

    assert audio1.shape == (TARGET_SAMPLES,)
    assert torch.allclose(audio1, audio2, atol=1e-6)
    assert label1.item() == label2.item()


def test_dataset_throughput_benchmark(manifest_path):
    """Benchmarks loading and augmentation throughput in samples per second."""
    batch_size = 8
    train_loader, _ = create_dataloaders(
        manifest_path=manifest_path,
        batch_size=batch_size,
        num_workers=0,
    )

    num_batches = min(5, len(train_loader))
    start_time = time.perf_counter()
    samples_processed = 0

    for idx, (audio, labels, _) in enumerate(train_loader):
        if idx >= num_batches:
            break
        samples_processed += audio.shape[0]

    elapsed = time.perf_counter() - start_time
    throughput = samples_processed / max(1e-4, elapsed)

    print(f"\n[BENCHMARK] Processed {samples_processed} samples in {elapsed:.3f}s -> Throughput: {throughput:.2f} samples/sec")
    assert throughput > 1.0, "DataLoader throughput should comfortably exceed 1 sample/sec."
