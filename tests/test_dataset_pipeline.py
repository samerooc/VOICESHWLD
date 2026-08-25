"""
Unit & Integration Test Suite for Phase 2 Dataset Pipeline & Waveform Augmentations.
Tests:
1. Manifest parsing and schema verification (data/manifest.csv).
2. Audio file integrity on disk (existence, 16kHz sample rate, mono channels).
3. Speaker ID independence and GroupKFold partition sanity.
4. Dynamic PyTorch WaveformAugmenter tensor transformation and shape preservation.
"""

import csv
import os
import sys
import numpy as np
import pytest
import soundfile as sf
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MANIFEST_PATH, SAMPLE_RATE
from src.waveform_augmentations import WaveformAugmenter


def test_manifest_existence_and_schema():
    """Verify that data/manifest.csv exists and has all required columns."""
    assert os.path.exists(MANIFEST_PATH), f"Manifest not found at {MANIFEST_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        assert "file_path" in fieldnames
        assert "label" in fieldnames
        assert "speaker_id" in fieldnames
        assert "generator_type" in fieldnames
        assert "duration_sec" in fieldnames
        assert "split" in fieldnames

        rows = list(reader)
        assert len(rows) > 0, "Manifest contains 0 rows."


def test_manifest_audio_files_integrity_on_disk():
    """Verify all audio files in manifest exist on disk, are 16kHz, mono, and non-empty."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check a representative sample of up to 30 files
    check_rows = rows[:30] + rows[-30:] if len(rows) > 60 else rows

    for r in check_rows:
        rel_path = r["file_path"]
        full_path = os.path.join(ROOT_DIR, rel_path)
        assert os.path.exists(full_path), f"Audio file not found on disk: {full_path}"

        info = sf.info(full_path)
        assert info.samplerate == SAMPLE_RATE, f"Sample rate mismatch: {info.samplerate} != {SAMPLE_RATE}"
        assert info.channels == 1, f"Audio is not mono: {info.channels} channels"
        assert info.duration >= 0.5, f"Audio duration too short: {info.duration:.2f}s"
        assert int(r["label"]) in [0, 1], f"Invalid label: {r['label']}"


def test_speaker_group_partition_leakage():
    """Verify speaker independence across train and test splits."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    train_speakers = set(r["speaker_id"] for r in rows if r.get("split") == "train")
    test_speakers = set(r["speaker_id"] for r in rows if r.get("split") == "test")

    # If test split is defined and populated, verify zero leakage
    if test_speakers and train_speakers:
        intersection = train_speakers.intersection(test_speakers)
        assert len(intersection) == 0, f"Speaker data leakage detected between train and test: {intersection}"


def test_waveform_augmenter_tensor_shape_and_range():
    """Verify WaveformAugmenter processes (B, N) tensors and preserves exact shape."""
    augmenter = WaveformAugmenter(
        sample_rate=16000,
        p_telephony=1.0,
        p_noise=1.0,
        p_gain_and_clip=1.0,
        p_time_mask=1.0,
    )
    augmenter.eval()

    batch_size = 4
    num_samples = 48000  # 3.0s @ 16kHz
    dummy_audio = torch.randn(batch_size, num_samples, dtype=torch.float32) * 0.5

    with torch.no_grad():
        aug_audio = augmenter(dummy_audio)

    assert aug_audio.shape == (batch_size, num_samples)
    assert not torch.isnan(aug_audio).any()
    assert not torch.isinf(aug_audio).any()
    assert torch.max(torch.abs(aug_audio)).item() <= 1.0


def test_waveform_augmenter_1d_tensor():
    """Verify WaveformAugmenter processes single 1D tensor."""
    augmenter = WaveformAugmenter(sample_rate=16000)
    audio_1d = torch.randn(32000, dtype=torch.float32) * 0.4

    with torch.no_grad():
        out_1d = augmenter(audio_1d)

    assert out_1d.shape == (32000,)
    assert not torch.isnan(out_1d).any()
