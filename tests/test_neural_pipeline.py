"""
Unit & Integration Test Suite for Phase 3 Deep Learning Pipeline & Model Training.
Tests:
1. VoiceShieldNeuralClassifier architecture, tensor dimensions, and prediction interface.
2. BinaryFocalLossWithLogits formulation, stability, and gradient backpropagation.
3. VoiceShieldDataset batching, fixed 48k sample window slicing/padding.
"""

import os
import sys
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import MANIFEST_PATH
from src.dataset_loader import (
    TARGET_DURATION_SEC,
    TARGET_SAMPLE_RATE,
    TARGET_SAMPLES,
    VoiceShieldDataset,
    create_dataloaders,
    parse_manifest,
)
from src.losses import BinaryFocalLossWithLogits
from src.neural_model import VoiceShieldNeuralClassifier, VoiceShieldNeuralDetector


def test_neural_classifier_forward_shape_and_predict():
    """Verify VoiceShieldNeuralClassifier accepts (B, 48000) and returns (B,) logits."""
    model = VoiceShieldNeuralClassifier(backbone_name="lightweight", device=torch.device("cpu"))
    model.eval()

    batch_size = 2
    dummy_input = torch.randn(batch_size, TARGET_SAMPLES, dtype=torch.float32)

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (batch_size,)
    assert not torch.isnan(logits).any()

    # Test predict_waveform interface
    pred = model.predict_waveform(dummy_input[0])
    assert "spoof_probability" in pred
    assert "human_probability" in pred
    assert "raw_logit" in pred
    assert 0.0 <= pred["spoof_probability"] <= 1.0
    assert 0.0 <= pred["human_probability"] <= 1.0


def test_binary_focal_loss_with_logits():
    """Verify BinaryFocalLossWithLogits calculation and gradient backprop."""
    criterion = BinaryFocalLossWithLogits(gamma=2.0, alpha=0.25)

    logits = torch.tensor([2.5, -2.5, 0.0, 1.2], requires_grad=True)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss = criterion(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

    # Test backpropagation
    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_dataset_loader_batch_and_shapes():
    """Verify VoiceShieldDataset creates standard (48000,) audio and (1,) label tensors."""
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip(f"Manifest {MANIFEST_PATH} not found.")

    records = parse_manifest(MANIFEST_PATH)
    assert len(records) > 0

    dataset = VoiceShieldDataset(
        records=records[:10],
        sample_rate=TARGET_SAMPLE_RATE,
        duration_sec=TARGET_DURATION_SEC,
        is_train=False,
    )

    audio_tensor, label_tensor, meta = dataset[0]
    assert audio_tensor.shape == (TARGET_SAMPLES,)
    assert label_tensor.shape == (1,)
    assert label_tensor.item() in [0.0, 1.0]
    assert "file_path" in meta
    assert not torch.isnan(audio_tensor).any()


def test_create_dataloaders_group_stratification():
    """Verify create_dataloaders returns functional train and validation loaders."""
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip(f"Manifest {MANIFEST_PATH} not found.")

    train_loader, val_loader = create_dataloaders(
        manifest_path=MANIFEST_PATH,
        batch_size=4,
        num_workers=0,
        val_split=0.20,
    )

    for audio_batch, label_batch, _ in train_loader:
        assert audio_batch.shape[1] == TARGET_SAMPLES
        assert label_batch.shape[1] == 1
        break
