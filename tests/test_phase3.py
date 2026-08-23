"""
Unit tests for Phase 3: Reproducible Baseline Training, Evaluation & Model Card.
"""

import os
import pytest

from scripts.evaluate_model import INDEPENDENT_METRICS_PATH, evaluate_baseline
from scripts.train_model import train_baseline
from src.config import (
    CONFUSION_MATRIX_PNG,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
)


def test_train_baseline_pipeline():
    """Verify that training pipeline executes, tunes hyperparameters, and saves model + metadata."""
    metadata = train_baseline()

    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(MODEL_METADATA_PATH)
    assert "training_dataset_hash" in metadata
    assert "optimal_decision_threshold" in metadata
    assert "best_hyperparameters" in metadata
    assert "feature_configuration" in metadata
    assert metadata["feature_configuration"]["total_features"] == 42


def test_evaluation_pipeline_and_reports():
    """Verify independent test evaluation, reports/metrics.json, and confusion matrix image."""
    report = evaluate_baseline()

    assert os.path.exists(METRICS_PATH)
    assert os.path.exists(INDEPENDENT_METRICS_PATH)
    assert os.path.exists(CONFUSION_MATRIX_PNG)
    assert "overall_metrics" in report
    assert "accuracy" in report["overall_metrics"]
    assert "macro_f1" in report["overall_metrics"]
    assert "roc_auc" in report["overall_metrics"]
    assert "false_positive_rate" in report["overall_metrics"]
    assert "false_negative_rate" in report["overall_metrics"]
    assert "confusion_matrix" in report
    assert "production_reliability_disclaimer" in report


def test_model_card_documentation():
    """Verify that docs/model_card.md exists with required sections."""
    assert os.path.exists("docs/model_card.md")
    with open("docs/model_card.md", "r", encoding="utf-8") as f:
        content = f.read()
    assert "VoiceShield Model Card" in content
    assert "Acoustic Feature Representation" in content
    assert "Intended Use" in content
