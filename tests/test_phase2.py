"""
Unit tests for Phase 2: Dataset Quality, Manifest Builder & Reproducibility Auditor.
"""

import os
import pytest

from scripts.build_manifest import build_manifest
from scripts.check_dataset import DATASET_REPORT_PATH, run_dataset_audit
from src.config import MANIFEST_PATH


def test_manifest_builder_and_required_columns():
    """Verify that build_manifest generates manifest.csv with all required columns."""
    records = build_manifest()

    assert os.path.exists(MANIFEST_PATH)
    assert len(records) > 0

    sample = records[0]
    required_cols = [
        "path",
        "label",
        "speaker_id",
        "language",
        "codec",
        "sample_rate",
        "duration_seconds",
        "source",
        "split",
        "sha256_hash",
        "is_valid",
    ]
    for col in required_cols:
        assert col in sample, f"Missing required manifest column: {col}"

    assert sample["label"] in ["bona_fide", "spoof"]


def test_dataset_check_and_report_generation():
    """Verify dataset auditor execution and reports/dataset_report.json creation."""
    report = run_dataset_audit(MANIFEST_PATH)

    assert os.path.exists(DATASET_REPORT_PATH)
    assert "total_files" in report
    assert "class_distribution" in report
    assert "duration_statistics" in report
    assert "sample_rate_distribution" in report
    assert "duplicate_hash_groups" in report
    assert "data_leakage_detected" in report
    assert report["data_leakage_detected"] is False  # Zero train/test leakage
    assert report["total_files"] == 24
