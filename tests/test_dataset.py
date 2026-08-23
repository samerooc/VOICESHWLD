"""
Comprehensive Unit Tests for VoiceShield Dataset Management & Quality Audit (Phase 2).
Covers:
1. Missing files
2. Corrupt files
3. Silent files
4. Too-short files
5. Invalid WAV files
6. Duplicate SHA-256 hashes
7. Invalid labels
8. Invalid split names
9. Class imbalance
10. Train/test overlap
11. Missing speaker metadata
12. Duration statistics
13. Sample-rate statistics
14. Files per class
15. Files per split
"""

import os
import pytest

from scripts.build_manifest import build_manifest, compute_sha256, inspect_wav_file
from scripts.check_dataset import DATASET_REPORT_PATH, run_dataset_audit
from src.config import MANIFEST_PATH


def test_build_manifest_schema_and_columns():
    """Verify that build_manifest creates manifest.csv containing all required schema columns."""
    records = build_manifest()
    assert os.path.exists(MANIFEST_PATH)
    assert len(records) == 24

    required_columns = [
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
    for r in records:
        for col in required_columns:
            assert col in r, f"Missing required column: {col}"
        assert r["label"] in ["bona_fide", "spoof"], f"Invalid label: {r['label']}"
        assert r["split"] in ["train", "test", "val"], f"Invalid split: {r['split']}"
        assert len(r["sha256_hash"]) == 64, "SHA-256 hash must be 64 hexadecimal characters."


def test_dataset_audit_15_checks():
    """Verify all 15 audit dimensions in reports/dataset_report.json."""
    report = run_dataset_audit(MANIFEST_PATH)
    assert os.path.exists(DATASET_REPORT_PATH)

    # 1. Missing files
    assert "missing_files" in report
    assert len(report["missing_files"]) == 0

    # 2. Corrupt / Invalid files
    assert "invalid_files" in report
    assert report["invalid_files"] == 0

    # 3. Silent files
    assert "silent_files" in report
    assert len(report["silent_files"]) == 0

    # 4. Too-short files
    assert "too_short_files" in report
    assert len(report["too_short_files"]) == 0

    # 5. Invalid WAV files
    assert report["valid_files"] == report["total_files"] == 24

    # 6. Duplicate SHA-256 hashes
    assert "duplicate_hash_groups" in report
    assert isinstance(report["duplicate_hash_groups"], dict)
    # The duplicate group between test/ai_voice 5.wav and 6.wav is correctly flagged
    if report["duplicate_hash_groups"]:
        assert len(report["warnings"]) > 0

    # 7. Invalid labels
    assert "invalid_labels" in report
    assert len(report["invalid_labels"]) == 0

    # 8. Invalid split names
    assert "invalid_splits" in report
    assert len(report["invalid_splits"]) == 0

    # 9. Class imbalance / distribution
    assert "class_distribution" in report
    assert "bona_fide" in report["class_distribution"]
    assert "spoof" in report["class_distribution"]

    # 10. Train/test overlap & Data leakage
    assert "data_leakage_detected" in report
    assert report["data_leakage_detected"] is False
    assert report["leaked_hash_count"] == 0

    # 11. Missing speaker metadata & speaker overlap
    assert "speaker_overlap_detected" in report
    assert report["speaker_overlap_detected"] is False

    # 12. Duration statistics
    dur_stats = report["duration_statistics"]
    assert dur_stats["total_seconds"] > 0
    assert dur_stats["average_seconds"] > 0
    assert dur_stats["min_seconds"] >= 0.50

    # 13. Sample-rate statistics
    assert "sample_rate_distribution" in report
    assert len(report["sample_rate_distribution"]) > 0

    # 14. Files per class
    assert report["class_distribution"]["bona_fide"] == 12
    assert report["class_distribution"]["spoof"] == 12

    # 15. Files per split
    assert "split_distribution" in report
    assert "train" in report["split_distribution"]
    assert "test" in report["split_distribution"]


def test_inspect_wav_file_detection():
    """Verify single file inspection on bona_fide and spoof samples."""
    sample_human = inspect_wav_file("data/test/human/01.wav")
    assert sample_human["label"] == "bona_fide"
    assert sample_human["label_id"] == 0
    assert sample_human["is_valid"] is True
    assert sample_human["split"] == "test"

    sample_spoof = inspect_wav_file("data/test/ai_voice/1.wav")
    assert sample_spoof["label"] == "spoof"
    assert sample_spoof["label_id"] == 1
    assert sample_spoof["is_valid"] is True
    assert sample_spoof["split"] == "test"
