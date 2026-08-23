"""
VoiceShield Split Disjointness & Zero-Leakage Unit Tests (Phase 3).
Verifies that speaker hashes and source recordings never cross train/test splits.
"""

import pytest
import pandas as pd
from src.dataset_manifest import load_validated_manifest, MANIFEST_PATH
from scripts.create_splits import TEST_GROUPS, create_speaker_independent_splits


def test_speaker_independence():
    df = load_validated_manifest(MANIFEST_PATH)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_spks = set(train_df["speaker_id_hash"].unique())
    test_spks = set(test_df["speaker_id_hash"].unique())

    overlap = train_spks.intersection(test_spks)
    assert len(overlap) == 0, f"Found speaker overlap across splits: {overlap}"


def test_source_recording_isolation():
    df = load_validated_manifest(MANIFEST_PATH)
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    train_hashes = set(train_df["sha256_hash"].unique())
    test_hashes = set(test_df["sha256_hash"].unique())

    overlap = train_hashes.intersection(test_hashes)
    assert len(overlap) == 0, f"Found identical source recording across splits: {overlap}"


def test_test_groups_defined():
    for grp in ["in_domain_test", "unseen_speaker_test", "unseen_source_test", "noisy_test", "compressed_test", "tts_test"]:
        assert grp in TEST_GROUPS
