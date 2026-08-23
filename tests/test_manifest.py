"""
VoiceShield Manifest Schema & Integrity Tests.
Validates required columns, allowed labels, allowed spoof types, and non-empty metadata.
"""

import pytest
import pandas as pd
from src.dataset_manifest import (
    load_validated_manifest,
    generate_manifest_csv,
    REQUIRED_COLUMNS,
    ALLOWED_LABELS,
    ALLOWED_SPOOF_TYPES,
    MANIFEST_PATH,
)


def test_manifest_generation_and_loading():
    df = generate_manifest_csv()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 20

    loaded = load_validated_manifest(MANIFEST_PATH)
    assert len(loaded) == len(df)


def test_manifest_columns_present():
    df = load_validated_manifest(MANIFEST_PATH)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Missing required manifest column: {col}"


def test_manifest_allowed_labels():
    df = load_validated_manifest(MANIFEST_PATH)
    for lbl in df["label"].unique():
        assert lbl in ALLOWED_LABELS, f"Invalid label in manifest: {lbl}"


def test_manifest_allowed_spoof_types():
    df = load_validated_manifest(MANIFEST_PATH)
    for st in df["spoof_type"].unique():
        assert st in ALLOWED_SPOOF_TYPES, f"Invalid spoof_type in manifest: {st}"


def test_manifest_no_private_metadata():
    df = load_validated_manifest(MANIFEST_PATH)
    forbidden_cols = ["phone_number", "person_name", "raw_transcript", "ssn", "user_id"]
    for f in forbidden_cols:
        assert f not in df.columns, f"Forbidden privacy column found: {f}"
