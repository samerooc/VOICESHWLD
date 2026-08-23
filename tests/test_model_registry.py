"""
VoiceShield Model Registry Tests.
Verifies secure model loading, cryptographic hash matching, and metadata validation.
"""

import os
import json
import pytest
from src.model_registry import verify_and_load_model


def test_verify_and_load_model_success():
    model, metadata = verify_and_load_model(
        model_path="models/voice_detector.pkl",
        metadata_path="models/model_metadata.json",
    )
    assert model is not None
    assert isinstance(metadata, dict)
    assert metadata["class_mapping"]["0"] == "bona_fide"
    assert metadata["class_mapping"]["1"] == "spoof"


def test_verify_and_load_missing_metadata():
    with pytest.raises(FileNotFoundError, match="Refusing to load unverified model"):
        verify_and_load_model(
            model_path="models/voice_detector.pkl",
            metadata_path="models/non_existent_metadata.json",
        )


def test_verify_and_load_corrupted_hash(tmp_path):
    temp_meta = tmp_path / "corrupted_metadata.json"
    with open("models/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta["model_artifact_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(temp_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    with pytest.raises(ValueError, match="Model Artifact Corruption: Hash mismatch"):
        verify_and_load_model(
            model_path="models/voice_detector.pkl",
            metadata_path=str(temp_meta),
        )
