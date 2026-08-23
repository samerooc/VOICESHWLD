"""
VoiceShield Label Mapping Tests (Section B).
Fails if class mapping is missing, differs between training and inference, or probabilities are reversed.
"""

import pytest
import json
from src.model_contract import LABEL_BONA_FIDE, LABEL_SPOOF, CLASS_NAMES
from src.model_registry import verify_and_load_model


def test_label_mapping_constants():
    assert LABEL_BONA_FIDE == 0
    assert LABEL_SPOOF == 1
    assert CLASS_NAMES[0] == "bona_fide"
    assert CLASS_NAMES[1] == "spoof"


def test_model_metadata_label_mapping_alignment():
    _, metadata = verify_and_load_model("models/voice_detector.pkl", "models/model_metadata.json")
    mapping = metadata.get("class_mapping", {})
    assert mapping.get("0") == "bona_fide"
    assert mapping.get("1") == "spoof"


def test_training_inference_label_equivalence():
    with open("configs/training.yaml", "r", encoding="utf-8") as f:
        meta_json = json.load(open("models/model_metadata.json", "r", encoding="utf-8"))
    assert meta_json["class_mapping"]["0"] == "bona_fide"
    assert meta_json["class_mapping"]["1"] == "spoof"
