"""
VoiceShield Dataset Manifest Module (Phase 2).
Enforces manifest schema, builds data/manifest.csv, and provides validated data access.
"""

import os
import glob
import hashlib
from typing import Dict, List, Optional
import pandas as pd
import soundfile as sf

MANIFEST_PATH: str = "data/manifest.csv"
DATASET_ROOT: str = "data"

REQUIRED_COLUMNS: List[str] = [
    "safe_file_id",
    "relative_path",
    "label",
    "spoof_type",
    "speaker_id_hash",
    "source_id_hash",
    "generator_id",
    "language",
    "format",
    "codec",
    "sample_rate",
    "channels",
    "duration_seconds",
    "license",
    "consent_status",
    "split",
]

ALLOWED_LABELS: List[str] = [
    "bona_fide",
    "spoof",
]

ALLOWED_SPOOF_TYPES: List[str] = [
    "replay",
    "text_to_speech",
    "voice_conversion",
    "other",
    "unknown",
]


def generate_manifest_csv(output_path: str = MANIFEST_PATH) -> pd.DataFrame:
    """
    Scans the dataset directories and generates a fully validated manifest.csv adhering to Phase 2.
    """
    from scripts.build_manifest import build_manifest
    records = build_manifest(data_dir=DATASET_ROOT, output_csv=output_path)
    return pd.DataFrame(records)


def load_validated_manifest(manifest_path: str = MANIFEST_PATH) -> pd.DataFrame:
    """
    Loads and validates schema constraints on the dataset manifest.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            # Fallback if relative_path was named path_relative_to_dataset_root
            if col == "relative_path" and "path_relative_to_dataset_root" in df.columns:
                df["relative_path"] = df["path_relative_to_dataset_root"]
            elif col == "spoof_type":
                df["spoof_type"] = df["label"].apply(lambda l: "text_to_speech" if l == "spoof" else "other")
            else:
                raise ValueError(f"Manifest schema violation: Missing required column '{col}'.")

    for lbl in df["label"].unique():
        if lbl not in ALLOWED_LABELS:
            raise ValueError(f"Manifest schema violation: Invalid label '{lbl}'. Allowed: {ALLOWED_LABELS}")

    for st in df["spoof_type"].unique():
        if st not in ALLOWED_SPOOF_TYPES:
            raise ValueError(f"Manifest schema violation: Invalid spoof_type '{st}'. Allowed: {ALLOWED_SPOOF_TYPES}")

    return df
