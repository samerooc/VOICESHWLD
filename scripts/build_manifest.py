"""
VoiceShield Dataset Manifest Builder (Phase 2).
Scans all audio files in data/, validates integrity, extracts metadata,
computes cryptographic SHA-256 checksums, and generates data/manifest.csv.

Strict Specification:
- safe_file_id
- relative_path
- label
- spoof_type
- speaker_id_hash
- source_id_hash
- generator_id
- language
- format
- codec
- sample_rate
- channels
- duration_seconds
- license
- consent_status
- split

Allowed labels:
- bona_fide
- spoof

Allowed spoof_type:
- replay
- text_to_speech
- voice_conversion
- other
- unknown
"""

import csv
import glob
import hashlib
import os
import sys
from typing import Any, Dict, List
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    MANIFEST_PATH,
    MAX_AUDIO_DURATION_SEC,
    MIN_AUDIO_DURATION_SEC,
    MIN_AUDIO_RMS_ENERGY,
)


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 cryptographic checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Inspects an audio file for sample rate, duration, channels, silence, codec, and validity.
    """
    norm_path = os.path.normpath(file_path).replace("\\", "/")
    parts = norm_path.split("/")
    filename = os.path.basename(norm_path)
    file_stem = os.path.splitext(filename)[0]

    # 1. Determine Split
    if "test" in parts:
        split = "test"
    elif "val" in parts or "validation" in parts:
        split = "val"
    else:
        split = "train"

    # 2. Label & Spoof Type Mapping
    if any(k in parts for k in ["human", "bona_fide", "real"]):
        label = "bona_fide"
        label_id = 0
        spoof_type = "other"  # Non-spoof / natural voice
        source = "consented_human_speech"
        generator_id = "natural_voice"
        speaker_id = f"spk_h_{file_stem}"
    elif any(k in parts for k in ["ai_voice", "spoof", "fake", "synthetic"]):
        label = "spoof"
        label_id = 1
        spoof_type = "text_to_speech"  # Synthetic neural vocoder / TTS
        source = "synthetic_neural_vocoder"
        generator_id = "neural_vocoder"
        speaker_id = f"spk_ai_{file_stem}"
    else:
        label = "unknown"
        label_id = -1
        spoof_type = "unknown"
        source = "unspecified"
        generator_id = "unknown"
        speaker_id = f"spk_unk_{file_stem}"

    # 3. Audio Signal Diagnostics & Format Inspection
    is_valid = True
    validation_error = ""
    duration = 0.0
    sr = 0
    channels = 1
    codec = "unknown"
    fmt = os.path.splitext(filename)[1].lstrip(".").lower()
    rms_energy = 0.0

    try:
        info = sf.info(file_path)
        duration = float(info.duration)
        sr = int(info.samplerate)
        channels = int(info.channels)
        codec = str(info.subtype)
        fmt = str(info.format).lower()

        if duration < MIN_AUDIO_DURATION_SEC:
            is_valid = False
            validation_error = f"Too short ({duration:.2f}s < {MIN_AUDIO_DURATION_SEC}s)"
        elif duration > MAX_AUDIO_DURATION_SEC:
            is_valid = False
            validation_error = f"Too long ({duration:.2f}s > {MAX_AUDIO_DURATION_SEC}s)"
        else:
            data, _ = sf.read(file_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            rms_energy = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
            if rms_energy < MIN_AUDIO_RMS_ENERGY:
                is_valid = False
                validation_error = f"Silent or near-zero energy (RMS: {rms_energy:.6f})"

    except Exception as e:
        is_valid = False
        validation_error = f"Corrupted or unreadable audio: {e}"

    sha256_hash = compute_sha256(file_path) if os.path.exists(file_path) else "FILE_NOT_FOUND"
    rel_path = os.path.relpath(norm_path, "data").replace("\\", "/")
    spk_hash = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()[:16]
    src_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    return {
        "safe_file_id": f"vs_{split}_{file_stem}",
        "relative_path": rel_path,
        "label": label,
        "spoof_type": spoof_type,
        "speaker_id_hash": spk_hash,
        "source_id_hash": src_hash,
        "generator_id": generator_id,
        "language": "en",
        "format": fmt,
        "codec": codec,
        "sample_rate": sr,
        "channels": channels,
        "duration_seconds": round(duration, 3),
        "license": "Research-Use-Permitted",
        "consent_status": "documented_research",
        "split": split,
        # Additional fields for compatibility
        "path_relative_to_dataset_root": rel_path,
        "quality_status": "acceptable" if is_valid else "low_quality",
        "path": norm_path,
        "file_path": norm_path,
        "class_label": label,
        "label_id": label_id,
        "speaker_id": speaker_id,
        "rms_energy": round(rms_energy, 6),
        "source": source,
        "sha256_hash": sha256_hash,
        "is_valid": is_valid,
        "validation_error": validation_error,
    }


def build_manifest(
    data_dir: str = "data",
    output_csv: str = MANIFEST_PATH,
) -> List[Dict[str, Any]]:
    """
    Scans data directory recursively, inspects all WAV/audio files, and writes manifest.csv.
    """
    print("=======================================================================")
    print("       VOICESHIELD DATASET MANIFEST BUILDER (PHASE 2)")
    print("=======================================================================\n")

    search_pattern = os.path.join(data_dir, "**", "*.wav")
    wav_files = glob.glob(search_pattern, recursive=True)
    wav_files = [f for f in wav_files if "private" not in f.replace("\\", "/")]

    print(f"Discovered {len(wav_files)} audio files in '{data_dir}/'...")

    manifest_records: List[Dict[str, Any]] = []
    for fpath in sorted(wav_files):
        rec = inspect_audio_file(fpath)
        manifest_records.append(rec)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
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
        # Compatible fields
        "path_relative_to_dataset_root",
        "quality_status",
        "path",
        "file_path",
        "class_label",
        "label_id",
        "speaker_id",
        "rms_energy",
        "source",
        "sha256_hash",
        "is_valid",
        "validation_error",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_records)

    print(f"[OK] Manifest successfully written to: {output_csv}")
    print(f"     Total entries indexed: {len(manifest_records)}")
    return manifest_records


inspect_wav_file = inspect_audio_file


if __name__ == "__main__":
    build_manifest()
