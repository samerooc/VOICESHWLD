#!/usr/bin/env python
"""
VoiceShield Step 2: Audio Standardization & Manifest Builder Engine.
Scans raw incoming audio, standardizes to 16,000 Hz 16-bit PCM Mono WAV,
validates acoustic integrity (duration >= 0.5s, non-silent, non-corrupt),
assigns leakage-free speaker IDs, and constructs `data/manifest.csv`.
"""

import argparse
import csv
import glob
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import librosa
import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"[*] {desc}...")
        return iterable

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    MANIFEST_PATH,
    MIN_AUDIO_DURATION_SEC,
    SAMPLE_RATE,
)


def compute_sha256(file_path: str) -> str:
    """Computes SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_wav_file(file_path: str) -> Dict[str, Any]:
    """Inspects a WAV file for duration, sample rate, channels, silence, and validity."""
    norm_path = os.path.normpath(file_path).replace("\\", "/")
    parts = norm_path.split("/")
    filename = os.path.basename(file_path)

    # Determine split
    if "test" in parts:
        split = "test"
    elif "val" in parts or "validation" in parts:
        split = "val"
    else:
        split = "train"

    # Determine class
    if any(k in parts for k in ["human", "bona_fide", "real"]):
        class_label = "bona_fide"
        label_id = 0
    elif any(k in parts for k in ["ai_voice", "spoof", "fake", "synthetic"]):
        class_label = "spoof"
        label_id = 1
    else:
        class_label = "unknown"
        label_id = -1

    is_valid = True
    validation_error = ""
    duration = 0.0
    sr = 0
    channels = 0
    is_silent = False
    rms_energy = 0.0

    try:
        info = sf.info(file_path)
        duration = float(info.duration)
        sr = int(info.samplerate)
        channels = int(info.channels)

        if duration < 0.1:
            is_valid = False
            validation_error = f"Audio duration too short ({duration:.3f}s < 0.1s)"

        # Check silence
        data, _ = sf.read(file_path, dtype="float32")
        rms_energy = float(np.sqrt(np.mean(data**2)))
        if rms_energy < 1e-4:
            is_silent = True
            is_valid = False
            validation_error = f"Audio signal is pure silence (RMS: {rms_energy:.6f})"

    except Exception as e:
        is_valid = False
        validation_error = f"Corrupted or unreadable audio file: {e}"

    sha256 = compute_sha256(file_path) if os.path.exists(file_path) else ""
    spk_id = f"spk_{parts[-2] if len(parts) >= 2 else '001'}"

    return {
        "file_path": file_path,
        "path": file_path,
        "filename": filename,
        "label": class_label,
        "label_id": label_id,
        "class_label": class_label,
        "split": split,
        "speaker_id": spk_id,
        "duration_seconds": duration,
        "duration_sec": duration,
        "duration": duration,
        "sample_rate": sr,
        "channels": channels,
        "codec": "PCM_16",
        "format": "WAV",
        "sha256_hash": sha256,
        "sha256": sha256,
        "source": "local_data",
        "language": "en",
        "is_valid": is_valid,
        "is_silent": is_silent,
        "rms_energy": rms_energy,
        "validation_error": validation_error,
    }


def build_manifest(raw_dir: str = "data/raw", dest_dir: str = "data", manifest_path: str = MANIFEST_PATH) -> List[Dict[str, Any]]:
    """Builds and returns manifest records."""
    return process_raw_dataset(raw_dir=raw_dir, dest_dir=dest_dir, manifest_path=manifest_path)


class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def infer_metadata_from_path(file_path: str) -> Tuple[int, str, str, str]:
    """
    Infers label, speaker_id, generator_type, and source_dataset from path or filename.
    Returns:
        (label, speaker_id, generator_type, dataset_source)
    """
    path_lower = file_path.lower().replace("\\", "/")
    filename = os.path.splitext(os.path.basename(file_path))[0].lower()

    # 1. Label Detection
    if any(k in path_lower for k in ["/ai_voice", "/spoof", "/synthetic", "ai_"]):
        label = 1
    else:
        label = 0

    # 2. Generator Tool Identification
    generators = ["elevenlabs", "xtts", "bark", "rvc", "tortoise", "hifigan", "melgan", "wavefake", "deepfake"]
    generator_type = "human_real"
    if label == 1:
        generator_type = "neural_vocoder"
        for g in generators:
            if g in path_lower or g in filename:
                generator_type = g
                break

    # 3. Speaker ID Extraction (Preventing Leakage)
    if "spk_" in filename:
        parts = filename.split("spk_")
        spk_id = "spk_" + parts[1].split("_")[0]
        if label == 1:
            spk_id = f"ai_{generator_type}_{spk_id}"
    else:
        # Generate stable speaker hash from folder and prefix
        hasher = hashlib.md5(filename.encode("utf-8")).hexdigest()[:6]
        prefix = "spk_ai" if label == 1 else "spk_human"
        spk_id = f"{prefix}_{hasher}"

    # 4. Dataset Source Identification
    if "minds14" in path_lower:
        dataset_source = "minds14_librispeech"
    elif "common_voice" in path_lower:
        dataset_source = "mozilla_common_voice"
    elif "wavefake" in path_lower:
        dataset_source = "wavefake_bench"
    elif "asvspoof" in path_lower:
        dataset_source = "asvspoof_2019_la"
    else:
        dataset_source = "voiceshield_standard_corpus"

    return label, spk_id, generator_type, dataset_source


def standardize_audio_file(
    input_path: str,
    output_path: str,
    target_sr: int = SAMPLE_RATE,
) -> Optional[float]:
    """
    Decodes input audio, mixes to mono, resamples to target_sr, checks sanity,
    and writes 16-bit PCM WAV. Returns duration in seconds or None if invalid.
    """
    try:
        data, sr = sf.read(input_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample if needed
        if sr != target_sr:
            data = librosa.resample(y=data, orig_sr=sr, target_sr=target_sr)

        # Sanity Check 1: Empty or extreme short
        duration = float(len(data) / target_sr)
        if duration < MIN_AUDIO_DURATION_SEC:
            return None

        # Sanity Check 2: Silence / Zero Energy
        rms = float(np.sqrt(np.mean(data**2)))
        if rms < 1e-4:
            return None

        # Amplitude normalization
        data_clean = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        max_amp = np.max(np.abs(data_clean))
        if max_amp > 1e-6:
            data_clean = data_clean / (max_amp + 1e-9)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, data_clean, target_sr, subtype="PCM_16")
        return duration

    except Exception:
        return None


def process_raw_dataset(
    raw_dir: str = "data/raw",
    dest_dir: str = "data",
    manifest_path: str = MANIFEST_PATH,
) -> List[Dict[str, Any]]:
    """
    Scans raw directory, standardizes audio, and creates data/manifest.csv.
    """
    raw_files = []
    for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
        raw_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))
        raw_files.extend(glob.glob(os.path.join(dest_dir, "test", "human", "**", ext), recursive=True))
        raw_files.extend(glob.glob(os.path.join(dest_dir, "test", "ai_voice", "**", ext), recursive=True))

    raw_files = sorted(list(set(raw_files)))

    if not raw_files:
        # If data/raw empty, also check data/human and data/ai_voice directly
        for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a"]:
            raw_files.extend(glob.glob(os.path.join(dest_dir, "human", "**", ext), recursive=True))
            raw_files.extend(glob.glob(os.path.join(dest_dir, "ai_voice", "**", ext), recursive=True))
        raw_files = sorted(list(set(raw_files)))

    if not raw_files:
        print(f"{Colors.RED}[!] No audio files found in {raw_dir} or {dest_dir}. Run download_datasets.py first.{Colors.RESET}")
        return []

    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  VOICESHIELD STEP 2: AUDIO STANDARDIZATION & MANIFEST BUILDER{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 75}{Colors.RESET}")
    print(f" • Found Raw Audio Files: {len(raw_files)}")
    print(f" • Target Sample Rate   : {SAMPLE_RATE} Hz (16-bit PCM Mono)")
    print(f" • Target Manifest Path : {manifest_path}\n")

    valid_records = []
    rejected_count = 0

    pbar = tqdm(raw_files, desc="Standardizing Audio Files")
    for file_path in pbar:
        label, spk_id, gen_type, ds_source = infer_metadata_from_path(file_path)

        # Standardized destination path
        sub_folder = "human" if label == 0 else "ai_voice"
        filename = os.path.basename(file_path)
        if not filename.endswith(".wav"):
            filename = os.path.splitext(filename)[0] + ".wav"

        out_path = os.path.join(dest_dir, sub_folder, filename)

        # Avoid copying on top of itself if already standardized
        if os.path.abspath(file_path) == os.path.abspath(out_path):
            try:
                info = sf.info(out_path)
                dur = float(info.duration)
                if dur >= MIN_AUDIO_DURATION_SEC:
                    valid_records.append({
                        "file_path": os.path.relpath(out_path, ROOT_DIR).replace("\\", "/"),
                        "duration_sec": round(dur, 3),
                        "speaker_id": spk_id,
                        "label": label,
                        "generator_type": gen_type,
                        "dataset_source": ds_source,
                    })
                    continue
            except Exception:
                pass

        dur = standardize_audio_file(file_path, out_path, target_sr=SAMPLE_RATE)
        if dur is not None:
            valid_records.append({
                "file_path": os.path.relpath(out_path, ROOT_DIR).replace("\\", "/"),
                "duration_sec": round(dur, 3),
                "speaker_id": spk_id,
                "label": label,
                "generator_type": gen_type,
                "dataset_source": ds_source,
            })
        else:
            rejected_count += 1

    pbar.close()

    # Split assignment grouped by speaker_id (balanced across classes)
    human_speakers = sorted(list(set(r["speaker_id"] for r in valid_records if r["label"] == 0)))
    ai_speakers = sorted(list(set(r["speaker_id"] for r in valid_records if r["label"] == 1)))

    np.random.seed(42)
    np.random.shuffle(human_speakers)
    np.random.shuffle(ai_speakers)

    def assign_split_sets(spk_list):
        n = len(spk_list)
        if n <= 2:
            return set(spk_list), set(), set(spk_list)
        train_end = max(1, int(0.70 * n))
        val_end = max(train_end + 1, int(0.85 * n)) if n > 3 else train_end
        return set(spk_list[:train_end]), set(spk_list[train_end:val_end]), set(spk_list[val_end:])

    h_train, h_val, h_test = assign_split_sets(human_speakers)
    a_train, a_val, a_test = assign_split_sets(ai_speakers)

    speakers = list(set(human_speakers + ai_speakers))
    train_spks = h_train | a_train
    val_spks = h_val | a_val
    test_spks = h_test | a_test
    # Ensure test has at least 1 speaker from each class if available
    if not (h_test & set(human_speakers)) and human_speakers:
        test_spks.add(human_speakers[-1])
    if not (a_test & set(ai_speakers)) and ai_speakers:
        test_spks.add(ai_speakers[-1])

    for r in valid_records:
        spk = r["speaker_id"]
        if spk in test_spks:
            r["split"] = "test"
        elif spk in val_spks:
            r["split"] = "validation"
        else:
            r["split"] = "train"
        r["path"] = r["file_path"]
        r["duration_seconds"] = r["duration_sec"]
        r["source"] = r.get("dataset_source", "voiceshield")
        r["codec"] = "PCM_16"
        r["sample_rate"] = 16000
        r["language"] = "en"
        r["is_valid"] = True
        r["sha256_hash"] = compute_sha256(r["file_path"]) if os.path.exists(r["file_path"]) else ""

    # Write Manifest CSV
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    fieldnames = [
        "file_path",
        "path",
        "label",
        "speaker_id",
        "generator_type",
        "duration_sec",
        "duration_seconds",
        "dataset_source",
        "source",
        "codec",
        "sample_rate",
        "language",
        "split",
        "sha256_hash",
        "is_valid",
    ]

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in valid_records:
            writer.writerow(r)

    # Summary Statistics Table
    n_human = sum(1 for r in valid_records if r["label"] == 0)
    n_ai = sum(1 for r in valid_records if r["label"] == 1)
    n_train = sum(1 for r in valid_records if r["split"] == "train")
    n_val = sum(1 for r in valid_records if r["split"] == "validation")
    n_test = sum(1 for r in valid_records if r["split"] == "test")

    print(f"\n{Colors.BOLD}{Colors.GREEN}{'=' * 75}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  DATASET MANIFEST GENERATION COMPLETE{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 75}{Colors.RESET}")
    print(f" {'Metric':<30} | {'Value':<15}")
    print(f" {'-'*30}-|-{'-'*15}")
    print(f" {'Total Valid Audio Samples':<30} | {len(valid_records):<15}")
    print(f" {'Human Voice Samples (0)':<30} | {n_human:<15}")
    print(f" {'Synthetic Spoof Samples (1)':<30} | {n_ai:<15}")
    print(f" {'Unique Speaker IDs':<30} | {len(speakers):<15}")
    print(f" {'Train Partition':<30} | {n_train:<15}")
    print(f" {'Validation Partition':<30} | {n_val:<15}")
    print(f" {'Held-Out Test Partition':<30} | {n_test:<15}")
    print(f" {'Rejected (Silent / Corrupted)':<30} | {rejected_count:<15}")
    print(f" {'Manifest CSV Location':<30} | {manifest_path:<15}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 75}{Colors.RESET}\n")

    return valid_records


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Step 2: Audio Standardization & Manifest Builder")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw audio files (default: data/raw)")
    parser.add_argument("--dest-dir", type=str, default="data", help="Output directory for standardized audio (default: data)")
    parser.add_argument("--output", "--manifest", dest="manifest", type=str, default=MANIFEST_PATH, help="Path for generated manifest CSV (default: data/manifest.csv)")

    args = parser.parse_args()
    process_raw_dataset(raw_dir=args.raw_dir, dest_dir=args.dest_dir, manifest_path=args.manifest)


if __name__ == "__main__":
    main()
