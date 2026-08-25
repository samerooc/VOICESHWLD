"""
VoiceShield Dataset Quality & Reproducibility Auditor (Phase 2).
Audits data/manifest.csv, validates signal properties, detects duplicates and data leakage,
generates reports/dataset_report.json, and prints a comprehensive terminal report.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.config import (
    MANIFEST_PATH,
    MIN_AUDIO_DURATION_SEC,
    REPORTS_DIR,
    RESEARCH_NOTICE,
    STATUTORY_DISCLAIMER,
)

DATASET_REPORT_PATH = os.path.join(REPORTS_DIR, "dataset_report.json")


def run_dataset_audit(manifest_path: str = MANIFEST_PATH) -> Dict[str, Any]:
    """
    Performs comprehensive dataset audit on the manifest records.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found: '{manifest_path}'. "
            f"Please run `python scripts/build_manifest.py` first."
        )

    records: List[Dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    total_files = len(records)
    if total_files == 0:
        raise ValueError(f"Manifest '{manifest_path}' contains zero records.")

    # 1. Validity, Missing, Corrupt, Silent & Short Detection
    valid_records = []
    invalid_records = []
    missing_files = []
    silent_files = []
    too_short_files = []

    for r in records:
        fpath = r.get("path") or r.get("file_path", "")
        is_valid_bool = r.get("is_valid", "True").lower() == "true"

        if not os.path.exists(fpath):
            missing_files.append(fpath)
            invalid_records.append(r)
            continue

        if not is_valid_bool:
            err = r.get("validation_error", "")
            if "silent" in err.lower():
                silent_files.append(fpath)
            elif "short" in err.lower():
                too_short_files.append(fpath)
            invalid_records.append(r)
        else:
            valid_records.append(r)

    # 2. Label & Split Validation
    valid_labels = {"bona_fide", "spoof", "0", "1", 0, 1}
    valid_splits = {"train", "val", "validation", "test"}
    invalid_labels = []
    invalid_splits = []
    warnings_list: List[str] = []

    for r in records:
        fpath = r.get("path") or r.get("file_path", "")
        raw_lbl = r.get("label") if r.get("label") is not None else r.get("class_label", "unknown")
        sp = r.get("split", "unknown")
        if str(raw_lbl).lower() in ["0", "bona_fide", "real", "human"]:
            lbl = "bona_fide"
        elif str(raw_lbl).lower() in ["1", "spoof", "fake", "synthetic", "ai"]:
            lbl = "spoof"
        else:
            lbl = str(raw_lbl)
            invalid_labels.append({"path": fpath, "label": raw_lbl})

        if sp not in valid_splits:
            invalid_splits.append({"path": fpath, "split": sp})

    # 3. Class Distribution (bona_fide vs spoof)
    class_counts: Dict[str, int] = {"bona_fide": 0, "spoof": 0}
    for r in records:
        raw_lbl = r.get("label") if r.get("label") is not None else r.get("class_label", "unknown")
        if str(raw_lbl).lower() in ["0", "bona_fide", "real", "human"]:
            lbl = "bona_fide"
        elif str(raw_lbl).lower() in ["1", "spoof", "fake", "synthetic", "ai"]:
            lbl = "spoof"
        else:
            lbl = str(raw_lbl)
        class_counts[lbl] = class_counts.get(lbl, 0) + 1

    # Class imbalance check
    bona_count = class_counts.get("bona_fide", 0)
    spoof_count = class_counts.get("spoof", 0)
    if total_files > 0 and (bona_count == 0 or spoof_count == 0):
        warnings_list.append("Severe class imbalance: one class has 0 samples.")
    elif min(bona_count, spoof_count) / max(bona_count, spoof_count) < 0.33:
        warnings_list.append(f"Moderate class imbalance detected: bona_fide={bona_count}, spoof={spoof_count}")

    # 4. Split Breakdown
    split_counts: Dict[str, Dict[str, int]] = {}
    for r in records:
        sp = r.get("split", "unknown")
        raw_lbl = r.get("label") if r.get("label") is not None else r.get("class_label", "unknown")
        lbl = "bona_fide" if str(raw_lbl).lower() in ["0", "bona_fide", "real", "human"] else "spoof"
        if sp not in split_counts:
            split_counts[sp] = {}
        split_counts[sp][lbl] = split_counts[sp].get(lbl, 0) + 1

    # 5. Duration Statistics
    durations = [
        float(r.get("duration_seconds") or r.get("duration_sec") or r.get("duration") or 0.0)
        for r in valid_records
        if float(r.get("duration_seconds") or r.get("duration_sec") or r.get("duration") or 0.0) > 0
    ]
    dur_total = float(sum(durations)) if durations else 0.0
    dur_avg = float(np.mean(durations)) if durations else 0.0
    dur_min = float(np.min(durations)) if durations else 0.0
    dur_max = float(np.max(durations)) if durations else 0.0
    dur_std = float(np.std(durations)) if durations else 0.0

    # 6. Sample Rate & Codec Statistics
    sr_counts: Dict[str, int] = {}
    codec_counts: Dict[str, int] = {}
    for r in valid_records:
        sr_key = f"{r.get('sample_rate', '0')} Hz"
        codec_key = r.get("codec", "unknown")
        sr_counts[sr_key] = sr_counts.get(sr_key, 0) + 1
        codec_counts[codec_key] = codec_counts.get(codec_key, 0) + 1

    # 7. Duplicate SHA-256 Hash Detection
    hash_to_files: Dict[str, List[str]] = {}
    for r in records:
        h = r.get("sha256_hash", "")
        fpath = r.get("path") or r.get("file_path", "")
        if h and h != "FILE_NOT_FOUND":
            hash_to_files.setdefault(h, []).append(fpath)

    duplicate_groups = {h: flist for h, flist in hash_to_files.items() if len(flist) > 1}
    if duplicate_groups:
        warnings_list.append(f"Found {len(duplicate_groups)} duplicate SHA-256 hash group(s).")

    # 8. Data Leakage Check (Train vs Test hash & path overlap)
    train_hashes = {r["sha256_hash"] for r in records if r.get("split") == "train" and r.get("sha256_hash")}
    test_hashes = {r["sha256_hash"] for r in records if r.get("split") == "test" and r.get("sha256_hash")}
    leaked_hashes = train_hashes.intersection(test_hashes)

    train_paths = {r.get("path") or r.get("file_path", "") for r in records if r.get("split") == "train"}
    test_paths = {r.get("path") or r.get("file_path", "") for r in records if r.get("split") == "test"}
    leaked_paths = train_paths.intersection(test_paths)

    # 9. Speaker Overlap Check
    train_speakers = {r.get("speaker_id", "") for r in records if r.get("split") == "train" and r.get("speaker_id")}
    test_speakers = {r.get("speaker_id", "") for r in records if r.get("split") == "test" and r.get("speaker_id")}
    overlapping_speakers = train_speakers.intersection(test_speakers)

    is_leakage_free = (len(leaked_hashes) == 0) and (len(leaked_paths) == 0)
    audit_passed = (
        len(invalid_records) == 0
        and len(missing_files) == 0
        and len(invalid_labels) == 0
        and len(invalid_splits) == 0
        and is_leakage_free
    )

    report_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path,
        "audit_passed": audit_passed,
        "total_files": total_files,
        "valid_files": len(valid_records),
        "invalid_files": len(invalid_records),
        "missing_files": missing_files,
        "silent_files": silent_files,
        "too_short_files": too_short_files,
        "invalid_labels": invalid_labels,
        "invalid_splits": invalid_splits,
        "warnings": warnings_list,
        "class_distribution": class_counts,
        "split_distribution": split_counts,
        "duration_statistics": {
            "total_seconds": round(dur_total, 2),
            "total_minutes": round(dur_total / 60.0, 2),
            "average_seconds": round(dur_avg, 2),
            "min_seconds": round(dur_min, 2),
            "max_seconds": round(dur_max, 2),
            "std_dev_seconds": round(dur_std, 2),
        },
        "sample_rate_distribution": sr_counts,
        "codec_distribution": codec_counts,
        "duplicate_hash_groups": duplicate_groups,
        "data_leakage_detected": not is_leakage_free,
        "leaked_hash_count": len(leaked_hashes),
        "speaker_overlap_detected": len(overlapping_speakers) > 0,
        "overlapping_speakers": list(overlapping_speakers),
        "statutory_disclaimer": STATUTORY_DISCLAIMER,
        "research_notice": RESEARCH_NOTICE,
    }

    # Save to reports/dataset_report.json
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(DATASET_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Print Terminal Summary
    print("=======================================================================")
    print("       VOICESHIELD DATASET QUALITY & REPRODUCIBILITY AUDIT")
    print("=======================================================================\n")
    print(f"  • Total Audio Files       : {total_files}")
    print(f"  • Valid Audio Files       : {len(valid_records)}")
    print(f"  • Invalid / Broken Files  : {len(invalid_records)}")
    print(f"  • Missing Files on Disk   : {len(missing_files)}")
    print(f"  • Invalid Labels          : {len(invalid_labels)}")
    print(f"  • Invalid Splits          : {len(invalid_splits)}")
    print(f"  • Total Duration          : {dur_total:.2f} seconds ({dur_total/60:.2f} mins)")
    print(f"  • Duration Range (Min-Max): {dur_min:.2f}s - {dur_max:.2f}s (Avg: {dur_avg:.2f}s)")

    if warnings_list:
        print("\n--- Warnings ---")
        for w in warnings_list:
            print(f"  [!] {w}")

    print("\n--- Class Balance ---")
    for cls, count in sorted(class_counts.items()):
        pct = (count / total_files) * 100
        print(f"  • {cls:<12}: {count:02d} files ({pct:5.1f}%)")

    print("\n--- Split Partitioning ---")
    for sp, sdict in sorted(split_counts.items()):
        for cls, count in sorted(sdict.items()):
            print(f"  • Split: {sp:<6} | Class: {cls:<10} -> {count:02d} files")

    print("\n--- Sample Rate Breakdown ---")
    for sr_val, count in sorted(sr_counts.items()):
        print(f"  • {sr_val:<10}: {count:02d} files")

    print("\n--- Integrity & Data Leakage ---")
    if duplicate_groups:
        print(f"  [!] Found {len(duplicate_groups)} duplicate hash group(s):")
        for h, flist in duplicate_groups.items():
            print(f"      Hash {h[:16]}... -> {flist}")
    else:
        print("  [OK] No duplicate files detected.")

    if is_leakage_free:
        print("  [OK] Zero Data Contamination: Train and Test splits are strictly independent.")
    else:
        print(f"  [!] DATA LEAKAGE DETECTED: {len(leaked_hashes)} common hashes between Train and Test.")

    if not overlapping_speakers:
        print("  [OK] Speaker Independence: No speaker IDs shared across Train and Test splits.")
    else:
        print(f"  [!] Speaker Overlap: {list(overlapping_speakers)}")

    print(f"\n[OK] Dataset Report written to: {DATASET_REPORT_PATH}")
    print("\n=======================================================================")
    print(f"                 AUDIT RESULT: {'PASSED' if audit_passed else 'ATTENTION NEEDED'}")
    print("=======================================================================\n")

    return report_data


if __name__ == "__main__":
    try:
        run_dataset_audit()
    except Exception as e:
        print(f"[ERROR] Dataset audit failed: {e}", file=sys.stderr)
        sys.exit(1)
