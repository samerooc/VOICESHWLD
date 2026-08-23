"""
VoiceShield Evaluation Manifest Builder.
Constructs a comprehensive, standardized evaluation manifest across multiple acoustic conditions.
"""

import os
import glob
import hashlib
import pandas as pd
import soundfile as sf


def build_evaluation_manifest():
    print("Building comprehensive evaluation manifest...")
    rows = []

    files = [
        # Train
        *[(f, "bona_fide", 0, "train", "clean_bona_fide") for f in sorted(glob.glob("data/human/*.wav"))],
        *[(f, "spoof", 1, "train", "telephony_spoof") for f in sorted(glob.glob("data/ai_voice/*.wav"))],
        # Test
        *[(f, "bona_fide", 0, "test", "clean_bona_fide") for f in sorted(glob.glob("data/test/human/*.wav"))],
        *[(f, "spoof", 1, "test", "telephony_spoof") for f in sorted(glob.glob("data/test/ai_voice/*.wav"))],
    ]

    for idx, (path, label, label_id, split, condition) in enumerate(files, start=1):
        info = sf.info(path)
        content = open(path, "rb").read()
        file_hash = hashlib.sha256(content).hexdigest()

        rows.append({
            "safe_file_id": f"eval_{split}_{idx:03d}",
            "file_path": path.replace("\\", "/"),
            "label": label,
            "label_id": label_id,
            "condition": condition,
            "split": split,
            "format": info.format,
            "channels": info.channels,
            "native_sample_rate": info.samplerate,
            "duration_sec": round(info.duration, 3),
            "sha256_hash": file_hash[:16],
            "language": "en",
            "speaker_id": f"speaker_{label_id:02d}",
            "generator_id": "elevenlabs_or_tts" if label == "spoof" else "human_recording",
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/evaluation_manifest.csv", index=False)
    print(f"[OK] data/evaluation_manifest.csv created with {len(df)} entries.")


if __name__ == "__main__":
    build_evaluation_manifest()
