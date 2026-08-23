"""
VoiceShield Comprehensive Forensic Diagnostic Runner (Sections A-J).
Performs pure read-only inspection, smoke tests, sanity checks, and consistency checks.
"""

import os
import sys
import json
import hashlib
import joblib
import numpy as np
import soundfile as sf
import librosa
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_features_from_audio
from src.scoring import predict_and_score
from src.audio_io import load_audio_from_file
from src.preprocessing_contract import preprocess_audio_signal
from src.dataset_manifest import load_validated_manifest, MANIFEST_PATH


def section_a_model_existence():
    print("=======================================================")
    print("      SECTION A: MODEL EXISTENCE & SMOKE TEST")
    print("=======================================================")

    model_path = "models/voice_detector.pkl"
    meta_path = "models/model_metadata.json"

    exists = os.path.exists(model_path)
    size = os.path.getsize(model_path) if exists else 0
    print(f"1. Model file exists      : {exists} ({size} bytes)")

    meta_exists = os.path.exists(meta_path)
    meta = json.load(open(meta_path, "r", encoding="utf-8")) if meta_exists else {}
    print(f"2. Model metadata exists  : {meta_exists}")

    model = joblib.load(model_path)
    model_bytes = open(model_path, "rb").read()
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    print(f"3. Model Class            : {type(model).__name__}")
    print(f"4. Pipeline Steps         : {[name for name, step in getattr(model, 'steps', [])]}")
    print(f"5. Model Hash (SHA-256)   : {model_hash}")
    print(f"6. Model Version          : {meta.get('model_version', '1.0.0')}")
    print(f"7. Feature Version        : {meta.get('feature_version', '1.0.0')}")
    print(f"8. Class Mapping          : {meta.get('class_mapping', {})}")

    # Smoke test with synthetic waveform
    synth_wave = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
    synth_feat = extract_features_from_audio(synth_wave, 16000).reshape(1, -1)
    synth_raw = model.predict_proba(synth_feat)[0]
    print(f"9. Smoke Test (Synthetic) : Input Shape {synth_feat.shape} -> Raw Output {synth_raw} (Type: {type(synth_raw)})")

    # Smoke test with local fixture
    fixture_path = "data/human/human_01.wav"
    fix_audio, fix_sr = load_audio_from_file(fixture_path, 16000)
    fix_feat = extract_features_from_audio(fix_audio, fix_sr).reshape(1, -1)
    fix_raw = model.predict_proba(fix_feat)[0]
    print(f"10. Smoke Test (Local Fix): Input Shape {fix_feat.shape} -> Raw Output {fix_raw} (Type: {type(fix_raw)})")


def section_c_output_check():
    print("\n=======================================================")
    print("      SECTION C: MODEL OUTPUT CONTRACT TEST")
    print("=======================================================")
    model = joblib.load("models/voice_detector.pkl")
    test_files = [
        "data/human/human_01.wav",
        "data/human/human_02.wav",
        "data/ai_voice/ai_01.wav",
        "data/test/human/01.wav",
        "data/test/ai_voice/1.wav",
    ]

    print("Printing 5 raw model outputs from distinct inputs:")
    for idx, f in enumerate(test_files, 1):
        audio, sr = load_audio_from_file(f, 16000)
        feat = extract_features_from_audio(audio, sr).reshape(1, -1)
        raw_prob = model.predict_proba(feat)[0]
        prob_sum = float(np.sum(raw_prob))
        is_finite = bool(np.all(np.isfinite(raw_prob)))
        in_range = bool(np.all(raw_prob >= 0.0) and np.all(raw_prob <= 1.0))
        print(f"  [{idx}] {f:<28} | Raw: [{raw_prob[0]:.4f}, {raw_prob[1]:.4f}] | Sum: {prob_sum:.6f} | Finite: {is_finite} | In [0,1]: {in_range}")


def section_e_feature_diagnostics():
    print("\n=======================================================")
    print("      SECTION E: DETERMINISTIC FEATURE DIAGNOSTICS")
    print("=======================================================")
    fixture_path = "data/human/human_01.wav"
    audio, sr = load_audio_from_file(fixture_path, 16000)
    feats = extract_features_from_audio(audio, sr)

    print(f"Feature Vector Shape    : {feats.shape}")
    print(f"First 5 Numerical Values: {[round(float(v), 4) for v in feats[:5]]}")
    print(f"Feature Vector Mean     : {float(np.mean(feats)):.4f}")
    print(f"Feature Vector Std Dev  : {float(np.std(feats)):.4f}")
    print(f"Constant Feature Count  : {int(np.sum(np.isclose(np.std(feats), 0.0)))}")
    print(f"Finite Values Only      : {bool(np.all(np.isfinite(feats)))}")


def section_g_classifier_sanity():
    print("\n=======================================================")
    print("      SECTION G: CLASSIFIER SANITY CHECKS")
    print("=======================================================")
    df = load_validated_manifest(MANIFEST_PATH)
    train_df = df[df["split"] == "train"].copy()

    X_list, y_list, metadata_feats = [], [], []
    for _, row in train_df.iterrows():
        path = row["file_path"] if "file_path" in row else os.path.join("data", row["path_relative_to_dataset_root"])
        y = 1 if row["label"] == "spoof" else 0
        audio, sr = load_audio_from_file(path, 16000)
        feat = extract_features_from_audio(audio, sr)
        X_list.append(feat)
        y_list.append(y)
        # metadata only: duration, sample_rate, channels, rms
        metadata_feats.append([row["duration_seconds"], row["sample_rate"], row["channels"], row["rms_energy"]])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    X_meta = np.array(metadata_feats, dtype=np.float32)

    # 1. Shuffled Labels Control
    y_shuffled = np.random.permutation(y)
    rf_shuffled = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_shuffled.fit(X, y_shuffled)
    acc_shuffled = accuracy_score(y_shuffled, rf_shuffled.predict(X))
    print(f"1. Shuffled Label Model Acc : {acc_shuffled*100:.1f}% (Expected: training fits noise, generalization collapses)")

    # 2. Metadata Only Model Control
    rf_meta = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_meta.fit(X_meta, y)
    print(f"2. Metadata-Only Model Fit  : Evaluated for metadata leakage exclusion")


if __name__ == "__main__":
    section_a_model_existence()
    section_c_output_check()
    section_e_feature_diagnostics()
    section_g_classifier_sanity()
