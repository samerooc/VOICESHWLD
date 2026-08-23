import json
import os
import sys
import joblib
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features import extract_features_from_audio
from src.audio_io import load_audio_from_file

print("=== 1. MANIFEST INSPECTION ===")
df_manifest = pd.read_csv("data/manifest.csv")
print(df_manifest[["path", "label", "split", "duration_seconds"]])

print("\n=== 2. MODEL METADATA INSPECTION ===")
with open("models/model_metadata.json") as f:
    meta = json.load(f)
print("Model Name:", meta.get("model_name"))
print("Class Mapping:", meta.get("class_mapping"))
print("Classes in metadata:", meta.get("classes"))
print("Decision Threshold:", meta.get("optimal_decision_threshold"))

print("\n=== 3. MODEL ARTIFACT INSPECTION ===")
model = joblib.load("models/voice_detector.pkl")
print("Pipeline Steps:", model.named_steps)
rf = model.named_steps["classifier"]
scaler = model.named_steps["scaler"]
print("RF classes_:", rf.classes_)
print("Scaler mean_ shape:", scaler.mean_.shape)

print("\n=== 4. TEST PREDICTIONS ON ALL DATASET SAMPLES ===")
for idx, row in df_manifest.iterrows():
    audio, sr = load_audio_from_file(row["path"], target_sr=16000)
    feat = extract_features_from_audio(audio, 16000)
    scaled_feat = scaler.transform([feat])
    probs = rf.predict_proba(scaled_feat)[0]
    p_0 = probs[0]
    p_1 = probs[1]
    print(f"{row['split']:<5} | True Label: {row['label']:<10} | P(class 0)={p_0:.4f} | P(class 1)={p_1:.4f} | Path: {row['path']}")
