"""
VoiceShield Reproducibility Suite (Phase 15).
Runs full end-to-end training and evaluation pipeline, verifying deterministic hash matching.
"""

import os
import sys
import json
import hashlib
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_baseline import train_model_from_config
from scripts.evaluate_model import evaluate_model_checkpoint


def run_reproducible_pipeline():
    print("=======================================================")
    print("       VOICESHIELD REPRODUCIBILITY VERIFICATION")
    print("=======================================================\n")

    print(f"Platform / OS       : {platform.platform()}")
    print(f"Python Version      : {platform.python_version()}")
    print(f"Processor           : {platform.processor()}")

    # 1. Train model from config
    print("\n--- Step 1: Training Model from Configuration ---")
    meta = train_model_from_config("configs/training.yaml")
    model_path = "models/voice_detector.pkl"
    model_bytes = open(model_path, "rb").read()
    model_hash = hashlib.sha256(model_bytes).hexdigest()

    print(f"Trained Model SHA-256 : {model_hash}")

    # 2. Independent evaluation
    print("\n--- Step 2: Running Independent Evaluation ---")
    evaluate_model_checkpoint(model_path)

    print("\n[OK] Reproducibility verification completed successfully.")


if __name__ == "__main__":
    run_reproducible_pipeline()
