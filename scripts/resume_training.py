"""
VoiceShield Resume Training CLI Script (Section 6).
Resumes checkpoint optimization from specified artifact path.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_pretrained import train_pretrained_pipeline


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Resume Training CLI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/pretrained_detector.pkl",
        help="Path to checkpoint artifact",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_training.yaml",
        help="Path to training configuration",
    )
    args = parser.parse_args()

    print(f"Resuming training pipeline from checkpoint: {args.checkpoint}...")
    train_pretrained_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
