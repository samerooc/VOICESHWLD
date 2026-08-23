"""
VoiceShield Pretrained Training CLI Entrypoint (Phase 6 & 8).
Executes pretrained training from YAML configuration.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_pretrained import train_pretrained_pipeline


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Pretrained Model Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to pretrained model configuration YAML",
    )
    args = parser.parse_args()

    print("=======================================================")
    print("      VOICESHIELD PRETRAINED MODEL TRAINING")
    print(f"      Configuration: {args.config}")
    print("=======================================================\n")

    train_pretrained_pipeline(config_path=args.config)


if __name__ == "__main__":
    main()
