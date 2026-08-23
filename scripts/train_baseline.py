"""
VoiceShield Baseline Model Training CLI Entrypoint (Phase 6 & 8).
Executes configuration-driven baseline training from YAML configuration.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_baseline import train_model_from_config


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Baseline Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration YAML file",
    )
    args = parser.parse_args()

    print("=======================================================")
    print("      VOICESHIELD BASELINE MODEL TRAINING")
    print(f"      Configuration: {args.config}")
    print("=======================================================\n")

    train_model_from_config(config_path=args.config)
    print("\n[OK] Baseline training process complete.")


if __name__ == "__main__":
    main()
