"""
VoiceShield Main Training CLI Entrypoint (Phase 7).
Executes reproducible training from YAML configuration.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train_baseline import train_model_from_config


def main():
    parser = argparse.ArgumentParser(description="VoiceShield ML Training CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training.yaml",
        help="Path to training configuration YAML file",
    )
    args = parser.parse_args()

    print("=======================================================")
    print("      VOICESHIELD REPRODUCIBLE TRAINING PIPELINE")
    print(f"      Configuration: {args.config}")
    print("=======================================================\n")

    train_model_from_config(config_path=args.config)
    print("\n[OK] Training process complete.")


if __name__ == "__main__":
    main()
