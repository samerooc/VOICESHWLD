"""
VoiceShield Model Training Entrypoint (Phase 3).
Executes scripts/train_model.py pipeline.
"""

import sys
from scripts.train_model import train_baseline

if __name__ == "__main__":
    try:
        train_baseline()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}", file=sys.stderr)
        sys.exit(1)