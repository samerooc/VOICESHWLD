"""
VoiceShield Model Evaluation Entrypoint (Phase 3).
Executes scripts/evaluate_model.py pipeline.
"""

import sys
from scripts.evaluate_model import evaluate_baseline

if __name__ == "__main__":
    try:
        evaluate_baseline()
    except Exception as e:
        print(f"[ERROR] Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)