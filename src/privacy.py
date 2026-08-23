"""
VoiceShield Privacy & Data Hygiene Module (Phase 1).
Enforces zero raw audio retention, ephemeral buffer isolation, and cryptographic auditing.
"""

import hashlib
import os
from typing import Optional

from src.config import STATUTORY_DISCLAIMER


def compute_sha256(data: bytes) -> str:
    """Computes SHA-256 hash of raw byte buffers for audit trails without saving audio."""
    return hashlib.sha256(data).hexdigest()


def safe_delete_file(file_path: Optional[str]) -> bool:
    """Safely removes an ephemeral temporary file with exception suppression."""
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            return True
        except OSError:
            return False
    return False


def get_privacy_statement() -> str:
    """Returns the statutory privacy assurance."""
    return (
        f"{STATUTORY_DISCLAIMER} All audio is processed locally in temporary memory buffers "
        "and immediately purged. No raw audio, voiceprints, or customer credentials are ever "
        "written to permanent storage or external networks."
    )
