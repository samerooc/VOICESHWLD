"""
Security & Input Sanitization Tests for VoiceShield (Phase 9).
Verifies resilience against path traversal, DoS, malformed headers, and information leakage.
"""

import io
from fastapi.testclient import TestClient
import pytest

from api import app
from src.config import MAX_FILE_SIZE_BYTES

client = TestClient(app)


def test_security_path_traversal_rejection():
    """Verify filenames with path traversal tokens are rejected."""
    traversal_names = [
        "../../secrets.wav",
        "..\\..\\boot.ini.wav",
        "audio/../../../etc/passwd.wav",
        "sample|calc.exe.wav",
    ]
    for name in traversal_names:
        res = client.post(
            "/predict",
            files={"file": (name, io.BytesIO(b"DUMMY_DATA"), "audio/wav")},
        )
        assert res.status_code in [400, 415]
        detail = res.json().get("detail", "")
        assert "Traceback" not in detail
        assert "C:\\" not in detail


def test_security_no_internal_paths_in_errors():
    """Verify internal filesystem paths are never leaked in error messages."""
    res = client.post(
        "/predict",
        files={"file": ("corrupt.wav", io.BytesIO(b"MALFORMED_HEADER"), "audio/wav")},
    )
    assert res.status_code == 400
    detail = res.json().get("detail", "")
    assert "/Users/" not in detail
    assert "C:\\Users\\" not in detail


def test_security_oversized_payload_protection():
    """Verify oversized payload is blocked before memory exhaustion."""
    oversized = b"\x00" * (MAX_FILE_SIZE_BYTES + 4096)
    res = client.post(
        "/predict",
        files={"file": ("huge.wav", io.BytesIO(oversized), "audio/wav")},
    )
    assert res.status_code == 413
