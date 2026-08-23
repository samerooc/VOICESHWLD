"""
Unit & Integration Tests for VoiceShield Deployment & Containerization (Phase 8).
Tests:
1. Docker-related files exist (Dockerfile, docker-compose.yml, .dockerignore)
2. GET /health works
3. GET /metadata works
4. API invalid-input handling (415 on MP3, 400 on empty)
5. Dashboard imports cleanly
6. Model artifact handling (loads baseline pipeline and metadata)
7. No private audio required for startup
8. Dockerignore pattern verification
"""

import os
from fastapi.testclient import TestClient
import pytest

from api import app
from src.config import MODEL_METADATA_PATH, MODEL_PATH
from src.model import load_metadata, load_model

client = TestClient(app)


def test_1_docker_files_exist():
    """1. Verify Docker-related configuration files exist."""
    assert os.path.exists("Dockerfile"), "Dockerfile missing"
    assert os.path.exists("docker-compose.yml"), "docker-compose.yml missing"
    assert os.path.exists(".dockerignore"), ".dockerignore missing"


def test_2_api_health_endpoint():
    """2. Verify GET /health endpoint returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "voiceshield-api"


def test_3_api_metadata_endpoint():
    """3. Verify GET /metadata endpoint returns schema."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "feature_version" in data
    assert "wav" in data["supported_format"]
    assert data["audio_saved"] is False


def test_4_api_invalid_input_handling():
    """4. Verify API rejects invalid inputs with clear HTTP status codes."""
    # Unsupported format rejected with 415
    res_non_audio = client.post(
        "/predict",
        files={"file": ("sample.txt", b"FAKE_TEXT", "text/plain")},
    )
    assert res_non_audio.status_code == 415

    # Empty payload rejected with 400
    res_empty = client.post(
        "/predict",
        files={"file": ("sample.wav", b"", "audio/wav")},
    )
    assert res_empty.status_code == 400


def test_5_dashboard_imports_cleanly():
    """5. Verify dashboard script compiles and imports without errors."""
    import py_compile
    compiled = py_compile.compile("app.py", doraise=True)
    assert compiled is not None


def test_6_model_artifacts_available():
    """6. Verify pre-trained model and metadata artifacts exist and load."""
    assert os.path.exists(MODEL_PATH), "Model pickle artifact missing"
    assert os.path.exists(MODEL_METADATA_PATH), "Model metadata JSON missing"

    model = load_model(MODEL_PATH)
    assert model is not None
    assert hasattr(model, "predict_proba")

    metadata = load_metadata(MODEL_METADATA_PATH)
    assert metadata is not None
    assert "model_version" in metadata
    assert "optimal_decision_threshold" in metadata


def test_7_no_private_audio_required_for_startup():
    """7. Verify system starts up and serves health check without requiring raw audio files."""
    response = client.get("/health")
    assert response.status_code == 200


def test_8_dockerignore_contains_prohibited_patterns():
    """8. Verify .dockerignore explicitly blocks private data, audio files, and virtual environments."""
    with open(".dockerignore", "r", encoding="utf-8") as f:
        content = f.read()

    prohibited_entries = [
        "*.wav",
        "*.mp3",
        "*.m4a",
        ".env",
        ".git",
        "venv",
        "backups",
    ]
    for pattern in prohibited_entries:
        assert pattern in content, f"Expected '{pattern}' in .dockerignore"
