"""
Unit & Static Configuration Tests for VoiceShield Production Deployment.
Tests:
1. Dockerfile multi-stage syntax and security directives.
2. Nginx configuration syntax, upstreams, and WebSocket upgrade headers.
3. Docker Compose multi-service architecture and health checks.
4. Environment template (.env.example) completeness.
"""

import os
import sys
import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def test_dockerfile_configuration():
    dockerfile_path = os.path.join(ROOT_DIR, "Dockerfile")
    assert os.path.exists(dockerfile_path), "Dockerfile must exist at repository root."

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "AS builder" in content, "Dockerfile must feature a multi-stage builder."
    assert "AS runtime" in content, "Dockerfile must feature a minimal runtime stage."
    assert "appuser" in content, "Dockerfile must configure a non-root application user for compliance."
    assert "EXPOSE 8000 8501" in content or ("8000" in content and "8501" in content)


def test_nginx_configuration():
    nginx_path = os.path.join(ROOT_DIR, "nginx", "nginx.conf")
    assert os.path.exists(nginx_path), "Nginx configuration file must exist."

    with open(nginx_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "upstream backend_api" in content
    assert "upstream frontend_ui" in content
    assert "proxy_set_header Upgrade $http_upgrade;" in content
    assert 'proxy_set_header Connection "upgrade";' in content
    assert "client_max_body_size 50M;" in content
    assert "X-Frame-Options" in content


def test_docker_compose_configuration():
    compose_path = os.path.join(ROOT_DIR, "docker-compose.yml")
    assert os.path.exists(compose_path), "docker-compose.yml must exist at repository root."

    with open(compose_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "voiceshield-redis:" in content
    assert "voiceshield-api:" in content
    assert "voiceshield-dashboard:" in content
    assert "voiceshield-proxy:" in content
    assert "redis_data:" in content
    assert "healthcheck:" in content


def test_env_example_configuration():
    env_path = os.path.join(ROOT_DIR, ".env.example")
    assert os.path.exists(env_path), ".env.example must exist at repository root."

    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "REDIS_URL" in content
    assert "DEVICE" in content
    assert "BACKEND_API_URL" in content
    assert "BACKEND_WS_URL" in content
