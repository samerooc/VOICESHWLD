"""
VoiceShield Production Deployment Verification Script.
Asserts that Nginx reverse proxy routes all REST endpoints, file uploads, and WebSockets:
1. GET /health
2. GET /metadata
3. POST /predict
4. WebSocket ws://<host>/ws/live-stream
"""

import argparse
import asyncio
import io
import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import requests
import soundfile as sf
import websockets

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def generate_synthetic_wav_bytes(duration_sec: float = 2.0, sr: int = 16000) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()


def test_rest_endpoints(base_url: str) -> bool:
    print("\n" + "=" * 75)
    print("  1. TESTING REST API ENDPOINTS VIA REVERSE PROXY")
    print("=" * 75)

    all_passed = True

    # 1. Health Check
    health_url = f"{base_url}/health"
    try:
        r = requests.get(health_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f" [PASS] GET {health_url} -> Status 200 OK | Health: {data.get('status')}")
        else:
            print(f" [FAIL] GET {health_url} -> Status {r.status_code}")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] GET {health_url} -> Exception: {e}")
        all_passed = False

    # 2. Metadata Endpoint
    meta_url = f"{base_url}/metadata"
    try:
        r = requests.get(meta_url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f" [PASS] GET {meta_url} -> Status 200 OK | Model: {data.get('model_type', 'Neural')}")
        else:
            print(f" [FAIL] GET {meta_url} -> Status {r.status_code}")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] GET {meta_url} -> Exception: {e}")
        all_passed = False

    # 3. Audio Prediction Endpoint
    predict_url = f"{base_url}/predict"
    try:
        wav_bytes = generate_synthetic_wav_bytes(duration_sec=2.0)
        files = {"file": ("test_verify.wav", wav_bytes, "audio/wav")}
        r = requests.post(predict_url, files=files, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f" [PASS] POST {predict_url} -> Status 200 OK | Risk: {data.get('risk_score')}/100 ({data.get('risk_band')})")
        else:
            print(f" [FAIL] POST {predict_url} -> Status {r.status_code} ({r.text[:100]})")
            all_passed = False
    except Exception as e:
        print(f" [FAIL] POST {predict_url} -> Exception: {e}")
        all_passed = False

    return all_passed


async def test_websocket_streaming(ws_base_url: str) -> bool:
    print("\n" + "=" * 75)
    print("  2. TESTING LIVE WEBSOCKET AUDIO STREAMING VIA REVERSE PROXY")
    print("=" * 75)

    ws_url = f"{ws_base_url}/ws/live-stream"
    print(f" • Connecting to WebSocket: {ws_url}...")

    try:
        async with websockets.connect(ws_url, ping_timeout=10) as ws:
            # 1. Receive handshake frame
            greeting = await asyncio.wait_for(ws.recv(), timeout=5.0)
            greeting_data = json.loads(greeting) if isinstance(greeting, str) else {}
            print(f" [PASS] WebSocket Connected -> Initial handshake: {greeting_data.get('status', 'connected')}")

            # 2. Stream 40ms binary linear PCM audio frames
            sample_rate = 16000
            chunk_samples = int(sample_rate * 0.04)  # 640 samples
            t = np.linspace(0, 0.04, chunk_samples, endpoint=False)
            pcm_chunk = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32).tobytes()

            print(" • Streaming 25 continuous audio frames (1.0 second)...")
            for _ in range(25):
                await ws.send(pcm_chunk)
                await asyncio.sleep(0.04)

            # 3. Receive streaming risk assessment frame
            assessment_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            assessment = json.loads(assessment_raw)
            print(f" [PASS] Streaming Assessment Received -> Risk: {assessment.get('smoothed_risk_score', assessment.get('risk_score', 0))}/100 | Band: {assessment.get('risk_band')}")

            return True
    except Exception as e:
        print(f" [FAIL] WebSocket Streaming Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Production Deployment Verifier")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="Base HTTP URL for API / Proxy")
    parser.add_argument("--ws-url", type=str, default="ws://localhost:8000", help="Base WebSocket URL for API / Proxy")

    args = parser.parse_args()

    rest_ok = test_rest_endpoints(args.base_url)
    ws_ok = asyncio.run(test_websocket_streaming(args.ws_url))

    print("\n" + "=" * 75)
    if rest_ok and ws_ok:
        print("  ✅ ALL PRODUCTION DEPLOYMENT HEALTH CHECKS PASSED")
    else:
        print("  ⚠️ SOME DEPLOYMENT HEALTH CHECKS FAILED OR SERVICES OFFLINE")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
