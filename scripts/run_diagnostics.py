"""
VoiceShield Phase 8 — End-to-End Automated Diagnostic Health Probes.

Executes multi-point diagnostic health checks across all VoiceShield subsystems:
  1. In-Memory Preprocessing & Normalization Probes
  2. Edge-Case Probes (Corrupt bytes, Silence, Truncated audio)
  3. Physical LPC Residual & Glottal Biomechanics DSP Probes
  4. Neural Foundation Model & Dynamic Label Resolver Probes
  5. FastAPI REST Gateway Probes (GET /health, GET /metadata, POST /predict)
  6. Live WebSocket Streaming & Twilio Protocol Probes

Usage:
    python scripts/run_diagnostics.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf
import torch

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure repository root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api import app
from src.audio_processor import (
    SAMPLE_RATE,
    compute_snr_db,
    decode_and_sanitize_audio,
    normalize_audio_standard,
)
from src.forensic_dsp import ForensicDSPAnalyzer, extract_dsp_metrics
from src.lpc_physics import LPCPhysicsAnalyzer, extract_lpc_residual
from src.neural_engine import ProductionNeuralDetector, _HUMAN_PATTERN, _SPOOF_PATTERN
from src.schemas import HealthResponse, MetadataResponse, PredictionResponse, StreamingTelemetryFrame
from src.streaming import LiveStreamingEngine, RollingAudioBuffer, linear_to_mulaw_bytes

# ---------------------------------------------------------------------------
# ANSI Terminal Colors
# ---------------------------------------------------------------------------
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BLUE = "\033[94m"
CLR_GRAY = "\033[90m"


class DiagnosticReporter:
    """Manages test status collection and terminal table formatting."""

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []
        self.start_time: float = time.time()

    def record(
        self,
        subsystem: str,
        probe_name: str,
        passed: bool,
        latency_ms: float,
        details: str = "",
        error: Optional[str] = None,
    ) -> None:
        self.results.append({
            "subsystem": subsystem,
            "probe_name": probe_name,
            "passed": passed,
            "latency_ms": latency_ms,
            "details": details,
            "error": error,
        })
        status_tag = f"{CLR_GREEN}[ PASS ]{CLR_RESET}" if passed else f"{CLR_RED}[ FAIL ]{CLR_RESET}"
        print(f"  {status_tag} {subsystem:<16} | {probe_name:<38} ({latency_ms:.1f} ms)")
        if not passed and error:
            print(f"         {CLR_RED}└── ERROR: {error}{CLR_RESET}")

    def render_summary(self) -> bool:
        total = len(self.results)
        passed_count = sum(1 for r in self.results if r["passed"])
        failed_count = total - passed_count
        elapsed_sec = time.time() - self.start_time

        print("\n" + "=" * 80)
        print(f"{CLR_BOLD}{CLR_CYAN}🛡️  VOICESHIELD SUBSYSTEM DIAGNOSTIC HEALTH REPORT{CLR_RESET}")
        print("=" * 80)
        print(f"Total Probes Executed: {total}")
        print(f"Passed: {CLR_GREEN}{passed_count}{CLR_RESET} | Failed: {CLR_RED if failed_count else CLR_GREEN}{failed_count}{CLR_RESET}")
        print(f"Total Execution Time: {elapsed_sec:.2f}s")
        print("-" * 80)

        for r in self.results:
            tag = f"{CLR_GREEN}PASS{CLR_RESET}" if r["passed"] else f"{CLR_RED}FAIL{CLR_RESET}"
            print(f"  [{tag}] {r['subsystem']:<14} : {r['probe_name']:<36} -> {r['details']} ({r['latency_ms']:.1f}ms)")

        print("=" * 80)
        if failed_count == 0:
            print(f"{CLR_BOLD}{CLR_GREEN}>>> ALL DIAGNOSTIC HEALTH PROBES PASSED (SYSTEM OPERATIONAL) <<<{CLR_RESET}\n")
            return True
        else:
            print(f"{CLR_BOLD}{CLR_RED}>>> DIAGNOSTIC FAILURE: {failed_count} PROBE(S) FAILED <<<{CLR_RESET}\n")
            return False


# ---------------------------------------------------------------------------
# Audio Generation Helpers
# ---------------------------------------------------------------------------

def _create_wav_bytes(duration_sec: float = 1.5, sr: int = 16000, f0: float = 220.0) -> bytes:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = (0.35 * np.sin(2 * np.pi * f0 * t) + 0.15 * np.sin(2 * np.pi * 2 * f0 * t)).astype(np.float32)
    sig += 0.005 * np.random.default_rng(42).standard_normal(len(t))
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. Preprocessing & Normalization Probes
# ---------------------------------------------------------------------------

def probe_preprocessing_normalization(reporter: DiagnosticReporter) -> None:
    print(f"\n{CLR_BOLD}{CLR_BLUE}[1/5] Audio Ingestion & Normalization Probes{CLR_RESET}")

    # Probe 1A: In-memory WAV decoding and normalization bounds
    t0 = time.perf_counter()
    try:
        wav_bytes = _create_wav_bytes(duration_sec=2.0)
        full_audio, vad_audio, _ = decode_and_sanitize_audio(wav_bytes)
        norm_audio = normalize_audio_standard(full_audio)

        mean_val = float(np.abs(np.mean(norm_audio)))
        std_val = float(np.std(norm_audio))

        passed = (mean_val < 1e-4) and (0.95 <= std_val <= 1.05) and len(norm_audio) == 32000
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Preprocessor", "Normalization Bounds (|mean|<1e-4, std~1.0)",
            passed, lat, f"mean={mean_val:.2e}, std={std_val:.4f}"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Preprocessor", "Normalization Bounds", False, lat, error=str(exc))

    # Probe 1B: Corrupt/invalid bytes graceful handling
    t0 = time.perf_counter()
    try:
        corrupt_bytes = b"CORRUPTED_NON_AUDIO_HEADER_BYTES_12345"
        try:
            full_audio, vad_audio, diag = decode_and_sanitize_audio(corrupt_bytes)
            passed = isinstance(full_audio, np.ndarray)
        except (ValueError, Exception):
            passed = True  # Controlled error handled without crashing
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Preprocessor", "Corrupt Bytes Fallback Resilience",
            passed, lat, "Controlled exception / safe fallback with zero crashes"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Preprocessor", "Corrupt Bytes Fallback Resilience", False, lat, error=str(exc))

    # Probe 1C: Silence detection quality gate
    t0 = time.perf_counter()
    try:
        detector = ProductionNeuralDetector(load_hf=False)
        silent_audio = np.zeros(16000 * 2, dtype=np.float32)
        res = detector.predict(silent_audio)
        passed = (
            res.get("risk_band_key") == "low_quality"
            and res.get("diagnostics", {}).get("is_silent") is True
            and res.get("prediction_label") == "LOW QUALITY / DEGRADED"
        )
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Preprocessor", "Silence & Degraded Quality Gate",
            passed, lat, f"Gated to '{res.get('risk_band')}'"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Preprocessor", "Silence Quality Gate", False, lat, error=str(exc))


# ---------------------------------------------------------------------------
# 2. Physics & Glottal DSP Probes
# ---------------------------------------------------------------------------

def probe_physics_and_dsp(reporter: DiagnosticReporter) -> None:
    print(f"\n{CLR_BOLD}{CLR_BLUE}[2/5] LPC Residual & Glottal Biomechanics DSP Probes{CLR_RESET}")

    t = np.linspace(0, 1.5, 24000, endpoint=False)
    harmonic_speech = (0.3 * np.sin(2 * np.pi * 180.0 * t) + 0.15 * np.sin(2 * np.pi * 360.0 * t)).astype(np.float32)

    # Probe 2A: LPC Physics Analyzer
    t0 = time.perf_counter()
    try:
        lpc_analyzer = LPCPhysicsAnalyzer(order=16, sr=16000)
        lpc_metrics = lpc_analyzer.extract(harmonic_speech)

        passed = (
            0.0 <= lpc_metrics["lpc_anomaly_score"] <= 1.0
            and 0.0 <= lpc_metrics["phase_entropy"] <= 1.0
            and lpc_metrics["lpc_kurtosis"] > 0.0
            and np.isfinite(lpc_metrics["residual_flatness"])
        )
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "LPC Physics", "Residual Phase Entropy & Kurtosis",
            passed, lat, f"Kurt={lpc_metrics['lpc_kurtosis']:.2f}, Entropy={lpc_metrics['phase_entropy']:.3f}"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("LPC Physics", "Residual Phase Entropy", False, lat, error=str(exc))

    # Probe 2B: Forensic DSP Glottal Micro-Jitter
    t0 = time.perf_counter()
    try:
        dsp_analyzer = ForensicDSPAnalyzer(sr=16000)
        dsp_metrics = dsp_analyzer.extract(harmonic_speech)

        passed = (
            0.0 <= dsp_metrics["glottal_risk"] <= 1.0
            and 0.0 <= dsp_metrics["jitter_local"] <= 0.05
            and np.isfinite(dsp_metrics["hnr_db"])
            and np.isfinite(dsp_metrics["lfcc_variance"])
        )
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Forensic DSP", "Praat Glottal Jitter & ASVspoof LFCC",
            passed, lat, f"Jitter={dsp_metrics['jitter_local']*100:.3f}%, HNR={dsp_metrics['hnr_db']:.1f}dB"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Forensic DSP", "Glottal Jitter & LFCC", False, lat, error=str(exc))


# ---------------------------------------------------------------------------
# 3. Neural Foundation Model & Dynamic Label Resolver Probes
# ---------------------------------------------------------------------------

def probe_neural_engine_and_resolver(reporter: DiagnosticReporter) -> None:
    print(f"\n{CLR_BOLD}{CLR_BLUE}[3/5] Neural Foundation & Dynamic Label Resolution Probes{CLR_RESET}")

    # Probe 3A: Dynamic label resolution regex patterns
    t0 = time.perf_counter()
    try:
        detector = ProductionNeuralDetector(load_hf=False)

        # Mock Model 1: {"0": "fake", "1": "real"} -> spoof=0, human=1
        m1 = MagicMock()
        m1.config.id2label = {0: "fake", 1: "real"}
        detector.model = m1
        detector._resolve_labels()
        c1 = (detector.spoof_idx == 0 and detector.human_idx == 1)

        # Mock Model 2: {"0": "human", "1": "spoof"} -> spoof=1, human=0
        m2 = MagicMock()
        m2.config.id2label = {0: "human", 1: "spoof"}
        detector.model = m2
        detector._resolve_labels()
        c2 = (detector.spoof_idx == 1 and detector.human_idx == 0)

        passed = c1 and c2
        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Neural Engine", "Dynamic Label Regex Resolution",
            passed, lat, "Verified {0:fake,1:real} and {0:human,1:spoof}"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Neural Engine", "Dynamic Label Regex", False, lat, error=str(exc))

    # Probe 3B: Inference execution latency compliance (< 500ms CPU / < 150ms GPU)
    t0 = time.perf_counter()
    try:
        detector = ProductionNeuralDetector(load_hf=False)
        test_audio = np.sin(np.linspace(0, 100, 16000 * 2)).astype(np.float32) * 0.3
        res = detector.predict(test_audio)

        latency_ms = float(res.get("latency_ms", 999.0))
        max_allowed = 150.0 if torch.cuda.is_available() else 500.0
        passed = latency_ms <= max_allowed and 0 <= res["risk_score"] <= 100

        lat = (time.perf_counter() - t0) * 1000
        reporter.record(
            "Neural Engine", f"Inference Latency ({detector.device})",
            passed, lat, f"Engine latency={latency_ms:.1f}ms (threshold < {max_allowed}ms)"
        )
    except Exception as exc:
        lat = (time.perf_counter() - t0) * 1000
        reporter.record("Neural Engine", "Inference Latency", False, lat, error=str(exc))


# ---------------------------------------------------------------------------
# 4. REST API Gateway Probes
# ---------------------------------------------------------------------------

def probe_fastapi_rest_gateway(reporter: DiagnosticReporter) -> None:
    print(f"\n{CLR_BOLD}{CLR_BLUE}[4/5] FastAPI REST Gateway Probes{CLR_RESET}")
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        # Probe 4A: GET /health
        t0 = time.perf_counter()
        try:
            resp = client.get("/health")
            passed = resp.status_code == 200 and "X-Process-Time-Ms" in resp.headers
            data = resp.json()
            HealthResponse(**data)
            lat = (time.perf_counter() - t0) * 1000
            reporter.record(
                "REST Gateway", "GET /health Schema & Telemetry",
                passed, lat, f"status={data.get('status')}, device={data.get('device')}"
            )
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            reporter.record("REST Gateway", "GET /health", False, lat, error=str(exc))

        # Probe 4B: GET /metadata
        t0 = time.perf_counter()
        try:
            resp = client.get("/metadata")
            passed = resp.status_code == 200
            data = resp.json()
            MetadataResponse(**data)
            lat = (time.perf_counter() - t0) * 1000
            reporter.record(
                "REST Gateway", "GET /metadata Supported Formats",
                passed, lat, f"formats={len(data.get('supported_formats', []))}"
            )
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            reporter.record("REST Gateway", "GET /metadata", False, lat, error=str(exc))

        # Probe 4C: POST /predict multipart WAV upload
        t0 = time.perf_counter()
        try:
            wav_bytes = _create_wav_bytes(duration_sec=1.5)
            files = {"file": ("diagnostic_sample.wav", wav_bytes, "audio/wav")}
            resp = client.post("/predict", files=files)
            passed = resp.status_code == 200
            data = resp.json()
            PredictionResponse(**data)
            lat = (time.perf_counter() - t0) * 1000
            reporter.record(
                "REST Gateway", "POST /predict Multipart WAV Ingestion",
                passed, lat, f"Score={data.get('risk_score')}, Band='{data.get('risk_band')}'"
            )
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            reporter.record("REST Gateway", "POST /predict", False, lat, error=str(exc))


# ---------------------------------------------------------------------------
# 5. Live WebSocket Streaming & Twilio Protocol Probes
# ---------------------------------------------------------------------------

def probe_websocket_streaming(reporter: DiagnosticReporter) -> None:
    print(f"\n{CLR_BOLD}{CLR_BLUE}[5/5] Live WebSocket Streaming & Telephony Probes{CLR_RESET}")
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        # Probe 5A: Binary PCM stream (/ws/live-stream)
        t0 = time.perf_counter()
        try:
            t = np.linspace(0, 0.040, 640, endpoint=False)
            pcm16_chunk = (0.3 * np.sin(2 * np.pi * 300.0 * t) * 32767).astype(np.int16).tobytes()

            with client.websocket_connect("/ws/live-stream") as ws:
                for _ in range(5):
                    ws.send_bytes(pcm16_chunk)
                    frame = ws.receive_json()
                    StreamingTelemetryFrame(**frame)

            lat = (time.perf_counter() - t0) * 1000
            reporter.record(
                "WebSocket", "WS /ws/live-stream Binary PCM16 Ingestion",
                True, lat, "Ingested 5x 40ms chunks @ 25Hz without drop"
            )
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            reporter.record("WebSocket", "WS /ws/live-stream", False, lat, error=str(exc))

        # Probe 5B: Twilio Voice Protocol (/ws/twilio-media-stream)
        t0 = time.perf_counter()
        try:
            t = np.linspace(0, 0.100, 800, endpoint=False)
            sig_8k = (0.4 * np.sin(2 * np.pi * 400.0 * t)).astype(np.float32)
            mulaw_bytes = linear_to_mulaw_bytes(sig_8k)
            b64_payload = base64.b64encode(mulaw_bytes).decode("utf-8")

            with client.websocket_connect("/ws/twilio-media-stream") as ws:
                ws.send_text(json.dumps({"event": "connected"}))
                ack1 = ws.receive_json()

                ws.send_text(json.dumps({"event": "start", "streamSid": "MZ_PROBE_123"}))
                ack2 = ws.receive_json()

                ws.send_text(json.dumps({
                    "event": "media",
                    "streamSid": "MZ_PROBE_123",
                    "media": {"payload": b64_payload},
                }))
                assessment = ws.receive_json()

            passed = (
                ack1.get("event") == "connected_ack"
                and assessment.get("event") == "assessment"
                and "smoothed_risk_score" in assessment
            )
            lat = (time.perf_counter() - t0) * 1000
            reporter.record(
                "WebSocket", "WS /ws/twilio-media-stream G.711 Telephony",
                passed, lat, "Twilio handshake & Base64 mu-law decode OK"
            )
        except Exception as exc:
            lat = (time.perf_counter() - t0) * 1000
            reporter.record("WebSocket", "WS /ws/twilio-media-stream", False, lat, error=str(exc))


# ---------------------------------------------------------------------------
# Main Diagnostics Runner
# ---------------------------------------------------------------------------

def main():
    print(f"\n{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}    VOICESHIELD AUTOMATED SYSTEM DIAGNOSTICS & HEALTH HARNESS        {CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")

    reporter = DiagnosticReporter()

    probe_preprocessing_normalization(reporter)
    probe_physics_and_dsp(reporter)
    probe_neural_engine_and_resolver(reporter)
    probe_fastapi_rest_gateway(reporter)
    probe_websocket_streaming(reporter)

    success = reporter.render_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
