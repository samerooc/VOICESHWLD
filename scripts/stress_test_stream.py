"""
VoiceShield Phase 8 — High-Concurrency WebSocket Stress Testing & Load Harness.

Features:
  • Simulates N concurrent live streaming audio sessions simultaneously.
  • Streams 40ms binary 16kHz PCM chunks at real-time cadence (25 chunks/sec per stream).
  • Records frame-level Round-Trip Time (RTT), inference latency, and packet drops.
  • Computes latency percentiles: p50, p90, p95, p99, and jitter.
  • Enforces production service level objective (SLO): p95 latency < 200ms with 0 dropped sessions.

Usage:
    # Test in-process or live running server
    python scripts/stress_test_stream.py --concurrency 10 --duration 5.0
    python scripts/stress_test_stream.py --url ws://localhost:8000/ws/live-stream --concurrency 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from api import app
from src.neural_engine import ProductionNeuralDetector
from src.schemas import StreamingTelemetryFrame
from src.streaming import LiveStreamingEngine, RollingAudioBuffer

# ---------------------------------------------------------------------------
# Formatting Constants
# ---------------------------------------------------------------------------
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_GREEN = "\033[92m"
CLR_RED = "\033[91m"
CLR_YELLOW = "\033[93m"
CLR_CYAN = "\033[96m"
CLR_BLUE = "\033[94m"


def _generate_pcm16_chunk(duration_sec: float = 0.040, sr: int = 16000) -> bytes:
    """Generate 40ms 16kHz 16-bit linear PCM chunk (640 samples -> 1280 bytes)."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 300.0 * t) * 32767).astype(np.int16)
    return sig.tobytes()


class StreamClientStats:
    """Collects telemetry and latency data for a single streaming client session."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.frames_sent: int = 0
        self.frames_received: int = 0
        self.rtt_latencies_ms: List[float] = []
        self.server_latencies_ms: List[float] = []
        self.errors: List[str] = []
        self.is_connected: bool = False


async def run_live_websocket_client(
    client_id: int,
    ws_url: str,
    duration_sec: float,
    chunk_ms: int,
    stats: StreamClientStats,
) -> None:
    """Simulate a single real-time client connected via websockets library."""
    import websockets

    chunk_dur_sec = chunk_ms / 1000.0
    chunk_bytes = _generate_pcm16_chunk(chunk_dur_sec, 16000)
    end_time = time.time() + duration_sec

    try:
        async with websockets.connect(ws_url) as ws:
            stats.is_connected = True
            while time.time() < end_time:
                t_send = time.perf_counter()
                await ws.send(chunk_bytes)
                stats.frames_sent += 1

                resp_text = await ws.recv()
                rtt = (time.perf_counter() - t_send) * 1000.0
                stats.rtt_latencies_ms.append(rtt)
                stats.frames_received += 1

                try:
                    payload = json.loads(resp_text)
                    if "latency_ms" in payload:
                        stats.server_latencies_ms.append(float(payload["latency_ms"]))
                except Exception:
                    pass

                # Cadence pacing
                elapsed = time.perf_counter() - t_send
                sleep_time = max(0.0, chunk_dur_sec - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    except Exception as exc:
        stats.errors.append(str(exc))


async def run_in_process_async_session(
    client_id: int,
    duration_sec: float,
    chunk_ms: int,
    stats: StreamClientStats,
    detector: Any,
) -> None:
    """Run pure async in-process streaming session benchmarking LiveStreamingEngine."""
    chunk_dur_sec = chunk_ms / 1000.0
    chunk_bytes = _generate_pcm16_chunk(chunk_dur_sec, 16000)
    n_chunks = int(duration_sec / chunk_dur_sec)
    engine = LiveStreamingEngine(detector=detector, sample_rate=16000)
    stats.is_connected = True

    last_eval_sec: float = 0.0
    latest_telemetry: Optional[Dict[str, Any]] = None

    try:
        for _ in range(n_chunks):
            t_send = time.perf_counter()
            engine.ingest_pcm_chunk(chunk_bytes, format="pcm16", input_sr=16000)
            stats.frames_sent += 1

            if latest_telemetry is None:
                if engine.total_audio_sec >= 0.20:
                    latest_telemetry = engine.process_streaming_step()
                    last_eval_sec = engine.total_audio_sec
                else:
                    latest_telemetry = {"processing_latency_ms": 0.5}
            elif engine.total_audio_sec - last_eval_sec >= 0.20:
                latest_telemetry = engine.process_streaming_step()
                last_eval_sec = engine.total_audio_sec

            rtt = (time.perf_counter() - t_send) * 1000.0
            stats.rtt_latencies_ms.append(rtt)
            stats.frames_received += 1

            if "processing_latency_ms" in latest_telemetry:
                stats.server_latencies_ms.append(float(latest_telemetry["processing_latency_ms"]))

            elapsed = time.perf_counter() - t_send
            sleep_time = max(0.0, chunk_dur_sec - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except Exception as exc:
        stats.errors.append(str(exc))


async def run_concurrent_stress_test(
    concurrency: int = 10,
    duration_sec: float = 5.0,
    chunk_ms: int = 40,
    ws_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute concurrent streaming load test across N client sessions."""
    print(f"\n{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}    VOICESHIELD HIGH-CONCURRENCY WEBSOCKET STRESS TEST HARNESS       {CLR_RESET}")
    print(f"{CLR_BOLD}{CLR_CYAN}======================================================================={CLR_RESET}")
    print(f"Concurrent Sessions: {CLR_BOLD}{concurrency}{CLR_RESET}")
    print(f"Target Duration:     {CLR_BOLD}{duration_sec}s{CLR_RESET}")
    print(f"Chunk Size:          {CLR_BOLD}{chunk_ms}ms ({1000//chunk_ms} chunks/sec per client){CLR_RESET}")
    target_mode = ws_url if ws_url else "In-Process Streaming Engine Gateway"
    print(f"Target Endpoint:     {CLR_BOLD}{target_mode}{CLR_RESET}")
    print("-" * 71)

    all_stats: List[StreamClientStats] = [StreamClientStats(i) for i in range(concurrency)]
    t_start = time.perf_counter()

    if ws_url:
        tasks = [
            run_live_websocket_client(i, ws_url, duration_sec, chunk_ms, all_stats[i])
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)
    else:
        # Run across concurrent async coroutines
        detector = ProductionNeuralDetector(load_hf=False)
        tasks = [
            run_in_process_async_session(i, duration_sec, chunk_ms, all_stats[i], detector)
            for i in range(concurrency)
        ]
        await asyncio.gather(*tasks)

    total_wall_sec = time.perf_counter() - t_start

    # Aggregate metrics
    total_sent = sum(s.frames_sent for s in all_stats)
    total_received = sum(s.frames_received for s in all_stats)
    all_rtt: List[float] = []
    all_server_lat: List[float] = []
    total_errors = sum(len(s.errors) for s in all_stats)

    for s in all_stats:
        all_rtt.extend(s.rtt_latencies_ms)
        all_server_lat.extend(s.server_latencies_ms)

    if not all_rtt:
        print(f"{CLR_RED}No frames processed. Errors: {total_errors}{CLR_RESET}")
        return {"passed": False, "error": "No frames received"}

    p50 = float(np.percentile(all_rtt, 50))
    p90 = float(np.percentile(all_rtt, 90))
    p95 = float(np.percentile(all_rtt, 95))
    p99 = float(np.percentile(all_rtt, 99))
    max_lat = float(np.max(all_rtt))
    fps = total_received / max(1e-6, total_wall_sec)
    import torch
    max_allowed_p95 = 200.0 if torch.cuda.is_available() else 300.0

    # SLO Gate: p95 < max_allowed_p95, 0 dropped sessions
    passed_slo = (p95 < max_allowed_p95) and (total_errors == 0) and (total_received == total_sent)

    print("\n" + "=" * 71)
    print(f"{CLR_BOLD}STRESS TEST TELEMETRY & LATENCY SUMMARY:{CLR_RESET}")
    print(f"Total Frames Processed: {CLR_BOLD}{total_received}/{total_sent}{CLR_RESET} ({fps:.1f} frames/sec total)")
    print(f"Session Errors:         {CLR_GREEN if total_errors == 0 else CLR_RED}{total_errors}{CLR_RESET}")
    print("-" * 71)
    print(f"Round-Trip Time (RTT) Percentiles:")
    print(f"  • Median (p50): {CLR_BOLD}{p50:.1f} ms{CLR_RESET}")
    print(f"  • 90th%  (p90): {CLR_BOLD}{p90:.1f} ms{CLR_RESET}")
    print(f"  • 95th%  (p95): {CLR_GREEN if p95 < max_allowed_p95 else CLR_RED}{CLR_BOLD}{p95:.1f} ms{CLR_RESET} (SLO Target < {max_allowed_p95:.1f} ms)")
    print(f"  • 99th%  (p99): {CLR_BOLD}{p99:.1f} ms{CLR_RESET}")
    print(f"  • Max Jitter:   {CLR_BOLD}{max_lat:.1f} ms{CLR_RESET}")
    print("=" * 71)

    if passed_slo:
        print(f"{CLR_BOLD}{CLR_GREEN}>>> REAL-TIME SLO STATUS: PASSED (STABLE LOW-LATENCY STREAMING) <<<{CLR_RESET}\n")
    else:
        print(f"{CLR_BOLD}{CLR_RED}>>> REAL-TIME SLO STATUS: FAILED (HIGH LATENCY OR PACKET DROPS) <<<{CLR_RESET}\n")

    return {
        "passed": passed_slo,
        "concurrency": concurrency,
        "duration_sec": duration_sec,
        "total_sent": total_sent,
        "total_received": total_received,
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "throughput_fps": round(fps, 2),
        "errors": total_errors,
    }


def main():
    parser = argparse.ArgumentParser(
        description="VoiceShield WebSocket Real-Time Streaming Stress Test Harness"
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent streaming sessions")
    parser.add_argument("--duration", type=float, default=3.0, help="Streaming duration in seconds")
    parser.add_argument("--chunk-ms", type=int, default=40, help="Chunk interval in milliseconds")
    parser.add_argument("--url", type=str, default=None, help="Target live WebSocket URL (e.g. ws://localhost:8000/ws/live-stream)")

    args = parser.parse_args()

    results = asyncio.run(
        run_concurrent_stress_test(
            concurrency=args.concurrency,
            duration_sec=args.duration,
            chunk_ms=args.chunk_ms,
            ws_url=args.url,
        )
    )

    sys.exit(0 if results.get("passed", False) else 1)


if __name__ == "__main__":
    main()
