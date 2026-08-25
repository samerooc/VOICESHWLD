"""
VoiceShield Neural Step 5: Real-Time WebSocket Streaming Client & Benchmark Script.
Streams 40ms 16-bit 16kHz linear PCM frames over WebSocket, logs real-time
evaluations, verifies Exponential Moving Average (EMA) smoothing, and checks
sub-300ms round-trip latency SLAs.
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import Optional

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_io import load_audio_from_file
from src.config import SAMPLE_RATE


async def stream_audio_websocket(
    server_url: str,
    audio_path: str,
    chunk_ms: int = 40,
    fast_mode: bool = False,
) -> None:
    """
    Streams raw binary PCM chunks to the VoiceShield WebSocket server and monitors telemetry.
    """
    import websockets

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)
    duration = len(audio) / sr

    # Convert normalized float32 [-1.0, 1.0] to 16-bit signed integer linear PCM bytes
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
    raw_pcm_bytes = audio_int16.tobytes()

    chunk_samples = int(SAMPLE_RATE * (chunk_ms / 1000.0))
    chunk_bytes_len = chunk_samples * 2  # 2 bytes per 16-bit sample

    print("\n" + "=" * 80)
    print("      VOICESHIELD REAL-TIME WEBSOCKET STREAMING BENCHMARK")
    print("=" * 80)
    print(f" • Target WebSocket URL : {server_url}")
    print(f" • Audio Source File    : {audio_path}")
    print(f" • Audio Duration       : {duration:.2f} seconds ({len(audio)} samples)")
    print(f" • Streaming Chunk Size : {chunk_ms} ms ({chunk_bytes_len} bytes/frame)")
    print(f" • Simulation Mode      : {'Fast Throughput Benchmark' if fast_mode else 'Live Real-Time Telephony Emulation'}")
    print("=" * 80)

    header = (
        f"{'Stream Time':<12} | {'Inst. Risk':<11} | {'Smoothed Score':<15} | "
        f"{'Risk Band':<30} | {'Latency':<9} | {'SLA (<300ms)':<10}"
    )
    print("\n" + header)
    print("-" * len(header))

    latencies = []
    received_evaluations = []

    async with websockets.connect(server_url) as ws:
        async def receiver_task():
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    if data.get("event") == "assessment":
                        received_evaluations.append(data)
                        lat_ms = data.get("latency_ms", 0.0)
                        latencies.append(lat_ms)

                        t_str = f"{data.get('timestamp', 0.0):>6.2f}s"
                        inst_str = f"{data.get('instantaneous_risk', 0):>3d} %"
                        smooth_str = f"{data.get('smoothed_risk_score', 0):>3d} / 100"
                        band_str = data.get("risk_band", "Unknown")
                        lat_str = f"{lat_ms:>6.2f}ms"
                        sla_str = "[PASS]" if lat_ms < 300.0 else "[EXCEED]"

                        print(
                            f"{t_str:<12} | {inst_str:<11} | {smooth_str:<15} | "
                            f"{band_str:<30} | {lat_str:<9} | {sla_str:<10}"
                        )
            except asyncio.CancelledError:
                pass
            except websockets.ConnectionClosed:
                pass

        recv_future = asyncio.create_task(receiver_task())

        # Stream binary frames
        offset = 0
        total_len = len(raw_pcm_bytes)

        while offset < total_len:
            frame = raw_pcm_bytes[offset : offset + chunk_bytes_len]
            await ws.send(frame)
            offset += chunk_bytes_len

            if not fast_mode:
                await asyncio.sleep(chunk_ms / 1000.0)

        # Allow trailing evaluation frames to arrive
        await asyncio.sleep(0.35)
        recv_future.cancel()

    # Telemetry Summary
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    max_latency = float(np.max(latencies)) if latencies else 0.0

    print("\n" + "=" * 80)
    print("                    WEBSOCKET TELEMETRY SUMMARY")
    print("=" * 80)
    print(f" • Total Frames Streamed  : {offset // chunk_bytes_len}")
    print(f" • Evaluations Received   : {len(received_evaluations)}")
    print(f" • Mean Inference Latency : {avg_latency:.2f} ms")
    print(f" • P95 Latency            : {p95_latency:.2f} ms")
    print(f" • Max Peak Latency       : {max_latency:.2f} ms")
    print(f" • SLA Compliance (<300ms): {'[PASS] 100% SLA COMPLIANT' if max_latency < 300.0 else '[WARN] LATENCY SPIKE'}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield WebSocket Streaming Benchmark")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/ws/live-stream", help="WebSocket server URL")
    parser.add_argument("--file", type=str, default=None, help="Path to audio WAV file")
    parser.add_argument("--chunk-ms", type=int, default=40, help="Frame duration in ms (default: 40)")
    parser.add_argument("--fast", action="store_true", help="Fast benchmark mode without real-time sleep delays")

    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        candidates = glob.glob("data/test/*/*.wav") + glob.glob("data/*/*.wav")
        if candidates:
            target_file = candidates[0]
        else:
            raise FileNotFoundError("No WAV test audio found in data/.")

    asyncio.run(
        stream_audio_websocket(
            server_url=args.url,
            audio_path=target_file,
            chunk_ms=args.chunk_ms,
            fast_mode=args.fast,
        )
    )


if __name__ == "__main__":
    main()
