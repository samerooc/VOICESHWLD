"""
VoiceShield Phase 6: High-Concurrency Redis Streams & Worker Benchmark Script.
Simulates N concurrent live calls pushing 40ms chunks in parallel, verifies
decoupled worker consumption, and profiles end-to-end round-trip scoring latencies.
"""

import argparse
import asyncio
import glob
import json
import os
import sys
import time
from typing import List

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
from src.queue_manager import DEFAULT_REDIS_URL, RedisStreamManager


async def simulate_single_caller(
    manager: RedisStreamManager,
    stream_id: str,
    raw_pcm_bytes: bytes,
    chunk_bytes_len: int,
    chunk_ms: int,
    num_chunks: int,
    latencies: List[float],
) -> None:
    """Simulates a single live call pushing 40ms audio chunks and listening for risk assessments."""
    # Start consumer listener task
    received_count = 0

    async def listener():
        nonlocal received_count
        async for assessment in manager.subscribe_risk_channel(stream_id):
            sent_t = assessment.get("timestamp", time.time())
            rtt_ms = max(1.0, (time.time() - sent_t) * 1000.0)
            latencies.append(rtt_ms)
            received_count += 1
            if received_count >= 3:
                break

    listen_task = asyncio.create_task(listener())

    pos = 0
    for i in range(num_chunks):
        if pos + chunk_bytes_len > len(raw_pcm_bytes):
            pos = 0
        chunk = raw_pcm_bytes[pos : pos + chunk_bytes_len]
        pos += chunk_bytes_len

        meta = {"timestamp": time.time(), "format": "pcm_float32", "sample_rate": 16000}
        await manager.push_audio_chunk(stream_id, chunk, metadata=meta)
        await asyncio.sleep(chunk_ms / 1000.0)

    await asyncio.sleep(0.5)
    listen_task.cancel()
    try:
        await listen_task
    except asyncio.CancelledError:
        pass


async def run_concurrency_benchmark(
    redis_url: str = DEFAULT_REDIS_URL,
    num_concurrent_calls: int = 20,
    duration_sec: float = 3.0,
    audio_path: Optional[str] = None,
) -> None:
    """Runs high-concurrency multi-stream benchmark."""
    manager = RedisStreamManager(redis_url=redis_url)

    if not audio_path:
        candidates = glob.glob(os.path.join(ROOT_DIR, "data", "**", "*.wav"), recursive=True)
        audio_path = candidates[0] if candidates else None

    if not audio_path or not os.path.exists(audio_path):
        # Generate synthetic sine wave if no audio file present
        t = np.linspace(0, 5.0, 80000, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    else:
        audio, _ = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)

    raw_pcm_bytes = audio.tobytes()
    chunk_ms = 40
    chunk_samples = int(SAMPLE_RATE * (chunk_ms / 1000.0))
    chunk_bytes_len = chunk_samples * 4  # 4 bytes per float32 sample
    num_chunks = int(duration_sec * 1000 / chunk_ms)

    print("\n" + "=" * 80)
    print("       VOICESHIELD DISTRIBUTED REDIS STREAMING BENCHMARK")
    print("=" * 80)
    print(f" • Redis Connection URL   : {redis_url}")
    print(f" • Concurrent Call Streams: {num_concurrent_calls}")
    print(f" • Stream Duration        : {duration_sec} seconds ({num_chunks} frames/stream)")
    print(f" • Total Frames to Queue  : {num_concurrent_calls * num_chunks} chunks")
    print("=" * 80 + "\n")

    latencies: List[float] = []
    t_start = time.perf_counter()

    tasks = [
        simulate_single_caller(
            manager=manager,
            stream_id=f"bench_stream_{i:03d}",
            raw_pcm_bytes=raw_pcm_bytes,
            chunk_bytes_len=chunk_bytes_len,
            chunk_ms=chunk_ms,
            num_chunks=num_chunks,
            latencies=latencies,
        )
        for i in range(num_concurrent_calls)
    ]

    await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t_start

    total_chunks = num_concurrent_calls * num_chunks
    throughput = total_chunks / max(0.01, elapsed)

    print("\n" + "=" * 80)
    print("                    BENCHMARK RESULTS & METRICS")
    print("=" * 80)
    print(f" • Total Elapsed Time     : {elapsed:.2f} s")
    print(f" • Ingestion Throughput   : {throughput:.1f} chunks / second")
    if latencies:
        print(f" • Round-Trip P50 Latency : {np.percentile(latencies, 50):.2f} ms")
        print(f" • Round-Trip P95 Latency : {np.percentile(latencies, 95):.2f} ms")
        print(f" • Round-Trip P99 Latency : {np.percentile(latencies, 99):.2f} ms")
    else:
        print(" • Note: Round-trip latency profiling requires an active background worker daemon.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Redis Streaming Concurrency Benchmark")
    parser.add_argument("--redis-url", type=str, default=DEFAULT_REDIS_URL, help="Redis server URL")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of concurrent live streams (default: 20)")
    parser.add_argument("--duration", type=float, default=3.0, help="Simulation duration in seconds (default: 3.0)")
    parser.add_argument("--audio-file", type=str, default=None, help="Audio file path to use for streaming")

    args = parser.parse_args()

    asyncio.run(
        run_concurrency_benchmark(
            redis_url=args.redis_url,
            num_concurrent_calls=args.concurrency,
            duration_sec=args.duration,
            audio_path=args.audio_file,
        )
    )


if __name__ == "__main__":
    main()
