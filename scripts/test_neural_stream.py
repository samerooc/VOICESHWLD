"""
VoiceShield Neural Step 4: Real-Time Telephony Simulation & Benchmark Tool.
Feeds 40ms streaming chunks into the RollingAudioBuffer, computes rolling neural
EMA risk scores, and outputs a dynamic live console telemetry dashboard.
"""

import argparse
import glob
import os
import sys
import time
from typing import Optional

# Ensure UTF-8 output encoding for Windows command line compatibility
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_io import load_audio_from_file
from src.config import SAMPLE_RATE
from src.neural_scoring import (
    DEFAULT_NEURAL_CHECKPOINT,
    NeuralStreamingScoreEngine,
    RollingAudioBuffer,
)


def run_neural_streaming_simulation(
    audio_path: str,
    checkpoint_path: str = DEFAULT_NEURAL_CHECKPOINT,
    chunk_ms: int = 40,
    window_eval_interval_ms: int = 200,
    fast_mode: bool = False,
    device: Optional[str] = None,
) -> None:
    """
    Simulates real-time chunked telephony audio ingestion and neural scoring.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Load test audio
    audio, sr = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)
    duration = len(audio) / sr

    print("\n" + "=" * 80)
    print("      VOICESHIELD NEURAL REAL-TIME TELEPHONY STREAMING SIMULATOR")
    print("=" * 80)
    print(f" • Audio Source File    : {audio_path}")
    print(f" • Audio Duration       : {duration:.2f} seconds ({len(audio)} samples @ {sr}Hz)")
    print(f" • Streaming Chunk Size : {chunk_ms} ms ({int(sr * (chunk_ms/1000.0))} samples)")
    print(f" • Evaluation Stride    : Every {window_eval_interval_ms} ms")
    print(f" • Neural Checkpoint    : {checkpoint_path}")
    print(f" • Hardware Device      : {device or 'Auto-Detect'}")
    print(f" • Simulation Mode      : {'Fast Benchmark' if fast_mode else 'Real-Time Telephony Emulation'}")
    print("=" * 80)

    # 2. Instantiate Buffer & Score Engine
    buffer = RollingAudioBuffer(capacity_seconds=6.0, sample_rate=sr)
    engine = NeuralStreamingScoreEngine(checkpoint_path=checkpoint_path, smoothing_alpha=0.35, device=device)
    engine.reset()

    chunk_samples = int(sr * (chunk_ms / 1000.0))
    eval_interval_chunks = max(1, window_eval_interval_ms // chunk_ms)

    # 3. Print Header
    header = (
        f"{'Time':<8} | {'Chunk':<6} | {'Raw Prob':<10} | {'EMA Score':<11} | "
        f"{'Risk Status':<32} | {'Latency':<9} | {'SLA (<200ms)':<10}"
    )
    print("\n" + header)
    print("-" * len(header))

    latencies = []
    chunk_idx = 0
    start_pos = 0

    sim_start_time = time.perf_counter()

    while start_pos < len(audio):
        chunk = audio[start_pos : start_pos + chunk_samples]
        buffer.add_samples(chunk)
        chunk_idx += 1
        curr_time = min(duration, chunk_idx * (chunk_ms / 1000.0))

        # Evaluate window every eval_interval_chunks (e.g. every 200ms)
        if chunk_idx % eval_interval_chunks == 0:
            window_audio = buffer.get_latest_window(window_samples=48000)
            res = engine.predict_stream_window(window_audio, sample_rate=sr)
            latencies.append(res["inference_latency_ms"])

            # Colorized pill badge simulation
            raw_p_str = f"{res['instantaneous_spoof_prob']*100:>5.1f}%"
            ema_score_str = f"{res['risk_score']:>3d} / 100"
            band_str = res["risk_band"]
            lat_str = f"{res['inference_latency_ms']:>6.2f}ms"
            sla_str = "✅ PASS" if res["is_realtime_compliant"] else "❌ EXCEEDED"

            print(
                f"{curr_time:>6.2f}s  | #{chunk_idx:<5d} | {raw_p_str:<10} | {ema_score_str:<11} | "
                f"{band_str:<32} | {lat_str:<9} | {sla_str:<10}"
            )

        start_pos += chunk_samples

        if not fast_mode:
            # Emulate real-time 40ms frame arrival
            time.sleep(chunk_ms / 1000.0)

    total_sim_time = time.perf_counter() - sim_start_time

    # 4. Telemetry & Benchmark Summary
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    max_latency = float(np.max(latencies)) if latencies else 0.0

    print("\n" + "=" * 80)
    print("                    STREAMING TELEMETRY SUMMARY")
    print("=" * 80)
    print(f" • Total Chunks Ingested  : {chunk_idx}")
    print(f" • Total Evaluations Done : {len(latencies)}")
    print(f" • Final Rolling Score    : {engine.rolling_risk_score} / 100")
    print(f" • Mean Inference Latency : {avg_latency:.2f} ms")
    print(f" • P95 Inference Latency  : {p95_latency:.2f} ms")
    print(f" • Max Inference Latency  : {max_latency:.2f} ms")
    print(f" • Real-Time Compliant    : {'✅ 100% SUB-200ms COMPLIANT' if max_latency < 200.0 else '⚠️ SLA SPIKE DETECTED'}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Real-Time Telephony Neural Stream Simulator")
    parser.add_argument("--file", type=str, default=None, help="Path to test WAV file")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_NEURAL_CHECKPOINT, help="Model weights path")
    parser.add_argument("--chunk-ms", type=int, default=40, help="Chunk duration in ms (default: 40)")
    parser.add_argument("--stride-ms", type=int, default=200, help="Inference evaluation interval (default: 200)")
    parser.add_argument("--fast", action="store_true", help="Run in high-speed benchmark mode without sleep delays")
    parser.add_argument("--device", type=str, default=None, help="Hardware device (cuda/cpu)")

    args = parser.parse_args()

    # Find a default test audio file if none provided
    target_file = args.file
    if not target_file:
        candidates = glob.glob("data/test/*/*.wav") + glob.glob("data/*/*.wav")
        if candidates:
            target_file = candidates[0]
        else:
            raise FileNotFoundError("No test audio files found in data/.")

    run_neural_streaming_simulation(
        audio_path=target_file,
        checkpoint_path=args.checkpoint,
        chunk_ms=args.chunk_ms,
        window_eval_interval_ms=args.stride_ms,
        fast_mode=args.fast,
        device=args.device,
    )


if __name__ == "__main__":
    main()
