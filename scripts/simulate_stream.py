"""
VoiceShield Sandbox Streaming Audio Simulator (Phase 7).

===============================================================================
                SANDBOX SIMULATION — NOT A LIVE CALL
This simulation demonstrates the processing flow only.
It is not a telecom integration or production latency benchmark.
===============================================================================
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_io import load_audio_from_file
from src.config import SAMPLE_RATE, STATUTORY_DISCLAIMER
from src.streaming import (
    STREAMING_DISCLAIMER,
    SandboxStreamAnalyzer,
    slice_streaming_windows,
)

# Aliases for backward compatibility
StreamSimulator = SandboxStreamAnalyzer
chunk_audio_stream = slice_streaming_windows

SANDBOX_BANNER: str = """
===============================================================================
       VOICESHIELD STREAM SIMULATOR (SANDBOX CHUNK PROCESSING)
              *** SANDBOX SIMULATION — NOT A LIVE CALL ***
===============================================================================
"""


def run_stream_simulation(
    audio_path: str,
    window_ms: int = 160,
    stride_ms: int = 40,
    max_windows: int = 25,
    simulated_delay_sec: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Executes streaming simulation over a prerecorded audio file with rolling risk computation.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: '{audio_path}'")

    print(SANDBOX_BANNER)
    print(f"  • Source WAV File       : {audio_path}")
    print(f"  • Window Size           : {window_ms} ms ({int(SAMPLE_RATE * (window_ms / 1000))} samples)")
    print(f"  • Stride Step           : {stride_ms} ms ({int(SAMPLE_RATE * (stride_ms / 1000))} samples)")
    print(f"  • Statutory Notice      : {STATUTORY_DISCLAIMER}")
    print("-------------------------------------------------------------------------------")
    print(f" {'Win#':<5} | {'Time (s)':<8} | {'Spoof Prob':<12} | {'Rolling Score':<14} | {'Risk Band':<16} | {'Proc (ms)':<9} | {'Status / Reason'}")
    print("-------------------------------------------------------------------------------")

    analyzer = SandboxStreamAnalyzer()
    audio, sr = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)
    generator = slice_streaming_windows(audio, sample_rate=sr, window_ms=window_ms, stride_ms=stride_ms)

    results: List[Dict[str, Any]] = []
    start_sim_time = time.perf_counter()

    try:
        for idx, timestamp, chunk in generator:
            if idx >= max_windows:
                print(f"\n[INFO] Reached maximum window limit ({max_windows}).")
                break

            res = analyzer.process_chunk(idx, timestamp, chunk, sample_rate=sr)
            results.append(res)

            prob_str = f"{res['instantaneous_spoof_prob']:.4f}" if res['is_valid'] else "---"
            roll_str = f"{res['rolling_risk_score']:.1f} / 100"
            status_str = "OK (Analyzed)" if res["is_valid"] else f"SKIPPED ({res['skipped_reason']})"

            print(
                f" {res['window_number']:<5} | "
                f"{res['timestamp_sec']:<8.3f} | "
                f"{prob_str:<12} | "
                f"{roll_str:<14} | "
                f"{res['risk_band']:<16} | "
                f"{res['processing_ms']:<9.2f} | "
                f"{status_str}"
            )

            if simulated_delay_sec > 0:
                time.sleep(simulated_delay_sec)

    except KeyboardInterrupt:
        print("\n[INFO] Simulation stopped by user (KeyboardInterrupt).")

    total_sim_sec = round(time.perf_counter() - start_sim_time, 3)

    print("-------------------------------------------------------------------------------")
    print(f" Simulation Summary:")
    print(f"  • Total Windows Processed : {analyzer.processed_windows}")
    print(f"  • Skipped Windows Count   : {analyzer.skipped_windows}")
    print(f"  • Final Rolling Risk Score: {analyzer.rolling_score:.1f} / 100")
    print(f"  • Total Simulation Time   : {total_sim_sec:.3f} s")
    print(f"  • Zero Raw Audio Retained : True (audio_saved: false)")
    print("===============================================================================\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceShield Sandbox Streaming Simulator")
    parser.add_argument(
        "--audio",
        type=str,
        default="data/test/ai_voice/1.wav",
        help="Path to local prerecorded WAV file.",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=160,
        help="Window size in milliseconds (default: 160 ms).",
    )
    parser.add_argument(
        "--stride-ms",
        type=int,
        default=40,
        help="Stride step size in milliseconds (default: 40 ms).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=25,
        help="Maximum windows to simulate (default: 25).",
    )
    args = parser.parse_args()

    audio_file = args.audio
    if not os.path.exists(audio_file):
        if os.path.exists("data/test/human/01.wav"):
            audio_file = "data/test/human/01.wav"

    run_stream_simulation(
        audio_path=audio_file,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        max_windows=args.max_windows,
    )


if __name__ == "__main__":
    main()
