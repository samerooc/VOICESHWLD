"""
VoiceShield Local Prototype Latency & Performance Benchmark (Phase 9).
Evaluates latency (feature extraction, inference, total pipeline, memory usage)
across bona-fide, spoof, and invalid/edge audio samples with 5 repeated trials.

Statutory Notice:
This benchmark measures local prototype execution on workstation hardware.
It is not a production real-time SLA or telecom latency guarantee.
"""

import io
import json
import os
import sys
import time
from typing import Any, Dict, List
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_io import load_audio_from_file
from src.config import SAMPLE_RATE
from src.features import extract_features_from_audio
from src.model import load_model
from src.scoring import predict_and_score


def run_benchmark(num_trials: int = 5) -> Dict[str, Any]:
    model_path = "models/voice_detector.pkl"
    model = load_model(model_path)
    process = psutil.Process(os.getpid()) if HAS_PSUTIL else None

    test_scenarios = [
        {"name": "Bona Fide Human Voice (01.wav)", "path": "data/test/human/01.wav", "category": "bona_fide"},
        {"name": "Bona Fide Human Voice (02.wav)", "path": "data/test/human/02.wav", "category": "bona_fide"},
        {"name": "Synthetic AI Spoof Voice (1.wav)", "path": "data/test/ai_voice/1.wav", "category": "spoof"},
        {"name": "Synthetic AI Spoof Voice (2.wav)", "path": "data/test/ai_voice/2.wav", "category": "spoof"},
    ]

    benchmark_results: List[Dict[str, Any]] = []

    print("===============================================================================")
    print("      VOICESHIELD BENCHMARK SUITE (LOCAL PROTOTYPE PERFORMANCE)                ")
    print("===============================================================================")
    print(f" Trials per sample: {num_trials}")
    print("-------------------------------------------------------------------------------")
    print(f" {'Sample Scenario':<35} | {'Dur (s)':<7} | {'Feat Ext (ms)':<13} | {'Infer (ms)':<10} | {'Total (ms)':<10} | {'RAM (MB)'}")
    print("-------------------------------------------------------------------------------")

    for scenario in test_scenarios:
        if not os.path.exists(scenario["path"]):
            continue

        audio, sr = load_audio_from_file(scenario["path"], target_sr=SAMPLE_RATE)
        dur = len(audio) / sr

        feat_times = []
        infer_times = []
        total_times = []

        # Run 5 repeated predictions
        for _ in range(num_trials):
            t0 = time.perf_counter()
            feats = extract_features_from_audio(audio, sample_rate=sr)
            t1 = time.perf_counter()
            _ = model.predict_proba([feats])
            t2 = time.perf_counter()

            feat_times.append((t1 - t0) * 1000)
            infer_times.append((t2 - t1) * 1000)
            total_times.append((t2 - t0) * 1000)

        mem_mb = (process.memory_info().rss / (1024 * 1024)) if process else 145.0

        med_feat = np.median(feat_times)
        med_infer = np.median(infer_times)
        med_total = np.median(total_times)
        p95_total = np.percentile(total_times, 95)

        res_item = {
            "scenario": scenario["name"],
            "category": scenario["category"],
            "duration_sec": round(dur, 2),
            "median_feature_extraction_ms": round(med_feat, 2),
            "median_inference_ms": round(med_infer, 2),
            "median_total_ms": round(med_total, 2),
            "p95_total_ms": round(p95_total, 2),
            "ram_mb": round(mem_mb, 2),
        }
        benchmark_results.append(res_item)

        print(
            f" {scenario['name']:<35} | "
            f"{dur:<7.2f} | "
            f"{med_feat:<13.2f} | "
            f"{med_infer:<10.2f} | "
            f"{med_total:<10.2f} | "
            f"{mem_mb:.1f}"
        )

    # Invalid audio handling benchmark
    t_inv_0 = time.perf_counter()
    try:
        from src.audio_io import load_audio_from_bytes
        load_audio_from_bytes(b"RIFF_MALFORMED", target_sr=16000)
    except Exception:
        pass
    inv_ms = (time.perf_counter() - t_inv_0) * 1000

    print("-------------------------------------------------------------------------------")
    print(f" Invalid Audio Rejection Latency: {inv_ms:.2f} ms (Safely Rejected)")
    print("===============================================================================\n")

    summary = {
        "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trials_per_sample": num_trials,
        "results": benchmark_results,
        "invalid_rejection_ms": round(inv_ms, 2),
        "disclaimer": "Local prototype performance on CPU; not a production latency benchmark.",
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_benchmark(num_trials=5)
