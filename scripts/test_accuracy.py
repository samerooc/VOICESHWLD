"""
VoiceShield Production Hybrid Detector Accuracy Benchmark Tool.
Tests audio sample against the Hybrid Pretrained Transformer + Praat Vocal-Tract DSP Engine.
"""

import argparse
import glob
import os
import sys
import time

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
from src.neural_engine import ProductionNeuralDetector


def run_benchmark(audio_path: str, device: str = None):
    if not os.path.exists(audio_path):
        print(f"[!] Error: Audio file not found at: {audio_path}")
        return

    print("\n" + "=" * 80)
    print("       VOICESHIELD HYBRID FORENSIC ACCURACY BENCHMARK")
    print("=" * 80)
    print(f" • Input Audio File     : {os.path.abspath(audio_path)}")

    # 1. Load Audio
    audio, sr = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)
    print(f" • Standardized Audio   : {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.2f}s)")

    # 2. Initialize Detector
    detector = ProductionNeuralDetector(device=device)

    # 3. Execute Inference
    res = detector.predict(audio, sample_rate=sr)
    fb = res["forensic_breakdown"]
    diag = res["diagnostics"]

    print("\n" + "-" * 80)
    print("                     FORENSIC DIAGNOSTIC READOUT")
    print("-" * 80)
    print(f" • Final Assessment     : {res['prediction_label'].upper()}")
    print(f" • Risk Score           : {res['risk_score']} / 100 ({res['risk_band']})")
    print(f" • Fused Spoof Prob     : {res['spoof_probability'] * 100:.2f}%")
    print(f" • Human Voice Prob     : {res['human_probability'] * 100:.2f}%")
    print("\n [MODALITY DECOMPOSITION]")
    print(f"   ├─ Fine-Tuned Transformer Prob : {fb['transformer_spoof_prob'] * 100:.2f}%")
    print(f"   ├─ Physical Vocal-Tract DSP    : {fb['dsp_physics_prob'] * 100:.2f}%")
    print(f"   ├─ Praat Local Jitter          : {fb['local_jitter']:.5f} (Normal: 0.005 - 0.015)")
    print(f"   ├─ Praat Local Shimmer         : {fb['local_shimmer']:.5f}")
    print(f"   ├─ Harmonics-to-Noise (HNR)    : {fb['hnr_db']:.1f} dB")
    print(f"   ├─ HF Spectral Cutoff Ratio    : {fb['hf_spectral_cutoff'] * 100:.2f}%")
    print(f"   └─ Glottal Anomaly Flag        : {'DETECTED ⚠️' if fb['has_glottal_anomaly'] else 'NONE (Natural) ✅'}")
    print("\n [SIGNAL DIAGNOSTICS & TELEMETRY]")
    print(f"   ├─ Audio Duration              : {diag['duration_sec']} s")
    print(f"   ├─ Estimated SNR               : {diag['snr_db']} dB")
    print(f"   └─ Total Processing Latency    : {res['latency_ms']} ms")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Hybrid Forensic Accuracy Benchmark")
    parser.add_argument("--file", type=str, required=False, default=None, help="Path to audio file (WAV/MP3/M4A/FLAC/OGG)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")

    args = parser.parse_args()

    audio_file = args.file
    if not audio_file:
        # Pick sample from dataset
        candidates = glob.glob(os.path.join(ROOT_DIR, "data", "**", "*.wav"), recursive=True)
        if candidates:
            audio_file = candidates[0]
        else:
            print("[!] Please specify --file <path_to_audio_file>")
            return

    run_benchmark(audio_file, device=args.device)


if __name__ == "__main__":
    main()
