"""
VoiceShield 4-Tier Orthogonal Forensic Robustness Verification Tool.
Tests authentic speech, vocoder deepfakes, and reverberant microphone recordings.
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


def run_robustness_test(file_path: str, is_mic: bool = False):
    if not os.path.exists(file_path):
        print(f"[!] Error: File not found: {file_path}")
        return

    print("\n" + "=" * 80)
    print("      VOICESHIELD 4-TIER ORTHOGONAL FORENSIC ROBUSTNESS BENCHMARK")
    print("=" * 80)
    print(f" • Target Audio File : {os.path.abspath(file_path)}")
    print(f" • Input Mode        : {'Live Mic / Reverb Calibrated' if is_mic else 'Standard Studio Upload'}")

    audio, sr = load_audio_from_file(file_path, target_sr=SAMPLE_RATE)
    detector = ProductionNeuralDetector(device="cpu")

    res = detector.predict(audio, sample_rate=sr, is_live_mic=is_mic)
    fb = res["forensic_breakdown"]
    diag = res["diagnostics"]

    print("\n" + "-" * 80)
    print("                       4-TIER FORENSIC AUDIT READOUT")
    print("-" * 80)
    print(f" • Final Assessment     : {res['prediction_label'].upper()}")
    print(f" • Threat Risk Score     : {res['risk_score']} / 100 ({res['risk_band']})")
    print(f" • Calibrated Spoof Prob : {res['spoof_probability'] * 100:.2f}%")
    print(f" • Authentic Human Prob  : {res['human_probability'] * 100:.2f}%")

    print("\n [TIER 1: VOICED SPEECH VAD & NORMALIZATION]")
    print(f"   ├─ Total Audio Duration     : {diag['duration_sec']} s")
    print(f"   ├─ Active Voiced Speech     : {diag['voiced_duration_sec']} s ({fb['voiced_ratio']*100:.1f}%)")
    print(f"   └─ Estimated Signal-to-Noise: {diag['snr_db']} dB")

    print("\n [TIER 2: ASVSPOOF-GRADE LFCC & VOCODER SPECTRAL ANALYSIS]")
    print(f"   ├─ Linear Cepstral (LFCC)   : {fb['lfcc_spoof_prob']*100:.2f}% Spoof Prob")
    print(f"   ├─ High-Freq Cutoff (>5.5k) : {fb['hf_cutoff_ratio']*100:.2f}% (Vocoders: <0.15%)")
    print(f"   └─ Spectral Flatness        : {fb['spectral_flatness']:.6f}")

    print("\n [TIER 3: BIOMECHANICAL GLOTTAL PHYSICS (VOICED-ONLY)]")
    print(f"   ├─ Physics Spoof Probability: {fb['physics_spoof_prob']*100:.2f}%")
    print(f"   ├─ Local Glottal Jitter     : {fb['local_jitter']:.5f} (Human: 0.006 - 0.022)")
    print(f"   ├─ Local Glottal Shimmer    : {fb['local_shimmer']:.5f}")
    print(f"   ├─ Harmonics-to-Noise (HNR) : {fb['hnr_db']:.1f} dB")
    print(f"   ├─ Formant Dispersion       : {fb['formant_dispersion']:.1f} Hz")
    print(f"   └─ Glottal Anomaly Marker   : {'DETECTED ⚠️' if fb['has_glottal_anomaly'] else 'NONE (Natural) ✅'}")

    print("\n [TIER 4: FOUNDATION MODEL WITH TEMPERATURE SCALING]")
    print(f"   ├─ Calibrated Neural Score  : {fb['transformer_spoof_prob']*100:.2f}%")
    print(f"   └─ Execution Latency        : {res['latency_ms']} ms")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Robustness Benchmark")
    parser.add_argument("--file", type=str, required=False, default=None, help="Path to audio file")
    parser.add_argument("--mic", action="store_true", help="Enable microphone noise / reverb calibration mode")

    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        candidates = glob.glob(os.path.join(ROOT_DIR, "data", "**", "*.wav"), recursive=True)
        target_file = candidates[0] if candidates else None

    if target_file:
        run_robustness_test(target_file, is_mic=args.mic)
    else:
        print("[!] Please provide an audio file with --file <path>")


if __name__ == "__main__":
    main()
