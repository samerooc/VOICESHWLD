"""
VoiceShield Production Forensic Verification CLI Tool.
Evaluates any audio recording (human mic or AI voice clone) with dynamic label resolution,
voiced-frame Praat biomechanics, and transformer consensus.
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


def run_forensic_verification(audio_path: str, is_mic: bool = False, device: str = None):
    if not os.path.exists(audio_path):
        print(f"[!] Error: Audio file not found at '{audio_path}'")
        return

    print("\n" + "=" * 80)
    print("         VOICESHIELD ENTERPRISE FORENSIC VERIFICATION TOOL")
    print("=" * 80)
    print(f" • Audio Target File     : {os.path.abspath(audio_path)}")
    print(f" • Input Mode            : {'Live Microphone Recording' if is_mic else 'Uploaded Clean Audio File'}")

    # 1. Load Audio
    audio, sr = load_audio_from_file(audio_path, target_sr=SAMPLE_RATE)
    print(f" • Ingested Audio Signal : {len(audio)} samples @ {sr} Hz ({len(audio)/sr:.2f}s)")

    # 2. Instantiate Hybrid Detector
    detector = ProductionNeuralDetector(device=device)

    # 3. Run Inference
    res = detector.predict(audio, sample_rate=sr, is_live_mic=is_mic)
    fb = res["forensic_breakdown"]
    diag = res["diagnostics"]

    print("\n" + "-" * 80)
    print("                  FORENSIC CONSENSUS & THREAT ASSESSMENT")
    print("-" * 80)
    print(f" • Final Verdict         : {res['prediction_label'].upper()}")
    print(f" • Threat Risk Score     : {res['risk_score']} / 100 ({res['risk_band']})")
    print(f" • Fused Spoof Prob      : {res['spoof_probability'] * 100:.2f}%")
    print(f" • Authentic Human Prob  : {res['human_probability'] * 100:.2f}%")

    print("\n [MODALITY DECOMPOSITION & BIOMECHANICS]")
    print(f"   ├─ Fine-Tuned Transformer Prob : {fb['transformer_spoof_prob'] * 100:.2f}%")
    print(f"   ├─ Voiced-Only Praat DSP Prob  : {fb['dsp_physics_prob'] * 100:.2f}%")
    print(f"   ├─ Local Glottal Jitter        : {fb['local_jitter']:.5f} (Normal Human: 0.005 - 0.015)")
    print(f"   ├─ Local Glottal Shimmer       : {fb['local_shimmer']:.5f}")
    print(f"   ├─ Harmonics-to-Noise Ratio    : {fb['hnr_db']:.1f} dB")
    print(f"   ├─ Cepstral Peak Prominence    : {fb['cpp']:.2f} dB")
    print(f"   ├─ HF Spectral Cutoff (>6kHz)  : {fb['hf_spectral_cutoff'] * 100:.2f}%")
    print(f"   └─ Glottal Periodicity Anomaly : {'DETECTED ⚠️' if fb['has_glottal_anomaly'] else 'NONE (Natural Periodicity) ✅'}")

    print("\n [ACOUSTIC DIAGNOSTICS & TELEMETRY]")
    print(f"   ├─ Total Duration              : {diag['duration_sec']} s")
    print(f"   ├─ Voiced Frame Ratio          : {fb['voiced_ratio'] * 100:.1f}%")
    print(f"   ├─ Estimated SNR               : {diag['snr_db']} dB")
    print(f"   └─ Total Processing Latency    : {res['latency_ms']} ms")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Enterprise Forensic Verification CLI")
    parser.add_argument("--file", type=str, required=False, default=None, help="Path to audio file (WAV/MP3/M4A/FLAC/OGG/WebM)")
    parser.add_argument("--mic", action="store_true", help="Enable microphone reverberation calibration mode")
    parser.add_argument("--device", type=str, default=None, help="Compute device ('cpu' or 'cuda')")

    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        candidates = glob.glob(os.path.join(ROOT_DIR, "data", "**", "*.wav"), recursive=True)
        if candidates:
            target_file = candidates[0]
        else:
            print("[!] Please specify an audio file using --file <path_to_audio>")
            return

    run_forensic_verification(target_file, is_mic=args.mic, device=args.device)


if __name__ == "__main__":
    main()
