"""
VoiceShield Single-File & Batch Forensic Accuracy Verification CLI.
Accepts an audio file path or directory and outputs a comprehensive multi-tier forensic audit report.
"""

import argparse
import glob
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.neural_engine import ProductionNeuralDetector


def run_audit(file_path: str, detector: ProductionNeuralDetector):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    with open(file_path, "rb") as fp:
        audio_bytes = fp.read()

    res = detector.predict_bytes(audio_bytes)
    fb = res["forensic_breakdown"]
    diag = res["diagnostics"]
    windows = res.get("window_breakdown", [])

    print("=" * 80)
    print(f"VOICESHIELD FORENSIC AUDIT REPORT: {os.path.basename(file_path)}")
    print("=" * 80)
    print(f" • File Path              : {file_path}")
    print(f" • Duration / Voiced Sec  : {diag['duration_sec']:.2f}s (Voiced: {diag['voiced_duration_sec']:.2f}s, {diag['voiced_ratio']*100:.1f}%)")
    print(f" • Signal-to-Noise Ratio  : {diag['snr_db']:.1f} dB")
    print(f" • Inference Latency      : {res['latency_ms']} ms")
    print("-" * 80)
    print(f" • THREAT VERDICT         : {res['prediction_label'].upper()}")
    print(f" • RISK SCORE (0-100)     : {res['risk_score']} / 100 ({res['risk_band']})")
    print(f" • AI Clone Probability   : {res['spoof_probability']*100:.2f}%")
    print(f" • Authentic Human Prob   : {res['human_probability']*100:.2f}%")
    print("-" * 80)
    print(" 🔬 MULTI-TIER FORENSIC EVIDENCE BREAKDOWN:")
    print(f"  [Tier 1: Transformer]  : {fb['transformer_spoof_prob']*100:.1f}% Spoof Probability")
    print(f"  [Tier 2: LPC Physics]  : Anomaly Score = {fb['lpc_anomaly_score']:.2f} | Residual Kurtosis = {fb['residual_kurtosis']:.2f} | Phase Entropy = {fb['hf_phase_entropy']:.3f}")
    print(f"  [Tier 3: Glottal & LFCC]: Glottal Risk = {fb['glottal_spoof_prob']:.2f} | Local Jitter = {fb['local_jitter']:.5f} | HNR = {fb['hnr_db']:.1f} dB | HF Cutoff (>5.5kHz) = {fb['hf_cutoff_ratio']*100:.3f}%")
    
    if windows and len(windows) > 1:
        print("-" * 80)
        print(f" ⏱️ SLIDING WINDOW TIMELINE ({len(windows)} Windows, 4.0s Window / 50% Overlap):")
        for win in windows:
            print(f"  • Window {win['window_index']} [{win['time_range']}]: AI Probability = {win['spoof_probability']*100:.1f}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="VoiceShield Forensic CLI Audit")
    parser.add_argument("--file", "-f", type=str, help="Path to audio file (WAV, MP3, M4A, FLAC, OGG, WebM)")
    parser.add_argument("--dir", "-d", type=str, help="Path to directory containing audio files")
    args = parser.parse_args()

    detector = ProductionNeuralDetector()

    if args.file:
        run_audit(args.file, detector)
    elif args.dir:
        files = []
        for ext in ["*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg"]:
            files.extend(glob.glob(os.path.join(args.dir, ext)))
        print(f"Found {len(files)} audio files in {args.dir}...")
        for f in sorted(files):
            run_audit(f, detector)
    else:
        # Default test run on human sample and ai sample
        sample_h = os.path.join(ROOT_DIR, "data", "test", "human", "01.wav")
        sample_ai = os.path.join(ROOT_DIR, "data", "test", "ai_voice", "1.wav")
        if os.path.exists(sample_h):
            run_audit(sample_h, detector)
        if os.path.exists(sample_ai):
            run_audit(sample_ai, detector)


if __name__ == "__main__":
    main()
