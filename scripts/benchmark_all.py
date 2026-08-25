"""
VoiceShield Batch Benchmark Evaluation Script.
Evaluates all human and AI test audio files in data/test/
"""

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


def main():
    print("=" * 80)
    print("VOICESHIELD BATCH BENCHMARK EVALUATION")
    print("=" * 80)

    det = ProductionNeuralDetector()

    human_files = sorted(glob.glob(os.path.join(ROOT_DIR, "data", "test", "human", "*.wav")))
    ai_files = sorted(glob.glob(os.path.join(ROOT_DIR, "data", "test", "ai_voice", "*.wav")))

    print(f"\nEvaluating {len(human_files)} Human Voice Files and {len(ai_files)} AI Voice Files...\n")

    correct_human = 0
    total_human = len(human_files)
    print("[1] AUTHENTIC HUMAN SAMPLES:")
    print("-" * 80)
    for f in human_files:
        with open(f, "rb") as fp:
            data = fp.read()
        res = det.predict(data)
        fb = res["forensic_breakdown"]
        diag = res["diagnostics"]
        score = res["risk_score"]
        is_pass = score <= 35
        if is_pass:
            correct_human += 1
        status_str = "PASS [HUMAN]" if is_pass else "FAIL [FALSE ALARM]"
        print(f" • {os.path.basename(f):<10} | Score: {score:3d}/100 | DL: {fb['transformer_spoof_prob']*100:5.1f}% | Jitter: {fb['local_jitter']:.5f} | HNR: {fb['hnr_db']:4.1f}dB | HF: {fb['hf_cutoff_ratio']*100:4.2f}% | SNR: {diag['snr_db']:4.1f}dB | {status_str}")

    correct_ai = 0
    total_ai = len(ai_files)
    print("\n[2] SYNTHETIC AI CLONED SAMPLES:")
    print("-" * 80)
    for f in ai_files:
        with open(f, "rb") as fp:
            data = fp.read()
        res = det.predict(data)
        fb = res["forensic_breakdown"]
        diag = res["diagnostics"]
        score = res["risk_score"]
        is_pass = score >= 50
        if is_pass:
            correct_ai += 1
        status_str = "PASS [DETECTED AI]" if is_pass else "FAIL [MISSED AI]"
        print(f" • {os.path.basename(f):<10} | Score: {score:3d}/100 | DL: {fb['transformer_spoof_prob']*100:5.1f}% | Jitter: {fb['local_jitter']:.5f} | HNR: {fb['hnr_db']:4.1f}dB | HF: {fb['hf_cutoff_ratio']*100:4.2f}% | SNR: {diag['snr_db']:4.1f}dB | {status_str}")

    print("\n" + "=" * 80)
    print(f"SUMMARY ACCURACY: Human Accuracy = {correct_human}/{total_human} ({correct_human/max(1, total_human)*100:.1f}%), AI Detection = {correct_ai}/{total_ai} ({correct_ai/max(1, total_ai)*100:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
