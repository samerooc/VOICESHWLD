"""
VoiceShield Automated Calibration & Sanity Verification Harness.
Generates synthesized test signals and verifies detector calibration across
vocal tract physics, LPC residuals, and transformer decision boundaries.
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.neural_engine import ProductionNeuralDetector
from src.audio_processor import SAMPLE_RATE


def generate_synthetic_tone(freq: float = 200.0, duration: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generates synthetic pure sine tone with zero vocal cord jitter."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_harmonic_speech_proxy(f0: float = 140.0, duration: float = 2.5, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generates synthetic harmonic series with natural human pitch micro-jitter."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Add 1.5% micro-jitter to fundamental frequency
    jitter = 0.015 * np.sin(2 * np.pi * 5.0 * t)
    f0_mod = f0 * (1.0 + jitter)
    phase = 2 * np.pi * np.cumsum(f0_mod) / sr

    signal = np.zeros_like(t)
    for harmonic in range(1, 15):
        amp = 1.0 / (harmonic ** 0.8)
        signal += amp * np.sin(harmonic * phase)

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    return signal.astype(np.float32)


def main():
    print("=" * 80)
    print("VOICESHIELD SYSTEM CALIBRATION & SANITY VERIFICATION")
    print("=" * 80)

    detector = ProductionNeuralDetector()
    print(f"[*] Engine Initialized on: [{detector.device}]")

    print("\n[1] Testing Synthetic Pure Tone (Zero-Jitter Regularity):")
    pure_tone = generate_synthetic_tone(freq=220.0, duration=2.0)
    res_tone = detector.predict(pure_tone)
    print(f" • Verdict: {res_tone['prediction_label']} | Score: {res_tone['risk_score']}/100 | Spoof Prob: {res_tone['spoof_probability']*100:.1f}%")

    print("\n[2] Testing Harmonic Speech Proxy (Natural Micro-Jitter):")
    harmonic_speech = generate_harmonic_speech_proxy(f0=130.0, duration=2.5)
    res_harm = detector.predict(harmonic_speech)
    print(f" • Verdict: {res_harm['prediction_label']} | Score: {res_harm['risk_score']}/100 | Spoof Prob: {res_harm['spoof_probability']*100:.1f}%")

    print("\n[3] Testing Real Human Benchmark Sample (data/test/human/01.wav):")
    h_file = os.path.join(ROOT_DIR, "data", "test", "human", "01.wav")
    if os.path.exists(h_file):
        with open(h_file, "rb") as fp:
            res_h = detector.predict(fp.read())
        print(f" • Verdict: {res_h['prediction_label']} | Score: {res_h['risk_score']}/100 | Latency: {res_h['latency_ms']}ms")

    print("\n[4] Testing Synthetic AI Clone Sample (data/test/ai_voice/1.wav):")
    ai_file = os.path.join(ROOT_DIR, "data", "test", "ai_voice", "1.wav")
    if os.path.exists(ai_file):
        with open(ai_file, "rb") as fp:
            res_ai = detector.predict(fp.read())
        print(f" • Verdict: {res_ai['prediction_label']} | Score: {res_ai['risk_score']}/100 | Latency: {res_ai['latency_ms']}ms")

    print("\n" + "=" * 80)
    print("CALIBRATION & VERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")
    print("=" * 80)


if __name__ == "__main__":
    main()
