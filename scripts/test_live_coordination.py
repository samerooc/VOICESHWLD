import os
import sys
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.neural_engine import ProductionNeuralDetector

def simulate_live_mic(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Simulates live room acoustics: mild room reverb, ambient noise, and mic gain."""
    # 1. Add ambient acoustic room noise (SNR ~25 dB)
    noise = np.random.normal(0, 0.005, len(audio)).astype(np.float32)
    # 2. Simple acoustic reflection / reverb
    reverb_delay = int(sr * 0.035)  # 35ms reflection
    reverb = np.zeros_like(audio)
    if len(audio) > reverb_delay:
        reverb[reverb_delay:] = audio[:-reverb_delay] * 0.18
    # 3. Combine and normalize
    mic_audio = audio + reverb + noise
    max_amp = np.max(np.abs(mic_audio))
    if max_amp > 1e-6:
        mic_audio = (mic_audio / max_amp) * 0.85
    return mic_audio.astype(np.float32)

def main():
    print("=" * 80)
    print("TESTING ENHANCED MULTI-TIER MODEL COORDINATION ACROSS LIVE & CLEAN AUDIO")
    print("=" * 80)

    detector = ProductionNeuralDetector()

    human_files = [f"data/test/human/{f}" for f in sorted(os.listdir("data/test/human")) if f.endswith(".wav")]
    ai_files = [f"data/test/ai_voice/{f}" for f in sorted(os.listdir("data/test/ai_voice")) if f.endswith(".wav")]

    print("\n--- 1. CLEAN BENCHMARK EVALUATION ---")
    for f in human_files:
        res = detector.predict(f)
        print(f"Human clean: {os.path.basename(f):<10} -> Score: {res['risk_score']:>2}/100 | Band: {res['risk_band']}")
    
    for f in ai_files:
        res = detector.predict(f)
        print(f"AI clean:    {os.path.basename(f):<10} -> Score: {res['risk_score']:>2}/100 | Band: {res['risk_band']}")

    print("\n--- 2. SIMULATED LIVE MICROPHONE EVALUATION (Room Reverb + Ambient Noise) ---")
    for f in human_files:
        audio, sr = sf.read(f)
        mic_audio = simulate_live_mic(audio, sr=sr)
        res = detector.predict(mic_audio, sample_rate=sr, is_live_mic=True)
        print(f"Human [Live Mic]: {os.path.basename(f):<10} -> Score: {res['risk_score']:>2}/100 | Band: {res['risk_band']}")

    for f in ai_files:
        audio, sr = sf.read(f)
        mic_audio = simulate_live_mic(audio, sr=sr)
        res = detector.predict(mic_audio, sample_rate=sr, is_live_mic=True)
        print(f"AI [Live Mic]:    {os.path.basename(f):<10} -> Score: {res['risk_score']:>2}/100 | Band: {res['risk_band']}")

if __name__ == "__main__":
    main()
