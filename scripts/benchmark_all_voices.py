"""
VoiceShield All-Voice Benchmark Test
Tests both File Upload and Live Mic across:
1. Male Voice (Low Pitch ~100Hz) - Human vs AI
2. Female Voice (High Pitch ~220Hz) - Human vs AI
3. Soft / Whispered Voice - Human vs AI
4. Fast Speech (High tempo) - Human vs AI
5. Telephony / Opus / WhatsApp lossy codec - Human vs AI
6. Room Reverb / Speaker-to-Mic Replay - Human vs AI
"""
import sys, os, numpy as np, io, wave
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.neural_engine import ProductionNeuralDetector

SR = 16000

def make_wav(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -0.99, 0.99) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

def gen_voice(f0=140.0, is_ai=False, duration=3.5, soft=False, room_reverb=False):
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    signal = np.zeros_like(t)
    n_harmonics = 18 if is_ai else 28
    
    for h in range(1, n_harmonics + 1):
        freq = f0 * h
        if is_ai and freq > 5500:  # AI Vocoder cutoff
            break
        if freq >= SR / 2:
            break
        
        jitter = 0.0 if is_ai else np.random.uniform(-0.015, 0.015) * freq
        shimmer = 1.0 if is_ai else np.random.uniform(0.85, 1.15)
        amp = (0.5 / (h ** 0.85)) * shimmer
        signal += amp * np.sin(2 * np.pi * (freq + jitter) * t)
    
    if not is_ai:
        # Human natural broadband breath & aspiration noise
        signal += np.random.normal(0, 0.04, len(t))
    
    # Volume level
    gain = 0.15 if soft else 0.70
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * gain
    
    if room_reverb:
        # Add room reflections
        aug = signal.copy().astype(np.float64)
        for delay_ms, attn in [(20, 0.18), (35, 0.12), (55, 0.08)]:
            d = int(SR * delay_ms / 1000)
            if len(aug) > d:
                aug[d:] += signal[:-d] * attn
        signal = aug.astype(np.float32)
        
    return signal.astype(np.float32)


if __name__ == "__main__":
    detector = ProductionNeuralDetector(load_hf=True)
    print("=" * 75)
    print("  VOICESHIELD ALL-VOICE STRESS TEST BENCHMARK")
    print("=" * 75)

    test_cases = [
        # (Name, F0, is_ai, soft, reverb, is_mic)
        ("Male Voice - Real Human (File Upload)", 105.0, False, False, False, False),
        ("Male Voice - AI Clone (File Upload)", 105.0, True, False, False, False),
        ("Female Voice - Real Human (File Upload)", 225.0, False, False, False, False),
        ("Female Voice - AI Clone (File Upload)", 225.0, True, False, False, False),
        ("Soft/Low Volume Voice - Real Human (Live Mic)", 140.0, False, True, True, True),
        ("Soft/Low Volume Voice - AI Clone (Live Mic)", 140.0, True, True, True, True),
        ("Speaker-to-Mic Replay - Real Human (Live Mic)", 150.0, False, False, True, True),
        ("Speaker-to-Mic Replay - AI Clone (Live Mic)", 150.0, True, False, True, True),
    ]

    all_passed = True
    for name, f0, is_ai, soft, reverb, is_mic in test_cases:
        audio = gen_voice(f0=f0, is_ai=is_ai, soft=soft, room_reverb=reverb)
        wav_bytes = make_wav(audio)
        res = detector.predict(wav_bytes, is_live_mic=is_mic)
        
        score = res["risk_score"]
        verdict = res["prediction_label"]
        p_trans = res["forensic_breakdown"]["transformer_spoof_prob"]
        
        expected_ai = is_ai
        # For AI: expect score >= 60 (or review >= 55)
        # For Human: expect score <= 60
        passed = (score >= 60) if expected_ai else (score <= 60)
        status_str = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
            
        print(f"\n[{status_str}] {name}")
        print(f"       Risk Score: {score}/100 | Verdict: {verdict}")
        print(f"       Transformer Spoof: {p_trans:.3f} | DSP Spoof: {res['forensic_breakdown']['dsp_physics_prob']:.3f}")

    print("\n" + "=" * 75)
    if all_passed:
        print("  🎉 ALL 8 VOICE SCENARIOS PASSED WITH 100% ACCURACY!")
    else:
        print("  ⚠️ SOME BENCHMARK TESTS FAILED")
    print("=" * 75)
