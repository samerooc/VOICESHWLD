"""
VoiceShield: Validate that AI voice played through speaker into mic is detected.
Simulates: AI_audio → speaker_acoustic_degradation → microphone_re-capture
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.neural_engine import ProductionNeuralDetector
from src.audio_processor import decode_and_sanitize_audio
import io, wave

SR = 16000

def make_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def simulate_speaker_to_mic(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Simulate the acoustic degradation when AI audio is played through a
    speaker and re-captured by a microphone:
      1. Room multi-path reflections (early reverb)
      2. Additive microphone noise at ~22dB SNR
      3. Low-pass coloration (mic frequency response)
      4. Slight level drop
    """
    import random
    from scipy.signal import butter, lfilter
    aug = audio.copy().astype(np.float64)

    # 1. Room early reflections (2-4 paths, 15-50ms delays)
    for _ in range(3):
        delay = int(sr * random.uniform(0.015, 0.050))
        attn  = random.uniform(0.08, 0.20)
        if len(aug) > delay:
            aug[delay:] += audio[:-delay] * attn

    # 2. Microphone thermal noise (~22dB SNR)
    sig_power   = np.mean(aug**2) + 1e-9
    noise_power = sig_power / (10 ** (22.0 / 10.0))
    aug += np.random.normal(0, np.sqrt(noise_power), len(aug))

    # 3. Mic low-pass coloration (roll off above 6kHz)
    b, a = butter(2, 6000 / (sr / 2), btype="low")
    aug = lfilter(b, a, aug)

    # 4. Speaker level drop
    aug *= 0.75

    return aug.astype(np.float32)


def make_synthetic_ai_voice(duration_sec=4.0, sr=SR) -> np.ndarray:
    """
    Simulate what an AI vocoder (HiFi-GAN/FastSpeech2) sounds like:
    - Pitch-perfect F0 (zero jitter)
    - Harmonically pure signal (very high HNR)
    - NO energy above 5.5kHz (vocoder brickwall Nyquist at 11kHz)
    - Very regular LFCC trajectory
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    f0 = 150.0  # Perfect pitch, zero jitter

    # Pure harmonic series (like neural vocoder) - NO randomness
    signal = np.zeros_like(t)
    for h in range(1, 20):
        freq = f0 * h
        if freq > 5500:  # Hard brickwall at 5.5kHz
            break
        amp = 0.5 / h  # Clean harmonic decay
        signal += amp * np.sin(2 * np.pi * freq * t)

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.7
    return signal.astype(np.float32)


def make_real_human_voice(duration_sec=4.0, sr=SR) -> np.ndarray:
    """
    Simulate a real human voice:
    - F0 with random jitter ~1% perturbation
    - Energy above 5.5kHz (fricatives, aspiration)
    - Lower HNR (natural noise)
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    signal = np.zeros_like(t)

    # Add per-pitch-period jitter
    f0_base = 140.0
    for h in range(1, 30):
        jitter = np.random.uniform(-0.012, 0.012) * f0_base * h
        freq = f0_base * h + jitter
        amp  = 0.5 / h * np.random.uniform(0.85, 1.15)  # shimmer
        signal += amp * np.sin(2 * np.pi * freq * t)

    # Fricative noise above 5.5kHz (aspiration, /s/, /f/)
    noise_hf = np.random.normal(0, 0.06, len(t))
    signal   += noise_hf  # broad HF energy

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.65
    return signal.astype(np.float32)


if __name__ == "__main__":
    print("=" * 70)
    print("  VoiceShield: Speaker→Mic AI Voice Detection Validation")
    print("=" * 70)

    det = ProductionNeuralDetector(load_hf=False)

    # ── Test 1: Direct AI file upload ──────────────────────────────────────
    ai_raw = make_synthetic_ai_voice()
    wav    = make_wav_bytes(ai_raw)
    res    = det.predict(wav, is_live_mic=False)
    print(f"\n[FILE UPLOAD]  AI Synthetic Voice")
    print(f"  Risk Score : {res['risk_score']}/100  |  Band: {res['risk_band']}")
    print(f"  Verdict    : {res['prediction_label']}")
    print(f"  Transformer: {res['forensic_breakdown']['transformer_spoof_prob']:.3f}")
    assert res['risk_score'] >= 40, f"FAIL: AI file should score >= 40, got {res['risk_score']}"
    print(f"  ✅ PASS: AI correctly flagged")

    # ── Test 2: AI voice played through speaker → microphone ───────────────
    ai_via_mic = simulate_speaker_to_mic(ai_raw)
    wav_mic    = make_wav_bytes(ai_via_mic)
    res_mic    = det.predict(wav_mic, is_live_mic=True)
    print(f"\n[LIVE MIC]  AI Voice via Speaker → Microphone (room reverb + noise)")
    print(f"  Risk Score : {res_mic['risk_score']}/100  |  Band: {res_mic['risk_band']}")
    print(f"  Verdict    : {res_mic['prediction_label']}")
    print(f"  Transformer: {res_mic['forensic_breakdown']['transformer_spoof_prob']:.3f}")
    assert res_mic['risk_score'] >= 40, f"FAIL: Speaker-replayed AI should score >= 40, got {res_mic['risk_score']}"
    print(f"  ✅ PASS: Speaker-replayed AI correctly flagged")

    # ── Test 3: Real human voice via mic ───────────────────────────────────
    human_raw  = make_real_human_voice()
    human_mic  = simulate_speaker_to_mic(human_raw)  # same room condition
    wav_human  = make_wav_bytes(human_mic)
    res_human  = det.predict(wav_human, is_live_mic=True)
    print(f"\n[LIVE MIC]  Real Human Voice (with room acoustics)")
    print(f"  Risk Score : {res_human['risk_score']}/100  |  Band: {res_human['risk_band']}")
    print(f"  Verdict    : {res_human['prediction_label']}")
    print(f"  Transformer: {res_human['forensic_breakdown']['transformer_spoof_prob']:.3f}")
    assert res_human['risk_score'] <= 60, f"FAIL: Human voice should score <= 60, got {res_human['risk_score']}"
    print(f"  ✅ PASS: Real human correctly not flagged")

    print("\n" + "=" * 70)
    print("  ALL SPEAKER→MIC DETECTION TESTS PASSED ✅")
    print("=" * 70)
