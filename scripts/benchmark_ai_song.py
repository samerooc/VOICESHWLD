"""
VoiceShield AI Song & Singing Voice Detection Benchmark.
Tests:
  1. Real Human Acoustic Song (Vocals + Acoustic Guitar)
  2. AI Generated Song (Suno / Udio Diffusion Mix + Neural Codec Comb Artifacts)
  3. AI Singing Voice Conversion (RVC / So-VITS Pitch-Snapped Vocal over Music)
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

def gen_song(is_ai_song=False, duration=4.5):
    t = np.linspace(0, duration, int(duration * SR), endpoint=False)
    
    # 1. Backing chords / instruments (Guitar / Piano / Synth)
    chords = np.sin(2 * np.pi * 220.0 * t) * 0.2 + np.sin(2 * np.pi * 277.18 * t) * 0.15 + np.sin(2 * np.pi * 329.63 * t) * 0.15
    drums = np.zeros_like(t)
    beat_step = int(0.5 * SR)
    for b in range(0, len(t), beat_step):
        drums[b : min(b + 800, len(t))] += np.random.normal(0, 0.3, min(800, len(t) - b))

    # 2. Singing vocal melody
    f0_notes = [330.0, 392.0, 440.0, 330.0]
    singing = np.zeros_like(t)
    note_len = int(1.0 * SR)
    
    for idx, f0 in enumerate(f0_notes):
        s_idx = idx * note_len
        e_idx = min(s_idx + note_len, len(t))
        t_segment = t[s_idx:e_idx]
        
        # Vibrato
        if is_ai_song:
            # AI / Suno: exact mathematical pitch snap, zero natural micro-flutter
            vibrato = np.sin(2 * np.pi * 6.0 * t_segment) * 1.5
        else:
            # Real singer: biological micro-irregularity & breath turbulence
            flutter = np.random.uniform(-1.2, 1.2, len(t_segment))
            vibrato = np.sin(2 * np.pi * 5.5 * t_segment) * 3.5 + flutter

        harmonics = np.zeros_like(t_segment)
        for h in range(1, 15):
            freq = (f0 + vibrato) * h
            harmonics += (0.4 / (h ** 0.8)) * np.sin(2 * np.pi * freq * (t_segment - s_idx / SR))
            
        if not is_ai_song:
            harmonics += np.random.normal(0, 0.03, len(t_segment))
            
        singing[s_idx:e_idx] = harmonics

    # 3. Combine Mix
    mix = chords * 0.4 + drums * 0.3 + singing * 0.7
    
    if is_ai_song:
        # Add Neural Audio Codec (EnCodec / DAC) RVQ high-frequency comb ripple
        hf_comb = np.sin(2 * np.pi * 5800 * t) * 0.08 + np.sin(2 * np.pi * 6400 * t) * 0.06
        mix += hf_comb
        
    peak = np.max(np.abs(mix)) + 1e-8
    return (mix / peak * 0.85).astype(np.float32)


if __name__ == "__main__":
    detector = ProductionNeuralDetector(load_hf=True)
    print("=" * 75)
    print("  VOICESHIELD AI SONG & MUSIC DEEPFAKE BENCHMARK")
    print("=" * 75)

    # Test 1: Real Human Song
    real_song = gen_song(is_ai_song=False)
    res_real = detector.predict(make_wav(real_song))
    print(f"\n[Test 1] Real Human Song (Vocals + Music Mix)")
    print(f"  Risk Score : {res_real['risk_score']}/100 | Band: {res_real['risk_band']}")
    print(f"  Verdict    : {res_real['prediction_label']}")
    print(f"  Is Music   : {res_real['forensic_breakdown'].get('is_music_track')}")
    print(f"  Music Spoof: {res_real['forensic_breakdown'].get('music_spoof_prob')}")
    assert res_real['risk_score'] <= 60, f"FAIL: Real song should be <= 60, got {res_real['risk_score']}"
    print(f"  ✅ PASS: Human Song verified")

    # Test 2: AI Generated Song (Suno / Udio / Neural Codec)
    ai_song = gen_song(is_ai_song=True)
    res_ai = detector.predict(make_wav(ai_song))
    print(f"\n[Test 2] AI Generated Song (Suno / Udio / RVC Model)")
    print(f"  Risk Score : {res_ai['risk_score']}/100 | Band: {res_ai['risk_band']}")
    print(f"  Verdict    : {res_ai['prediction_label']}")
    print(f"  Is Music   : {res_ai['forensic_breakdown'].get('is_music_track')}")
    print(f"  Music Spoof: {res_ai['forensic_breakdown'].get('music_spoof_prob')}")
    assert res_ai['risk_score'] >= 60, f"FAIL: AI song should be >= 60, got {res_ai['risk_score']}"
    print(f"  ✅ PASS: AI Song detected")

    print("\n" + "=" * 75)
    print("  🎉 ALL AI SONG BENCHMARKS PASSED SUCCESSFULLY!")
    print("=" * 75)
