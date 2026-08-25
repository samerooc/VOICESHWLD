import os
import sys
import numpy as np
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import decode_and_sanitize_audio
from src.forensic_dsp import ForensicDSPAnalyzer
from src.lpc_physics import LPCPhysicsAnalyzer

def main():
    dsp_a = ForensicDSPAnalyzer(sr=16000)
    lpc_a = LPCPhysicsAnalyzer(order=16, sr=16000)

    print("=" * 90)
    print("FORENSIC PHYSICS AUDIT: F0 STD, LFCC VAR, LPC KURTOSIS")
    print("=" * 90)

    for d, name in [("data/test/human", "HUMAN"), ("data/test/ai_voice", "AI")]:
        print(f"\n--- {name} UTTERANCES ---")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            raw, voiced, diag = decode_and_sanitize_audio(os.path.join(d, f))
            eval_a = voiced if len(voiced) > 8000 else raw
            dsp = dsp_a.extract_dsp_metrics(eval_a)
            lpc = lpc_a.extract_lpc_residual(eval_a)

            print(
                f"{f:<10} | F0_std: {dsp['f0_std']:>5.1f}Hz | "
                f"LFCC_var: {dsp['lfcc_variance']:>6.3f} | "
                f"Kurtosis: {lpc['residual_kurtosis']:>7.1f} | "
                f"Jitter: {dsp['jitter_local']:.5f} | "
                f"Shimmer: {dsp['shimmer_local']:.5f}"
            )

if __name__ == "__main__":
    main()
