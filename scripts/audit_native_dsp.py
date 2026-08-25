import os
import sys
import numpy as np
import torch
import soundfile as sf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.audio_processor import decode_and_sanitize_audio
from src.forensic_dsp import ForensicDSPAnalyzer
from src.lpc_physics import LPCPhysicsAnalyzer
from src.neural_model import VoiceShieldNeuralClassifier

def main():
    dsp_a = ForensicDSPAnalyzer(sr=16000)
    lpc_a = LPCPhysicsAnalyzer(order=16, sr=16000)
    nat_m = VoiceShieldNeuralClassifier(backbone_name="lightweight")
    ckpt = torch.load("models/voiceshield_neural_best.pt", map_location="cpu")
    nat_m.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    nat_m.eval()

    print("=" * 90)
    print("NATIVE ACOUSTIC NEURAL MODEL + DSP ACCURACY AUDIT")
    print("=" * 90)

    for d, name in [("data/test/human", "HUMAN"), ("data/test/ai_voice", "AI")]:
        print(f"\n--- {name} SAMPLES ---")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            raw, voiced, diag = decode_and_sanitize_audio(os.path.join(d, f))
            eval_a = voiced if len(voiced) > 8000 else raw
            res_nat = nat_m.predict_waveform(raw)
            p_nat = float(res_nat["spoof_probability"])
            dsp = dsp_a.extract_dsp_metrics(eval_a)
            lpc = lpc_a.extract_lpc_residual(eval_a)

            print(
                f"{f:<10} | Nat_Spoof: {p_nat:.4f} | "
                f"LPC_Anom: {lpc['lpc_anomaly_score']:.4f} | "
                f"GlottalRisk: {dsp['glottal_risk']:.4f} | "
                f"Jitter: {dsp['jitter_local']:.5f} | "
                f"HNR: {dsp['hnr_db']:>4.1f}dB | "
                f"SNR: {diag['snr_db']:>4.1f}dB"
            )

if __name__ == "__main__":
    main()
