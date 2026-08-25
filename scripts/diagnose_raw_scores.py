import os
import sys
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.neural_engine import ProductionNeuralDetector

def main():
    det = ProductionNeuralDetector()
    print("\n" + "=" * 90)
    print("DETAILED COMPONENT SCORE AUDIT (Human vs AI Clone)")
    print("=" * 90)

    for d, name in [("data/test/human", "HUMAN"), ("data/test/ai_voice", "AI")]:
        print(f"\n--- {name} UTTERANCES ---")
        for f in sorted(os.listdir(d)):
            if not f.endswith(".wav"):
                continue
            path = os.path.join(d, f)
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # 1. HF Transformer raw logits
            win_samples = min(len(audio), 48000)
            inputs = det.feature_extractor(audio[:win_samples], sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(det.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = det.model(**inputs).logits
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

            # 2. Native Model
            res_nat = det.native_model.predict_waveform(audio) if det.native_model else {"spoof_probability": 0.5}

            # 3. DSP Metrics
            dsp = det.dsp_analyzer.extract_dsp_metrics(audio, sr=16000)
            lpc = det.lpc_analyzer.extract_lpc_residual(audio, sr=16000)

            print(
                f"{f:<10} | HF_0: {probs[0]:.4f} | HF_1: {probs[1]:.4f} | "
                f"Nat_Spoof: {res_nat['spoof_probability']:.4f} | "
                f"LPC_Anom: {lpc['lpc_anomaly_score']:.4f} | "
                f"Jitter: {dsp['jitter_local']:.5f} | "
                f"GlottalHuman: {str(dsp['is_human_glottal']):<5} | "
                f"Cutoff: {dsp['hf_cutoff_ratio']:.6f}"
            )

if __name__ == "__main__":
    main()
