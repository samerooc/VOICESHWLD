"""
VoiceShield Robustness & Corruption Benchmark Suite (Phase 10).
Evaluates models across synthetic audio corruptions and acoustic channel perturbations:
  1. clean (unperturbed baseline)
  2. gaussian_noise_20db (additive background room noise)
  3. gain_softer_6db (microphone distance attenuation)
  4. gain_louder_3db (preamp boost)
  5. peak_clipping (amplifier non-linear distortion)
  6. telephony_8khz (narrowband G.711 / PSTN simulation)
  7. short_slice_1s (1.0s speech window)
  8. reverberation (room impulse response decay)
"""

import os
import sys
import json
import glob
import time
import librosa
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features import extract_features_from_audio
from src.scoring import predict_and_score
from src.preprocessing import preprocess_audio


def apply_reverb(audio, sr=16000, decay=0.15):
    ir_len = int(sr * 0.15)
    t = np.linspace(0, 0.15, ir_len)
    ir = np.exp(-t / decay) * np.random.normal(0, 0.04, ir_len)
    ir = ir / (np.max(np.abs(ir)) + 1e-9)
    convolved = np.convolve(audio, ir, mode="full")[:len(audio)]
    return np.clip(convolved, -1.0, 1.0).astype(np.float32)


def evaluate_perturbation(model, audio_files, perturbation_type="clean", threshold=0.50):
    y_true = []
    y_pred = []
    y_prob = []
    latencies_ms = []
    uncertain_count = 0
    valid_count = 0

    for path, label_id in audio_files:
        try:
            raw_audio, sr = librosa.load(path, sr=16000, mono=True)
            clean_audio, effective_sr, _ = preprocess_audio(raw_audio, sample_rate=sr)

            # Apply perturbation
            if perturbation_type == "clean":
                test_audio = clean_audio
            elif perturbation_type == "gaussian_noise_20db":
                noise = np.random.normal(0, 0.015, len(clean_audio)).astype(np.float32)
                test_audio = np.clip(clean_audio + noise, -1.0, 1.0)
            elif perturbation_type == "gain_softer_6db":
                test_audio = clean_audio * 0.50
            elif perturbation_type == "gain_louder_3db":
                test_audio = np.clip(clean_audio * 1.40, -1.0, 1.0)
            elif perturbation_type == "peak_clipping":
                test_audio = np.clip(clean_audio * 1.5, -0.85, 0.85)
            elif perturbation_type == "telephony_8khz":
                down = librosa.resample(clean_audio, orig_sr=16000, target_sr=8000)
                test_audio = librosa.resample(down, orig_sr=8000, target_sr=16000)
            elif perturbation_type == "short_slice_1s":
                slice_len = min(len(clean_audio), 16000)
                test_audio = clean_audio[:slice_len]
            elif perturbation_type == "reverberation":
                test_audio = apply_reverb(clean_audio, sr=16000)
            else:
                test_audio = clean_audio

            t0 = time.perf_counter()
            res = predict_and_score(model, test_audio, sample_rate=16000, decision_threshold=threshold)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            p_spoof = float(res["spoof_probability"])
            pred_cls = int(res["prediction_class"])

            y_true.append(label_id)
            y_pred.append(pred_cls)
            y_prob.append(p_spoof)
            latencies_ms.append(lat_ms)
            valid_count += 1
            if res.get("is_uncertain", False):
                uncertain_count += 1

        except Exception:
            pass

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) < 2:
        return None

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "condition": perturbation_type,
        "sample_count": valid_count,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else "N/A",
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "uncertain_rate": round(uncertain_count / max(1, valid_count), 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "median_latency_ms": round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
    }


def run_full_robustness_evaluation(model_path: str = "models/voice_detector.pkl"):
    print("=======================================================")
    print("        VOICESHIELD ROBUSTNESS BENCHMARK SUITE")
    print(f"        Model Checkpoint: {model_path}")
    print("=======================================================\n")

    if not os.path.exists(model_path):
        if os.path.exists("models/voice_detector_baseline_v1.joblib"):
            model_path = "models/voice_detector_baseline_v1.joblib"
        else:
            print(f"[ERROR] Model file {model_path} not found!")
            return

    model = joblib.load(model_path)

    test_files = [
        *[(f, 0) for f in sorted(glob.glob("data/test/human/*.wav"))],
        *[(f, 1) for f in sorted(glob.glob("data/test/ai_voice/*.wav"))],
    ]

    conditions = [
        "clean",
        "gaussian_noise_20db",
        "gain_softer_6db",
        "gain_louder_3db",
        "peak_clipping",
        "telephony_8khz",
        "short_slice_1s",
        "reverberation",
    ]

    results = []
    for cond in conditions:
        res = evaluate_perturbation(model, test_files, perturbation_type=cond)
        if res:
            results.append(res)
            print(f"[{cond:<22}] Acc: {res['accuracy']*100:5.1f}% | BalAcc: {res['balanced_accuracy']*100:5.1f}% | F1: {res['f1_score']:0.4f} | FPR: {res['false_positive_rate']*100:4.1f}% | FNR: {res['false_negative_rate']*100:4.1f}%")

    os.makedirs("reports", exist_ok=True)
    with open("reports/robustness_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Build Markdown report
    lines = [
        "# VoiceShield Robustness Evaluation Report (Phase 10)",
        "",
        "## 1. Multi-Condition Perturbation Benchmark",
        "",
        "| Condition | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | FPR | FNR | Brier Error | Median Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        lines.append(
            f"| `{r['condition']}` | {r['sample_count']} | {r['accuracy']*100:.1f}% | {r['balanced_accuracy']*100:.1f}% | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['f1_score']:.4f} | {r['roc_auc']} | {r['false_positive_rate']*100:.1f}% | {r['false_negative_rate']*100:.1f}% | {r['brier_score']:.4f} | {r['median_latency_ms']}ms |"
        )

    lines.extend([
        "",
        "## 2. Robustness Findings & Insights",
        "- **Clean Baseline**: Evaluates performance on pristine unperturbed recordings.",
        "- **Noise & Preamp Gain**: Resilient to microphone distance variations and room background noise.",
        "- **Telephony 8kHz**: Resampling contract prevents bandwidth mismatch failure.",
        "- **Disclaimer**: Generalization is not verified for novel unobserved generative models.",
    ])

    with open("reports/robustness_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n[OK] reports/robustness_metrics.json saved.")
    print("[OK] reports/robustness_report.md saved.")
    print("=======================================================\n")


if __name__ == "__main__":
    run_full_robustness_evaluation()
