# VoiceShield Current Baseline Benchmark Metrics (Section 1)

## 1. Frozen Benchmark Performance

- **Evaluated Checkpoint**: `models/voice_detector.pkl`
- **Total Test Samples**: 14 (7 Bona Fide, 7 Spoof)
- **Accuracy**: `100.00%`
- **Macro F1-Score**: `1.0000`
- **ROC-AUC**: `1.0000`
- **False Positive Rate**: `0.0%`
- **False Negative Rate**: `0.0%`
- **Brier Calibration Error**: `0.0210`

## 2. Multi-Condition Stress Matrix

| Condition | Accuracy | Macro F1 | FPR | FNR | Robustness Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`clean`** | 100.0% | 1.0000 | 0.0% | 0.0% | Stable |
| **`gain_louder_3db`** | 100.0% | 1.0000 | 0.0% | 0.0% | Stable |
| **`gain_softer_6db`** | 100.0% | 1.0000 | 0.0% | 0.0% | Stable |
| **`peak_clipping`** | 100.0% | 1.0000 | 0.0% | 0.0% | Stable |
| **`reverberation`** | 100.0% | 1.0000 | 0.0% | 0.0% | Stable |
| **`short_slice_1s`** | 85.7% | 0.8750 | 28.6% | 0.0% | Flagged as Uncertainty |
| **`telephony_8khz`** | 50.0% | 0.6667 | 100.0% | 0.0% | Bandwidth Limited |
| **`gaussian_noise_20db`**| 42.9% | 0.0000 | 14.3% | 100.0% | Flagged as Low Quality |
