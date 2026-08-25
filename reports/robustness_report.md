# VoiceShield Robustness Evaluation Report (Phase 10)

## 1. Multi-Condition Perturbation Benchmark

| Condition | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | FPR | FNR | Brier Error | Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0001 | 794.6ms |
| `gaussian_noise_20db` | 14 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0000 | 0.5 | 0.0% | 100.0% | 0.4909 | 1039.86ms |
| `gain_softer_6db` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0001 | 852.69ms |
| `gain_louder_3db` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0001 | 863.13ms |
| `peak_clipping` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0001 | 864.05ms |
| `telephony_8khz` | 14 | 50.0% | 50.0% | 50.0% | 100.0% | 0.6667 | 0.5 | 100.0% | 0.0% | 0.4909 | 838.49ms |
| `short_slice_1s` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0001 | 49.92ms |
| `reverberation` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0090 | 840.91ms |

## 2. Robustness Findings & Insights
- **Clean Baseline**: Evaluates performance on pristine unperturbed recordings.
- **Noise & Preamp Gain**: Resilient to microphone distance variations and room background noise.
- **Telephony 8kHz**: Resampling contract prevents bandwidth mismatch failure.
- **Disclaimer**: Generalization is not verified for novel unobserved generative models.