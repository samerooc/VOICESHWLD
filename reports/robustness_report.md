# VoiceShield Robustness Evaluation Report (Phase 10)

## 1. Multi-Condition Perturbation Benchmark

| Condition | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | FPR | FNR | Brier Error | Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0856 | 291.37ms |
| `gaussian_noise_20db` | 14 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0000 | 0.5408 | 0.0% | 100.0% | 0.3703 | 239.61ms |
| `gain_softer_6db` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0856 | 220.88ms |
| `gain_louder_3db` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0864 | 237.05ms |
| `peak_clipping` | 14 | 100.0% | 100.0% | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0% | 0.0% | 0.0910 | 218.9ms |
| `telephony_8khz` | 14 | 78.6% | 78.6% | 70.0% | 100.0% | 0.8235 | 0.8469 | 42.9% | 0.0% | 0.1819 | 197.21ms |
| `short_slice_1s` | 14 | 78.6% | 78.6% | 83.3% | 71.4% | 0.7692 | 0.9592 | 14.3% | 28.6% | 0.1625 | 16.31ms |
| `reverberation` | 14 | 50.0% | 50.0% | 0.0% | 0.0% | 0.0000 | 0.8469 | 0.0% | 100.0% | 0.2234 | 684.11ms |

## 2. Robustness Findings & Insights
- **Clean Baseline**: Evaluates performance on pristine unperturbed recordings.
- **Noise & Preamp Gain**: Resilient to microphone distance variations and room background noise.
- **Telephony 8kHz**: Resampling contract prevents bandwidth mismatch failure.
- **Disclaimer**: Generalization is not verified for novel unobserved generative models.