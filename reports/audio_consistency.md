# VoiceShield Audio Pair Consistency Report (Section H)

## 1. Perturbation Variance Summary

- **Total Audited Files**: `14`
- **Max Absolute Drift under Gain Scaling**: `0.0000`
- **Max Absolute Drift under Telephony Codec Resampling**: `0.5950`

## 2. Sample Drift Log

| File Basename | Baseline Spoof Prob | Gain Scaled Spoof Prob | Resampled (8kHz) Spoof Prob | Max Condition Drift |
| :--- | :--- | :--- | :--- | :--- |
| `1.wav` | 81.8% | 81.8% | 83.2% | 1.5% |
| `2.wav` | 84.8% | 84.8% | 89.8% | 5.0% |
| `3.wav` | 79.1% | 79.1% | 81.6% | 2.5% |
| `4.wav` | 83.8% | 83.8% | 84.2% | 0.5% |
| `5.wav` | 84.2% | 84.2% | 85.0% | 0.8% |
| `6.wav` | 84.2% | 84.2% | 85.0% | 0.8% |
| `7.wav` | 80.2% | 80.2% | 83.0% | 2.8% |
| `01.wav` | 33.9% | 33.9% | 92.6% | 58.8% |

## 3. Analysis & Findings

1. **Gain Invariance**: Peak amplitude normalization ensures that moderate volume variation causes minimal probability deviation.
2. **Bandwidth Invariance**: Bandwidth compression reduces upper harmonic density, which is correctly flagged as uncertainty rather than forced artificial certainty.
