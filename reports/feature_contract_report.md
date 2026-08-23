# VoiceShield Feature Contract Report

## 1. Feature Schema Overview (Version 1.0.0)

| Index Range | Feature Group | Description | Extraction Method |
| :--- | :--- | :--- | :--- |
| `00 - 19` | `mfcc_mean_01` to `mfcc_mean_20` | Spectral Formants & Harmonics means | `np.mean(librosa.feature.mfcc(y, sr, n_mfcc=20), axis=1)` |
| `20 - 39` | `mfcc_std_01` to `mfcc_std_20` | Pitch & Prosodic Phase Modulation stds | `np.std(librosa.feature.mfcc(y, sr, n_mfcc=20), axis=1)` |
| `40` | `rms_energy_mean` | Global Signal Root-Mean-Square Energy | `float(np.mean(librosa.feature.rms(y)))` |
| `41` | `zero_crossing_rate_mean` | Global Zero Crossing Rate (Fricative/Noise) | `float(np.mean(librosa.feature.zero_crossing_rate(y)))` |

- **Total Feature Count**: Fixed at exactly `42`.
- **Target Dimensions**: `(42,)` 1D array for single sample inference; `(N, 42)` for batch training.

## 2. Leakage and Contamination Policy

1. **NO Identifier Leakage**:
   - File path, filename, parent directory name, file creation timestamp, hash, and caller metadata are strictly isolated from the feature vector.
   - Only raw audio time-series samples are passed into `extract_features_from_audio`.
2. **Numeric Stability**:
   - All extracted feature arrays are sanitized via `np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)`.
   - Pipeline uses standard scaling `StandardScaler` fitted exclusively on training split data.
