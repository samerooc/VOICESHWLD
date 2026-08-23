# VoiceShield Feature Diagnostics Report (Section E)

## 1. Feature Representation Specification (Schema v1.0.0)

| Index Range | Feature Name Group | Dimension | Acoustic Property Captured |
| :--- | :--- | :--- | :--- |
| `0 - 19` | `mfcc_mean_01` to `mfcc_mean_20` | 20 | Spectral Envelope, Vocal Formants & Glottal Filter |
| `20 - 39` | `mfcc_std_01` to `mfcc_std_20` | 20 | Micro-Pitch Modulation, Jitter & Spectral Flux |
| `40` | `rms_energy` | 1 | Signal Power & Vocal Effort Dynamics |
| `41` | `zcr` | 1 | Zero-Crossing Rate & Transition Frictional Quality |
| **Total** | **42 Features** | **42** | **Full Physical Waveform Representation** |

## 2. Zero-Leakage Policy Enforcement

- **Filename Leakage**: `0%` (Paths and filenames are never ingested).
- **Directory / Split Leakage**: `0%` (Folder names are never parsed for features).
- **File Size Leakage**: `0%` (Raw file sizes are excluded; features computed on normalized windows).
- **Constant / Degenerate Features**: `0` constant features detected.
