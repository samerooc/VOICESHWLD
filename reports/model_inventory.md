# VoiceShield Model Inventory & Artifact Registry

## 1. Registered Model Artifacts

| Model Name / Identifier | Artifact Path | Size (Bytes) | SHA-256 Hash | Framework | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline_v1`** | `models/voice_detector_baseline_v1.joblib` | 94,266 | `b2a7616d5fdf407b5dd6bbc9d40aeb2dc04e12e5ba3b1da579ecee28376b9609` | scikit-learn (Pipeline: StandardScaler + RandomForest) | **Active Baseline** |
| **`voice_detector_current`** | `models/voice_detector.pkl` | 94,266 | `b2a7616d5fdf407b5dd6bbc9d40aeb2dc04e12e5ba3b1da579ecee28376b9609` | scikit-learn (Pipeline: StandardScaler + RandomForest) | **Active Production** |

## 2. Model Metadata & Lineage

- **Feature Schema Version**: `1.0.0` (42 acoustic features: 20 MFCC means, 20 MFCC stds, 1 RMS energy, 1 Zero Crossing Rate)
- **Preprocessing Contract Version**: `1.0.0` (Target sample rate 16kHz, mono conversion, DC offset subtraction, peak normalization, silence trimming)
- **Class Mapping Contract**: `0 -> bona_fide (Likely Human Voice)`, `1 -> spoof (Likely Spoof / AI Voice)`
- **Decision Threshold**: `0.500`
- **Training Dataset Hash**: `8efb6e96e97ac41a910e7bfb73cfa49d33cdc1ab5280a0b68c5246512eb856c8`
- **License / Consent Policy**: Training dataset audio is verified for research prototyping. Model artifact requires verification prior to deployment.
- **Safety Policy**: Does NOT perform automated blocking, voice cloning, or identity proofing.
