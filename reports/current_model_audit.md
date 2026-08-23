# VoiceShield Current Model Audit Report (Phase 1)

## 1. Executive Summary & Model Artifact Specifications

| Parameter | Baseline Specification | Audit Verification Status |
| :--- | :--- | :--- |
| **Model Artifact** | `models/voice_detector.pkl` / `models/voice_detector_baseline_v1.joblib` | Verified |
| **Model Architecture** | `StandardScaler` + `RandomForestClassifier(n_estimators=200, random_state=42)` | Verified |
| **Feature Schema Version** | `1.0.0` (42 acoustic features: 20 MFCC means, 20 MFCC stds, 1 RMS, 1 ZCR) | Verified |
| **Preprocessing Contract** | Mono downmix, 16,000 Hz resampling, DC removal, peak normalization, silence trimming | Verified |
| **Class 0 Mapping** | `bona_fide` (Natural human speech) | Verified |
| **Class 1 Mapping** | `spoof` (Synthetic / cloned speech) | Verified |
| **Output Type** | Direct probability distribution $P \in [0.0, 1.0]$, $\sum P = 1.0$ | Verified |
| **Decision Threshold** | `0.500` (Tuned strictly on validation CV partition) | Verified |
| **Speaker Independence** | 100% disjoint speaker hashes across train and test partitions | Verified |
| **Feature / Metadata Leakage** | Zero metadata, filepath, or label leakage into features | Verified |

---

## 2. Limitations of Current Baseline

1. **Spectral Summary Compression**: 42 summary statistics condense temporal modulation and vocoder phase anomalies.
2. **Telephony Bandwidth Vulnerability**: 8kHz narrowband audio causes slight spectral flattening.
3. **Pretrained Backbone Analysis**:
   - Pretrained self-supervised audio encoders (e.g. WavLM, Wav2Vec2, HuBERT) can extract rich temporal representations.
   - When external weights are not pre-cached locally on disk, the system reports `BLOCKED: pretrained weights unavailable.` to strictly prevent unauthorized background downloads.
