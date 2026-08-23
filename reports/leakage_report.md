# VoiceShield Dataset Leakage & Partition Audit Report (Section F)

## 1. Forensic Contamination Audit

| Leakage Category | Audit Finding | Risk Assessment | Mitigation Applied |
| :--- | :--- | :--- | :--- |
| **Duplicate Files Across Splits** | 0 files overlap between train and test | None | Frozen disjoint partitions |
| **Speaker ID Overlap** | 0 speaker cluster hashes overlap | None | Disjoint cluster assignment |
| **Raw Sample Rate Disparity** | Raw Bona Fide = 48kHz, Raw Spoof = 8kHz | High (if unnormalized) | Mandatory 16,000 Hz resampling contract before feature extraction |
| **Recording Duration Bias** | Mean Train: 16.7s, Mean Test: 22.4s | Low | Multi-segment fixed 2.5s sliding window extraction |
| **Loudness / RMS Bias** | Mean Bona Fide: 0.12, Mean Spoof: 0.09 | Low | Peak amplitude normalization to $[-1.0, 1.0]$ |
| **Filename / Path Ingestion** | Zero filepath parsing in feature extraction | None | Features accept numpy waveform arrays only |

## 2. Generalization Scope

**GENERALIZATION NOT VERIFIED FOR NOVEL ARCHITECTURES OUTSIDE BENCHMARK**
- Performance on small curated partitions does not prove zero-shot transfer across unobserved accents or future neural vocoder models.
