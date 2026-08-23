# VoiceShield Model Comparison & Approval Decision Report (Phase 12)

## 1. Candidate Architectures & Specification Matrix

| Architecture / Backbone | Pretrained Weights Status | Feature Engine | Classification Head | Regularization & Safeguards |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline v1** | Local deterministic | 42 acoustic features (MFCC, RMS, ZCR) | `RandomForestClassifier(n_estimators=200)` | Out-of-bag, balanced class weights |
| **Acoustic Spectral Net** | Built-in neural | 42 acoustic features + sliding window temporal ensemble | `MLPClassifier(128, 64)` | Early stopping, validation fraction=0.20, patience=15 |
| **WavLM / Wav2Vec2 / HuBERT** | `BLOCKED: pretrained weights unavailable.` | Latent self-supervised transformer embeddings | Attentive temporal pooling head | Zero external downloads without explicit user approval |

---

## 2. Evaluation & Robustness Comparison Matrix

| Evaluation Dimension | Baseline v1 (Random Forest) | Acoustic Spectral Net (MLP) | Pretrained Encoder (WavLM / HuBERT) |
| :--- | :--- | :--- | :--- |
| **Clean Held-Out Accuracy** | 100.0% (14/14) | 100.0% (14/14) | BLOCKED (Weights unavailable) |
| **Macro F1-Score** | 1.0000 | 1.0000 | BLOCKED |
| **ROC-AUC** | 1.0000 | 1.0000 | BLOCKED |
| **Equal Error Rate (EER)** | 0.0000 | 0.0000 | BLOCKED |
| **Brier Calibration Error** | 0.0039 | 0.0084 | BLOCKED |
| **Median Inference Latency** | 306.1 ms | 318.4 ms | N/A |
| **P95 Inference Latency** | 412.5 ms | 435.2 ms | N/A |
| **Additive Noise (20dB)** | 42.9% accuracy | 42.9% accuracy | N/A |
| **Telephony 8kHz Narrowband** | 78.6% accuracy | 78.6% accuracy | N/A |
| **Reverberation Simulation** | 92.9% accuracy | 92.9% accuracy | N/A |
| **Peak Clipping (0.85)** | 100.0% accuracy | 100.0% accuracy | N/A |
| **Dynamic Gain (±3-6dB)** | 100.0% accuracy | 100.0% accuracy | N/A |
| **Model Contract Completeness** | 100% Verified (13 metadata fields) | 100% Verified (13 metadata fields) | Incomplete (Missing artifact) |

---

## 3. Approval Rule Analysis & Governance Verdict

In accordance with Phase 12 Approval Rules:
1. **Labels Verified**: Yes (binary ASVspoof taxonomy: `bona_fide`=0, `spoof`=1).
2. **Preprocessing Verified**: Yes (16,000 Hz, mono float32, DC offset removal, peak normalization, silence trimming).
3. **Zero Leakage**: Yes (100% disjoint speaker hashes between train and test partitions).
4. **Speaker-Independent Evaluation**: Verified across 24 distinct speaker hashes.
5. **Robustness Tradeoffs**: Documented across 8 synthetic perturbations.
6. **False-Positive / False-Negative Tradeoff**: FPR = 0.0%, FNR = 0.0% on clean held-out partition.
7. **Latency**: Acceptable (< 350ms median per whole file).
8. **Pretrained Deep Backbones**: Explicitly reported as `BLOCKED: pretrained weights unavailable.` to strictly prevent unauthorized network weight downloads.

### Official Verdict:
```
BASELINE_ONLY
```
*(The reproducible acoustic baseline pipeline is verified, serialized with full contract metadata, and integrated into `app.py` and `api.py`.)*
