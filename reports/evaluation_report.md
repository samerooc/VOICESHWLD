# VoiceShield Independent Evaluation Report (Phase 10)

## 1. Frozen Test Group Performance

| Test Group | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | EER | FPR | FNR | Brier | Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean_audio (Held-Out In-Domain)` | 32 | 6.2% | N/A (Single class) | 0.0% | 0.0% | 0.0000 | N/A (Single class) | N/A (Single class) | 93.8% | 0.0% | N/A | 191.68ms |
| `unseen_speakers` | 32 | 6.2% | N/A (Single class) | 0.0% | 0.0% | 0.0000 | N/A (Single class) | N/A (Single class) | 93.8% | 0.0% | N/A | 201.31ms |
| `unseen_sources` | 32 | 6.2% | N/A (Single class) | 0.0% | 0.0% | 0.0000 | N/A (Single class) | N/A (Single class) | 93.8% | 0.0% | N/A | 188.18ms |
| `text_to_speech` | 30 | 0.0% | N/A (Single class) | 0.0% | 0.0% | 0.0000 | N/A (Single class) | N/A (Single class) | 100.0% | 0.0% | N/A | 178.87ms |
| `voice_conversion` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `replay` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 2. Confusion Matrix (Clean Held-Out Test)

| | Predicted Bona Fide (0) | Predicted Spoof (1) |
| :--- | :--- | :--- |
| **Actual Bona Fide (0)** | 2 | 30 |
| **Actual Spoof (1)** | 0 | 0 |

## 3. Mandatory Governance & Generalization Notice
- **Status**: `GENERALIZATION_UNVERIFIED`
- **Notice**: High performance on small held-out research partitions is not proof of production-level zero-shot generalization across unobserved commercial speech synthesisers or hostile telephony channels.