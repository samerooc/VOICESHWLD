# VoiceShield Independent Evaluation Report (Phase 10)

## 1. Frozen Test Group Performance

| Test Group | Samples | Accuracy | Bal. Acc | Precision | Recall | F1 Score | ROC-AUC | EER | FPR | FNR | Brier | Median Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean_audio (Held-Out In-Domain)` | 14 | 100.0% | 1.0 | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0 | 0.0% | 0.0% | 0.0856 | 258.92ms |
| `unseen_speakers` | 14 | 100.0% | 1.0 | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0 | 0.0% | 0.0% | 0.0856 | 248.1ms |
| `unseen_sources` | 14 | 100.0% | 1.0 | 100.0% | 100.0% | 1.0000 | 1.0 | 0.0 | 0.0% | 0.0% | 0.0856 | 241.78ms |
| `text_to_speech` | 7 | 100.0% | N/A (Single class) | 100.0% | 100.0% | 1.0000 | N/A (Single class) | N/A (Single class) | 0.0% | 0.0% | N/A | 337.51ms |
| `voice_conversion` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `replay` | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## 2. Confusion Matrix (Clean Held-Out Test)

| | Predicted Bona Fide (0) | Predicted Spoof (1) |
| :--- | :--- | :--- |
| **Actual Bona Fide (0)** | 7 | 0 |
| **Actual Spoof (1)** | 0 | 7 |

## 3. Mandatory Governance & Generalization Notice
- **Status**: `GENERALIZATION_UNVERIFIED`
- **Notice**: High performance on small held-out research partitions is not proof of production-level zero-shot generalization across unobserved commercial speech synthesisers or hostile telephony channels.