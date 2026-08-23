# VoiceShield Data Partitioning & Leakage-Safe Split Report (Phase 3)

## 1. Split Partition Summary

| Partition | Total Samples | Bona Fide Samples | Spoof Samples | Unique Speaker Clusters | Partition SHA-256 Hash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 10 | 5 | 5 | 10 | `8efb6e96e97ac41a...` |
| **Held-Out Test** | 14 | 7 | 7 | 14 | `55c215ee3594347a...` |
| **Total** | 24 | 12 | 12 | 24 | Verified Disjoint |

---

## 2. Frozen Evaluation Test Groups

| Test Group | Sample Count | Description |
| :--- | :--- | :--- |
| `in_domain_test` | 14 | Standard held-out benchmark recordings |
| `unseen_speaker_test` | 14 | Speakers strictly absent from training set |
| `unseen_source_test` | 14 | Acoustic source profiles absent from training set |
| `noisy_test` | 14 | Additive Gaussian room noise stress evaluation |
| `compressed_test` | 14 | Narrowband telephony (8kHz) & compression stress |
| `replay_test` | 0 | Labeled acoustic physical replay (0 in current partition) |
| `tts_test` | 7 | Synthetic text-to-speech / neural vocoder attacks |
| `voice_conversion_test` | 0 | Voice conversion attacks (0 in current partition) |

---

## 3. Leakage Prevention Rules Enforcement

1. **Speaker Disjointness**: Train and Test speaker hashes have **0 overlap** (verified).
2. **Recording Isolation**: No source recording or segment is shared across splits.
3. **Reproducibility**: Partitioning is fixed with seed `42`.
4. **Generalization Scope**:
   - Status: **GENERALIZATION_UNVERIFIED**
   - High test accuracy on this partition indicates consistency on the benchmark, but zero-shot generalization across unobserved commercial voice cloning architectures is not verified.
