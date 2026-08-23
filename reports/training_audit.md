# VoiceShield Training & Dataset Forensic Audit Report (Phase 1)

## 1. Dataset Population Statistics

| Metric Category | Count / Value | Details |
| :--- | :--- | :--- |
| **Total Audio Files** | 24 | Complete research benchmark dataset |
| **Bona Fide Files** | 12 | 5 Train (`data/human/*.wav`), 7 Test (`data/test/human/*.wav`) |
| **Spoof Files** | 12 | 5 Train (`data/ai_voice/*.wav`), 7 Test (`data/test/ai_voice/*.wav`) |
| **Unique Speakers** | 24 | 10 Train (`spk_h_human_01..05`, `spk_ai_ai_01..05`), 14 Test (`spk_h_01..07`, `spk_ai_1..07`) |
| **Unique Sources / Generators** | 2 | `natural_voice` (consented human speech), `neural_vocoder` (synthetic neural vocoder) |
| **Containers & Codecs** | WAV (PCM_16) | Standard uncompressed linear PCM |
| **Raw Sample Rates** | 8,000 Hz & 48,000 Hz | 8,000 Hz (Spoof / telephony vocoders), 48,000 Hz (Bona fide studio recordings) |
| **Duration Range** | 3.968s – 51.670s | Train: 3.968s – 16.583s (mean 10.12s); Test: 12.313s – 51.670s (mean 27.11s) |
| **Dataset Licenses** | Research-Use-Permitted | Permitted for security benchmarking and algorithmic evaluation |
| **Consent Status** | documented_research | Curated research audio; zero unconsented or scraped audio |
| **Corrupt Files** | 0 | 100% files decoded and verified via soundfile/librosa |
| **Duplicate Files** | 0 | All SHA-256 file hashes are unique |

---

## 2. Integrity, Compliance & Leakage Audit

1. **Non-Negotiable Ethical Standards**:
   - **No Internet Scraping**: No audio was scraped from social media, YouTube, or private calls.
   - **No Unconsented Audio**: All recordings are approved for research benchmarking.
   - **No External Service Uploads**: All audio is processed purely in-memory and locally on-device.
   - **No Final Test Set Training**: The 14 held-out test files are frozen and strictly excluded from training, validation, or threshold tuning.
   - **No Speaker or Recording Overlap**: Train and test speaker hashes are 100% disjoint.

2. **Sample Rate Disparity Mitigation**:
   - Spoof recordings are 8 kHz while genuine recordings are 48 kHz.
   - **Mandatory Preprocessing Contract**: All audio is resampled to **16,000 Hz** prior to feature extraction. Classifiers never receive raw sample rate metadata as an input feature.

3. **Generalization Scope**:
   - **GENERALIZATION_UNVERIFIED**: While speaker disjointness is verified within the dataset partition, zero-shot generalization across unobserved commercial TTS architectures or high-loss mobile telecom codecs cannot be claimed.
