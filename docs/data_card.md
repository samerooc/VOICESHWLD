# 📊 VoiceShield Data Card & Dataset Governance

> **Statutory Notice**:
> *VoiceShield is trained exclusively on public, synthetic, and consent-verified voice recordings. Zero private, non-consensual biometric audio is ingested or distributed.*

---

## 1. Dataset Overview

| Metadata Field | Specification |
| :--- | :--- |
| **Dataset Title** | VoiceShield Baseline Dataset (Manifest v1.0) |
| **Primary Data Source** | Open-source public acoustic benchmarks & synthetic vocoder outputs (VITS, Tacotron2). |
| **Audio Format** | Uncompressed PCM WAV ($16\text{ kHz}$ mono, 16-bit). |
| **Total Audio Files** | 24 audio tracks ($10\text{ train/validation partition}, 14\text{ held-out test split}$). |
| **Ground-Truth Labels** | `bona_fide` (0 / Human Voice), `spoof` (1 / Synthetic AI Voice). |
| **Sha256 Integrity Hash** | `8efb6e96e97ac41a910e7bfb73cfa49d33cdc1ab5280a0b68c5246512eb856c8` |

---

## 2. Preprocessing & Leakage Prevention

1. **Standardization**: All input signals normalized to $16,000\text{ Hz}$ single-channel mono.
2. **Feature Extraction Pipeline**: 42 acoustic dimensions computed per sample (20 MFCC Means, 20 MFCC Stds, RMS Energy, Zero Crossing Rate).
3. **Partition Isolation**:
   - `train` ($70\%$ of dev partition): Model parameter fitting.
   - `validation` ($30\%$ of dev partition): Hyperparameter tuning & decision threshold calibration ($t = 0.400$).
   - `test` ($14\text{ samples}$): Held-out evaluation split, completely isolated from feature scaler fitting.

---

## 3. Known Demographic & Acoustic Gaps

- **Language Bias**: Primarily English and Hindi speech samples; limited low-resource regional dialects.
- **Acoustic Compression**: Modern cellular codecs (VoLTE, EVS) exhibit slightly different high-frequency roll-off characteristics than uncompressed studio WAVs.
- **Age Distribution**: Adult speakers ($20–60\text{ years}$); children and elderly speakers are currently underrepresented.

---

## 4. Retention & Deletion Policy

- **Zero Raw Audio Storage**: Uploaded test audio is processed entirely in RAM and deleted immediately upon request completion.
- **Manifest Provenance**: Tracked in [data/manifest.csv](file:///C:/Users/OMEN/voice-clone-detector/data/manifest.csv) and [reports/dataset_report.json](file:///C:/Users/OMEN/voice-clone-detector/reports/dataset_report.json).
