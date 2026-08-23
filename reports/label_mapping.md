# VoiceShield Label Mapping Forensic Report (Section B)

## 1. Class Mapping Definition

- **Class `0`**: `bona_fide` (**Likely Human Voice**) — Natural biological speech produced via physiological vocal tracts.
- **Class `1`**: `spoof` (**Likely Spoof / AI Voice**) — Synthetic speech generated via neural vocoders, voice conversion, or TTS.

## 2. End-to-End Alignment Audit

| System Component | File Path | Class 0 Definition | Class 1 Definition | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Model Contract** | `src/model_contract.py` | `0: bona_fide` | `1: spoof` | **MATCH** ✅ |
| **Dataset Manifest** | `data/manifest.csv` | `bona_fide` | `spoof` | **MATCH** ✅ |
| **Training Pipeline** | `src/train_baseline.py` | `0: bona_fide` | `1: spoof` | **MATCH** ✅ |
| **Saved Model Checkpoint**| `models/model_metadata.json` | `0: bona_fide` | `1: spoof` | **MATCH** ✅ |
| **Inference & Scoring** | `src/scoring.py` | `0: bona_fide` | `1: spoof` | **MATCH** ✅ |
| **REST API** | `api.py` | `0: bona_fide` | `1: spoof` | **MATCH** ✅ |
| **Dashboard UI** | `app.py` | `0: Likely Human Voice` | `1: Likely Spoof / AI Voice` | **MATCH** ✅ |

## 3. Probability Directionality

- $P(\text{bona\_fide}) = P(y = 0) \in [0.0, 1.0]$
- $P(\text{spoof}) = P(y = 1) \in [0.0, 1.0]$
- $P(\text{bona\_fide}) + P(\text{spoof}) = 1.0$ (Strictly normalized, non-inverted).
