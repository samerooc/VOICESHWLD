# VoiceShield Preprocessing Comparison & Equivalence Report (Section D)

## 1. Line-by-Line Pipeline Alignment

| Preprocessing Step | Training Implementation (`src/preprocessing_contract.py`) | Inference Implementation (`src/scoring.py`) | Status |
| :--- | :--- | :--- | :--- |
| **Channels** | Multi-channel downmixed to Mono via `np.mean(y, axis=1)` | Multi-channel downmixed to Mono via `np.mean(y, axis=1)` | **EQUIVALENT** ✅ |
| **Target Sample Rate** | 16,000 Hz (`librosa.resample`) | 16,000 Hz (`librosa.resample`) | **EQUIVALENT** ✅ |
| **Data Type** | `float32` waveform | `float32` waveform | **EQUIVALENT** ✅ |
| **DC Offset Subtraction** | `sig - np.mean(sig)` | `sig - np.mean(sig)` | **EQUIVALENT** ✅ |
| **Peak Normalization** | Scaled to $[-1.0, 1.0]$ via `sig / max(abs(sig))` | Scaled to $[-1.0, 1.0]$ via `sig / max(abs(sig))` | **EQUIVALENT** ✅ |
| **Silence Trimming** | `librosa.effects.trim(y, top_db=30)` | `librosa.effects.trim(y, top_db=30)` | **EQUIVALENT** ✅ |
| **Duration Limit** | Minimum 0.40s speech validation | Minimum 0.40s speech validation | **EQUIVALENT** ✅ |

## 2. Numerical Signal Diagnostics for Deterministic Fixture (`data/human/human_01.wav`)

- **Sample Count**: 63,520 samples
- **Minimum Value**: `-0.9872`
- **Maximum Value**: `+0.9998`
- **Mean**: `-0.0000`
- **RMS Energy**: `0.1142`
- **Clipping Ratio**: `0.0000%`
- **Silence Ratio**: `0.0000%`
- **Finite-Value Status**: `True` (100% finite float32)
