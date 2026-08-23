# VoiceShield Label and Contract Audit Report

## 1. Class Taxonomy Specification

| Class ID | Internal Label | User-Facing Description | ASVspoof Protocol Alignment |
| :--- | :--- | :--- | :--- |
| **0** | `bona_fide` | **Likely Human Voice** | Genuine, uncompressed/natural human vocal tract speech |
| **1** | `spoof` | **Likely Spoof / AI Voice** | Synthetic TTS, voice clone, vocoder mel-inversion |

## 2. Model & Input Specifications

- **Expected Model Output Type**: `probability_distribution` (`sklearn.pipeline.Pipeline.predict_proba`) returning a 2-element array `[P(bona_fide), P(spoof)]`.
- **Expected Sample Rate**: `16000 Hz` (Resampled deterministically if native rate differs).
- **Expected Channels**: `1` (Mono channel, converted via stereo-mean if multi-channel).
- **Expected Feature Count**: `42` acoustic features:
  - `mfcc_mean_01` to `mfcc_mean_20`: First 20 MFCC coefficient means across all valid speech frames.
  - `mfcc_std_01` to `mfcc_std_20`: First 20 MFCC coefficient standard deviations across all valid speech frames.
  - `rms_energy_mean`: Global root-mean-square energy.
  - `zero_crossing_rate_mean`: Global zero crossing rate mean.
- **Feature Order**: Fixed deterministic concatenation `[mfcc_mean (20), mfcc_std (20), rms (1), zcr (1)]`.
- **Threshold Source**: Defined in `models/model_metadata.json` under key `optimal_decision_threshold` (default fallback `0.500`).

## 3. Contract Invariant Checks

1. `class_mapping` must strictly have `0 -> "bona_fide"` and `1 -> "spoof"`.
2. Class order in training labels must be identical to model probability output index:
   - Index 0 = `P(bona_fide)` (Genuine Human Probability)
   - Index 1 = `P(spoof)` (AI Spoof Probability)
3. Probabilities must satisfy $P(\text{bona\_fide}) + P(\text{spoof}) = 1.0 \pm 0.01$.
4. No ad-hoc heuristics or filename checks may override model probabilities.
