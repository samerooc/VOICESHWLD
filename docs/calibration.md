# VoiceShield Calibration & Risk State Documentation (Phase 10)

## 1. Probability Calibration

VoiceShield applies temperature-scaled logits and validation threshold tuning:
$$\hat{p} = \sigma\left(\frac{\text{logit}(p)}{T}\right)$$
Where $T$ is tuned on the held-out validation split.

## 2. Standard 5-State Risk Taxonomy

| Risk State | Condition / Range | Badge / Visual | Operational Advisory Action |
| :--- | :--- | :--- | :--- |
| **`low`** | Score `0 - 25` | 🟢 Green | Normal workflow. Authentic human speech pattern. |
| **`review`** | Score `26 - 65` | 🟡 Yellow | Advisory alert. Acoustic anomalies require secondary check. |
| **`high`** | Score `66 - 100` | 🔴 Red | High risk. Elevated synthetic/cloned pattern detected. |
| **`uncertain`** | Prob `0.40 - 0.60` | ⚠️ Yellow | Insufficient evidence. Do not force automatic conclusion. |
| **`low_quality`** | Degraded / Faint / Clipped | ⚠️ Yellow | Request clean re-recording under quiet conditions. |
