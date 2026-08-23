# VoiceShield Model Output & Probability Conversion Report

## 1. Model Output Type Determination

VoiceShield uses a scikit-learn pipeline consisting of `StandardScaler` followed by `RandomForestClassifier` (or an ensemble of regularized tree classifiers).
- **Primary Method**: `model.predict_proba(X)`
- **Native Return Type**: 2D `np.ndarray` of shape `(n_samples, 2)`.
- **Classification Output**: Normalized probability estimates:
  - Column Index `0`: $P(\text{class}=0 \mid X) = P(\text{bona\_fide})$
  - Column Index `1`: $P(\text{class}=1 \mid X) = P(\text{spoof})$

## 2. Mathematical Conversion Path

Since the model natively outputs normalized class probabilities directly via `predict_proba`:
1. **NO double sigmoid**: We do NOT apply sigmoid ($\sigma(x)$) on top of already-normalized probabilities.
2. **NO redundant softmax**: We do NOT apply softmax on probabilities.
3. **Finite and Bound Checks**:
   - Every probability $p \in [0.0, 1.0]$.
   - $\sum p_i = 1.0 \pm 1e-4$.
4. **Dual Output Tracking**:
   - `raw_model_score`: Uncalibrated raw probability $P(\text{spoof})$ directly from the classifier.
   - `calibrated_probability`: Formally calibrated risk probability (using isotropic scaling or isotonic/Platt regression without heuristic overrides).
   - `risk_score`: Integer representation $\text{round}(p \times 100) \in [0, 100]$.
