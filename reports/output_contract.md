# VoiceShield Model Output Contract Report (Section C)

## 1. Mathematical Output Representation

The VoiceShield classifier output is formally defined as a **2-element normalized probability distribution vector**:
$$\mathbf{p} = [p_{\text{bona\_fide}}, p_{\text{spoof}}] \in [0.0, 1.0]^2$$
Subject to the constraint:
$$p_{\text{bona\_fide}} + p_{\text{spoof}} = 1.0 \pm 10^{-5}$$

## 2. Mathematical Integrity Properties

- **Range Enforcement**: Explicitly bounded in $[0.0, 1.0]$.
- **Sigmoid / Softmax Invariance**: Output is obtained directly from calibrated probability estimators. No double-sigmoid or redundant logit conversions.
- **Finite Value Verification**: Zero NaN, infinite, or complex numbers permitted.
- **Batch vs Single**: Batch and single-sample inference yield mathematically identical results ($L_\infty < 10^{-7}$).
