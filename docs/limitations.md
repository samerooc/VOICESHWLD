# ⚠️ VoiceShield Operational Limitations & Ethical Boundaries

> **Statutory Notice**:
> *VoiceShield is an experimental decision-support prototype. It does not provide definitive legal, forensic, or biometric proof of identity.*

---

## 1. Key Operational Limitations

1. **Non-Causal Statistical Correlation**: Feature importances (MFCC formants, pitch jitter, spectral centroid) measure statistical correlation with known synthetic training markers. They do not constitute deterministic causal proof of deepfake generation.
2. **Adversarial & Perturbation Sensitivity**: Audio heavily distorted by extreme background noise or synthetic adversarial noise can affect model confidence. VoiceShield flags these cases with an `OUT-OF-DISTRIBUTION` banner rather than forcing a classification.
3. **Bandwidth-Limited Narrowband Signals**: 8 kHz telephony channels (G.711, AMR-NB) discard frequencies above 4 kHz. While VoiceShield analyzes available spectral cues, confidence is uncalibrated compared to studio 16 kHz audio.
4. **No Hardware Telecom Interception**: The platform does not directly interface with live cellular base stations, SS7/Diameter networks, or PBX SIP trunks. Live streaming is emulated via in-memory chunking in the Sandbox Simulator.
5. **No Automatic Action**: VoiceShield provides advisory signals only. It must never be configured to automatically disconnect calls or freeze user accounts without human review.
