# 🎙️ VoiceShield 3-Minute SIH Jury Demo Script

> **Statutory Notice**:
> *“This is an experimental decision-support prototype; not identity proof. VoiceShield outputs advisory risk signals to assist human analysts.”*

---

## ⏱️ Exact 3-Minute Presentation Timeline

### 1. 0:00–0:20 | Problem Statement & The Threat Landscape
- *"Respected Jury Members, deepfake voice cloning is growing exponentially. Scammers clone an executive or family member's voice with just 3 seconds of audio to authorize fraudulent wire transfers and bypass verification."*
- *"Black-box AI models fail in security operations because they lack transparency. Today we present **VoiceShield**: an explainable, privacy-preserving voice authenticity and deepfake risk analysis system."*

### 2. 0:20–0:45 | Ingestion & Forensics Inspector
- *"Let us open the VoiceShield SOC Dashboard ([http://localhost:8502](http://localhost:8502))."*
- *"We upload a test call audio sample (`data/test/ai_voice/1.wav`). VoiceShield decodes the audio strictly in volatile RAM — zero raw audio bytes are ever written to disk or logs."*

### 3. 0:45–1:15 | Signal Diagnostics & 42 Feature Pipeline
- *"Within 40 milliseconds, VoiceShield's pipeline extracts 42 acoustic dimensions across formants, temporal prosody, pitch micro-jitter, RMS energy, and zero-crossing noise floor."*
- *"The Signal Diagnostics panel immediately displays the duration, sample rate (16 kHz), silence ratio (15.5%), and prosodic pitch variance ($F_0$)."*

### 4. 1:15–1:45 | Calibrated Risk Scoring & Risk Bands
- *"The model outputs a calibrated Spoof Probability of 0.62 and maps it to a **Risk Score of 62/100 (Review Required)**."*
- *"VoiceShield operates on a safety threshold of $t = 0.400$. It provides immediate, non-blocking guidance: 'Possible spoof-risk signal detected. Recommend out-of-band secondary verification via registered callback or passkey.' Zero automatic blocking is executed."*

### 5. 1:45–2:15 | Explainability (5 Canonical Groups) & Uncertainty
- *"In **Tab 2 (Forensic Explainability)**, we demystify the prediction using 5 canonical functional groups: Spectral Formants (75.4% share), Timing Dynamics (12.6%), Pitch Jitter (9.3%), Energy (2.7%), and Acoustic Quality."*
- *"If spoof probability falls in the borderline zone ($0.40 \le P \le 0.60$), the system flags **UNCERTAIN — MANUAL REVIEW REQUIRED**, preventing false automated conclusions."*

### 6. 2:15–2:40 | Sandbox Streaming Simulator (160ms / 40ms)
- *"In **Tab 4 (Streaming Simulator)**, we demonstrate real-time chunked stream evaluation. The audio is sliced into 160 ms analysis windows with 40 ms stride, calculating an Exponential Moving Average (EMA) rolling risk score in real time as the speaker talks."*

### 7. 2:40–3:00 | Limitations, Privacy & Responsible AI
- *"To conclude: VoiceShield is DPDP-aligned, uses a non-root Docker container, stores zero raw audio, and assists human analysts with transparent decision support. Thank you, and we welcome your questions."*
