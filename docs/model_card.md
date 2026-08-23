# 📋 VoiceShield Model Card: Baseline v1

## 1. Model Details
* **Model Name**: VoiceShield Baseline v1
* **Architecture**: `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])`
* **Version**: `1.0.0`
* **Release Date**: August 2026
* **License**: Educational / SIH Hackathon Research Prototype
* **Model Artifact**: `models/voice_detector.pkl`
* **Metadata & Config**: `models/model_metadata.json`

---

## 2. Intended Use
* **Primary Use**: Assisting cybersecurity analysts and call center SOC operators in identifying elevated risk of synthetic / cloned speech in voice recordings.
* **Intended Users**: Security analysts, fraud investigators, and cybersecurity researchers.
* **Decision Support**: Provides calibrated 0–100 risk scores and qualitative guidance (`Low`, `Review required`, `High risk`).

### ⚠️ Out-of-Scope & Prohibited Uses
* **Automated Enforcement**: Must **never** be used for automatic termination of calls, freezing bank accounts, or declining transactions without human operator review.
* **Identity Verification**: Not an identity-proofing or biometric voice-matching system.
* **Unauthorized Surveillance**: Must not be used for unauthorized wiretapping or network packet sniffing.

---

## 3. Acoustic Feature Representation (42 Features)
The model takes a normalized 42-dimensional feature vector extracted at a standard 16,000 Hz sampling rate:
1. **MFCC Means (Indices 0–19)**: 20 coefficients capturing spectral vocal tract envelope shape.
2. **MFCC Standard Deviations (Indices 20–39)**: 20 dynamic temporal variance metrics across speech frames.
3. **Root Mean Square (RMS) Energy (Index 40)**: Signal amplitude and power distribution.
4. **Zero Crossing Rate (ZCR) (Index 41)**: Signal noisiness and high-frequency spectral transitions.

---

## 4. Training & Validation Protocol
* **Source of Truth**: `data/manifest.csv` with SHA-256 integrity verification.
* **Data Partitions**:
  * **Train Split**: 10 audio files (5 bona_fide, 5 spoof).
  * **Validation Split**: Stratified 30% holdout for hyperparameter tuning & decision threshold calibration ($t = 0.400$).
  * **Independent Test Split**: 14 untouched audio files (7 bona_fide, 7 spoof).
* **Cross-Validation**: Stratified K-Fold CV over parameter grid (`n_estimators`, `max_depth`, `min_samples_split`).

---

## 5. Performance Metrics (Independent Test Split)
* **Test Accuracy**: `92.86%` (13 / 14 correct)
* **Macro F1-Score**: `0.9282`
* **ROC-AUC Score**: `1.0000`
* **Spoof Recall (Attack Catch Rate)**: `100.0%` (0 False Negatives on spoof attacks)
* **Spoof Precision**: `87.5%`
* **Bona Fide Precision / Recall**: `100.0% / 85.7%`
* **False Positive Rate (FPR)**: `14.3%` (1 out of 7 human samples flagged for review)
* **False Negative Rate (FNR)**: `0.0%`

---

## 6. Limitations & Mitigations
1. **Small Research Dataset**: Metrics are measured on our local research set. Unseen generative vocoders (e.g. latest diffusion models) require continuous retraining.
2. **Telephony Codec Distortion**: Narrowband 8 kHz telephony compression (G.711) attenuates high frequencies; features must be resampled to 16 kHz.
3. **Acoustic Noise**: Environmental background noise can skew zero-crossing rates and MFCCs; mitigated via multi-channel analyst review.

---

## ⚖️ Statutory Disclaimer
*Experimental decision-support prototype; not proof of identity. Predictions are advisory signals designed to support human investigation.*
