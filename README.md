# 🛡️ VoiceShield — AI Voice Spoofing Risk Detector

> **Statutory Notice**: *This is an experimental decision-support prototype; not identity proof. VoiceShield outputs advisory risk signals to support human analysts and must never be used for automatic call termination, transaction blocking, or identity decisions without independent verification.*

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [How It Works — Step by Step](#3-how-it-works--step-by-step)
4. [Technology Stack](#4-technology-stack)
5. [Model Architecture and Training](#5-model-architecture-and-training)
6. [Feature Engineering](#6-feature-engineering)
7. [API Reference](#7-api-reference)
8. [Dashboard Tabs](#8-dashboard-tabs)
9. [Privacy and Security Design](#9-privacy-and-security-design)
10. [Project Structure](#10-project-structure)
11. [Quick Start — Local Setup](#11-quick-start--local-setup)
12. [Deploy with Docker](#12-deploy-with-docker)
13. [Public Access via Cloudflare Tunnel](#13-public-access-via-cloudflare-tunnel)
14. [Model Performance Metrics](#14-model-performance-metrics)
15. [Scaling Guide — From Laptop to Production](#15-scaling-guide--from-laptop-to-production)
16. [Limitations and Honest Disclosures](#16-limitations-and-honest-disclosures)
17. [Contributing and Roadmap](#17-contributing-and-roadmap)

---

## 1. Project Overview

**VoiceShield** is a privacy-first, explainable AI system that analyzes audio recordings to estimate the probability that a voice is **synthetically generated, cloned, or spoofed** using AI voice synthesis tools.

Built for:
- 🏦 **Fraud Prevention** — Call center voice verification support
- 🔒 **Security Operations Centers (SOC)** — Real-time spoofing risk alerts
- 🎓 **Research and Hackathon Demo** (Smart India Hackathon)

**What VoiceShield Does:**
- Accepts an audio file (WAV, MP3, M4A, OGG, FLAC) or microphone recording
- Extracts 42 acoustic features in milliseconds
- Runs a calibrated ML inference pipeline
- Returns a **0–100 Risk Score** + **5-state Risk Band** + **advisory recommendations**
- Never stores or persists any audio data

---

## 2. Problem Statement

AI voice cloning technology (ElevenLabs, Tortoise TTS, RVC, XTTS, etc.) has become freely accessible. Attackers can:
- Clone a CEO's voice from a 3-minute YouTube clip
- Impersonate bank customers over phone calls
- Bypass voice-based authentication systems

Traditional audio-based verification cannot distinguish between a real person and a high-quality AI clone. VoiceShield addresses this by analyzing **acoustic artifacts** left by AI synthesis pipelines that are invisible to the human ear.

---

## 3. How It Works — Step by Step

```
Audio File / Microphone
        |
        v
+-----------------------------+
|  1. Audio Loading & Decode  |  <- Supports WAV, MP3, M4A, OGG, FLAC
|     (src/audio_io.py)       |  <- Stereo->Mono, Resample to 16kHz
+-----------------------------+
        |
        v
+-----------------------------+
|  2. Preprocessing           |  <- DC offset removal, amplitude normalize
|     (src/preprocessing.py)  |  <- Silence trim, NaN sanitization
+-----------------------------+
        |
        v
+-----------------------------+
|  3. Feature Extraction      |  <- 20 MFCC Means + 20 MFCC Stds
|     (src/features.py)       |  <- 1 RMS Energy + 1 ZCR = 42 features
+-----------------------------+
        |
        v
+-----------------------------+
|  4. Multi-Segment Ensemble  |  <- Sliding 2.5s windows, 1.0s hop
|     (src/scoring.py)        |  <- Global + segment median blended
+-----------------------------+
        |
        v
+-----------------------------+
|  5. Calibrated ML Model     |  <- StandardScaler + RandomForest
|     (models/voice_detector) |  <- Predicts P(spoof) in [0.0, 1.0]
+-----------------------------+
        |
        v
+-----------------------------+
|  6. Risk Scoring Engine     |  <- 5-state calibration
|     (src/calibration.py)    |  <- Low / Review / High / Uncertain / Low-Quality
+-----------------------------+
        |
        v
+-----------------------------+
|  7. Explainability Report   |  <- Feature importance + diagnostics
|     (src/explainability.py) |  <- OOD detection, uncertainty flags
+-----------------------------+
        |
        v
  Dashboard / REST API
  (app.py / api.py)
```

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Language** | Python 3.10+ | Core runtime |
| **ML Model** | scikit-learn RandomForest + StandardScaler Pipeline | Classification |
| **Audio Processing** | librosa 0.10+, soundfile, pydub | Audio loading, MFCC, RMS, ZCR |
| **REST API** | FastAPI + Uvicorn | /health, /predict, /metadata endpoints |
| **Dashboard** | Streamlit | SOC analyst UI with explainability |
| **Containerization** | Docker + Docker Compose | Portable deployment |
| **Tunneling** | Cloudflare Tunnel (cloudflared) | Public HTTPS access without port forwarding |
| **Testing** | pytest (141 tests) | Unit, integration, security, end-to-end |
| **Serialization** | joblib | Model artifact persistence |
| **Config** | YAML (configs/) | Training hyperparameters, feature config |

---

## 5. Model Architecture and Training

### Model Pipeline

```python
Pipeline([
    ('scaler', StandardScaler()),           # Z-score normalization per feature
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',            # Handles class imbalance
        random_state=42
    ))
])
```

### Training Data
- **10 base training files** (5 human, 5 AI-synthesized)
- **Augmented to 132+ samples** via multi-condition augmentations

### Augmentation Pipeline (Training only, never applied to test data)

| Augmentation | Purpose |
|:---|:---|
| Sliding window segmentation (2.5s, 1.0s hop) | Temporal diversity |
| Gain +25% / -25% | Microphone volume robustness |
| Gaussian noise (sigma=0.005) | Background noise robustness |
| Telephony 8kHz down/up resample | Phone call quality simulation |
| Acoustic reverberation | Room echo simulation |
| Spectral masking (SpecAugment) | Frequency dropout regularization |
| Mild peak clipping | Compression artifact simulation |

### Cross-Validation
- **5-fold Stratified K-Fold** on augmented training data
- Threshold tuned on validation split (NOT test split)
- Final model fit on 100% of training partition

### Decision Threshold Tuning
Scans 0.20 to 0.80 in 61 steps, selects threshold maximizing F1 balanced closest to 0.50.

---

## 6. Feature Engineering

VoiceShield extracts a **42-dimensional acoustic feature vector**:

| Feature Group | Count | What It Captures |
|:---|:---:|:---|
| MFCC Means (coefficients 1-20) | 20 | Spectral envelope / vocal tract shape |
| MFCC Std Devs (coefficients 1-20) | 20 | Temporal variation / pitch micro-jitter |
| RMS Energy (mean) | 1 | Signal power / loudness |
| Zero Crossing Rate (mean) | 1 | Noisiness / fricative content |
| **Total** | **42** | |

**Why MFCCs?**

AI synthesizers produce speech that sounds natural but lacks the subtle micro-jitter, formant modulation, and phase coherence of biological vocal tracts. MFCC temporal statistics capture these differences.

**Feature Importance (from trained model):**

```
Spectral Formants & Harmonics (MFCC Means 1-20)    84.5%
Pitch Micro-Jitter & Phase Modulation (MFCC Stds)    9.6%
Macro Timing & Prosody Dynamics (MFCC Stds 1-10)     5.1%
Acoustic Noise Floor & Transitions (ZCR)              0.5%
Signal Energy Distribution (RMS)                      0.3%
```

---

## 7. API Reference

Base URL: `http://localhost:8000`

### GET /health

```json
{ "status": "ok", "service": "voiceshield-api" }
```

### GET /metadata

Returns model version, feature config, training hash, and performance metrics.

### POST /predict

Upload audio file for analysis.

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@your_audio.wav"
```

**Response:**
```json
{
  "prediction_label": "Likely Human Voice",
  "human_probability": 0.872,
  "spoof_probability": 0.128,
  "risk_score": 13,
  "risk_band": "Low",
  "risk_description": "LOW RISK — no strong spoof signal detected",
  "is_uncertain": false,
  "recommendations": ["Acoustic features align with typical human voice characteristics."],
  "disclaimer": "Experimental prototype. Not identity proof.",
  "audio_saved": false
}
```

**Interactive Swagger Docs:** `http://localhost:8000/docs`

---

## 8. Dashboard Tabs

Open the dashboard at `http://localhost:8501`

| Tab | Name | What It Does |
|:---:|:---|:---|
| 1 | Voice Authenticity Inspector | Upload file / record mic → full analysis with risk score, badge, and diagnostics |
| 2 | Explainability and Signal Diagnostics | Feature importance chart, pitch/energy signals, calibration state, OOD flags |
| 3 | Independent Evaluation and Benchmarks | Confusion matrix, precision/recall/F1/AUC loaded live from reports/metrics.json |
| 4 | Live Call Streaming Simulator | 160ms window / 40ms stride EMA rolling risk timeline chart |

### Risk Score Bands

| Score Range | Result | Meaning |
|:---:|:---:|:---|
| 0 – 25 | Low Risk | Consistent with natural human voice |
| 26 – 65 | Review Required | Borderline acoustic evidence |
| 66 – 100 | High Risk | Elevated synthetic/cloning markers |
| 45 – 55 | Inconclusive | Insufficient evidence — re-record in quiet environment |
| Low quality audio | Warning | Clipped/silent/faint — result unreliable |

---

## 9. Privacy and Security Design

| Policy | Implementation |
|:---|:---|
| **Zero raw audio persistence** | Audio processed in RAM only; never written to disk post-analysis |
| **In-memory processing** | load_audio_from_bytes() uses io.BytesIO — no temp files by default |
| **Temp file fallback** | If needed, tempfile.NamedTemporaryFile with guaranteed cleanup via safe_delete_file() |
| **No external calls** | Zero audio uploaded to any external API |
| **No identity claims** | Output is probability signal, not identity verification |
| **Statutory disclaimer** | Every API response and UI includes research-only disclaimer |
| **Input validation** | File size limit (max 50MB), format allowlist, minimum duration checks |
| **Oversized payload protection** | HTTP 413 returned for files above limit |

---

## 10. Project Structure

```
voice-clone-detector/
|
+-- app.py                    # Streamlit SOC dashboard (4 tabs)
+-- api.py                    # FastAPI REST service
+-- train_model.py            # Training entrypoint
+-- evaluate_model.py         # Evaluation entrypoint
|
+-- src/                      # Core library
|   +-- audio_io.py           # Multi-format audio loader
|   +-- features.py           # 42-feature MFCC extractor
|   +-- preprocessing.py      # Normalize, trim, validate
|   +-- model.py              # Model load/save utilities
|   +-- scoring.py            # predict_and_score() main function
|   +-- calibration.py        # 5-state risk band engine
|   +-- explainability.py     # Feature importance + OOD detection
|   +-- privacy.py            # Safe file deletion utilities
|   +-- streaming.py          # Chunked stream processing
|   +-- config.py             # Constants & paths
|   +-- schemas.py            # Pydantic request/response schemas
|
+-- scripts/                  # Standalone utilities
|   +-- train_model.py        # Full training pipeline
|   +-- evaluate_model.py     # Evaluation on test split
|   +-- evaluate_robustness.py # Noise/reverb robustness test
|   +-- simulate_stream.py    # Streaming simulator
|   +-- error_analysis.py     # Misclassification analysis
|   +-- build_manifest.py     # Dataset manifest builder
|
+-- models/                   # Trained model artifacts
|   +-- voice_detector.pkl          # Current production model
|   +-- voice_detector_baseline_v1.joblib
|   +-- model_metadata.json         # Version, hash, metrics
|   +-- eval_metrics.json
|
+-- data/                     # Audio dataset
|   +-- human/                # Training: real human voices
|   +-- ai_voice/             # Training: AI synthesized voices
|   +-- test/human/           # Test: held-out human voices
|   +-- test/ai_voice/        # Test: held-out AI voices
|   +-- manifest.csv          # Dataset index with splits + hashes
|   +-- evaluation_manifest.csv
|
+-- reports/                  # Auto-generated reports
|   +-- metrics.json
|   +-- validation_metrics.json
|   +-- feature_importance.csv
|   +-- robustness_report.json
|   +-- confusion_matrix.png
|
+-- configs/                  # YAML configurations
|   +-- training.yaml         # Hyperparameters + augmentation
|   +-- preprocessing.yaml
|   +-- config.yaml
|
+-- tests/                    # 141 automated tests
|   +-- test_api.py
|   +-- test_scoring.py
|   +-- test_calibration.py
|   +-- test_privacy.py
|   +-- test_security.py
|   +-- test_end_to_end.py
|
+-- docs/                     # Technical documentation
+-- Dockerfile                # Container build
+-- docker-compose.yml        # Multi-service orchestration
+-- requirements.txt          # Production dependencies
+-- cloudflared.exe           # Cloudflare tunnel binary (Windows)
```

---

## 11. Quick Start — Local Setup

### Prerequisites
- Python 3.10+
- pip
- Git

### Windows (PowerShell)

```powershell
# Clone the repo
git clone https://github.com/yourname/voice-clone-detector.git
cd voice-clone-detector

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train model (if not already trained)
python train_model.py

# Terminal 1 — Start FastAPI backend
.\venv\Scripts\uvicorn api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — Start Streamlit dashboard
.\venv\Scripts\streamlit run app.py --server.port 8501 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py

uvicorn api:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.port 8501 --server.enableCORS false --server.enableXsrfProtection false
```

### Run All 141 Tests

```bash
pytest -v
```

---

## 12. Deploy with Docker

```bash
# Build and start all services
docker compose up --build

# API      -> http://localhost:8000
# Dashboard -> http://localhost:8502
```

`docker-compose.yml` defines:
- `voiceshield-api` service (FastAPI on port 8000)
- `voiceshield-dashboard` service (Streamlit on port 8502)
- Health checks with `curl /health`
- Non-root `appuser` for security
- Volume mounts for model artifacts

---

## 13. Public Access via Cloudflare Tunnel

To expose the dashboard to any device on any network (no port forwarding, no router config):

```powershell
# Windows — run from project root
.\cloudflared.exe tunnel --url http://localhost:8501
```

```bash
# Linux/Mac
cloudflared tunnel --url http://localhost:8501
```

Output will show:

```
Your quick Tunnel has been created!
Visit: https://random-words.trycloudflare.com
```

Share that HTTPS URL — it works on any phone, tablet, or laptop worldwide. The tunnel stays active while your PC is running.

**For a permanent public URL (no expiry):**
1. Create free Cloudflare account at cloudflare.com
2. Set up a named tunnel with your own custom subdomain
3. Or deploy to Streamlit Community Cloud (share.streamlit.io) for 24/7 hosting

---

## 14. Model Performance Metrics

Evaluated on **held-out test set** (never seen during training):

| Metric | Value |
|:---|:---:|
| **Accuracy** | 96.3% |
| **Balanced Accuracy** | 94.4% |
| **Precision** | 94.7% |
| **Recall** | 100.0% |
| **F1 Score** | 0.973 |
| **ROC-AUC** | 0.9938 |
| Decision Threshold | 0.500 |

**Test Set Sample Results:**

| File | True Label | Predicted | Spoof % | Band |
|:---|:---:|:---:|:---:|:---:|
| test/ai_voice/1.wav | Spoof | Correct | 66% | High Risk |
| test/ai_voice/2.wav | Spoof | Correct | 70% | High Risk |
| test/human/01.wav | Human | Correct | 28% | Low |
| test/human/03.wav | Human | Correct | 8% | Low |
| test/human/4.wav | Human | Correct | 4% | Low |

> **Note**: Metrics are on the small internal dataset. `GENERALIZATION_UNVERIFIED` — performance on truly out-of-domain audio may vary.

---

## 15. Scaling Guide — From Laptop to Production

### Current State (Demo / Laptop)

```
Browser -> Cloudflare Tunnel -> Streamlit (localhost:8501)
                             -> FastAPI   (localhost:8000)
```

- Works for demos, hackathons, small teams
- Goes down when PC shuts down
- Not fault tolerant

---

### Scale Level 1: Cloud VM (1–10 concurrent users)

Deploy to a single cloud VM (AWS EC2 t3.medium / GCP e2-medium / Azure B2s):

```bash
git clone your-repo
pip install -r requirements.txt
python train_model.py

# Use systemd to auto-restart on crash
# Use nginx as reverse proxy
# Use certbot for free SSL certificate
```

**Cost:** ~$20–$40/month

---

### Scale Level 2: Docker + Managed Cloud (10–100 concurrent users)

```bash
# Build and push container
docker build -t voiceshield .
docker tag voiceshield gcr.io/YOUR_PROJECT/voiceshield
docker push gcr.io/YOUR_PROJECT/voiceshield

# Deploy to Google Cloud Run (auto-scales to zero)
gcloud run deploy voiceshield \
  --image gcr.io/YOUR_PROJECT/voiceshield \
  --platform managed \
  --allow-unauthenticated \
  --port 8000
```

Other options: AWS ECS, Azure Container Apps, Railway.app, Render.com

**Cost:** Pay-per-request, ~$0–$50/month depending on load

---

### Scale Level 3: Microservices + Load Balancer (100–10,000 users)

```
Internet --> Load Balancer (nginx / AWS ALB)
                   |
        +----------+-----------+
        |          |           |
    API Pod 1  API Pod 2  API Pod 3     (FastAPI replicas)
        |          |           |
        +----------+-----------+
                   |
            Message Queue (Redis / RabbitMQ)
                   |
            Worker Pool (ML inference workers)
```

Key changes needed:
1. Separate ML inference workers from API servers
2. Redis queue for async audio processing
3. Kubernetes (K8s) for container orchestration
4. Horizontal Pod Autoscaler based on queue depth
5. Shared model storage on S3 / GCS
6. Centralized logging (ELK Stack / CloudWatch)
7. Prometheus + Grafana for metrics dashboards

**Cost:** $200–$2000/month depending on load

---

### Scale Level 4: Enterprise Production (10,000+ users)

Additional requirements:
- Replace RandomForest with deep learning (wav2vec 2.0, RawNet2, AASIST)
- GPU inference (NVIDIA T4 / A10) for large audio files
- Model versioning with MLflow or DVC
- A/B testing infrastructure for model updates
- Audio streaming integration (WebRTC, Twilio Media Streams, SIP/RTP)
- Multi-region deployment for low latency globally
- SOC2 / ISO 27001 compliance for enterprise customers
- Full audit logging for all predictions

---

### Model Upgrade Path

| Stage | Model | Expected Accuracy* | Latency |
|:---|:---|:---:|:---:|
| Current (Demo) | MFCC + RandomForest | ~96% (internal) | ~250ms |
| Level 2 | MFCC + XGBoost / LightGBM | ~97% | ~100ms |
| Level 3 | wav2vec 2.0 fine-tuned | ~99%+ | ~500ms (GPU) |
| Level 4 | AASIST / RawGAT-ST | ~99.5%+ | ~200ms (GPU) |

*Accuracy on published ASVspoof 2019/2021 benchmarks

---

## 16. Limitations and Honest Disclosures

| Limitation | Detail |
|:---|:---|
| **Small training dataset** | 10 base files (5 human, 5 AI). Generalization to diverse voices is UNVERIFIED |
| **Gaussian noise weakness** | Model shows reduced confidence on heavily noise-corrupted audio |
| **Reverberation sensitivity** | Heavy room echo can shift spoof probability into borderline range |
| **Language bias** | Training data may not represent all languages equally |
| **New synthesizers** | A new AI voice synthesizer not seen in training may evade detection |
| **Not identity proof** | Low risk score does NOT confirm the speaker is who they claim to be |
| **No live call integration** | Not connected to any telephone network, SIP, or RTP stream |

---

## 17. Contributing and Roadmap

### Immediate Improvements
- [ ] Add more diverse human voices (multilingual, different ages and genders)
- [ ] Add more AI synthesizer samples (RVC, XTTS, Bark, Voicebox)
- [ ] Implement wav2vec 2.0 backbone for better feature extraction
- [ ] Add pitch/prosody features (F0 jitter, shimmer, HNR)

### Future Roadmap
- [ ] Real-time WebRTC audio stream processing
- [ ] REST API authentication (API keys / OAuth2)
- [ ] Batch prediction endpoint for bulk audio files
- [ ] Model drift monitoring dashboard
- [ ] Multi-language support validation
- [ ] Public benchmark evaluation on ASVspoof 2019 LA track

---

## License

Research and educational use only. Not for production identity verification.

---

## Acknowledgements

- **librosa** — Audio feature extraction
- **scikit-learn** — ML pipeline
- **Streamlit** — Dashboard framework
- **FastAPI** — REST API framework
- **Cloudflare Tunnel** — Public HTTPS access
- **ASVspoof Challenge** — Inspiration for evaluation methodology

---

*Built for Smart India Hackathon 2024 | VoiceShield v2.0.0*
