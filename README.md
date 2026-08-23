# 🛡️ VoiceShield: Explainable AI Voice Authenticity & Deepfake Detection

> **Statutory Notice & Operational Policy**:
> *“This is an experimental decision-support prototype; not identity proof. VoiceShield outputs advisory risk signals to support human analysts and must never be used for automatic call termination or transaction blocking.”*

---

## 📌 1. Project Overview

VoiceShield is an explainable, privacy-preserving voice clone and synthetic speech risk assessment platform built for Security Operations Centers (SOC), fraud prevention analysts, and hackathon demonstrations (Smart India Hackathon).

### Core Capabilities:
- **Fast In-Memory Analysis**: Inspects uploaded WAV files in volatile memory without persisting raw audio to disk.
- **42 Acoustic Features**: 20 MFCC Means, 20 MFCC Standard Deviations, RMS Energy, and Zero Crossing Rate.
- **5 Canonical Functional Categories**: *Spectral Formants*, *Timing & Prosody*, *Pitch Micro-Jitter*, *Signal Energy*, and *Acoustic Quality*.
- **Uncertainty & Anomaly Policy**: Automatically highlights ambiguous signals ($0.40 \le P(\text{spoof}) \le 0.60$) and warns against out-of-distribution (OOD) audio anomalies.
- **Sandbox Streaming Simulator**: Simulates real-time chunked stream evaluation ($160\text{ ms}$ windows, $40\text{ ms}$ stride) with an Exponential Moving Average (EMA) rolling risk score.

---

## 🚀 2. Quick Start & Setup

### Option A: Local Non-Docker Run (Python 3.10+)

#### Windows (PowerShell):
```powershell
# 1. Clone repository and navigate to root
cd voice-clone-detector

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start FastAPI REST Service (Port 8000)
python -m uvicorn api:app --reload --port 8000

# 5. In a separate terminal, start Streamlit Dashboard (Port 8502)
python -m streamlit run app.py --server.port 8502
```

#### Linux / macOS:
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start FastAPI Service
uvicorn api:app --reload --port 8000 &

# 4. Start Streamlit Dashboard
streamlit run app.py --server.port 8502
```

---

### Option B: Docker & Docker Compose Deployment (Phase 8)

#### Prerequisites:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

#### Build & Run:
```bash
# 1. Build container images (Zero raw audio included)
docker compose build

# 2. Start services in background
docker compose up -d

# 3. View service status & logs
docker compose ps
docker compose logs --tail=100

# 4. Stop and remove containers
docker compose down
```

---

## 🌐 3. Service Endpoints & URLs

| Service | Port | Endpoint URL | Description |
| :--- | :--- | :--- | :--- |
| **Streamlit SOC Dashboard** | `8502` | [http://localhost:8502](http://localhost:8502) | Interactive UI with forensic inspector, explainability panel, and stream simulator. |
| **FastAPI REST Service** | `8000` | [http://localhost:8000](http://localhost:8000) | Core ML prediction & risk scoring microservice. |
| **System Health Probe** | `8000` | [http://localhost:8000/health](http://localhost:8000/health) | Container health check probe. |
| **Swagger Interactive Docs** | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive OpenAPI testing UI. |
| **ReDoc Documentation** | `8000` | [http://localhost:8000/redoc](http://localhost:8000/redoc) | Clean API specification guide. |

---

## 🧪 4. WAV Sample Testing Instructions

### Via REST API (`curl` or Python):
```bash
# Test with a Human voice sample
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/human/01.wav;type=audio/wav"

# Test with a Synthetic AI voice sample
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/test/ai_voice/1.wav;type=audio/wav"
```

### Via Terminal Simulator:
```powershell
python scripts/simulate_stream.py --audio data/test/ai_voice/1.wav --max-windows 30
```

---

## 🔒 5. Privacy, Ethical Guardrails & DPDP Alignment

1. **Zero Raw Audio Retention**: The API processes audio arrays in volatile memory and purges buffers in `finally` blocks (`audio_saved: false`).
2. **No Automatic Action**: VoiceShield does not trigger automatic call drops or banking locks. All alerts are advisory.
3. **No Private Storage**: `.dockerignore` blocks `.wav`, `.mp3`, and raw datasets from ever entering container images.
4. **No Identity Claims**: Outputs describe statistical feature correlations, never "identity proof".

---

## 🛠️ 6. Troubleshooting

- **Port Conflict (8502 or 8000 in use)**: Ensure no leftover processes are bound to ports:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8502).OwningProcess | Stop-Process
  ```
- **WAV Decode Error**: Ensure audio is standard PCM 16-bit or 32-bit float WAV format. Compressed formats (MP3/AAC) must be converted to WAV prior to upload.

---

## 📜 7. Automated Test Suite

```powershell
# Run complete test suite (Unit, Integration, API, Deployment)
python -m pytest -q
```
