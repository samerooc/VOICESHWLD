# 🏆 Smart India Hackathon (SIH) — VoiceShield Project Guide

> **Statutory Notice**:
> *“This is an experimental decision-support prototype; not identity proof. VoiceShield outputs advisory risk signals to support human analysts and must never be used for automatic call termination or transaction blocking.”*

---

## 🎯 1. Problem & Solution Overview

| Problem | VoiceShield Solution |
| :--- | :--- |
| **Black-Box AI Models**: High false alarms and zero transparency for security analysts. | **5 Canonical Explainability Groups**: Decomposes 42 acoustic dimensions into Spectral Formants, Timing, Pitch Jitter, Energy, and Acoustic Quality. |
| **Biometric Privacy Leaks**: Traditional systems store caller voice recordings on disk. | **Zero-Retention Architecture**: Audio is decoded strictly in volatile RAM and purged upon inference completion (`audio_saved: false`). |
| **Overconfident Errors in Borderline Calls**: Hard 0.50 thresholds fail on compressed audio. | **Uncertainty Band Policy**: Automatically flags ambiguous scores ($0.40 \le P \le 0.60$) with `"UNCERTAIN — MANUAL REVIEW REQUIRED"`. |
| **Need for Real-Time Inspection**: Call centers need continuous risk scores. | **Sandbox Streaming Simulator**: Slices 160 ms windows with 40 ms stride, calculating an Exponential Moving Average (EMA) rolling risk score in real time. |

---

## ⚡ 2. Quick Execution for SIH Jury Evaluation

### 1. Launch Services (Local Non-Docker or Docker)
```powershell
# Start Streamlit SOC Dashboard on Port 8502
python -m streamlit run app.py --server.port 8502

# Start FastAPI Microservice on Port 8000
python -m uvicorn api:app --reload --port 8000
```

### 2. Open Endpoints in Browser
- **Streamlit SOC Dashboard**: [http://localhost:8502](http://localhost:8502)
- **FastAPI OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Container Health Check Probe**: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Run Benchmark & Verification Suite
```powershell
# Run full automated test suite (76 tests)
python -m pytest -q

# Run local prototype latency benchmark
python scripts/benchmark.py

# Run streaming CLI simulation
python scripts/simulate_stream.py --audio data/test/ai_voice/1.wav --max-windows 25
```
