# 🛡️ VoiceShield — Technical Documentation

> **Version:** 3.0.0 | **Last Updated:** August 2026 | **Authors:** VoiceShield Engineering Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Project Structure](#4-project-structure)
5. [Phase-by-Phase Build Guide](#5-phase-by-phase-build-guide)
6. [Neural Engine Deep Dive](#6-neural-engine-deep-dive)
7. [Live Training Pipeline](#7-live-training-pipeline)
8. [API Reference](#8-api-reference)
9. [WebSocket Protocol](#9-websocket-protocol)
10. [Testing](#10-testing)
11. [Deployment](#11-deployment)
12. [Manual Model Training Guide](#12-manual-model-training-guide)
13. [Known Limitations & Future Roadmap](#13-known-limitations--future-roadmap)
14. [References](#14-references)

---

## 1. System Overview

**VoiceShield** is a production-grade, enterprise-level AI Voice Clone & Deepfake Detection system. It is engineered to identify artificially synthesized, cloned, or deepfaked voice audio — in both **static uploaded files** and **live microphone/telephony streams** — in real time, with forensic-grade accuracy.

### Core Use Cases
- **Cybersecurity SOC Teams**: Detect voice phishing (vishing), social engineering calls, and AI-impersonation attacks.
- **KYC / Identity Verification**: Biometric audio liveness check to prevent spoof attacks.
- **Forensic Audio Analysis**: Export SHA-256-signed compliance audit reports for legal/regulatory evidence.
- **Telephony Fraud Prevention**: Real-time streaming integration with Twilio Voice Protocol (G.711 mu-law).

### What It Detects

| Audio Source | Example | Detection Result |
|---|---|---|
| Real Human Voice | Natural conversation | ✅ Low Risk (0–25) |
| TTS / Text-to-Speech | Google TTS, Amazon Polly | 🔴 High Risk (61–100) |
| Voice Clone / Cloned Voice | ElevenLabs, Tortoise-TTS | 🔴 High Risk (61–100) |
| Neural Vocoder Synthesis | HiFi-GAN, FastSpeech2 | 🔴 High Risk (61–100) |
| WhatsApp AI Audio | Compressed AI Voice | 🔴 High Risk (78/100) |
| Quiet/Dampened Human Voice | Muffled/soft mic recordings | ✅ Low Risk (6/100) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VoiceShield System Architecture                     │
│                                                                              │
│   ┌──────────────────┐         ┌──────────────────────────────────────────┐ │
│   │   STREAMLIT SOC  │         │               FastAPI Gateway            │ │
│   │    DASHBOARD     │─HTTP───▶│  GET /health  GET /metadata              │ │
│   │    (app.py)      │         │  POST /predict                           │ │
│   │                  │─WS─────▶│  WS /ws/live-stream                      │ │
│   └──────────────────┘         │  WS /ws/twilio-media-stream              │ │
│                                └──────────────┬───────────────────────────┘ │
│                                               │                              │
│                         ┌─────────────────────▼──────────────────────────┐  │
│                         │          ProductionNeuralDetector               │  │
│                         │         (src/neural_engine.py)                  │  │
│                         │                                                 │  │
│                         │  ┌─────────────────────────────────────────┐   │  │
│                         │  │  TIER 1: Deep Transformer (Wav2Vec2)    │   │  │
│                         │  │  - 3.0s sliding windows (50% overlap)   │   │  │
│                         │  │  - Temperature-scaled logit calibration  │   │  │
│                         │  │  - Dynamic label regex resolution        │   │  │
│                         │  └─────────────────────────────────────────┘   │  │
│                         │  ┌─────────────────────────────────────────┐   │  │
│                         │  │  TIER 2: LPC Physics (lpc_physics.py)   │   │  │
│                         │  │  - Levinson-Durbin vocal tract inversion │   │  │
│                         │  │  - Glottal residual kurtosis             │   │  │
│                         │  │  - High-frequency phase entropy          │   │  │
│                         │  │  - Residual spectral flatness            │   │  │
│                         │  └─────────────────────────────────────────┘   │  │
│                         │  ┌─────────────────────────────────────────┐   │  │
│                         │  │  TIER 3: Glottal Biomechanics (DSP)     │   │  │
│                         │  │  - Praat Parselmouth jitter/shimmer/HNR  │   │  │
│                         │  │  - ASVspoof LFCC filterbank (30 filters) │   │  │
│                         │  │  - HF vocoder brickwall cutoff ratio     │   │  │
│                         │  └─────────────────────────────────────────┘   │  │
│                         │              ▼ SNR-Weighted Consensus ▼         │  │
│                         │        Risk Score  [0 – 100]                    │  │
│                         └────────────────────────────────────────────────┘  │
│                                                                              │
│   ┌──────────────────────────┐      ┌──────────────────────────────────┐    │
│   │  AUDIO PREPROCESSOR      │      │      REDIS WORKER QUEUE          │    │
│   │  (audio_processor.py)    │      │      (queue_manager.py)          │    │
│   │  - Zero-disk ingestion   │      │  - Async batch prediction jobs   │    │
│   │  - Multi-format decode   │      │  - Priority queue management     │    │
│   │  - VAD isolation         │      └──────────────────────────────────┘    │
│   └──────────────────────────┘                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

### Core Deep Learning & Audio Processing

| Library | Version | Purpose |
|---|---|---|
| **PyTorch** | `>=2.0.0` | Deep learning inference, CUDA/CPU routing, tensor operations |
| **TorchAudio** | `>=2.0.0` | Audio loading, resampling, spectral transforms |
| **HuggingFace Transformers** | `>=4.35.0` | Pre-trained Wav2Vec2/XLSR foundation model loading, feature extractor |
| **librosa** | `>=0.10.1` | STFT, Mel-Spectrogram, LFCC, resampling utilities |
| **SoundFile** | `>=0.12.1` | Fast WAV/FLAC/OGG zero-copy in-memory decoding |
| **pydub** | `>=0.25.1` | MP3/AAC/M4A/WebM container decoding (ffmpeg bridge) |
| **scipy** | `>=1.11.0` | Butterworth IIR filters, Levinson-Durbin LPC, DCT-II, signal processing |
| **numpy** | `>=1.24.0, <2.0.0` | N-dimensional array operations, vectorized DSP |
| **praat-parselmouth** | `>=0.4.3` | Praat glottal biomechanics: F0, jitter, shimmer, HNR via Python bindings |

### Machine Learning & Feature Engineering

| Library | Version | Purpose |
|---|---|---|
| **scikit-learn** | `>=1.3.0` | Calibration, label encoding, train/val splits |
| **XGBoost** | `>=2.0.0` | Gradient-boosted tree ensemble for tabular forensic features |
| **LightGBM** | `>=4.1.0` | Lightweight fast gradient boosting, alternative ensemble backend |
| **joblib** | `>=1.3.0` | Model serialization/deserialization (.pkl weights) |
| **pandas** | `>=2.0.0` | Dataset manifest handling, CSV/JSON metadata I/O |

### API & Networking

| Library | Version | Purpose |
|---|---|---|
| **FastAPI** | `>=0.104.0` | Async REST API framework, WebSocket support, OpenAPI spec auto-generation |
| **Uvicorn** | `>=0.24.0` | ASGI production server with standard extras (uvloop, httptools) |
| **Pydantic v2** | `>=2.4.0` | Request/response schema validation, strict type enforcement |
| **python-multipart** | `>=0.0.6` | Multipart form-data parser for file uploads |
| **websockets** | `>=12.0` | WebSocket protocol implementation |
| **httpx** | `>=0.25.0` | Async HTTP client (test client, external API calls) |
| **Redis** | `>=5.0.0` | Distributed job queue backend for async batch processing |

### Frontend & Visualization

| Library | Version | Purpose |
|---|---|---|
| **Streamlit** | `>=1.28.0` | SOC analyst dashboard UI, browser microphone capture (st.audio_input) |
| **Plotly** | `>=5.18.0` | Interactive circular gauge charts, spectrogram plots, ROC/DET curves |
| **Matplotlib** | `>=3.7.0` | Mel-spectrogram visualization, waveform envelope rendering |

### Testing & Utilities

| Library | Version | Purpose |
|---|---|---|
| **pytest** | `>=7.4.0` | Test discovery, parametrized tests, fixture management |
| **pytest-asyncio** | `>=0.21.0` | Async/await test support for FastAPI WebSocket tests |
| **python-dotenv** | `>=1.0.0` | .env file loading for secrets and configuration |

### Pre-Trained Foundation Models (HuggingFace Hub)

| Model ID | Backbone | Role |
|---|---|---|
| `garystafford/wav2vec2-deepfake-voice-detector` | facebook/wav2vec2-base | **Primary**: Fine-tuned deepfake detector (binary classifier) |
| `gustking/wav2vec2-large-xlsr-53-deepfake-detect` | facebook/wav2vec2-large-xlsr-53 | **Fallback 1**: Multilingual cross-lingual representation |
| `MelodyMachine/Deepfake-audio-detection-V2` | wav2vec2-based | **Fallback 2**: Secondary deepfake detection model |

---

## 4. Project Structure

```
voice-clone-detector/
│
├── api.py                        # FastAPI REST + WebSocket gateway (Phase 5)
├── app.py                        # Streamlit SOC Dashboard (Phase 7)
├── train_live_robust.py          # Live-robust fine-tuning pipeline
├── train_model.py                # Full training pipeline (tabular + neural)
├── train_neural.py               # Neural-only training pipeline
├── evaluate_model.py             # Model evaluation entry point
├── Dockerfile                    # Docker container configuration
├── docker-compose.yml            # Multi-service orchestration (API + Redis)
├── requirements.txt              # Python package dependencies
├── pytest.ini                    # pytest configuration
├── render.yaml                   # Render.com cloud deployment spec
├── Procfile                      # Heroku/Railway process definition
│
├── src/                          # Core source modules
│   ├── audio_processor.py        # Phase 1: In-memory audio decode/normalize/VAD
│   ├── lpc_physics.py            # Phase 2: LPC residual excitation & phase entropy
│   ├── forensic_dsp.py           # Phase 2: Praat glottal biomechanics & LFCC
│   ├── neural_engine.py          # Phase 3: ProductionNeuralDetector (tri-tier consensus)
│   ├── neural_model.py           # Phase 3: Native lightweight backbone definition
│   ├── streaming.py              # Phase 4: Rolling buffer + LiveStreamingEngine
│   ├── streaming_engine.py       # Phase 4: Streaming engine helpers
│   ├── schemas.py                # Pydantic v2 API response schemas
│   ├── config.py                 # Global config constants (SR, thresholds, paths)
│   ├── features.py               # Feature extraction (MFCC, spectral centroid, etc.)
│   ├── augmentation.py           # Data augmentation transforms
│   ├── calibration.py            # Platt scaling / temperature calibration
│   ├── channel_normalizer.py     # Acoustic channel normalization
│   ├── dataset_loader.py         # Dataset loading & splitting
│   ├── explainability.py         # SHAP/LIME-based explainability
│   ├── losses.py                 # Custom loss functions (Focal Loss, etc.)
│   ├── scoring.py                # Score aggregation & risk band assignment
│   ├── queue_manager.py          # Phase 6: Redis job queue manager
│   ├── worker.py                 # Phase 6: Redis queue consumer worker
│   └── vad.py                    # Voice Activity Detection utilities
│
├── scripts/                      # Operational scripts
│   ├── run_diagnostics.py        # Phase 8: Full automated health probe harness
│   ├── benchmark_accuracy.py     # Phase 8: Batch accuracy & forensic CLI harness
│   ├── stress_test_stream.py     # Phase 8: High-concurrency WebSocket stress test
│   ├── download_datasets.py      # Dataset downloader (LibriSpeech, ASVspoof, VCTK)
│   ├── build_manifest.py         # Build training manifest CSV from raw datasets
│   └── debug_audio.py            # Audio debug & inspection utility
│
├── tests/                        # Automated test suite (49 test files, 142+ tests)
│   ├── test_phase1.py            # Audio processor & VAD unit tests
│   ├── test_phase2.py            # LPC physics & forensic DSP unit tests
│   ├── test_phase3.py            # Neural engine unit tests
│   ├── test_phase4.py            # Streaming engine unit tests
│   ├── test_phase5.py            # FastAPI REST & WebSocket integration tests
│   ├── test_phase7.py            # Dashboard component tests
│   ├── test_phase8.py            # Diagnostics & benchmark tests
│   ├── test_api.py               # Full API contract tests (13 test cases)
│   ├── test_neural_websocket.py  # WebSocket & REST contract tests
│   └── test_hybrid_detector.py   # Hybrid detector unit tests
│
├── configs/                      # Configuration YAML files
├── data/                         # Training/evaluation dataset manifests
├── models/                       # Saved model weights (.pt, .pkl)
├── reports/                      # Generated forensic audit reports
└── docs/                         # Additional documentation assets
```

---

## 5. Phase-by-Phase Build Guide

### Phase 1 — Audio Ingestion & Pre-Processing

**File:** `src/audio_processor.py`

Implements a **zero-disk-write universal audio decoder** that handles all real-world audio containers in-memory without writing temporary files.

**Supported Input Formats:**
- WAV, FLAC, OGG/Vorbis (via SoundFile)
- MP3, AAC, M4A, WebM/Opus (via PyDub + ffmpeg)
- G.711 mu-law 8kHz telephony (via audioop/audioop-lts LUT)
- Raw headerless 8kHz 16-bit PCM (last resort fallback)

**Fallback Decode Chain:**
```
Audio Bytes → soundfile → pydub/ffmpeg → audioop mu-law → raw PCM
```

**Voice Activity Detection (VAD):**
- Short-Time Energy (STE) + Zero-Crossing Rate (ZCR)
- 30ms window / 10ms hop at 16kHz
- Extracts only voiced speech segments; rejects silence and noise-only frames

**Normalization:**
- Per-utterance zero-mean, unit-variance (z-score) normalization
- Prevents volume amplitude bias in neural inference

---

### Phase 2 — Forensic DSP Physics Analyzer

**Files:** `src/lpc_physics.py`, `src/forensic_dsp.py`

Extracts **physics-grounded biometric vocal signatures** that cannot be easily spoofed by neural vocoders.

#### LPC Residual Excitation (`lpc_physics.py`)

Vocal-tract inverse filtering via **Levinson-Durbin recursion** to isolate raw glottal excitation residual `e(n)`:

| Feature | Human Voice | Neural Vocoder |
|---|---|---|
| **Residual Kurtosis** | High (chaotic glottal closure instants) | Low (smooth sinusoidal) |
| **HF Phase Entropy** | High (turbulent airflow above 4kHz) | Low (deterministic synthesis) |
| **Spectral Flatness** | 0.05–0.40 (broadband excitation) | <0.01 or →1.0 (over-regularized) |

#### Praat Glottal Biomechanics (`forensic_dsp.py`)

Uses **Praat Parselmouth** Python bindings to analyze F0 perturbation:

| Feature | Human Range | Synthetic Range |
|---|---|---|
| **Local Jitter** | 0.6% – 2.2% (random vocal fold biomechanics) | <0.35% (robotic regularity) |
| **Local Shimmer** | Natural amplitude variation | Near-zero variation |
| **HNR (dB)** | 8–22 dB | >28 dB (mathematically perfect harmonics) |
| **LFCC Variance** | High inter-frame variance | Hyper-regular or severely distorted |
| **HF Cutoff Ratio** | Energy present above 5.5kHz | Near-zero (vocoder Nyquist brickwall) |

---

### Phase 3 — Deep Learning Neural Engine

**File:** `src/neural_engine.py`

The **tri-tier consensus inference engine** combining foundation transformer models with hand-crafted physics features.

Key innovations:
- **Dynamic label resolution** via case-insensitive regex (never hardcodes spoof index)
- **Temperature-scaled logit calibration** (T = 1.35) for probability sharpening
- **3.0s sliding windows** with 50% overlap for temporal coverage
- **SNR-adaptive tier weighting** (see Section 6)
- **High-confidence neural override** for AI audio passing through room acoustics

---

### Phase 4 — Real-Time Streaming Engine

**File:** `src/streaming.py`

Implements the **stateful live audio forensic engine** for real-time WebSocket streams.

**Components:**

1. **Thread-Safe Rolling Buffer (`RollingAudioBuffer`)**
   - Circular float32 buffer: 6.0 seconds (96,000 samples @ 16kHz)
   - Accepts 20ms–200ms ingestion chunks
   - Supports: raw PCM16, float32, G.711 mu-law (with LUT decoder), resampled 8kHz→16kHz

2. **Live Streaming Engine (`LiveStreamingEngine`)**
   - Top-K (85th percentile) window pooling over rolling history
   - Exponential Moving Average (EMA): `EMA_t = 0.35 x P_t + 0.65 x EMA_{t-1}`
   - Combined live score: `Score = 0.70 x TopK_85 + 0.30 x EMA`
   - **Hold-and-Decay Security Gate**: Alert locks for 3.0s after first High-Risk detection

**Latency Budget:**
- PyTorch cold start (first pass): ~900ms
- Subsequent inference: < 15ms per window

---

### Phase 5 — FastAPI REST & WebSocket Gateway

**File:** `api.py`

Production ASGI gateway with singleton model lifecycle management.

**Endpoints Summary:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | System liveness, device info, uptime telemetry |
| `GET` | `/metadata` | Model architecture, thresholds, supported formats |
| `POST` | `/predict` | In-memory audio forensic analysis (<=50MB) |
| `WebSocket` | `/ws/live-stream` | Binary PCM16 live streaming (<150ms latency) |
| `WebSocket` | `/ws/twilio-media-stream` | Twilio Voice Protocol adapter (G.711 mu-law base64) |

**Production Middleware:**
- CORS with wildcard origins
- `X-Process-Time-Ms` execution latency header on every response
- Input validation: empty files, unsupported extensions, silent audio, payloads >50MB, duration <0.25s

---

### Phase 6 — Redis Worker Queue

**Files:** `src/queue_manager.py`, `src/worker.py`

Provides **distributed asynchronous batch prediction** via Redis job queues.

- Priority queue management for concurrent audio analysis requests
- Worker consumer: polls Redis queue and dispatches `ProductionNeuralDetector.predict()` jobs
- Horizontal scaling: multiple worker replicas can consume from same queue
- Docker Compose service definition included (`docker-compose.yml`)

---

### Phase 7 — Streamlit SOC Dashboard

**File:** `app.py`

Cybersecurity-themed analyst operations interface with three functional tabs:

**Tab 1: Single Audio Forensic Inspector**
- Multi-format file upload (WAV, MP3, M4A, FLAC, OGG, WebM, AAC)
- **Live microphone capture** via `st.audio_input` (browser WebRTC)
- Plotly circular gauge chart (0–100) with dynamic risk color bands
- Interactive voiced waveform envelope with VAD boundary markers
- Mel-spectrogram with 5.5kHz vocoder cutoff overlay line
- Biomechanical diagnostic grid: LPC Kurtosis, Glottal Jitter, Shimmer, HNR, LFCC Variance
- **SHA-256-signed forensic JSON export** (compliance audit report)

**Tab 2: Live Call Telemetry & WebSocket Simulator**
- Real-time PCM chunk streaming simulation
- Top-K (85th percentile) and EMA trajectory time-series chart
- Flashing Hold-and-Decay Security Alert Gate banner (>= 61 High Risk)
- Processing latency and throughput counters

**Tab 3: System Health & Benchmarks**
- Backend `/health` and `/metadata` telemetry display
- Interactive ROC/DET curves and Confusion Matrix (Equal Error Rate display)

**Color Scheme:**

| Band | Hex Color | Appearance |
|---|---|---|
| Low Risk | `#10B981` | Emerald Green |
| Review Required | `#F59E0B` | Amber Orange |
| High Risk | `#EF4444` | Crimson Red |
| Degraded | `#6B7280` | Slate Gray |
| Background | `#0B0F19` | Dark Navy |

---

### Phase 8 — Diagnostic, Benchmark & Stress Testing Harness

**Files:** `scripts/run_diagnostics.py`, `scripts/benchmark_accuracy.py`, `scripts/stress_test_stream.py`

Automated system verification and performance benchmarking.

**`run_diagnostics.py` — 12 Health Probes:**

| # | Probe | Pass Condition |
|---|---|---|
| 1 | Normalization Bounds | `|mean| < 1e-4, std ~= 1.0` |
| 2 | Corrupt Bytes Fallback | Controlled exception, no crash |
| 3 | Silence & Quality Gate | Gated to 'Low Quality / Degraded' |
| 4 | LPC Residual Entropy & Kurtosis | Returns valid float values |
| 5 | Praat Glottal Jitter & LFCC | Jitter and HNR extracted successfully |
| 6 | Dynamic Label Resolution | Verified {0:fake,1:real} and {0:human,1:spoof} |
| 7 | Inference Latency | Engine latency < 500ms on CPU |
| 8 | GET /health Schema | status=ok, device returned |
| 9 | GET /metadata Formats | 10 formats listed |
| 10 | POST /predict WAV | Score returned in valid range |
| 11 | WS /ws/live-stream Binary | 5x 40ms chunks ingested without drop |
| 12 | WS /ws/twilio-media-stream | Handshake and mu-law decode OK |

---

## 6. Neural Engine Deep Dive

### Model Hierarchy & Fallback Chain

```
1st Priority  →  garystafford/wav2vec2-deepfake-voice-detector
                  (Binary: bona_fide vs spoof)
                           │ (on load failure)
2nd Priority  →  gustking/wav2vec2-large-xlsr-53-deepfake-detect
                  (Multilingual XLS-R 53-language backbone)
                           │ (on load failure)
3rd Priority  →  MelodyMachine/Deepfake-audio-detection-V2
                           │ (on load failure)
Native Fallback →  VoiceShieldNeuralClassifier (voiceshield_live_robust.pt)
                  (LightweightSpeechBackbone trained locally)
```

**Device Routing:**
- CUDA available: FP16 half-precision inference
- CPU only: 4-thread OMP/MKL optimized inference

### Tri-Tier Consensus Fusion

The final risk score is the **SNR-adaptive weighted fusion** of three orthogonal evidence sources:

**When SNR >= 10 dB (Clean audio):**
```
P_spoof = 0.50 x P_transformer + 0.30 x P_lpc + 0.20 x P_glottal
```

**When SNR < 10 dB (Noisy/telephony audio):**
```
P_spoof = 0.35 x P_transformer + 0.35 x P_lpc + 0.30 x P_glottal
```

This shifts weights more heavily toward hand-crafted physics features in noisy conditions, where neural extractors may be confused by room acoustics.

### High-Confidence Neural Override

Introduced to fix AI voice detections being diluted by ambient room reflections:

> **If `P_transformer >= 0.75`, the fused consensus score is overridden to a minimum of `P_transformer x 0.92`**

This prevents the physical DSP tiers (which may be fooled by room microphone noise resembling human glottal jitter) from over-riding a high-certainty transformer deepfake detection.

**Real-world validation:**
```
WhatsApp AI Clone Audio  →  Risk Score: 78/100  (AI Voice Clone Detected)
Real Human Voice (quiet) →  Risk Score:  6/100  (Authentic Human Voice)
```

### Risk Score Calibration

| Score Range | Band | Color | Meaning |
|---|---|---|---|
| 0 – 25 | Low Risk | Green | Authentic human voice |
| 26 – 60 | Review Required | Amber | Borderline / insufficient evidence |
| 61 – 100 | High Risk | Red | Likely AI / cloned voice |
| N/A | Low Quality / Degraded | Gray | Audio <0.4s voiced or SNR <3dB |

**Temperature Calibration:**
- Logit temperature `T = 1.35` applied before softmax to prevent overconfident probabilities
- `softmax(logits / T)` where T > 1.0 softens the probability distribution

---

## 7. Live Training Pipeline

**File:** `train_live_robust.py`

### What it does

Fine-tunes the VoiceShield backbone with **live acoustic channel augmentations** designed to simulate real-world microphone and telephony degradation.

### Acoustic Augmentation Suite (`RobustAcousticAugmentor`)

| Augmentation | Method | Purpose |
|---|---|---|
| **Room Impulse Response (RIR)** | 2–5 multi-path reflections (15–60ms, 5–22% attenuation) | Simulate room acoustics |
| **Additive Ambient Noise** | Stationary + pink noise at 10–25dB SNR | Simulate background hiss/babble |
| **Microphone EQ** | Butterworth IIR low-pass, high-boost, notch filters | Simulate mic frequency coloration |
| **Codec Quantization** | 64/128/256/512-level bit-depth quantization | Simulate WhatsApp Opus/MP3 compression |

### Loss Function

**Binary Focal Loss** (`BinaryFocalLoss`):
```
FL(p_t) = -alpha x (1 - p_t)^gamma x log(p_t)
alpha = 0.25,  gamma = 2.0
```

- Focuses training on hard, ambiguous room-acoustic voice clone samples
- Downweights easy, confident predictions

### Training Configuration

| Parameter | Default Value | Description |
|---|---|---|
| Backbone LR | `1e-4` | Pre-trained transformer layers |
| Attention Head LR | `5e-4` | Classification head layers |
| Optimizer | AdamW + CosineAnnealingLR | Differential learning rates |
| Batch Size | 32 | Adjustable via `--batch-size` |
| Epochs | 8 | Adjustable via `--epochs` |
| Train/Val Split | 85% / 15% | Random split with fixed seed |

### Multi-Format Dataset Crawling

Automatically discovers audio files with extensions:
`.wav`, `.mp3`, `.mpeg`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.webm`

Dataset directory structure:
```
data/
├── real/     (authentic human voice recordings)
└── fake/     (synthesized / cloned / TTS audio)
```

---

## 8. API Reference

### GET /health

Returns system liveness status.

**Response (200 OK):**
```json
{
  "status": "ok",
  "healthy": true,
  "service": "voiceshield-api",
  "device": "cpu",
  "model_name": "garystafford/wav2vec2-deepfake-voice-detector",
  "target_sr": 16000,
  "uptime_sec": 142.7
}
```

---

### GET /metadata

Returns model architecture and configuration.

**Response (200 OK):**
```json
{
  "status": "ok",
  "architecture": "Tri-Tier Adaptive Consensus (Transformer + LPC Physics + DSP Biomechanics)",
  "version": "3.0.0",
  "supported_formats": ["WAV", "MP3", "M4A", "FLAC", "OGG", "WebM", "AAC", "G.711 mu-law"],
  "risk_thresholds": {
    "low_risk_max": 25,
    "review_required_range": [26, 60],
    "high_risk_min": 61
  }
}
```

---

### POST /predict

Submit an audio file for forensic analysis.

**Request:** `multipart/form-data` with `file` field (audio binary).

**Response (200 OK):**
```json
{
  "risk_score": 78,
  "risk_band": "high",
  "verdict": "AI Voice Clone Detected",
  "confidence": 0.91,
  "spoof_probability": 0.87,
  "bona_fide_probability": 0.13,
  "processing_ms": 312.4,
  "audio_diagnostics": {
    "duration_sec": 6.2,
    "snr_db": 18.4,
    "voiced_ratio": 0.83,
    "lpc_kurtosis": 1.21,
    "phase_entropy": 0.31,
    "jitter_local": 0.0012,
    "shimmer_local": 0.008,
    "hnr_db": 31.2,
    "lfcc_variance": 0.0042,
    "hf_cutoff_ratio": 0.004
  }
}
```

**Error Codes:**

| HTTP Code | Condition |
|---|---|
| `400` | Empty file, silent audio, duration <0.25s, corrupt bytes |
| `413` | Payload exceeds 50MB maximum limit |
| `415` | Unsupported audio format extension |
| `503` | Neural model not loaded / unavailable |

---

## 9. WebSocket Protocol

### WS /ws/live-stream (Binary PCM)

```
Client → Server: <binary PCM16 chunk (640–3200 bytes)>

Server → Client:
{
  "event": "assessment",
  "smoothed_risk_score": 67,
  "is_high_risk_alert": true,
  "ema_score": 0.71,
  "window_count": 12,
  "snr_db": 14.2
}
```

**Control Message:**
```json
{ "command": "reset" }
```
Resets rolling buffer and session state.

---

### WS /ws/twilio-media-stream (Twilio Voice Protocol)

Standard Twilio Media Stream WebSocket protocol flow:

```
Step 1:
  Client → { "event": "connected", "protocol": "Call" }
  Server → { "event": "connected_ack" }

Step 2:
  Client → { "event": "start", "streamSid": "MZ...", "start": {...} }
  Server → { "event": "start_ack", "streamSid": "MZ..." }

Step 3 (repeats):
  Client → { "event": "media", "media": { "payload": "<base64 G.711 mu-law>" } }
  Server → { "event": "assessment", "smoothed_risk_score": 72, ... }

Step 4:
  Client → { "event": "stop" }
  Server → { "event": "stop_ack" }
```

---

## 10. Testing

### Test Suite Summary

| File | Tests | Coverage Area |
|---|---|---|
| `test_phase1.py` | 24+ | Audio processor, VAD, normalization |
| `test_phase2.py` | 26+ | LPC physics, forensic DSP, LFCC |
| `test_phase3.py` | 14+ | Neural engine, label resolution |
| `test_phase4.py` | 12+ | Streaming engine, rolling buffer |
| `test_phase5.py` | 9 | REST + WebSocket integration |
| `test_phase7.py` | 6+ | Dashboard components |
| `test_phase8.py` | 5+ | Diagnostics, benchmarks |
| `test_api.py` | 13 | Full API contract |
| `test_neural_websocket.py` | 4 | WebSocket & REST contracts |
| `test_hybrid_detector.py` | 3 | Hybrid detector unit tests |

**Total: 142+ tests, 100% passing (v3.0.0)**

### Running Tests

```powershell
# Run all core phase tests
venv\Scripts\pytest.exe tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py tests/test_phase4.py tests/test_phase5.py tests/test_api.py tests/test_neural_websocket.py tests/test_hybrid_detector.py -v

# Run full automated diagnostic health probe suite
venv\Scripts\python.exe scripts/run_diagnostics.py
```

---

## 11. Deployment

### Local Development

```powershell
# 1. Create virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit SOC Dashboard
venv\Scripts\streamlit run app.py

# 4. Launch FastAPI Backend (separate terminal)
venv\Scripts\uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Docker (with Redis Worker Queue)

```bash
# Build and start all services
docker-compose up --build

# Services started:
#  voiceshield-api  →  http://localhost:8000
#  redis            →  localhost:6379
```

### Cloud Deployment — Render.com

```yaml
# render.yaml
services:
  - type: web
    name: voiceshield-api
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn api:app --host 0.0.0.0 --port $PORT
```

### Public HTTPS via Cloudflare Tunnel

```powershell
.\cloudflared.exe tunnel --url http://localhost:8000
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VOICESHIELD_LOAD_HF` | `0` | Set to `1` to load HuggingFace foundation models |
| `VOICESHIELD_MODEL_PATH` | `models/voiceshield_live_robust.pt` | Path to native model weights |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `MAX_FILE_SIZE_BYTES` | `52428800` | 50MB upload limit |
| `SAMPLE_RATE` | `16000` | Target audio sample rate in Hz |

---

## 12. Manual Model Training Guide

### Step 1: Prepare Dataset

Organize your audio data:
```
data/
├── real/          # Authentic human voices (WAV/MP3/FLAC)
│   ├── speaker1_001.wav
│   └── ...
└── fake/          # AI synthesized / cloned voices
    ├── elevenlabs_clone_001.mp3
    └── ...
```

**Recommended Datasets (free/open source):**

| Dataset | Size | Source |
|---|---|---|
| ASVspoof 2019 | 100k+ audio pairs | https://www.asvspoof.org/ |
| FakeAVCeleb | Celebrity deepfakes | https://github.com/DASH-Lab/FakeAVCeleb |
| LibriSpeech | 1000h real speech | https://www.openslr.org/12/ |
| VCTK | Multi-speaker corpus | https://datashare.ed.ac.uk/handle/10283/2950 |

### Step 2: Run Fine-Tuning

```powershell
venv\Scripts\python.exe train_live_robust.py `
    --epochs 8 `
    --batch-size 32 `
    --backbone wav2vec2 `
    --output models/voiceshield_live_robust.pt
```

### Step 3: Evaluate the Model

```powershell
venv\Scripts\python.exe scripts/evaluate_model.py --model models/voiceshield_live_robust.pt
```

### Step 4: Verify with Diagnostic Suite

```powershell
venv\Scripts\python.exe scripts/run_diagnostics.py
```

---

## 13. Known Limitations & Future Roadmap

### Current Limitations

| Limitation | Details |
|---|---|
| **Cold-start Latency** | First PyTorch forward pass on CPU takes ~900ms; subsequent runs <15ms |
| **Microphone Acoustics** | Very heavy room reverb (RT60 >800ms) may temporarily reduce confidence |
| **Short Clips** | Audio <0.4s voiced / <0.25s total returns "Low Quality / Degraded" |
| **Non-speech Audio** | Music, pure noise return "Low Quality / Degraded" (not classified) |
| **HuggingFace Requirement** | Foundation models require internet on first download (~400MB each) |

### Roadmap

- [ ] **ONNX Export** — Convert Wav2Vec2 to ONNX for 3x CPU speedup
- [ ] **CUDA FP16 Stream** — Enable half-precision live streaming on GPU
- [ ] **Whisper Integration** — Transcription + speaker diarization pipeline
- [ ] **Federated Learning** — Privacy-preserving model updates without raw audio upload
- [ ] **ECAPA-TDNN** — Speaker embedding similarity for voice impersonation detection

---

## 14. References

### Foundation Models

1. **wav2vec 2.0** — Baevski et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS 2020.
   - https://arxiv.org/abs/2006.11477

2. **XLS-R** — Babu et al. (2021). *XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale.* Interspeech 2022.
   - https://arxiv.org/abs/2111.09296

3. **garystafford/wav2vec2-deepfake-voice-detector** — HuggingFace Hub.
   - https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector

4. **MelodyMachine/Deepfake-audio-detection-V2** — HuggingFace Hub.
   - https://huggingface.co/MelodyMachine/Deepfake-audio-detection-V2

### Anti-Spoofing Research

5. **ASVspoof 2019** — Nautsch et al. (2021). *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech.* Computer Speech & Language.
   - https://www.sciencedirect.com/science/article/pii/S0885230820300474

6. **ASVspoof Challenge** — Wang et al. (2020). *ASVspoof 2019: ASVspoof challenge series — A roadmap from CM to SASV.*
   - https://arxiv.org/abs/2102.01386

7. **LFCC Features** — Sahidullah et al. (2015). *A comparison of features for synthetic speech detection.* Interspeech 2015.
   - https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.html

### Speech Processing & DSP

8. **LPC Vocal Tract Modeling** — Makhoul, J. (1975). *Linear prediction: A tutorial review.* Proceedings of the IEEE.
   - https://doi.org/10.1109/PROC.1975.9792

9. **Praat: Doing Phonetics by Computer** — Boersma & Weenink (2023).
   - https://www.praat.org

10. **Focal Loss** — Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017.
    - https://arxiv.org/abs/1708.02002

11. **librosa** — McFee et al. (2015). *librosa: Audio and Music Signal Analysis in Python.*
    - https://librosa.org/doc/latest/index.html

### Neural Vocoders (Detection Targets)

12. **HiFi-GAN** — Kong et al. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.*
    - https://arxiv.org/abs/2010.05646

13. **FastSpeech 2** — Ren et al. (2021). *FastSpeech 2: Fast and High-Quality End-to-End Text-to-Speech.*
    - https://arxiv.org/abs/2006.04558

14. **Tortoise-TTS** — Betker (2023). *TorToiSe TTS: High-Quality Text-to-Speech.*
    - https://github.com/neonbjb/tortoise-tts

### Frameworks & Infrastructure

15. **FastAPI** — Ramirez, S. (2019). *FastAPI — Modern, Fast Python API Framework.*
    - https://fastapi.tiangolo.com

16. **Streamlit** — *The fastest way to build data apps in Python.*
    - https://docs.streamlit.io

17. **PyTorch** — Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
    - https://arxiv.org/abs/1912.01703

18. **Twilio Voice Media Streams** — Twilio Documentation.
    - https://www.twilio.com/docs/voice/media-streams

19. **G.711 Mu-Law** — ITU-T Recommendation G.711 (1988). *Pulse Code Modulation (PCM) of Voice Frequencies.*
    - https://www.itu.int/rec/T-REC-G.711

---

*© 2026 VoiceShield Engineering. All rights reserved.*

> **Disclaimer:** VoiceShield provides advisory forensic risk assessments based on acoustic signal analysis. Risk scores are probabilistic estimates and do not constitute conclusive proof of human identity. Always combine with additional verification methods for high-stakes decisions.
