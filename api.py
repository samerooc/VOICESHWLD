"""
VoiceShield FastAPI REST Service (Phase 13).
Provides high-performance, in-memory, privacy-preserving REST API endpoints
for voice authenticity risk scoring, signal diagnostics, and explainability.

Statutory Notice:
Experimental decision-support prototype; not identity proof.
Must not be used for automatic call termination or transaction blocking.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from src.audio_io import load_audio_from_bytes
from src.preprocessing import detect_audio_container_and_codec
from src.config import (
    MAX_FILE_SIZE_BYTES,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from src.explainability import build_explainability_report
from src.model import load_metadata, load_model
from src.schemas import HealthResponse, MetadataResponse, PredictResponse
from src.scoring import predict_and_score
from src.model_registry import verify_and_load_model

app = FastAPI(
    title="VoiceShield API",
    description="Explainable AI Voice Deepfake & Impersonation Risk Detection REST Service",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://localhost:8502",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8502",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model = None
_metadata = None


def get_cached_model():
    global _model, _metadata
    if _model is None and os.path.exists(MODEL_PATH) and os.path.exists(MODEL_METADATA_PATH):
        try:
            _model, _metadata = verify_and_load_model(MODEL_PATH, MODEL_METADATA_PATH)
        except Exception:
            _model = load_model(MODEL_PATH)
            _metadata = load_metadata(MODEL_METADATA_PATH)
    return _model, _metadata


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check() -> Dict[str, str]:
    """
    Health check endpoint returning system status.
    """
    return {
        "status": "ok",
        "service": "voiceshield-api",
    }


@app.get("/metadata", response_model=MetadataResponse, tags=["System"])
def get_system_metadata() -> Dict[str, Any]:
    """
    Retrieves active model version, backbone, feature configuration, and supported formats.
    """
    _, metadata = get_cached_model()
    model_ver = metadata.get("model_version", "2.0.0") if metadata else "2.0.0"
    feat_ver = metadata.get("feature_version", "1.0.0") if metadata else "1.0.0"
    backbone = metadata.get("backbone", "acoustic_spectral_net") if metadata else "acoustic_spectral_net"

    return {
        "model_version": model_ver,
        "feature_version": feat_ver,
        "backbone": backbone,
        "class_mapping": {"0": "bona_fide", "1": "spoof"},
        "supported_format": "wav, mp3, mp4, m4a, ogg, flac",
        "audio_saved": False,
        "disclaimer": STATUTORY_DISCLAIMER,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_voice(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, MP4, M4A, OGG, FLAC)"),
) -> Dict[str, Any]:
    """
    Inspects an uploaded audio file in-memory and returns calibrated spoof risk scores,
    signal diagnostics, and explainability features. Zero raw audio is persisted to disk.
    """
    start_time = time.perf_counter()
    raw_bytes: Optional[bytes] = None
    audio_arr: Optional[np.ndarray] = None

    try:
        # 1. Filename validation & safe extension verification
        filename = file.filename or "audio.wav"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported audio format '{ext}'. Supported formats: {', '.join(SUPPORTED_AUDIO_EXTENSIONS)}.",
            )

        if re.search(r"[\\/<>:\"|?*]", filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename: Filename contains illegal or unsafe characters.",
            )

        # 2. Read bytes into memory and validate size
        raw_bytes = await file.read()
        if len(raw_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded audio file is empty (0 bytes).",
            )

        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio file size exceeds maximum limit of {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
            )

        container_fmt, codec_fmt = detect_audio_container_and_codec(raw_bytes, file_ext=ext)

        # 3. In-memory decoding & validation (Zero Disk Retention)
        try:
            audio_arr, sr = load_audio_from_bytes(raw_bytes, target_sr=SAMPLE_RATE, file_ext=ext)
        except ValueError as ve:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio validation failed: {str(ve)}",
            )
        except Exception as ex:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio decoding failed: {ex}. Ensure file is a valid, uncorrupted audio recording.",
            )

        # 4. Model inference & risk scoring
        model, metadata = get_cached_model()
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Please ensure model training has been completed.",
            )

        threshold = (
            metadata.get("optimal_decision_threshold", 0.50)
            if metadata
            else 0.50
        )
        pred_res = predict_and_score(
            model,
            audio_arr,
            sample_rate=sr,
            decision_threshold=threshold,
        )

        # 5. Explainability & Diagnostics Package
        train_mean = np.array(metadata.get("train_feature_mean")) if metadata and "train_feature_mean" in metadata else None
        train_std = np.array(metadata.get("train_feature_std")) if metadata and "train_feature_std" in metadata else None

        explain_res = build_explainability_report(
            model,
            audio_arr,
            sr,
            pred_res,
            train_mean=train_mean,
            train_std=train_std,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        raw_band = pred_res["risk_band"].lower()
        if explain_res["is_uncertain"] or pred_res["is_uncertain"]:
            risk_band_val = "uncertain"
        elif "low" in raw_band:
            risk_band_val = "low"
        elif "high" in raw_band:
            risk_band_val = "high"
        elif "degraded" in explain_res["signal_diagnostics"].get("audio_quality", "").lower():
            risk_band_val = "low_quality"
        else:
            risk_band_val = "review"

        explanation_list = [
            {
                "category": group.get("category", "general"),
                "feature_group": group.get("feature_group", "Unknown"),
                "importance_share": group.get("importance_share", 0.0),
            }
            for group in explain_res.get("top_feature_groups", [])
        ]

        return {
            "model_version": metadata.get("model_version", "2.0.0") if metadata else "2.0.0",
            "backbone": metadata.get("backbone", "acoustic_spectral_net") if metadata else "acoustic_spectral_net",
            "feature_version": metadata.get("feature_version", "1.0.0") if metadata else "1.0.0",
            "class_mapping": {"0": "bona_fide", "1": "spoof"},
            "original_format": container_fmt or ext.lstrip("."),
            "detected_codec": codec_fmt,
            "raw_model_score": float(pred_res["raw_model_score"]),
            "calibrated_probability": float(pred_res["calibrated_probability"]),
            "bona_fide_probability": round(float(pred_res["bona_fide_probability"]), 4),
            "spoof_probability": round(float(pred_res["spoof_probability"]), 4),
            "risk_score": int(pred_res["risk_score"]),
            "risk_band": risk_band_val,
            "uncertainty": bool(pred_res["is_uncertain"] or explain_res["is_uncertain"]),
            "quality_flags": pred_res.get("quality_flags", []),
            "valid_window_count": 1,
            "total_window_count": 1,
            "explanation": explanation_list,
            "processing_ms": elapsed_ms,
            "audio_saved": False,
            "reliability_notice": "Prediction reliability depends on audio quality and similarity to evaluation data.",
            "disclaimer": STATUTORY_DISCLAIMER,
        }

    finally:
        if raw_bytes is not None:
            del raw_bytes
        if audio_arr is not None:
            del audio_arr
        await file.close()
