"""
VoiceShield Configuration Module (Phase 1).
Central constants, paths, thresholds, and statutory disclaimers.
"""

from typing import Dict

# Audio Configuration
SAMPLE_RATE: int = 16000
N_MFCC: int = 20
TOTAL_FEATURES: int = 42

# Audio Validation Limits
MIN_AUDIO_DURATION_SEC: float = 0.50
MAX_AUDIO_DURATION_SEC: float = 300.00
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
MIN_AUDIO_RMS_ENERGY: float = 1e-5

# Multi-Format Support (WAV, MP3, MP4, M4A, OGG, FLAC)
SUPPORTED_AUDIO_EXTENSIONS: tuple = (".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac")
SUPPORTED_AUDIO_FORMATS: list = ["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac"]

# Feature Metadata
FEATURE_CONFIG: Dict = {
    "version": "1.0.0",
    "n_mfcc": N_MFCC,
    "sample_rate": SAMPLE_RATE,
    "include_rms": True,
    "include_zcr": True,
    "total_features": TOTAL_FEATURES,
    "description": "20 MFCC means + 20 MFCC stds + 1 RMS energy + 1 Zero Crossing Rate",
}

# File & Directory Paths
MANIFEST_PATH: str = "data/manifest.csv"
MODEL_PATH: str = "models/voice_detector.pkl"
MODEL_BASELINE_V1_PATH: str = "models/voice_detector_baseline_v1.joblib"
MODEL_METADATA_PATH: str = "models/model_metadata.json"
REPORTS_DIR: str = "reports"
METRICS_PATH: str = "reports/metrics.json"
VALIDATION_METRICS_PATH: str = "reports/validation_metrics.json"
FINAL_TEST_METRICS_PATH: str = "reports/final_test_metrics.json"
CONFUSION_MATRIX_PNG: str = "reports/confusion_matrix.png"

# Class Taxonomy & Mapping (Standard ASVspoof protocol)
LABEL_HUMAN: int = 0
LABEL_AI: int = 1

LABEL_MAP: Dict[int, str] = {
    0: "bona_fide",
    1: "spoof",
}

CLASS_MAPPING: Dict[str, str] = {
    "0": "bona_fide",
    "1": "spoof",
}

LABEL_NAMES: Dict[int, str] = {
    LABEL_HUMAN: "Likely Human Voice",
    LABEL_AI: "Likely Spoof / AI Voice",
}

FEATURE_SCHEMA_VERSION: str = "1.0.0"
MODEL_VERSION: str = "1.0.0"

# Calibrated Risk Score Bands (0 - 100)
RISK_LOW_THRESHOLD: int = 25
RISK_MEDIUM_THRESHOLD: int = 65
UNCERTAINTY_THRESHOLD_LOW: float = 0.40
UNCERTAINTY_THRESHOLD_HIGH: float = 0.60
DEFAULT_DECISION_THRESHOLD: float = 0.50

# Statutory Prototype Disclaimer
STATUTORY_DISCLAIMER: str = "Experimental decision-support prototype; not identity proof."

RESEARCH_NOTICE: str = (
    "RESEARCH PROTOTYPE NOTICE: Scores and accuracy metrics are benchmark figures on a "
    "small hackathon research dataset and DO NOT constitute proof of production reliability. "
    "Real-world telephony noise, codec compression, and novel generative vocoders require human-in-the-loop review."
)
