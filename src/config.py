"""
VoiceShield Configuration Module (Phase 1 Upgrade).
Central constants, paths, thresholds, acoustic taxonomy, and statutory disclaimers.
"""

from typing import Dict, List

# Audio Configuration
SAMPLE_RATE: int = 16000
N_MFCC: int = 20
LEGACY_TOTAL_FEATURES: int = 42
EXTENDED_TOTAL_FEATURES: int = 77
TOTAL_FEATURES: int = LEGACY_TOTAL_FEATURES  # Default for baseline compatibility

# Audio Validation Limits
MIN_AUDIO_DURATION_SEC: float = 0.50
MAX_AUDIO_DURATION_SEC: float = 300.00
MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB
MIN_AUDIO_RMS_ENERGY: float = 1e-5

# Multi-Format Support (WAV, MP3, MP4, M4A, OGG, FLAC, AAC)
SUPPORTED_AUDIO_EXTENSIONS: tuple = (".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac", ".aac")
SUPPORTED_AUDIO_FORMATS: list = ["wav", "mp3", "mp4", "m4a", "ogg", "flac", "aac"]

# Known AI Voice Synthesizers Taxonomy
SUPPORTED_SYNTHESIZERS: List[str] = [
    "elevenlabs",
    "xtts",
    "bark",
    "rvc",
    "tortoise",
    "openvoice",
    "valle",
    "coqui_tts",
    "styletts2",
    "vits",
    "diffsinger",
    "unknown",
]

# Supported Languages & Accents
SUPPORTED_LANGUAGES: List[str] = [
    "hi",        # Hindi
    "en-IN",     # Indian English
    "en-US",     # US English
    "en-GB",     # British English
    "bn",        # Bengali
    "te",        # Telugu
    "ta",        # Tamil
    "mr",        # Marathi
    "unknown",
]

# Feature Metadata
FEATURE_CONFIG: Dict = {
    "version": "2.0.0",
    "legacy_features": LEGACY_TOTAL_FEATURES,
    "extended_features": EXTENDED_TOTAL_FEATURES,
    "total_features": TOTAL_FEATURES,
    "n_mfcc": N_MFCC,
    "sample_rate": SAMPLE_RATE,
    "include_rms": True,
    "include_zcr": True,
    "include_f0_jitter": True,
    "include_shimmer_hnr": True,
    "include_spectral_dynamics": True,
    "include_prosody_timing": True,
    "include_hf_artifacts": True,
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
DRIFT_LOGS_DIR: str = "reports/drift_logs"
DRIFT_METRICS_PATH: str = "reports/drift_metrics.json"

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

FEATURE_SCHEMA_VERSION: str = "2.0.0"
MODEL_VERSION: str = "2.0.0"

# Calibrated Risk Score Bands (0 - 100)
RISK_LOW_THRESHOLD: int = 25
RISK_MEDIUM_THRESHOLD: int = 65
UNCERTAINTY_THRESHOLD_LOW: float = 0.40
UNCERTAINTY_THRESHOLD_HIGH: float = 0.60
DEFAULT_DECISION_THRESHOLD: float = 0.60

# Drift Monitoring Thresholds (Population Stability Index / KS-pvalue)
DRIFT_PSI_WARNING_THRESHOLD: float = 0.10
DRIFT_PSI_ALERT_THRESHOLD: float = 0.25
DRIFT_KS_PVAL_THRESHOLD: float = 0.05

# Statutory Prototype Disclaimer
STATUTORY_DISCLAIMER: str = "Experimental decision-support prototype; not identity proof."

RESEARCH_NOTICE: str = (
    "RESEARCH PROTOTYPE NOTICE: Scores and accuracy metrics are benchmark figures on a "
    "small hackathon research dataset and DO NOT constitute proof of production reliability. "
    "Real-world telephony noise, codec compression, and novel generative vocoders require human-in-the-loop review."
)
