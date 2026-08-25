"""
VoiceShield: Privacy-First AI Voice Deepfake & Impersonation Risk Detection.
"""

from src.audio_io import (
    get_audio_metadata,
    load_audio_from_bytes,
    load_audio_from_file,
)
from src.audio_processor import (
    compute_snr_db,
    load_audio_from_bytes as process_audio_bytes,
)
from src.config import (
    CLASS_MAPPING,
    CONFUSION_MATRIX_PNG,
    DEFAULT_DECISION_THRESHOLD,
    FEATURE_CONFIG,
    LABEL_AI,
    LABEL_HUMAN,
    LABEL_NAMES,
    MANIFEST_PATH,
    MAX_AUDIO_DURATION_SEC,
    MAX_FILE_SIZE_BYTES,
    METRICS_PATH,
    MIN_AUDIO_DURATION_SEC,
    MIN_AUDIO_RMS_ENERGY,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    N_MFCC,
    REPORTS_DIR,
    RESEARCH_NOTICE,
    RISK_LOW_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    SAMPLE_RATE,
    STATUTORY_DISCLAIMER,
    TOTAL_FEATURES,
)
from src.explainability import (
    EXPLAINABILITY_DISCLAIMER,
    build_explainability_report,
    check_out_of_distribution,
    compute_signal_diagnostics,
    get_feature_summary_table,
    get_global_feature_importance,
)
from src.features import (
    TOTAL_DIMENSIONS,
    extract_features,
    extract_features_from_audio,
    extract_features_from_file,
    get_feature_names,
)
from src.model import (
    build_pipeline,
    load_metadata,
    load_model,
    load_model_and_metadata,
    save_model,
)
from src.privacy import (
    compute_sha256,
    get_privacy_statement,
    safe_delete_file,
)
from src.scoring import (
    calculate_risk_score,
    get_risk_band,
    predict_and_score,
)
from src.vad import VoiceActivityDetector
from src.validation import (
    validate_audio_signal,
    validate_wav_bytes,
)

__all__ = [
    "SAMPLE_RATE",
    "N_MFCC",
    "TOTAL_FEATURES",
    "TOTAL_DIMENSIONS",
    "FEATURE_CONFIG",
    "MANIFEST_PATH",
    "MODEL_PATH",
    "MODEL_METADATA_PATH",
    "REPORTS_DIR",
    "METRICS_PATH",
    "CONFUSION_MATRIX_PNG",
    "LABEL_HUMAN",
    "LABEL_AI",
    "LABEL_NAMES",
    "CLASS_MAPPING",
    "RISK_LOW_THRESHOLD",
    "RISK_MEDIUM_THRESHOLD",
    "DEFAULT_DECISION_THRESHOLD",
    "STATUTORY_DISCLAIMER",
    "RESEARCH_NOTICE",
    "EXPLAINABILITY_DISCLAIMER",
    "MIN_AUDIO_DURATION_SEC",
    "MAX_AUDIO_DURATION_SEC",
    "MAX_FILE_SIZE_BYTES",
    "MIN_AUDIO_RMS_ENERGY",
    "load_audio_from_bytes",
    "load_audio_from_file",
    "process_audio_bytes",
    "get_audio_metadata",
    "compute_snr_db",
    "VoiceActivityDetector",
    "validate_wav_bytes",
    "validate_audio_signal",
    "extract_features",
    "extract_features_from_audio",
    "extract_features_from_file",
    "get_feature_names",
    "build_pipeline",
    "load_model",
    "load_metadata",
    "load_model_and_metadata",
    "save_model",
    "calculate_risk_score",
    "get_risk_band",
    "predict_and_score",
    "compute_sha256",
    "safe_delete_file",
    "get_privacy_statement",
    "compute_signal_diagnostics",
    "check_out_of_distribution",
    "get_feature_summary_table",
    "get_global_feature_importance",
    "build_explainability_report",
]
