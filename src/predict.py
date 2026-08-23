"""
VoiceShield Legacy Prediction Bridge (Phase 1).
Maintains backwards compatibility for legacy modules by bridging to src.scoring and src.model.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.config import DEFAULT_DECISION_THRESHOLD, SAMPLE_RATE, STATUTORY_DISCLAIMER
from src.model import load_metadata, load_model, load_model_and_metadata
from src.scoring import predict_and_score

PHASE4_DISCLAIMER: str = STATUTORY_DISCLAIMER


def predict_audio(
    model: Any,
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    decision_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Compatibility bridge for predict_audio -> predict_and_score."""
    res = predict_and_score(
        model=model,
        audio=audio,
        sample_rate=sample_rate,
        decision_threshold=decision_threshold,
    )
    # Bridge aliases
    res["ai_probability"] = res["spoof_probability"]
    return res
