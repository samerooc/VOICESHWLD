"""
VoiceShield Continuous Retraining & Drift Monitoring Module (Phase 1 Upgrade).
Provides:
  1. Privacy-Preserving Inference Logging (Logs acoustic feature vectors, predictions,
     and metadata without persisting raw audio recordings).
  2. Misclassification & Review Buffer (Collects ambiguous and flagged samples for audited retraining).
  3. Feature Drift Detector (Computes Kolmogorov-Smirnov p-values and Population Stability Index (PSI)
     against training baseline distributions).
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from src.config import (
    DRIFT_KS_PVAL_THRESHOLD,
    DRIFT_LOGS_DIR,
    DRIFT_METRICS_PATH,
    DRIFT_PSI_ALERT_THRESHOLD,
    DRIFT_PSI_WARNING_THRESHOLD,
    MODEL_METADATA_PATH,
    REPORTS_DIR,
)


class InferenceAuditLogger:
    """
    Logs inference feature vectors, predicted risk scores, and operational telemetry
    to append-only JSONL files without saving raw audio payloads.
    """
    def __init__(self, log_dir: str = DRIFT_LOGS_DIR):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.current_log_file = os.path.join(
            self.log_dir, f"inference_log_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        )

    def log_prediction(
        self,
        features: np.ndarray,
        spoof_probability: float,
        risk_score: int,
        risk_band: str,
        is_uncertain: bool,
        quality_status: str = "acceptable",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Appends a privacy-safe audit record.
        """
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "spoof_probability": round(float(spoof_probability), 4),
            "human_probability": round(float(1.0 - spoof_probability), 4),
            "risk_score": int(risk_score),
            "risk_band": str(risk_band),
            "is_uncertain": bool(is_uncertain),
            "quality_status": str(quality_status),
            "feature_dim": int(len(features)),
            "feature_vector": [round(float(v), 5) for v in features],
            "metadata": metadata or {},
        }

        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Warning: Failed to write inference audit log: {e}")

        return record


class MisclassificationBuffer:
    """
    Maintains a candidate pool of uncertain, flagged, or analyst-corrected samples
    for scheduled retraining cycles.
    """
    def __init__(self, buffer_file: str = os.path.join(REPORTS_DIR, "retraining_candidates.jsonl")):
        self.buffer_file = buffer_file
        os.makedirs(os.path.dirname(self.buffer_file), exist_ok=True)

    def add_candidate(
        self,
        features: np.ndarray,
        predicted_spoof_prob: float,
        ground_truth_label: Optional[int] = None,
        reason: str = "uncertain_prediction",
        source_tag: str = "live_traffic",
    ) -> None:
        """
        Adds a candidate sample to the buffer with feature representation.
        """
        entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source_tag": source_tag,
            "predicted_spoof_prob": round(float(predicted_spoof_prob), 4),
            "ground_truth_label": ground_truth_label,
            "feature_vector": [round(float(v), 5) for v in features],
        }
        with open(self.buffer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline (expected)
    and live (actual) distributions for a single continuous feature.

    PSI < 0.10: No significant change (STABLE)
    0.10 <= PSI < 0.25: Moderate shift (WARNING)
    PSI >= 0.25: Significant distribution shift (DRIFT_DETECTED)
    """
    if len(expected) < 5 or len(actual) < 5:
        return 0.0

    # Compute bin boundaries based on baseline quantiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency counts
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Normalize to proportions with epsilon smoothing to prevent div by zero
    eps = 1e-4
    expected_pct = (expected_counts / len(expected)) + eps
    actual_pct = (actual_counts / len(actual)) + eps

    expected_pct = expected_pct / np.sum(expected_pct)
    actual_pct = actual_pct / np.sum(actual_pct)

    # PSI formula: sum((actual - expected) * ln(actual / expected))
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.nan_to_num(psi_val, nan=0.0))


class FeatureDriftDetector:
    """
    Monitors streaming or batch feature vectors against baseline training statistics.
    Computes Kolmogorov-Smirnov (KS) test p-values and Population Stability Index (PSI).
    """
    def __init__(self, metadata_path: str = MODEL_METADATA_PATH):
        self.metadata_path = metadata_path
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_std: Optional[np.ndarray] = None
        self.load_baseline()

    def load_baseline(self) -> None:
        """Loads baseline mean and std vectors from model metadata."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    if "train_feature_mean" in meta and "train_feature_std" in meta:
                        self.baseline_mean = np.array(meta["train_feature_mean"], dtype=np.float32)
                        self.baseline_std = np.array(meta["train_feature_std"], dtype=np.float32)
            except Exception as e:
                print(f"Warning: Failed to load baseline from {self.metadata_path}: {e}")

    def evaluate_batch_drift(
        self,
        live_feature_matrix: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates drift on a matrix of live incoming feature vectors (N_samples, N_features).
        """
        if live_feature_matrix is None or len(live_feature_matrix) < 5:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_count": len(live_feature_matrix) if live_feature_matrix is not None else 0,
                "overall_drift_status": "STABLE",
                "max_psi": 0.0,
                "drifted_features_count": 0,
                "details": [],
            }

        num_features = live_feature_matrix.shape[1]
        feature_names = feature_names or [f"feat_{i}" for i in range(num_features)]

        feature_reports = []
        max_psi = 0.0
        drifted_count = 0

        # Generate synthetic baseline samples from stored Gaussian parameters if raw data not loaded
        np.random.seed(42)
        n_samples = max(200, len(live_feature_matrix))

        for idx in range(num_features):
            feat_live = live_feature_matrix[:, idx]
            
            # Reconstruct baseline sample for feature idx
            if self.baseline_mean is not None and idx < len(self.baseline_mean):
                mu = self.baseline_mean[idx]
                sigma = max(1e-4, self.baseline_std[idx] if self.baseline_std is not None else 1.0)
                feat_baseline = np.random.normal(mu, sigma, n_samples)
            else:
                feat_baseline = np.random.normal(0, 1, n_samples)

            # 1. Compute PSI
            psi_score = calculate_psi(feat_baseline, feat_live)
            if psi_score > max_psi:
                max_psi = psi_score

            # 2. Compute KS-Test
            ks_stat, ks_pval = stats.ks_2samp(feat_baseline, feat_live)

            # Determine individual feature status
            if psi_score >= DRIFT_PSI_ALERT_THRESHOLD or ks_pval < 0.01:
                status = "DRIFT_DETECTED"
                drifted_count += 1
            elif psi_score >= DRIFT_PSI_WARNING_THRESHOLD or ks_pval < DRIFT_KS_PVAL_THRESHOLD:
                status = "WARNING"
            else:
                status = "STABLE"

            feature_reports.append({
                "feature_index": idx,
                "feature_name": feature_names[idx] if idx < len(feature_names) else f"feat_{idx}",
                "psi_score": round(float(psi_score), 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_pval), 5),
                "status": status,
            })

        # Overall Status
        if max_psi >= DRIFT_PSI_ALERT_THRESHOLD or drifted_count >= max(2, int(num_features * 0.15)):
            overall_status = "DRIFT_DETECTED"
        elif max_psi >= DRIFT_PSI_WARNING_THRESHOLD or drifted_count > 0:
            overall_status = "WARNING"
        else:
            overall_status = "STABLE"

        result = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": int(len(live_feature_matrix)),
            "num_features_evaluated": int(num_features),
            "overall_drift_status": overall_status,
            "max_psi": round(float(max_psi), 4),
            "drifted_features_count": int(drifted_count),
            "psi_warning_threshold": DRIFT_PSI_WARNING_THRESHOLD,
            "psi_alert_threshold": DRIFT_PSI_ALERT_THRESHOLD,
            "feature_details": feature_reports,
        }

        # Persist metrics summary
        try:
            os.makedirs(os.path.dirname(DRIFT_METRICS_PATH), exist_ok=True)
            with open(DRIFT_METRICS_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass

        return result
