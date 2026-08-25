"""
VoiceShield Phase 8 — Diagnostics, Benchmark & Stress Test Suite.

Verifies:
  1. scripts/run_diagnostics.py Probe Executions (Preprocessing, DSP, Neural, REST, WebSocket)
  2. scripts/benchmark_accuracy.py Single File & Batch Metric Calculations (EER, F1, Accuracy)
  3. scripts/stress_test_stream.py Concurrent WebSocket Load Testing & Percentile Latency SLOs

Run with:
    pytest tests/test_phase8.py -v
"""

from __future__ import annotations

import ast
import io
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.benchmark_accuracy import compute_binary_metrics, inspect_single_file, run_batch_benchmark
from scripts.run_diagnostics import (
    DiagnosticReporter,
    probe_fastapi_rest_gateway,
    probe_neural_engine_and_resolver,
    probe_physics_and_dsp,
    probe_preprocessing_normalization,
    probe_websocket_streaming,
)
from scripts.stress_test_stream import run_concurrent_stress_test
from src.neural_engine import ProductionNeuralDetector


# ---------------------------------------------------------------------------
# Test 1: run_diagnostics.py Health Probes
# ---------------------------------------------------------------------------

def test_run_diagnostics_all_probes():
    """Verify that all diagnostic probes execute and pass without exceptions."""
    reporter = DiagnosticReporter()

    probe_preprocessing_normalization(reporter)
    probe_physics_and_dsp(reporter)
    probe_neural_engine_and_resolver(reporter)
    probe_fastapi_rest_gateway(reporter)
    probe_websocket_streaming(reporter)

    assert len(reporter.results) >= 8
    failed = [r for r in reporter.results if not r["passed"]]
    assert len(failed) == 0, f"Failed probes: {failed}"


# ---------------------------------------------------------------------------
# Test 2: benchmark_accuracy.py Metric Calculations
# ---------------------------------------------------------------------------

def test_compute_binary_metrics():
    """Verify calculation of accuracy, precision, recall, f1, and equal error rate (EER)."""
    # 5 real (0) and 5 spoof (1)
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    y_scores = [0.1, 0.2, 0.15, 0.25, 0.30, 0.85, 0.90, 0.95, 0.80, 0.75]

    metrics = compute_binary_metrics(y_true, y_scores, threshold=0.50)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["equal_error_rate"] == 0.0
    assert metrics["true_positives"] == 5
    assert metrics["true_negatives"] == 5


def test_benchmark_single_file_inspection():
    """Verify single file inspection CLI logic with synthesized in-memory WAV."""
    detector = ProductionNeuralDetector(load_hf=False)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        temp_path = tf.name

    try:
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sig = (0.35 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
        sf.write(temp_path, sig, 16000, format="WAV", subtype="PCM_16")

        pred = inspect_single_file(temp_path, detector)
        assert "risk_score" in pred
        assert "forensic_breakdown" in pred
        assert "diagnostics" in pred
        assert 0 <= pred["risk_score"] <= 100
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_benchmark_batch_directory():
    """Verify batch evaluation across directory and JSON export."""
    detector = ProductionNeuralDetector(load_hf=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create 2 mock sample files
        f1 = os.path.join(tmp_dir, "sample_human_01.wav")
        f2 = os.path.join(tmp_dir, "sample_spoof_02.wav")

        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sf.write(f1, (0.3 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32), 16000)
        sf.write(f2, (0.35 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), 16000)

        out_json = os.path.join(tmp_dir, "out_metrics.json")
        out_csv = os.path.join(tmp_dir, "out_metrics.csv")

        results = run_batch_benchmark(
            tmp_dir,
            detector,
            output_json_path=out_json,
            output_csv_path=out_csv,
        )

        assert results["total_samples"] == 2
        assert os.path.exists(out_json)
        assert os.path.exists(out_csv)
        assert "metrics" in results


# ---------------------------------------------------------------------------
# Test 3: stress_test_stream.py Concurrent Load Testing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_streaming_stress_harness():
    """Verify stress harness executes with 5 concurrent sessions and passes SLO."""
    res = await run_concurrent_stress_test(
        concurrency=5,
        duration_sec=1.5,
        chunk_ms=40,
        ws_url=None,  # In-process TestClient mode
    )

    assert res["passed"] is True
    assert res["errors"] == 0
    assert res["total_received"] == res["total_sent"]
    max_p95 = 200.0 if torch.cuda.is_available() else 350.0
    assert res["p95_ms"] < max_p95
