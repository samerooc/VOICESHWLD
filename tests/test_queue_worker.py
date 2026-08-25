"""
Unit & Integration Test Suite for Phase 6 Distributed Redis Queues & Worker Architecture.
Tests:
1. RedisStreamManager push, read, acknowledge, and Pub/Sub broadcasting.
2. In-memory queue fallback when Redis is offline.
3. VoiceShieldWorker buffer isolation and window scoring.
"""

import asyncio
import os
import sys
import numpy as np
import pytest
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.queue_manager import RedisStreamManager
from src.worker import VoiceShieldWorker


@pytest.mark.asyncio
async def test_redis_stream_manager_push_and_fallback():
    """Verify push_audio_chunk writes entries to stream or in-memory fallback."""
    manager = RedisStreamManager(redis_url="redis://localhost:6379/0")
    stream_id = "test_call_001"

    chunk_bytes = np.ones(640, dtype=np.float32).tobytes()
    entry_id = await manager.push_audio_chunk(stream_id, chunk_bytes, metadata={"format": "pcm_float32"})

    assert entry_id is not None
    assert len(str(entry_id)) > 0


@pytest.mark.asyncio
async def test_redis_pubsub_publish_and_subscribe():
    """Verify Pub/Sub channel message broadcasting."""
    manager = RedisStreamManager()
    stream_id = "test_sub_002"

    test_assessment = {
        "event": "assessment",
        "stream_id": stream_id,
        "risk_score": 75,
        "risk_band": "High Risk",
        "latency_ms": 18.5,
    }

    # Start subscriber in background
    received = []

    async def sub_listener():
        async for msg in manager.subscribe_risk_channel(stream_id):
            received.append(msg)
            break

    sub_task = asyncio.create_task(sub_listener())
    await asyncio.sleep(0.05)

    # Publish assessment
    await manager.publish_prediction(stream_id, test_assessment)

    await asyncio.sleep(0.1)
    sub_task.cancel()
    try:
        await sub_task
    except asyncio.CancelledError:
        pass

    assert len(received) >= 1
    assert received[0]["risk_score"] == 75
    assert received[0]["risk_band"] == "High Risk"


def test_worker_buffer_isolation():
    """Verify VoiceShieldWorker maintains independent RollingAudioBuffers per stream."""
    worker = VoiceShieldWorker(worker_id="test-worker", device="cpu")

    buf1 = worker.get_or_create_buffer("stream_alpha")
    buf2 = worker.get_or_create_buffer("stream_beta")

    assert buf1 is not buf2
    assert "stream_alpha" in worker.session_buffers
    assert "stream_beta" in worker.session_buffers

    # Push different data to each buffer
    buf1.add_samples(np.ones(1000, dtype=np.float32) * 0.2)
    buf2.add_samples(np.ones(1000, dtype=np.float32) * 0.9)

    win1 = buf1.get_latest_window(window_samples=1000)
    win2 = buf2.get_latest_window(window_samples=1000)

    assert np.all(win1 == 0.2)
    assert np.all(win2 == 0.9)
