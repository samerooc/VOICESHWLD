"""
VoiceShield Background Worker Daemon.
Consumes real-time audio streams from Redis, maintains rolling buffers per session,
executes the Multi-Tier Forensic Engine, and publishes live threat assessments to Redis Pub/Sub.
"""

import asyncio
import os
import signal
import sys
from typing import Dict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.queue_manager import QueueManager
from src.streaming import NeuralStreamingScoreEngine, RollingAudioBuffer

RUNNING = True


def handle_sigterm(signum, frame):
    global RUNNING
    print("[*] Worker received shutdown signal. Terminating gracefully...")
    RUNNING = False


class VoiceShieldWorker:
    """
    Independent worker session manager managing isolated circular audio buffers per stream.
    """
    def __init__(self, worker_id: str = "default-worker", device: str = "cpu"):
        self.worker_id = worker_id
        self.device = device
        self.session_buffers: Dict[str, RollingAudioBuffer] = {}
        self.session_engines: Dict[str, NeuralStreamingScoreEngine] = {}

    def get_or_create_buffer(self, session_id: str) -> RollingAudioBuffer:
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = RollingAudioBuffer(capacity_seconds=8.0)
            self.session_engines[session_id] = NeuralStreamingScoreEngine(device=self.device)
        return self.session_buffers[session_id]


class StreamWorker:
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.qm = QueueManager(redis_url=redis_url)
        self.buffers: Dict[str, RollingAudioBuffer] = {}
        self.engines: Dict[str, NeuralStreamingScoreEngine] = {}

    async def run(self):
        connected = await self.qm.connect()
        if not connected:
            print("[!] Worker cannot connect to Redis. Ensure Redis is running.")
            return

        print("[*] VoiceShield Background Stream Worker Online & Listening...")

        while RUNNING:
            try:
                # Scan active stream keys: stream:audio:*
                stream_keys = []
                async for key in self.qm.redis.scan_iter("stream:audio:*"):
                    stream_keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)

                if not stream_keys:
                    await asyncio.sleep(0.05)
                    continue

                for skey in stream_keys:
                    session_id = skey.replace("stream:audio:", "")
                    if session_id not in self.buffers:
                        self.buffers[session_id] = RollingAudioBuffer(capacity_seconds=6.0, sample_rate=16000)
                        self.engines[session_id] = NeuralStreamingScoreEngine(smoothing_alpha=0.35)

                    # Read latest unread chunk from stream
                    entries = await self.qm.redis.xread({skey: "0-0"}, count=10)
                    for stream_name, messages in entries:
                        for msg_id, data in messages:
                            payload = data.get(b"payload", b"")
                            fmt = data.get(b"format", b"pcm16").decode("utf-8")

                            buf = self.buffers[session_id]
                            if fmt == "pcm16":
                                buf.add_bytes_pcm16(payload)
                            elif fmt == "mulaw":
                                buf.add_mulaw_bytes(payload)

                            # Delete processed entry
                            await self.qm.redis.xdel(skey, msg_id)

                    # If buffer has enough audio, evaluate 3.0s window
                    buf = self.buffers[session_id]
                    if buf.get_current_duration() >= 0.5:
                        window_samples = buf.get_latest_window(window_samples=48000)
                        engine = self.engines[session_id]
                        eval_res = engine.predict_stream_window(window_samples, sample_rate=16000)
                        eval_res["session_id"] = session_id
                        await self.qm.publish_risk_score(session_id, eval_res)

                await asyncio.sleep(0.02)
            except Exception as e:
                print(f"[!] Worker stream iteration error: {e}")
                await asyncio.sleep(0.1)

        await self.qm.disconnect()
        print("[OK] Stream Worker stopped.")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = os.getenv("REDIS_PORT", "6379")
    redis_uri = f"redis://{redis_host}:{redis_port}/0"

    worker = StreamWorker(redis_url=redis_uri)
    asyncio.run(worker.run())
