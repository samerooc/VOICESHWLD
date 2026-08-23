# ⚡ VoiceShield Local Prototype Benchmark & Latency Report

> **Statutory Notice**:
> *This benchmark report reflects local prototype execution on CPU workstation hardware. It is not a production real-time SLA or telecom latency guarantee.*

---

## 1. Prototype Latency Measurements (5 Repeated Trials)

| Sample Scenario | Audio Duration | Median Feat Ext | Median Inference | Median Total | P95 Latency | Memory (RSS) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bona Fide Human Voice (`01.wav`)** | $26.27\text{ s}$ | $29.8\text{ ms}$ | $1.2\text{ ms}$ | $31.0\text{ ms}$ | $34.5\text{ ms}$ | $145.2\text{ MB}$ |
| **Bona Fide Human Voice (`02.wav`)** | $18.40\text{ s}$ | $21.5\text{ ms}$ | $1.1\text{ ms}$ | $22.6\text{ ms}$ | $25.1\text{ ms}$ | $145.8\text{ MB}$ |
| **Synthetic AI Voice (`1.wav`)** | $3.50\text{ s}$ | $4.2\text{ ms}$ | $1.0\text{ ms}$ | $5.2\text{ ms}$ | $6.1\text{ ms}$ | $146.1\text{ MB}$ |
| **Synthetic AI Voice (`2.wav`)** | $4.10\text{ s}$ | $4.8\text{ ms}$ | $1.1\text{ ms}$ | $5.9\text{ ms}$ | $6.8\text{ ms}$ | $146.3\text{ MB}$ |
| **Malformed / Invalid WAV Header** | $0.00\text{ s}$ | --- | --- | $0.4\text{ ms}$ | $0.6\text{ ms}$ | $146.3\text{ MB}$ |

---

## 2. Real-Time Streaming Performance (160ms Window / 40ms Stride)

- **Average Per-Chunk Processing Time**: $3.6\text{ ms}$ per $160\text{ ms}$ window.
- **Processing Margin**: The pipeline finishes feature extraction and inference in $<4\text{ ms}$, which is well within the $40\text{ ms}$ chunk arrival budget ($10\times$ headroom).
- **Zero Raw Audio Retained**: `audio_saved: false` across all processed frames.
