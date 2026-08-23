# VoiceShield Production & SIH Demo Container (Phase 8)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install libsndfile for WAV audio processing and curl for health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Copy dependency manifests and install pinned requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, configs, and reports
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/
COPY reports/ ./reports/
COPY data/manifest.csv ./data/manifest.csv
COPY data/README.md ./data/README.md
COPY app.py .
COPY api.py .

# Set permissions for non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Expose ports for Streamlit Dashboard (8502) and FastAPI Service (8000)
EXPOSE 8502 8000

# Default healthcheck targeting FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command starts FastAPI service
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
