#!/usr/bin/env bash
set -e

echo "======================================================="
echo "       VOICESHIELD LINUX / CLOUD DEPLOYMENT SCRIPT"
echo "======================================================="

# System dependencies check
if command -v apt-get &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y libsndfile1 curl ffmpeg
fi

# Virtualenv setup
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start FastAPI background service
echo "[+] Starting FastAPI REST Service on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Start Streamlit Dashboard
echo "[+] Starting Streamlit Dashboard on port 8501..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
