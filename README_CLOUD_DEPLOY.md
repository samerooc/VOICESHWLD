# 🌐 VoiceShield 24/7 Permanent Cloud Deployment Guide

When running locally on your laptop, closing the laptop or turning it off terminates the local Python/Cloudflare process.

To have a **permanent public link that runs 24/7 even when your laptop is completely turned off**, deploy to any of the following 100% free cloud platforms.

---

## 🚀 Option 1: Hugging Face Spaces (Recommended — Free & 24/7 Persistent)

Hugging Face Spaces provides free 24/7 container hosting with a permanent `https://<your-username>-voiceshield.hf.space` link.

### Quick Setup Steps:
1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space).
2. Set Space Name: `voiceshield-soc`.
3. Select **Streamlit** as the Space SDK (or **Docker**).
4. License: `MIT` / Public.
5. In your local terminal, push this repository to Hugging Face:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/voiceshield-soc
   git push space main
   ```
6. Your permanent URL will be:
   **`https://<your-username>-voiceshield-soc.hf.space`**
   *(Runs 24/7 non-stop in the cloud even when your laptop is off).*

---

## ⚡ Option 2: Render.com (Free 24/7 Web Service)

1. Sign up at [https://render.com](https://render.com).
2. Click **New** -> **Web Service**.
3. Connect your GitHub repository `voice-clone-detector`.
4. Configure settings:
   - **Runtime**: `Python 3` (or `Docker`)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port 10000 --server.address 0.0.0.0`
5. Click **Create Web Service**.
6. Render gives you a permanent 24/7 HTTPS URL:
   **`https://voiceshield.onrender.com`**

---

## 🐳 Option 3: Cloud VPS (AWS EC2 / DigitalOcean / Oracle Cloud)

If deploying to an Ubuntu/Debian Cloud Virtual Machine:
1. Clone repository:
   ```bash
   git clone <repo_url>
   cd voice-clone-detector
   ```
2. Launch production stack in background:
   ```bash
   docker compose up -d --build
   ```
3. Set up auto-restart on system reboot:
   ```bash
   sudo systemctl enable docker
   ```
4. Accessible 24/7 on your server's Elastic Public IP or custom domain.
