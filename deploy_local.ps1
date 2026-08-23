# VoiceShield One-Click Windows Production Deployment Script
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         VOICESHIELD LOCAL PRODUCTION DEPLOYMENT" -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Check virtual environment
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "[!] Virtual environment not found. Creating venv..." -ForegroundColor Yellow
    python -m venv venv
    .\venv\Scripts\python.exe -m pip install --upgrade pip
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
}

# Start FastAPI in background
Write-Host "[+] Launching FastAPI REST Service on http://127.0.0.1:8000..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& '$ScriptDir\venv\Scripts\uvicorn.exe' api:app --host 127.0.0.1 --port 8000" -WindowStyle Minimized

# Start Streamlit Dashboard
Write-Host "[+] Launching Streamlit SOC Dashboard on http://127.0.0.1:8501..." -ForegroundColor Green
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "& '$ScriptDir\venv\Scripts\streamlit.exe' run app.py --server.port 8501"

Start-Sleep -Seconds 3
Write-Host "`n[OK] VoiceShield is fully deployed and accessible:" -ForegroundColor Green
Write-Host "   - SOC Dashboard : http://localhost:8501" -ForegroundColor Cyan
Write-Host "   - REST API Docs : http://127.0.0.1:8000/docs" -ForegroundColor Cyan
