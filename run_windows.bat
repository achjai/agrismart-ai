@echo off
echo ==========================================
echo   AgriSmart AI - 1-Click Local Setup
echo ==========================================
echo.

REM Step 1: Create virtual environment if it doesn't exist
if not exist venv (
    echo [1/4] Creating Python virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists, skipping...
)

REM Step 2: Activate venv and install dependencies
echo [2/4] Installing dependencies (this may take a few minutes on first run)...
call venv\Scripts\activate
pip install -r requirements.txt -q

REM Step 3: Download model weights if not already present
if not exist "model\weights\RESNET50_FINETUNED.weights.h5" (
    echo [3/4] Downloading model weights from Google Drive (~214MB)...
    python -c "import gdown; gdown.download(id='1Hw-9exEnsLYtFLqYeOKogVtH6_XATw-8', output='model/weights/RESNET50_FINETUNED.weights.h5')"
) else (
    echo [3/4] Model weights already downloaded, skipping...
)

REM Step 4: Start the server
echo.
echo [4/4] Starting AgriSmart AI Server...
echo ============================================
echo   Open this in your browser:
echo   http://127.0.0.1:8000
echo ============================================
echo.
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
pause
