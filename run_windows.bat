@echo off
echo ==========================================
echo   AgriSmart AI - 1-Click Local Setup
echo ==========================================
echo.

if not exist venv (
    echo [1/3] Creating Python virtual environment...
    python -m venv venv
)

echo [2/3] Activating virtual environment and checking requirements...
call venv\Scripts\activate
pip install -r requirements.txt -q

echo.
echo [3/3] Starting the AgriSmart AI Server...
echo ----------------------------------------------------
echo  Once you see "Application startup complete",
echo  open this link in your browser: http://127.0.0.1:8000
echo ----------------------------------------------------
echo.
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000
pause
