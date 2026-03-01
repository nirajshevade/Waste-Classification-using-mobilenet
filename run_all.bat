@echo off
echo ============================================
echo   Waste Classification System Launcher
echo ============================================
echo.
cd /d %~dp0
echo Working directory: %CD%
echo.

echo [1/2] Starting Backend API Server...
start "Waste Classification API" cmd /k "cd /d %~dp0 && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Waiting for API to start...
timeout /t 5 /nobreak > nul

echo.
echo [2/2] Starting Frontend Application...
start "Waste Classification Frontend" cmd /k "cd /d %~dp0 && streamlit run frontend/app.py --server.port 8501"

echo.
echo ============================================
echo   System Started Successfully!
echo ============================================
echo.
echo   API Server:    http://localhost:8000
echo   API Docs:      http://localhost:8000/docs
echo   Frontend:      http://localhost:8501
echo.
echo   Press any key to exit this launcher...
echo   (Servers will continue running)
echo ============================================
pause > nul
