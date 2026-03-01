@echo off
echo ============================================
echo   Starting Waste Classification API Server
echo ============================================
echo.
cd /d %~dp0
echo Working directory: %CD%
echo.
echo Starting FastAPI server on http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause
