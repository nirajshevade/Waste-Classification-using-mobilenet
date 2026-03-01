@echo off
echo ============================================
echo   Starting Waste Classification Frontend
echo ============================================
echo.
cd /d %~dp0
echo Working directory: %CD%
echo.
echo Starting Streamlit on http://localhost:8501
echo.
streamlit run frontend/app.py --server.port 8501
pause
