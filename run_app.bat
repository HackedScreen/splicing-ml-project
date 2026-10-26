@echo off
setlocal enabledelayedexpansion
title Splicing ML Classifier

cd /d "%~dp0"
echo Starting Streamlit server...
where streamlit >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "Streamlit Server" cmd /c "cd /d "%~dp0" && streamlit run app/streamlit_app.py --server.headless true --server.port 8501"
) else (
    start "Streamlit Server" wsl -d Ubuntu bash -c "source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true; conda activate splicing_project 2>/dev/null || true; streamlit run app/streamlit_app.py --server.headless true --server.port 8501"
)

echo Waiting for http://localhost:8501 ...
set RETRIES=0
set MAX_RETRIES=25

:POLL_LOOP
curl -s -o nul http://localhost:8501/
if %ERRORLEVEL% EQU 0 goto SERVER_READY

set /a RETRIES+=1
if %RETRIES% GEQ %MAX_RETRIES% goto SERVER_FAILED

ping -n 2 127.0.0.1 >nul
goto POLL_LOOP

:SERVER_READY
echo Server ready on http://localhost:8501. Opening browser...
start "" "http://localhost:8501"
goto END

:SERVER_FAILED
echo Error: Streamlit server did not respond within %MAX_RETRIES% seconds.
pause
exit /b 1

:END
