@echo off
setlocal enabledelayedexpansion
title Splicing ML Classifier - Web App Launcher
echo ======================================================================
echo Launching Splicing ML Streamlit App inside WSL2 Ubuntu...
echo Model: Random Forest Corrected Classifier (74.01% Acc, 0.6474 Macro-F1)
echo ======================================================================
echo.

echo Starting Streamlit server in background...
start "Streamlit Server" wsl -d Ubuntu -u brooklyn bash -c "source /home/brooklyn/miniforge3/etc/profile.d/conda.sh && conda activate splicing_project && cd /home/brooklyn/splicing-ml-project && streamlit run app/streamlit_app.py --server.headless true --server.port 8501"

echo Waiting for server to become ready on http://localhost:8501 ...
set RETRIES=0
set MAX_RETRIES=25

:POLL_LOOP
curl -s -o nul http://localhost:8501/
if %ERRORLEVEL% EQU 0 goto SERVER_READY

set /a RETRIES+=1
if %RETRIES% GEQ %MAX_RETRIES% goto SERVER_FAILED

echo [%RETRIES%s] Server starting up, polling http://localhost:8501 ...
ping -n 2 127.0.0.1 >nul
goto POLL_LOOP

:SERVER_READY
echo.
echo ======================================================================
echo Server is LIVE at http://localhost:8501 (started in %RETRIES%s)!
echo Opening browser...
echo ======================================================================
start "" "http://localhost:8501"
goto END

:SERVER_FAILED
echo.
echo ======================================================================
echo [ERROR] Streamlit server failed to respond within %MAX_RETRIES% seconds.
echo Please check WSL terminal or run the command manually to inspect logs.
echo ======================================================================
pause
exit /b 1

:END
