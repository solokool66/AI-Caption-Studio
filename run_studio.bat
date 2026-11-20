@echo off
title AI Caption Studio - Community Edition
cd /d "%~dp0"
echo ===================================================
echo   AI Caption Studio (Community Edition - Port 5200)
echo ===================================================
echo [*] Starting Python Flask server...
python caption_server.py
pause
