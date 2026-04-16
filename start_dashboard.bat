@echo off
setlocal

cd /d "%~dp0"

start "Weather Dashboard Service" cmd /k "python china_weather_spider_analysis.py --serve-dashboard --workers 1"
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8765/dashboard"

endlocal
