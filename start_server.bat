@echo off
title Smart Parking Server
cd /d "D:\Smart Parking\repo\backend"

echo Starting Smart Parking Server...
start /min "SmartParkingServer" "D:\Qwen 2.5 7B\env\python.exe" app_final.py

echo Waiting for server to be ready...
:wait
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/api/health >nul 2>&1
if errorlevel 1 goto wait

echo Server ready! Opening frontend...
start "" http://localhost:8000
echo.
echo Server is running. Close this window to stop the server.
echo To stop: taskkill /fi "WINDOWTITLE eq SmartParkingServer"
pause >nul
