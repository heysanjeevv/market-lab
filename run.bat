@echo off
rem Starts the dashboard at http://localhost:8810 and opens it in the browser.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (echo Run install.bat first. & pause & exit /b 1)
start "" http://localhost:8810
.venv\Scripts\python -u app.py
