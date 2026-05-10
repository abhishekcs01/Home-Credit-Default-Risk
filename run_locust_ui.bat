@echo off
REM Locust web UI — start API first in another terminal: python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
cd /d "%~dp0"
python scripts/run_locust_ui.py %*
