@echo off
REM Same as `make all` / `python scripts/run_all.py` — works without GNU Make on Windows.
cd /d "%~dp0"
python scripts/run_all.py %*
