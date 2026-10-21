@echo off
setlocal
set "DIR=%~dp0"
if exist "%DIR%\.venv\Scripts\python.exe" (
    "%DIR%\.venv\Scripts\python.exe" -m mandible_clarity %*
) else (
    python -m mandible_clarity %*
)
