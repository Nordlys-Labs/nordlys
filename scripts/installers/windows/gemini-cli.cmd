@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gemini-cli.ps1" %*
endlocal
