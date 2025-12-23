@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0grok-cli.ps1" %*
endlocal
