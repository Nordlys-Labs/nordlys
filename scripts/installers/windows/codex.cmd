@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0codex.ps1" %*
endlocal
