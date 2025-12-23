@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0claude-code.ps1" %*
endlocal
