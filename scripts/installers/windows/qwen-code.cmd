@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0qwen-code.ps1" %*
endlocal
