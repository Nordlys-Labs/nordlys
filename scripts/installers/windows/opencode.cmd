@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0opencode.ps1" %*
endlocal
