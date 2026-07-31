@echo off
REM Double-click entry point. Real work lives in make_installer.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0make_installer.ps1" %*
pause
