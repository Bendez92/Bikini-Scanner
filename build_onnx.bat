@echo off
setlocal

cd /d "%~dp0"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"
if not exist "%VENV_PY%" call :CreateVenv
if errorlevel 1 goto :End

"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :End
"%VENV_PY%" -m pip install -r requirements.txt -r requirements-onnx.txt -r requirements-build.txt
if errorlevel 1 goto :End
"%VENV_PY%" -m PyInstaller bikini_scanner_onnx.spec --clean --noconfirm
if errorlevel 1 goto :End

if defined BIKINI_SIGN_PFX (
    if not defined BIKINI_SIGN_PASS (
        echo BIKINI_SIGN_PASS is required when BIKINI_SIGN_PFX is set.
        goto :End
    )
    where signtool >nul 2>nul
    if not errorlevel 1 signtool sign /f "%BIKINI_SIGN_PFX%" /p "%BIKINI_SIGN_PASS%" /fd SHA256 "%~dp0dist\BikiniScanner-ONNX.exe"
) else (
    echo No BIKINI_SIGN_PFX supplied; producing an unsigned build.
)

echo ONNX build complete: %~dp0dist\BikiniScanner-ONNX.exe

:End
pause
exit /b %errorlevel%

:CreateVenv
call py -3.12 -m venv .venv
if not errorlevel 1 exit /b 0
call py -3 -m venv .venv
if not errorlevel 1 exit /b 0
call python -m venv .venv
if not errorlevel 1 exit /b 0
echo Failed to create virtual environment.
exit /b 1
