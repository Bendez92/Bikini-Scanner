<#
.SYNOPSIS
    Builds Bikini Scanner end to end and produces a Windows installer.

.DESCRIPTION
    Creates/reuses .venv, installs dependencies, runs PyInstaller (onedir), then
    compiles installer-output\BikiniScannerSetup.exe with Inno Setup. Inno Setup is
    installed via winget if it is missing.

    Authenticode signing is opt-in. Set both to sign the app and the installer:
        $env:BIKINI_SIGN_PFX  = "C:\path\to\certificate.pfx"
        $env:BIKINI_SIGN_PASS = "certificate-password"

.PARAMETER SkipDeps
    Skip pip install. Use when .venv is already provisioned and only code changed.

.PARAMETER SkipBuild
    Skip PyInstaller and only recompile the installer from the existing dist folder.
#>
[CmdletBinding()]
param(
    [switch]$SkipDeps,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$AppDir = Join-Path $Root "dist\BikiniScannerApp"
$AppExe = Join-Path $AppDir "BikiniScanner.exe"
$SetupExe = Join-Path $Root "installer-output\BikiniScannerSetup.exe"

function Write-Step($Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

# Signs a file only when the user supplied a certificate; unsigned is the default.
function Invoke-SignFile($Path) {
    if (-not $env:BIKINI_SIGN_PFX) { return }
    if (-not $env:BIKINI_SIGN_PASS) {
        throw "BIKINI_SIGN_PASS is required when BIKINI_SIGN_PFX is set."
    }
    $signtool = Get-Command signtool -ErrorAction SilentlyContinue
    if (-not $signtool) {
        Write-Warning "signtool not found; leaving $(Split-Path -Leaf $Path) unsigned."
        return
    }
    & $signtool.Source sign /f $env:BIKINI_SIGN_PFX /p $env:BIKINI_SIGN_PASS /fd SHA256 $Path
    if ($LASTEXITCODE -ne 0) { throw "signtool failed for $Path" }
}

function Resolve-Iscc {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
        (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $cmd = Get-Command ISCC -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

# --- 1. Virtual environment -------------------------------------------------
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$FreshVenv = $false
if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating virtual environment"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } else {
        python -m venv .venv
    }
    if (-not (Test-Path $VenvPython)) { throw "Failed to create .venv" }
    $FreshVenv = $true
}

# --- 2. Dependencies --------------------------------------------------------
# -SkipDeps means "the venv is already provisioned". If it demonstrably is not -
# because this run just created it, or PyInstaller is missing - install anyway.
# Honouring the flag there guarantees a confusing "PyInstaller failed" several
# minutes later instead of a clear message now.
& $VenvPython -c "import PyInstaller" 2>$null
$HasPyInstaller = $LASTEXITCODE -eq 0
if ($SkipDeps -and ($FreshVenv -or -not $HasPyInstaller)) {
    Write-Host ""
    Write-Host "-SkipDeps was requested, but this .venv has no build dependencies yet." -ForegroundColor Yellow
    Write-Host "Installing them once (roughly 1 GB of downloads); later builds can skip this." -ForegroundColor Yellow
    $SkipDeps = $false
}
if (-not $SkipDeps) {
    Write-Step "Installing dependencies"
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip self-upgrade failed" }
    & $VenvPython -m pip install -r requirements.txt -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

# --- 3. PyInstaller ---------------------------------------------------------
if (-not $SkipBuild) {
    Write-Step "Building application bundle (this takes several minutes)"
    & $VenvPython -m PyInstaller bikini_scanner.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}
if (-not (Test-Path $AppExe)) {
    throw "Expected $AppExe. Run without -SkipBuild first."
}
Invoke-SignFile $AppExe

# --- 4. Inno Setup ----------------------------------------------------------
$Iscc = Resolve-Iscc
if (-not $Iscc) {
    Write-Step "Inno Setup not found; installing via winget"
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "winget unavailable. Install Inno Setup 6 manually: https://jrsoftware.org/isdl.php"
    }
    winget install --id JRSoftware.InnoSetup --accept-source-agreements --accept-package-agreements --disable-interactivity
    $Iscc = Resolve-Iscc
    if (-not $Iscc) { throw "Inno Setup install did not produce ISCC.exe" }
}

Write-Step "Compiling installer"
& $Iscc "installer.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup compilation failed" }
Invoke-SignFile $SetupExe

if (-not $env:BIKINI_SIGN_PFX) {
    Write-Host "No BIKINI_SIGN_PFX supplied; the installer is unsigned." -ForegroundColor Yellow
}

$SizeMb = [math]::Round((Get-Item $SetupExe).Length / 1MB, 1)
Write-Host ""
Write-Host "Installer ready: $SetupExe ($SizeMb MB)" -ForegroundColor Green
