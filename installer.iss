; Inno Setup script for Bikini Scanner.
; Packages the PyInstaller onedir bundle from dist\BikiniScannerApp.
; Build end-to-end with make_installer.ps1; this script alone does not run PyInstaller.

#define MyAppName "Bikini Scanner"
; MyAppVersion can be overridden from the build script (make_installer.ps1 passes
; /DMyAppVersion=... so the installer and __version__.py never drift). This fallback
; is used only when ISCC is run standalone without that define.
#ifndef MyAppVersion
  #define MyAppVersion "1.3.0"
#endif
#define MyAppPublisher "Bikini Scanner"
#define MyAppExeName "BikiniScanner.exe"
#define MyAppSource "dist\BikiniScannerApp"

[Setup]
AppId={{B4E5B9E5-6A45-4E98-BE3C-1A5F7D64B4D8}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Bikini Scanner
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer-output
OutputBaseFilename=BikiniScannerSetup
SetupIconFile=assets\bikini_scanner.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The bundle ships native x64 binaries (torch), so refuse to install where it cannot run.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName} Setup

; Clear the previous payload before laying down the new one. Inno only overwrites files
; it is installing, so anything an older version shipped and this one does not would sit
; in {app} forever. Upgrading 1.1.0 -> 1.2.0 left 221 MB of orphans that way (scipy,
; sklearn, torch/include, 83 .lib files, the OpenCV ffmpeg DLL), cancelling out the size
; reduction on the target machine. Only the bundle directory is touched: user labels,
; caches and preferences live elsewhere.
[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#MyAppSource}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyAppSource}\*"; DestDir: "{app}"; Excludes: "{#MyAppExeName}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

; Scan caches live beside the scanned images and prefs live in %APPDATA%\bikini-scanner,
; so uninstall deliberately leaves both alone. Only the installed program files go.
