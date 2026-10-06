#define MyAppName "INVISIBLE³D"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "INVISIBLE³D Research"
#define MyAppExeName "INVISIBLE3D.exe"

[Setup]
AppId={{9A0B7B5C-6D62-4B6A-8E55-9E1A6F5E3D72}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\INVISIBLE3D
DefaultGroupName={#MyAppName}
OutputDir=dist-installer
OutputBaseFilename=INVISIBLE3D-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "dist\INVISIBLE3D\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
