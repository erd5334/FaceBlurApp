; ============================================================
; Yüz Bulanıklaştırma v1.2 — Inno Setup Installer Script
; Er Yazılım | https://eryazilimci.com
; ============================================================

[Setup]
AppId={{C8BA0F1C-3D5F-4E9A-A3D5-F4E9A3D5F4E9}
AppName=Yüz Bulanıklaştırma
AppVersion=1.2.0
AppVerName=Yüz Bulanıklaştırma v1.2.0
AppPublisher=Er Yazılım
AppPublisherURL=https://eryazilimci.com
AppSupportURL=https://eryazilimci.com
AppUpdatesURL=https://eryazilimci.com
DefaultDirName={autopf}\Yüz Bulanıklaştırma
DefaultGroupName=Er Yazılım\Yüz Bulanıklaştırma
AllowNoIcons=yes
LicenseFile=
PrivilegesRequired=lowest
OutputDir=dist\installer
OutputBaseFilename=Yuz_Bulaniklaştirma_v1.2_Setup
SetupIconFile=app_icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardResizable=yes
VersionInfoVersion=1.2.0.0
VersionInfoCompany=Er Yazılım
VersionInfoDescription=Yüz Bulanıklaştırma Uygulaması
VersionInfoCopyright=Copyright © 2026 Er Yazılım
UninstallDisplayName=Yüz Bulanıklaştırma
UninstallDisplayIcon={app}\Yüz Bulanıklaştırma.exe

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon";    Description: "Masaüstüne kısayol oluştur";    GroupDescription: "Ek seçenekler:"; Flags: unchecked
Name: "startupicon";   Description: "Başlangıçta otomatik başlat";   GroupDescription: "Ek seçenekler:"; Flags: unchecked

[Files]
; Ana EXE — tek dosya, tüm bağımlılıklar içinde
Source: "dist\Yüz Bulanıklaştırma.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Yüz Bulanıklaştırma";               Filename: "{app}\Yüz Bulanıklaştırma.exe"
Name: "{group}\Yüz Bulanıklaştırma'yı Kaldır";     Filename: "{uninstallexe}"
Name: "{autodesktop}\Yüz Bulanıklaştırma";          Filename: "{app}\Yüz Bulanıklaştırma.exe"; Tasks: desktopicon
Name: "{userstartup}\Yüz Bulanıklaştırma";          Filename: "{app}\Yüz Bulanıklaştırma.exe"; Tasks: startupicon

[Run]
Filename: "{app}\Yüz Bulanıklaştırma.exe"; \
  Description: "Yüz Bulanıklaştırma uygulamasını şimdi başlat"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
