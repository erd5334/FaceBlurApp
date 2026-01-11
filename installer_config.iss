; FaceBlurApp Inno Setup Script
; Er Yazılım - Yüz Bulanıklaştırma Kurulum Dosyası

[Setup]
AppId={{C8BA0F1C-3D5F-4E9A-A3D5-F4E9A3D5F4E9}}
AppName=Yüz Bulanıklaştırma
AppVersion=1.0
AppPublisher=Er Yazılım
DefaultDirName={autopf}\Yüz Bulanıklaştırma
DefaultGroupName=Yüz Bulanıklaştırma
AllowNoIcons=yes
; Çıktı klasörü ve adı
OutputDir=dist
OutputBaseFilename=Yüz_Bulanıklaştırma_Setup
SetupIconFile=app_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Yüz Bulanıklaştırma.exe"; DestDir: "{app}"; Flags: ignoreversion
; Gerekli ayar dosyalarını veya modelleri tek dosya yapısına gömdüğümüz için 
; buraya ek dosya eklemeye gerek yok (PyInstaller hallediyor).

[Icons]
Name: "{group}\Yüz Bulanıklaştırma"; Filename: "{app}\Yüz Bulanıklaştırma.exe"
Name: "{autodesktop}\Yüz Bulanıklaştırma"; Filename: "{app}\Yüz Bulanıklaştırma.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Yüz Bulanıklaştırma.exe"; Description: "{cm:LaunchProgram,Yüz Bulanıklaştırma}"; Flags: nowait postinstall skipifsilent
