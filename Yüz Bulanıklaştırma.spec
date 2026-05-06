# -*- mode: python ; coding: utf-8 -*-
# Yüz Bulanıklaştırma v1.2 — PyInstaller Spec
# Yeni eklemeler: tkinterdnd2, piexif, BatchManagerWindow, VideoProcessorWindow

from PyInstaller.utils.hooks import collect_all, collect_data_files
import os

# ---- Veri dosyaları ----
datas = [
    ('blaze_face_short_range.tflite', '.'),
    ('haarcascade_frontalface_default.xml', '.'),
    ('app_icon.ico', '.'),
]

binaries = []
hiddenimports = [
    'piexif',
    'tkinterdnd2',
    'cv2',
    'mediapipe',
    'PIL._tkinter_finder',
]

# customtkinter — temaları ve fontları da topla
tmp = collect_all('customtkinter')
datas    += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# tkinterdnd2 — DLL'leri dahil et
tmp = collect_all('tkinterdnd2')
datas    += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

# mediapipe — model dosyaları ve meta verileri
tmp = collect_all('mediapipe')
datas    += tmp[0]; binaries += tmp[1]; hiddenimports += tmp[2]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'pandas', 'jupyter',
        'IPython', 'notebook', 'tkinter.test',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Yüz Bulanıklaştırma',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'python3*.dll'],
    runtime_tmpdir=None,
    console=False,           # Pencere uygulaması, terminal açılmasın
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app_icon.ico',
    version_file=None,
)
