# FaceBlurApp Windows Build Script
# Bu script uygulamayı tek bir .exe dosyası haline getirir.

Write-Host "🚀 Derleme işlemi başlatılıyor..." -ForegroundColor Cyan

# Sanal ortamı kontrol et
if (!(Test-Path ".\venv_new")) {
    Write-Host "❌ venv_new bulunamadı! Lütfen önce sanal ortamı kurun." -ForegroundColor Red
    exit
}

# Bağımlılıkları kontrol et/yükle
Write-Host "📦 Bağımlılıklar kontrol ediliyor..." -ForegroundColor Yellow
.\venv_new\Scripts\python.exe -m pip install -r requirements.txt

# PyInstaller ile derle
Write-Host "🔨 PyInstaller çalıştırılıyor..." -ForegroundColor Yellow
.\venv_new\Scripts\pyinstaller.exe --noconsole --onefile --name "FaceBlur_v1.0" `
    --add-data "blaze_face_short_range.tflite;." `
    --add-data "haarcascade_frontalface_default.xml;." `
    --collect-all customtkinter `
    --icon=NONE `
    main.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Başarılı! Uygulama 'dist' klasöründe oluşturuldu." -ForegroundColor Green
    Write-Host "📂 Dosya: dist\FaceBlur_v1.0.exe" -ForegroundColor Green
} else {
    Write-Host "`n❌ Derleme sırasında bir hata oluştu." -ForegroundColor Red
}
