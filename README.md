# 🎭 Yüz Bulanıklaştırma Uygulaması

Fotoğraflardaki yüzleri **otomatik olarak algılayan** ve **bulanıklaştıran** modern bir Python masaüstü uygulaması.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2+-purple.svg)

---

## ✨ Özellikler

### 🔍 Akıllı Yüz Algılama
- **MediaPipe** - Google'ın gelişmiş yapay zeka modeli (önerilen)
- **OpenCV Haar Cascade** - Klasik ve hızlı yöntem
- Birden fazla yüzü aynı anda algılama

### ✏️ Manuel Yüz Çizimi
- Otomatik algılama başarısız olursa **kendiniz çizin**!
- Fotoğraf üzerinde sürükleyerek elips çizme
- Çizilen yüzleri silme ve düzenleme

### 🎯 Seçici Bulanıklaştırma
- Her yüz için ayrı seçim (checkbox)
- **Yeşil** = Bulanıklaştırılacak
- **Kırmızı** = Bulanıklaştırılmayacak
- Tümünü seç / Tümünü kaldır butonları

### ⭕ Oval Bulanıklaştırma
- Dikdörtgen yerine **doğal elips şekli**
- Yumuşak kenar geçişleri
- Ayarlanabilir bulanıklaştırma seviyesi (1-100)

### 💾 Kolay Kaydetme
- PNG, JPEG, BMP formatlarında kaydetme
- Yüksek kalite çıktı

---

## 📸 Ekran Görüntüleri

```
┌─────────────────────────────────────────────────────────────────────┐
│ 🎭 Yüz Bulanıklaştırıcı                                     - □ X  │
├──────────────────────┬──────────────────────────────────────────────┤
│                      │                                              │
│ [📁 Fotoğraf Seç]    │                                              │
│                      │         ┌─────────────────┐                  │
│ Algılama Yöntemi     │         │    #1           │                  │
│ ● MediaPipe          │         │  ┌───────┐      │                  │
│ ○ OpenCV             │         │  │ 😊    │      │                  │
│                      │         │  └───────┘      │                  │
│ Bulanıklaştırma: 30  │         │                 │                  │
│ ═══════════════      │         └─────────────────┘                  │
│                      │                                              │
│ [🔍 Yüzleri Algıla]  │                                              │
│ [✏️ Manuel Yüz Çiz]  │                                              │
│ [✨ Bulanıklaştır]   │                                              │
│ [💾 Farklı Kaydet]   │                                              │
│ [🔄 Sıfırla]         │                                              │
│                      │                                              │
│ 🎯 Yüz Listesi       │                                              │
│ [Tümünü Seç] [Kaldır]│                                              │
│ ☑ Yüz #1        [🗑️] │                                              │
│ ☑ Yüz #2        [🗑️] │                                              │
│                      │                                              │
│ ┌──────────────────┐ │                                              │
│ │ ✅ 2 yüz bulundu │ │                                              │
│ └──────────────────┘ │                                              │
└──────────────────────┴──────────────────────────────────────────────┘
```

---

## 🚀 Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- Windows 10/11 (test edildi)

### Adım 1: Projeyi İndirin
```bash
git clone https://github.com/kullanici/FaceBlurApp.git
cd FaceBlurApp
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

Veya tek tek:
```bash
pip install opencv-python customtkinter Pillow numpy mediapipe
```

### Adım 4: Modeli İndirin
MediaPipe yüz algılama modeli:
```powershell
# PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" -OutFile "blaze_face_short_range.tflite"
```

Veya [buradan](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite) manuel indirin ve proje klasörüne koyun.

---

## 🎮 Kullanım

### Uygulamayı Başlatın
```bash
python main.py
```

### Adım Adım Kullanım

1. **📁 Fotoğraf Seç** → Bulanıklaştırmak istediğiniz fotoğrafı yükleyin

2. **🔍 Yüzleri Algıla** → Otomatik yüz algılama (MediaPipe veya OpenCV)

3. **✏️ Manuel Yüz Çiz** (isteğe bağlı) → Algılanamayan yüzleri kendiniz çizin:
   - Butona tıklayın (kırmızıya döner)
   - Fotoğraf üzerinde sürükleyerek elips çizin
   - İstediğiniz kadar yüz ekleyin

4. **🎯 Yüz Seçimi** → Bulanıklaştırmak istemediğiniz yüzlerin işaretini kaldırın

5. **⚙️ Bulanıklık Ayarı** → Kaydırıcı ile seviyeyi ayarlayın (1-100)

6. **✨ Bulanıklaştır** → Seçili yüzlere efekt uygulayın

7. **💾 Farklı Kaydet** → Sonucu kaydedin

---

## 📁 Proje Yapısı

```
FaceBlurApp/
├── main.py                              # Ana uygulama
├── requirements.txt                     # Python bağımlılıkları
├── README.md                            # Bu dosya
├── TODO.md                              # Gelecek özellikler
├── blaze_face_short_range.tflite        # MediaPipe modeli
├── haarcascade_frontalface_default.xml  # OpenCV cascade
└── venv/                                # Sanal ortam (oluşturulacak)
```

---

## 🛠️ Teknik Detaylar

| Bileşen | Teknoloji |
|---------|-----------|
| **Arayüz** | CustomTkinter (Modern Tkinter) |
| **Yüz Algılama** | MediaPipe Tasks API, OpenCV Haar Cascade |
| **Görüntü İşleme** | PIL/Pillow, NumPy |
| **Bulanıklaştırma** | Gaussian Blur + Elips Maske |

---

## ⌨️ Planlanan Kısayollar

| Kısayol | İşlev |
|---------|-------|
| `Ctrl+O` | Fotoğraf aç |
| `Ctrl+S` | Kaydet |
| `Space` | Yüzleri algıla |
| `D` | Çizim modu |
| `Escape` | Çizim modundan çık |

---

## 🐛 Sorun Giderme

### "Model dosyası bulunamadı" hatası
- `blaze_face_short_range.tflite` dosyasının proje klasöründe olduğundan emin olun

### Yüz algılanamıyor
- Farklı algılama yöntemini deneyin (MediaPipe ↔ OpenCV)
- Manuel çizim özelliğini kullanın
- Fotoğrafın net ve aydınlık olduğundan emin olun

### Türkçe karakter sorunu
- Proje klasör yolunda Türkçe karakter olmamalı
- Örnek: `C:\Python\FaceBlurApp` ✅
- Örnek: `C:\Users\Erdoğan\...` ❌

---

## 📋 Gelecek Özellikler

Detaylı liste için [TODO.md](TODO.md) dosyasına bakın.

**Öne Çıkanlar:**
- 🎬 Video desteği
- 📷 Gerçek zamanlı kamera
- 📦 Toplu işlem
- 🎨 Farklı bulanıklaştırma stilleri (pikselleştirme, emoji)
- 🖥️ Tek dosya .exe

---

## 📝 Lisans

MIT License - Özgürce kullanabilirsiniz.

---

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

---

## 👨‍💻 Geliştirici

Bu uygulama **Antigravity AI Assistant** yardımıyla geliştirildi.

---

*Son güncelleme: 2026-01-11*
