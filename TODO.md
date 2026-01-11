# 🚀 Gelecek Özellikler ve İyileştirmeler

Bu dosya, Yüz Bulanıklaştırma Uygulaması için planlanan ve önerilebilecek özellikleri içerir.

---

## 🎯 Öncelikli Özellikler

### 1. Toplu İşlem (Batch Processing)
- [x] Birden fazla fotoğraf seçme
- [x] Tüm fotoğraflara otomatik yüz algılama ve bulanıklaştırma
- [x] İlerleme çubuğu
- [x] Sonuçları ayrı klasöre kaydetme
- [x] İşlem raporu (kaç yüz bulundu, hangi dosyalarda)

---

## ✨ Kullanıcı Deneyimi İyileştirmeleri

### 2. Bulanıklaştırma Stilleri
- [x] Gaussian Blur (mevcut)
- [x] Pikselleştirme (Mosaic/Pixelate)
- [x] Siyah dikdörtgen (Black box)
- [x] Emoji ile kapatma (😊, 🙈, ⭐)
- [ ] Özel görüntü ile kapatma (logo, sticker)
- [x] Renk dolgusu (Color fill)

### 3. Şekil Seçenekleri
- [ ] Elips (mevcut)
- [ ] Dikdörtgen
- [ ] Yuvarlak (tam daire)
- [ ] Serbest çizim (freehand)

### 4. Gelişmiş Seçim Araçları
- [x] Çoklu seçim modu (Tıklayarak seçim/iptal)
- [ ] Lasso seçim (serbest alan çizme)
- [ ] Magic wand (benzer renkleri seçme)
- [x] Seçimi büyüt/küçült (Yüz Alanı Genişletme Slider'ı)


### 5. Geri Al / Yinele (Undo/Redo)
- [x] Son işlemi geri alma (Ctrl+Z)
- [x] Geri alınan işlemi yineleme (Ctrl+Y / Ctrl+Shift+Z)
- [ ] İşlem geçmişi paneli (Gelecek sürüm için)


---

## 🔧 Teknik İyileştirmeler

### 6. Gelişmiş Yüz Algılama
- [x] Yan profil yüz algılama (Haar Profile Cascade entegrasyonu)
- [x] Maske/gözlük takan yüzleri algılama (MediaPipe Hibrit Mod ile)
- [ ] Çoklu model desteği (seçilebilir)
- [ ] Özel model yükleme
- [ ] Yüz landmark (göz, burun, ağız) algılama

## 🎨 Arayüz İyileştirmeleri

### 7. Tema ve Görünüm
- [x] Açık/Koyu tema geçişi
- [x] Özel renk temaları (Mavi, Yeşil, Koyu Mavi)
- [x] Yazı tipi boyutu ayarı (UI Ölçeklendirme ile)


### 8. Klavye Kısayolları
- [x] `Ctrl+O` - Fotoğraf aç
- [x] `Ctrl+S` - Kaydet
- [x] `F11` - Tam ekran (Aç/Kapat)
- [x] `Space` - Yüzleri algıla
- [x] `B` - Bulanıklaştır
- [x] `D` - Çizim modu
- [x] `Delete` - Seçili yüzü sil
- [x] `Ctrl+A` - Tüm yüzleri seç
- [x] `Escape` - Çizim modundan veya Tam ekrandan çık

### 9. Önizleme İyileştirmeleri
- [x] Zoom in/out (Mouse tekerleği ve +/- tuşları)
- [x] Pan (Sağ tık veya Shift+Sol tık ile sürükleme)
- [x] Orijinal/İşlenmiş karşılaştırma (👁️ butonu basılı tutularak)
- [x] Tam ekran önizleme (F11 modu ile)



### 10. Drag & Drop Desteği
- [ ] Dosyayı pencereye sürükleyip bırakma *(customtkinter ile uyumsuz - alternatif: Ctrl+O kısayolu)*
- [ ] Birden fazla dosya sürükleme (toplu işlem için)

---

## 📱 Platform ve Dağıtım

### 11. Çalıştırılabilir Dosya (Build & Deploy)
- [x] Windows: `Yüz Bulanıklaştırma.exe` oluşturma (Logo dahil ✅)
- [x] Windows: Installer (`setup.exe`) altyapısı (`installer_config.iss` hazır ✅)
- [x] GitHub Actions: Otomatik build sistemini kurma (İsimler güncellendi ✅)

### 12. Çapraz Platform Dağıtımı
- [x] Özel Uygulama Logosu (`app_icon.ico` oluşturuldu ✅)
- [x] macOS: `.app` paketi (Yüz_Bulanıklaştırma ✅)
- [x] Linux: `.AppImage` paketi (Yüz_Bulanıklaştırma ✅)



---

## 🔒 Gizlilik ve Güvenlik

### 13. Gizlilik Özellikleri
- [ ] İşlenen dosyaları otomatik silme seçeneği
- [ ] Metadata temizleme (EXIF verisi)
- [ ] Yerel işleme garantisi (internet bağlantısı gerektirmez)

---

## 📊 Ekstra Özellikler

### 14. Akıllı Öneriler
- [x] Yüz boyutuna göre otomatik blur seviyesi önerisi
- [x] Görüntü kalitesine göre ayar önerileri

### 15. Şablon ve Preset
- [ ] Sık kullanılan ayarları kaydetme

### 16. Raporlama
- [ ] İşlem özeti
- [ ] İstatistikler (toplam işlenen fotoğraf, algılanan yüz sayısı)

---

## 🐛 Bilinen Sorunlar ve Düzeltmeler

### Düzeltilecekler
- [x] Çok büyük resimlerde performans optimizasyonu (Hızlandırılmış Algılama)
- [x] Bellek kullanımını optimize etme (Sıkıştırılmış Geri Al/Yinele)
- [x] Hata mesajlarını daha açıklayıcı yapma (handle_error sistemi)


---

## 📝 Notlar

- Özellik önerileri için issue açabilirsiniz
- Her özellik için ayrı branch oluşturulmalı
- Yeni özellikler eklenmeden önce test edilmeli

---

## 🏆 Öncelik Sıralaması (Önerim)

1. **Toplu İşlem** - Pratik kullanım için önemli
2. **Bulanıklaştırma Stilleri** - Kullanıcı tercihlerini artırır
3. **Klavye Kısayolları** - Hızlı kullanım
4. **Drag & Drop** - Kolay kullanım
5. **Exe Dağıtım** - Kolay kurulum

---

*Son güncelleme: 2026-01-11*
