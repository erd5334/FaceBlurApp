"""
Yüz Bulanıklaştırma Uygulaması
Fotoğraflardaki yüzleri otomatik algılar ve bulanıklaştırır.
MediaPipe Tasks API ve OpenCV kullanarak yüz algılama yapar.
Kullanıcı manuel olarak da yüz bölgesi çizebilir.
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas
from PIL import Image, ImageFilter, ImageDraw, ImageTk
import cv2
import numpy as np
import os
from pathlib import Path
import threading
import json
import platform
import io
import sys

# Drag & Drop desteği
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# EXIF metadata temizleme
try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass
# MediaPipe Tasks API import

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe yüklenemedi.")


# Ayarlar dosyasını yükle
def load_settings():
    default_settings = {
        "appearance_mode": "dark",
        "color_theme": "blue",
        "ui_scaling": "100%"
    }
    try:
        if os.path.exists("settings.json"):
            with open("settings.json", "r") as f:
                return {**default_settings, **json.load(f)}
    except:
        pass
    return default_settings

def save_settings(settings):
    try:
        with open("settings.json", "w") as f:
            json.dump(settings, f)
    except:
        pass

# Başlangıç Ayarlarını Uygula
user_settings = load_settings()
ctk.set_appearance_mode(user_settings["appearance_mode"])
ctk.set_default_color_theme(user_settings["color_theme"])

# Drag & Drop için CTk ile TkinterDnD birleştirme
if DND_AVAILABLE:
    class _DnDBase(TkinterDnD.DnDWrapper, ctk.CTk):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
    _AppBase = _DnDBase
else:
    _AppBase = ctk.CTk

class FaceBlurApp(_AppBase):
    def __init__(self):
        super().__init__()
        
        # UI Ölçeklendirme ayarını yükle
        scaling = user_settings.get("ui_scaling", "100%")
        scaling_float = int(scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(scaling_float)
        ctk.set_window_scaling(scaling_float)

        # Pencere ayarları
        self.title("🎭 Yüz Bulanıklaştırıcı")
        self.geometry("1200x800")
        self.minsize(900, 600)
        
        # İşletim sistemi tespiti
        self.system = platform.system()
        self.is_fullscreen = False

        # Uygulama açıldığında ekranı kapla (Maximize)
        self.after(1000, self.maximize_window)
        
        # Pencere İkonu
        try:
            icon_path = self.get_resource_path('app_icon.ico')
            if os.path.exists(icon_path):
                if self.system == "Windows":
                    self.iconbitmap(icon_path)
                else:
                    img = ImageTk.PhotoImage(Image.open(icon_path))
                    self.wm_iconphoto(True, img)
        except:
            pass

    def get_resource_path(self, relative_path):
        """PyInstaller için kaynak dosyaların yolunu çöz (EXE uyumluluğu)"""
        try:
            # PyInstaller geçici klasör yolu (_MEIPASS)
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    def maximize_window(self):

        """İşletim sistemine göre en uygun ekranı kaplama yöntemi"""
        try:
            if self.system == "Windows":
                self.state("zoomed")
            elif self.system == "Darwin": # macOS
                self.state("zoomed") # macOS'ta da genelde çalışır
            else: # Linux
                self.attributes("-zoomed", True)
        except:
            # Fallback: Eğer hata verirse tam ekran yerine büyük bir pencere yap
            self.geometry("1400x900")


        
        # Değişkenler
        self.appearance_mode = ctk.StringVar(value=user_settings["appearance_mode"])

        self.color_theme = ctk.StringVar(value=user_settings["color_theme"])
        self.ui_scaling = ctk.StringVar(value=user_settings["ui_scaling"])
        
        self.original_image = None
        self.processed_image = None
        self.cv_image = None
        self.face_locations = []  # Tüm algılanan yüzler
        self.selected_faces = []  # Seçili yüzler (True/False listesi)
        self.blur_strength = ctk.IntVar(value=3)
        self.detection_method = ctk.StringVar(value="hybrid")
        self.blur_style = ctk.StringVar(value="gaussian")  # gaussian, pixelate, black, color, emoji, sticker
        self.blur_color = "#000000"  # Renk dolgusu için varsayılan renk
        self.face_margin = ctk.IntVar(value=15)  # Seçim alanı genişletme yüzdesi (%)
        self.sticker_image = None   # Özel sticker/logo (PIL Image)
        self.exif_strip = ctk.BooleanVar(value=True)  # EXIF metadata temizleme




        
        # Manuel çizim değişkenleri
        self.drawing_mode = False
        self.draw_start_x = None
        self.draw_start_y = None
        self.current_rect = None
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        
        # Zoom ve Pan değişkenleri
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Geri Al / Yinele (Undo/Redo) Sistem Değişkenleri
        self.undo_stack = []
        self.redo_stack = []
        self.max_stack_size = 20



        
        # Yüz algılama modelleri
        self.face_cascade = None
        self.face_detector = None
        self.load_detection_models()
        
        # UI oluştur
        self.create_ui()
        
        # Klavye kısayollarını bağla
        self.bind_keyboard_shortcuts()


        
    def load_detection_models(self):
        """Yüz algılama modellerini yükle"""
        # MediaPipe Face Detection (Tasks API)
        if MEDIAPIPE_AVAILABLE:
            try:
                # Model dosyasının yolu (EXE uyumlu)
                model_path = self.get_resource_path('blaze_face_short_range.tflite')
                
                if os.path.exists(model_path):
                    base_options = python.BaseOptions(model_asset_path=model_path)
                    options = vision.FaceDetectorOptions(
                        base_options=base_options,
                        min_detection_confidence=0.4,
                        min_suppression_threshold=0.3
                    )
                    self.face_detector = vision.FaceDetector.create_from_options(options)
                    print("MediaPipe yüz algılama hazır.")
                else:
                    print(f"Model dosyası bulunamadı: {model_path}")
                    self.face_detector = None
            except Exception as e:
                print(f"MediaPipe yükleme hatası: {e}")
                self.face_detector = None
        
        # OpenCV Haar Cascade (yedek olarak)
        self.profile_cascade = None
        try:
            # Önce frontal cascade yükle (EXE uyumlu)
            local_cascade = self.get_resource_path('haarcascade_frontalface_default.xml')
            if not os.path.exists(local_cascade):
                # Eğer yerelde yoksa cv2 içinden dene
                local_cascade = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

            if os.path.exists(local_cascade):
                self.face_cascade = cv2.CascadeClassifier(local_cascade)
                if not self.face_cascade.empty():
                    print("Frontal Haar Cascade hazır.")
            
            # Profile cascade yükle (Yan profiller için)
            profile_path = os.path.join(cv2.data.haarcascades, 'haarcascade_profileface.xml')
            if os.path.exists(profile_path):
                self.profile_cascade = cv2.CascadeClassifier(profile_path)
                if not self.profile_cascade.empty():
                    print("Profile Haar Cascade hazır.")
                    
        except Exception as e:
            print(f"Cascade yükleme hatası: {e}")
            self.face_cascade = None
            self.profile_cascade = None


    
    def bind_keyboard_shortcuts(self):
        """Klavye kısayollarını bağla"""
        # Ctrl+O - Fotoğraf aç
        self.bind("<Control-o>", lambda e: self.load_image())
        self.bind("<Control-O>", lambda e: self.load_image())
        
        # Ctrl+S - Kaydet (işlenmiş görüntü varsa)
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-S>", lambda e: self.save_image())
        
        # Ctrl+Shift+S - Farklı kaydet (aynı işlev)
        self.bind("<Control-Shift-S>", lambda e: self.save_image())
        self.bind("<Control-Shift-s>", lambda e: self.save_image())
        
        # Space - Yüzleri algıla
        self.bind("<space>", lambda e: self.detect_faces())
        
        # B - Bulanıklaştır
        self.bind("b", lambda e: self.apply_blur())
        self.bind("B", lambda e: self.apply_blur())
        
        # D - Çizim modu
        self.bind("d", lambda e: self.toggle_drawing_mode())
        self.bind("D", lambda e: self.toggle_drawing_mode())
        
        # Delete - Seçili yüzü sil (ilk seçili olanı)
        self.bind("<Delete>", lambda e: self.delete_first_selected_face())
        
        # Ctrl+A - Tüm yüzleri seç
        self.bind("<Control-a>", lambda e: self.select_all_faces())
        self.bind("<Control-A>", lambda e: self.select_all_faces())
        
        # Ctrl+Z - Geri Al
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-Z>", lambda e: self.undo())
        
        # Ctrl+Y / Ctrl+Shift+Z - Yinele
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-Y>", lambda e: self.redo())
        self.bind("<Control-Shift-Z>", lambda e: self.redo())
        self.bind("<Control-Shift-z>", lambda e: self.redo())

        # Escape - Çizim modundan çık veya Tam Ekrandan çık

        self.bind("<Escape>", self.on_escape_press)
        
        # F11 - Tam Ekran Toggle
        self.bind("<F11>", lambda e: self.toggle_fullscreen())
        
        # --- ZOOM & PAN BINDINGS ---

        # Mouse tekerleği ile Zoom (Windows/Linux/Mac uyumlu)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        
        # Klavyeden + ve - ile Zoom
        self.bind("<plus>", lambda e: self.zoom_in())
        self.bind("<KP_Add>", lambda e: self.zoom_in())
        self.bind("<minus>", lambda e: self.zoom_out())
        self.bind("<KP_Subtract>", lambda e: self.zoom_out())
        self.bind("0", lambda e: self.reset_zoom())
        
        # Sağ tık (veya Shift+Sol Tık) ile Sürükleme (Pan)
        self.canvas.bind("<Button-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-3>", self.stop_pan)
        
        # Alternatif: Shift + Sol Tık ile de sürüklenebilsin
        self.canvas.bind("<Shift-Button-1>", self.start_pan)
        self.canvas.bind("<Shift-B1-Motion>", self.do_pan)
        
        print("⌨️ Klavye kısayolları ve Zoom/Pan kontrolleri aktif!")

    
    def delete_first_selected_face(self):
        """İlk seçili yüzü sil"""
        if not self.face_locations:
            return
        
        # Seçili yüzleri bul
        for i, selected in enumerate(self.selected_faces):
            if selected:
                self.delete_face(i)
                return
        
        # Hiçbiri seçili değilse ilk yüzü sil
        if self.face_locations:
            self.delete_face(0)
    
    def exit_drawing_mode(self):
        """Çizim modundan güvenli çık"""
        if self.drawing_mode:
            self.toggle_drawing_mode()



    
    def create_ui(self):
        """Kullanıcı arayüzünü oluştur"""
        
        # Ana grid yapılandırması
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # ========== SOL PANEL - KONTROLLER ==========
        self.sidebar = ctk.CTkFrame(self, width=450, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # Scrollable frame for sidebar content
        self.sidebar_scroll = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.sidebar_scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Logo/Başlık
        self.logo_label = ctk.CTkLabel(
            self.sidebar_scroll, 
            text="🎭 Yüz\nBulanıklaştırıcı",
            font=ctk.CTkFont(size=22, weight="bold")
        )
        self.logo_label.pack(pady=(15, 8))
        
        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="Fotoğraflardaki yüzleri otomatik\nalgıla ve bulanıklaştır",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.subtitle_label.pack(pady=(0, 15))
        
        # Fotoğraf Yükle Butonu
        self.load_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="📁 Fotoğraf Seç",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            command=self.load_image
        )
        self.load_btn.pack(padx=15, pady=10, fill="x")
        
        # Toplu İşlem Butonu
        self.batch_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="📂 Toplu İşlem",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            fg_color="#F39C12",
            hover_color="#E67E22",
            command=self.batch_process
        )
        self.batch_btn.pack(padx=15, pady=5, fill="x")

        # Video Aç Butonu
        self.video_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="🎬 Video İşle",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            fg_color="#C0392B",
            hover_color="#96281B",
            command=self.open_video
        )
        self.video_btn.pack(padx=15, pady=5, fill="x")

        
        # Ayırıcı
        self.separator1 = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator1.pack(fill="x", padx=15, pady=10)

        
        # Algılama yöntemi hibrit olarak ayarlandı, seçim kutuları kaldırıldı 


        
        # Ayırıcı
        self.separator2 = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator2.pack(fill="x", padx=15, pady=15)
        
        # Bulanıklaştırma Stili
        self.style_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="🎨 Bulanıklaştırma Stili",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.style_label.pack(padx=15, anchor="w", pady=(0,5))
        
        self.style_gaussian = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="🌫️ Gaussian Blur",
            variable=self.blur_style,
            value="gaussian"
        )
        self.style_gaussian.pack(padx=25, pady=2, anchor="w")
        
        self.style_pixelate = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="🔲 Pikselleştirme",
            variable=self.blur_style,
            value="pixelate"
        )
        self.style_pixelate.pack(padx=25, pady=2, anchor="w")
        
        self.style_black = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="⬛ Siyah Kutu",
            variable=self.blur_style,
            value="black"
        )
        self.style_black.pack(padx=25, pady=2, anchor="w")
        
        self.style_color = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="🎨 Renk Dolgusu",
            variable=self.blur_style,
            value="color"
        )
        self.style_color.pack(padx=25, pady=2, anchor="w")
        
        self.style_emoji = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="😊 Emoji",
            variable=self.blur_style,
            value="emoji"
        )
        self.style_emoji.pack(padx=25, pady=2, anchor="w")

        # Sticker/Logo radio + seçim butonu
        self.style_sticker = ctk.CTkRadioButton(
            self.sidebar_scroll,
            text="🖼️ Sticker/Logo",
            variable=self.blur_style,
            value="sticker"
        )
        self.style_sticker.pack(padx=25, pady=2, anchor="w")

        self.sticker_select_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="📂 Logo Seç",
            font=ctk.CTkFont(size=11),
            height=28,
            fg_color="#2980B9",
            hover_color="#2471A3",
            command=self.select_sticker
        )
        self.sticker_select_btn.pack(padx=30, pady=(0, 4), anchor="w")

        self.sticker_name_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="(Logo seçilmedi)",
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.sticker_name_label.pack(padx=30, anchor="w")

        # Ayırıcı
        self.separator2b = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator2b.pack(fill="x", padx=15, pady=10)


        
        # Bulanıklaştırma Seviyesi
        self.blur_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="Bulanıklaştırma Seviyesi",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.blur_label.pack(padx=15, anchor="w")
        
        self.blur_value_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text=f"{self.blur_strength.get()}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#3B8ED0"
        )
        self.blur_value_label.pack(pady=5)
        
        self.blur_slider = ctk.CTkSlider(
            self.sidebar_scroll,
            from_=1,
            to=100,
            variable=self.blur_strength,
            command=self.on_blur_change
        )
        self.blur_slider.pack(padx=15, fill="x")
        
        # Akıllı Öneriler Label'ı
        self.suggestion_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.suggestion_frame.pack(padx=15, pady=(5, 0), fill="x")
        
        self.suggestion_label = ctk.CTkLabel(
            self.suggestion_frame,
            text="✨ Akıllı Öneri: (Henüz yok)",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color="#FFA500",
            justify="left",
            wraplength=200
        )
        self.suggestion_label.pack(side="left", anchor="w")
        
        self.apply_suggestion_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="🪄 Öneriyi Uygula",
            font=ctk.CTkFont(size=11),
            height=24,
            width=100,
            fg_color="#34495E",
            hover_color="#2C3E50",
            command=self.apply_smart_suggestion
        )
        self.apply_suggestion_btn.pack(padx=15, pady=2, anchor="e")
        self.apply_suggestion_btn.configure(state="disabled") # Yüz algılanana kadar pasif

        
        # Ayırıcı
        self.separator_margin = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator_margin.pack(fill="x", padx=15, pady=10)
        
        # Yüz Alanı Genişletme (Margin)
        self.margin_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="Yüz Alanı Genişletme (%)",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.margin_label.pack(padx=15, anchor="w")
        
        self.margin_value_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text=f"{self.face_margin.get()}%",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#2ECC71"
        )
        self.margin_value_label.pack(pady=5)
        
        self.margin_slider = ctk.CTkSlider(
            self.sidebar_scroll,
            from_=0,
            to=100,
            variable=self.face_margin,
            command=self.on_margin_change
        )
        self.margin_slider.pack(padx=15, fill="x")
        
        # Ayırıcı
        self.separator3 = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator3.pack(fill="x", padx=15, pady=10)

        
        # İşlem Butonları
        self.detect_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="🔍 Yüzleri Algıla",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#2D7D46",
            hover_color="#236B38",
            command=self.detect_faces
        )
        self.detect_btn.pack(padx=15, pady=5, fill="x")
        
        # Manuel Çizim Butonu
        self.draw_mode_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="✏️ Manuel Yüz Çiz",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#3498DB",
            hover_color="#2980B9",
            command=self.toggle_drawing_mode
        )
        self.draw_mode_btn.pack(padx=15, pady=5, fill="x")
        
        # Çizim modu bilgisi
        self.draw_mode_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="#3498DB"
        )
        self.draw_mode_label.pack(padx=15, anchor="w")
        
        self.blur_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="✨ Seçilenleri Bulanıklaştır",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#9B59B6",
            hover_color="#7D3C98",
            command=self.apply_blur
        )
        self.blur_btn.pack(padx=15, pady=5, fill="x")
        
        self.save_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="💾 Farklı Kaydet",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#E67E22",
            hover_color="#D35400",
            command=self.save_image
        )
        self.save_btn.pack(padx=15, pady=5, fill="x")

        # EXIF Metadata temizleme checkbox
        self.exif_checkbox = ctk.CTkCheckBox(
            self.sidebar_scroll,
            text="🔒 EXIF Metadata Temizle",
            variable=self.exif_strip,
            font=ctk.CTkFont(size=11)
        )
        self.exif_checkbox.pack(padx=15, pady=(0, 5), anchor="w")
        
        self.reset_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="🔄 Sıfırla",
            font=ctk.CTkFont(size=14),
            height=40,
            fg_color="#555555",
            hover_color="#444444",
            command=self.reset_image
        )
        self.reset_btn.pack(padx=15, pady=5, fill="x")

        
        self.compare_btn = ctk.CTkButton(
            self.sidebar_scroll,
            text="👁️ Orijinal/İşlenmiş (Basılı Tut)",
            font=ctk.CTkFont(size=12),
            height=30,
            fg_color="#34495E",
            hover_color="#2C3E50"
        )
        self.compare_btn.pack(padx=15, pady=5, fill="x")
        
        # Basılı tutma eventlerini bağla
        self.compare_btn.bind("<ButtonPress-1>", lambda e: self.show_original())
        self.compare_btn.bind("<ButtonRelease-1>", lambda e: self.show_processed())

        
        # Ayırıcı
        self.separator4 = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator4.pack(fill="x", padx=15, pady=15)
        
        # Yüz Seçimi Bölümü
        self.face_selection_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="🎯 Yüz Listesi",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.face_selection_label.pack(padx=15, anchor="w")
        
        self.face_selection_info = ctk.CTkLabel(
            self.sidebar_scroll,
            text="Yüzleri algılayın veya manuel çizin\nSonra bulanıklaştırmak istediklerinizi seçin",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.face_selection_info.pack(padx=15, anchor="w", pady=5)
        
        # Tümünü Seç / Hiçbirini Seçme butonları
        self.select_buttons_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.select_buttons_frame.pack(fill="x", padx=15, pady=5)
        
        self.select_all_btn = ctk.CTkButton(
            self.select_buttons_frame,
            text="Tümünü Seç",
            font=ctk.CTkFont(size=11),
            height=28,
            width=85,
            fg_color="gray40",
            hover_color="gray50",
            command=self.select_all_faces
        )
        self.select_all_btn.pack(side="left", padx=2)
        
        self.select_none_btn = ctk.CTkButton(
            self.select_buttons_frame,
            text="Tümünü Kaldır",
            font=ctk.CTkFont(size=11),
            height=28,
            width=85,
            fg_color="gray40",
            hover_color="gray50",
            command=self.deselect_all_faces
        )
        self.select_none_btn.pack(side="left", padx=2)
        
        # Yüz checkbox'ları için frame
        self.face_checkboxes_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="transparent")
        self.face_checkboxes_frame.pack(fill="x", padx=15, pady=5)
        
        # Checkbox listesi (dinamik olarak doldurulacak)
        self.face_checkbox_vars = []
        self.face_checkboxes = []
        self.face_delete_buttons = []
        
        # Durum Bilgisi
        self.status_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="gray20")
        self.status_frame.pack(fill="x", padx=10, pady=15)
        
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="📷 Bir fotoğraf seçin",
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        self.status_label.pack(pady=10)
        
        # --- GÖRÜNÜM AYARLARI ---
        self.separator_theme = ctk.CTkFrame(self.sidebar_scroll, height=2, fg_color="gray30")
        self.separator_theme.pack(fill="x", padx=15, pady=15)
        
        self.theme_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="🎨 Görünüm Ayarları",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.theme_label.pack(padx=15, anchor="w")
        
        # Tema Seçimi
        self.theme_mode_label = ctk.CTkLabel(self.sidebar_scroll, text="Tema Modu:", font=ctk.CTkFont(size=12))
        self.theme_mode_label.pack(padx=25, anchor="w", pady=(5, 0))
        
        self.theme_mode_menu = ctk.CTkOptionMenu(
            self.sidebar_scroll,
            values=["System", "Dark", "Light"],
            variable=self.appearance_mode,
            command=self.change_appearance_mode
        )
        self.theme_mode_menu.pack(padx=25, pady=5, fill="x")
        
        # Renk Teması
        self.color_theme_label = ctk.CTkLabel(self.sidebar_scroll, text="Renk Paleti:", font=ctk.CTkFont(size=12))
        self.color_theme_label.pack(padx=25, anchor="w", pady=(5, 0))
        
        self.color_theme_menu = ctk.CTkOptionMenu(
            self.sidebar_scroll,
            values=["blue", "green", "dark-blue"],
            variable=self.color_theme,
            command=self.change_color_theme
        )
        self.color_theme_menu.pack(padx=25, pady=5, fill="x")
        
        # UI Ölçeklendirme (Yazı Boyutu)
        self.scaling_label = ctk.CTkLabel(self.sidebar_scroll, text="Arayüz Ölçeği:", font=ctk.CTkFont(size=12))
        self.scaling_label.pack(padx=25, anchor="w", pady=(5, 0))
        
        self.scaling_menu = ctk.CTkOptionMenu(
            self.sidebar_scroll,
            values=["80%", "90%", "100%", "110%", "120%"],
            variable=self.ui_scaling,
            command=self.change_scaling
        )
        self.scaling_menu.pack(padx=25, pady=5, fill="x")
        
        # Sürüm Bilgisi
        self.version_label = ctk.CTkLabel(
            self.sidebar_scroll, 
            text="v1.2.0 | Er Yazılım", 
            font=ctk.CTkFont(size=10),
            text_color="gray40"
        )
        self.version_label.pack(pady=20)

        
        self.face_count_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#3B8ED0"
        )
        self.face_count_label.pack(pady=(0, 10))
        
        # Klavye Kısayolları Bilgisi
        self.shortcuts_frame = ctk.CTkFrame(self.sidebar_scroll, fg_color="gray20")
        self.shortcuts_frame.pack(fill="x", padx=10, pady=5)
        
        self.shortcuts_title = ctk.CTkLabel(
            self.shortcuts_frame,
            text="⌨️ Klavye Kısayolları",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3B8ED0"
        )
        self.shortcuts_title.pack(pady=(10, 5))
        
        shortcuts_text = (
            "Ctrl+O: Fotoğraf Aç\n"
            "Ctrl+S: Kaydet\n"
            "Space: Yüz Algıla\n"
            "B: Bulanıklaştır\n"
            "D: Çizim Modu\n"
            "Delete: Seçili Yüzü Sil\n"
            "Ctrl+A: Tümünü Seç\n"
            "Esc: Çizimden Çık"
        )
        
        self.shortcuts_label = ctk.CTkLabel(
            self.shortcuts_frame,
            text=shortcuts_text,
            font=ctk.CTkFont(size=10),
            text_color="gray70",
            justify="left"
        )
        self.shortcuts_label.pack(pady=(0, 10), padx=10)
        
        # Hızlı İpucu
        self.tip_label = ctk.CTkLabel(
            self.sidebar_scroll,
            text="💡 Hızlı Başlangıç: Ctrl+O ile fotoğraf açın!",
            font=ctk.CTkFont(size=10),
            text_color="#FFA500",
            wraplength=260
        )
        self.tip_label.pack(pady=10, padx=15)



        
        # ========== SAĞ PANEL - GÖRÜNTÜ ALANI ==========
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="gray10")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Görüntü frame
        self.image_frame = ctk.CTkFrame(self.main_frame, fg_color="gray15", corner_radius=15)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.image_frame.grid_columnconfigure(0, weight=1)
        self.image_frame.grid_rowconfigure(0, weight=1)
        
        # Canvas for image display and drawing
        self.canvas = Canvas(
            self.image_frame, 
            bg="#252525", 
            highlightthickness=0,
            cursor="arrow"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Placeholder text
        self.placeholder_text_id = self.canvas.create_text(
            400, 300,
            text="🖼️\n\nFotoğraf yüklemek için\n'Fotoğraf Seç' butonuna tıklayın\n\nveya Ctrl+O tuşlarına basın\n\nDosyayı buraya sürükleyip bırakabilirsiniz\n\nDesteklenen: PNG, JPG, JPEG, BMP, WEBP",
            fill="gray",
            font=("Segoe UI", 14),
            justify="center"
        )

        # Drag & Drop bağlantısı
        if DND_AVAILABLE:
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self._on_drop)
        
        # Canvas image reference
        self.canvas_image = None
        self.canvas_image_id = None


        
    def toggle_drawing_mode(self):
        """Manuel çizim modunu aç/kapat"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir fotoğraf yükleyin!")
            return
        
        self.drawing_mode = not self.drawing_mode
        
        if self.drawing_mode:
            self.draw_mode_btn.configure(
                text="✏️ Çizim Modu: AÇIK",
                fg_color="#E74C3C",
                hover_color="#C0392B"
            )
            self.draw_mode_label.configure(
                text="📌 Fotoğraf üzerinde sürükleyerek\n     yüz bölgesi çizin"
            )
            self.canvas.configure(cursor="crosshair")
            self.status_label.configure(text="✏️ Manuel çizim modu aktif")
        else:
            self.draw_mode_btn.configure(
                text="✏️ Manuel Yüz Çiz",
                fg_color="#3498DB",
                hover_color="#2980B9"
            )
            self.draw_mode_label.configure(text="")
            self.canvas.configure(cursor="arrow")
            self.status_label.configure(text="📷 Hazır")


    def toggle_fullscreen(self, event=None):
        """Pencereyi gerçek tam ekran moduna al/çıkar"""
        self.is_fullscreen = not self.is_fullscreen
        self.attributes("-fullscreen", self.is_fullscreen)
        
        if self.is_fullscreen:
            self.status_label.configure(text="📺 Tam ekran modu (Çıkmak için F11 veya ESC)")
        else:
            self.status_label.configure(text="🔲 Normal görünüm")
        return "break"

    def on_escape_press(self, event=None):
        """Escape tuşu davranışı: Önce çizim modundan çık, sonra tam ekrandan"""
        if self.drawing_mode:
            self.exit_drawing_mode()
        elif self.is_fullscreen:
            self.toggle_fullscreen()
        return "break"

    
    def show_original(self):
        """Orijinal görüntüyü geçici olarak göster"""
        if self.original_image:
            self.display_image(self.original_image)
            self.status_label.configure(text="👁️ Orijinal Görüntü (İşlenmemiş)")

    def show_processed(self):
        """İşlenmiş (veya önizlemeli) görüntüye geri dön"""
        self.refresh_display()
        if self.processed_image:
            self.status_label.configure(text="✅ İşlenmiş Görüntü")

    def canvas_to_img_coords(self, canvas_x, canvas_y):
        """Canvas koordinatlarını orijinal resim koordinatlarına çevir (Zoom/Pan uyumlu)"""
        img_x = (canvas_x - self.display_offset_x) / self.display_scale
        img_y = (canvas_y - self.display_offset_y) / self.display_scale
        return img_x, img_y

    def on_canvas_press(self, event):
        """Canvas'a tıklandığında"""
        if self.original_image is None:
            return
            
        # Eğer çizim modunda değilsek, tıklanan yerdeki yüzün seçimini değiştir
        if not self.drawing_mode:
            img_x, img_y = self.canvas_to_img_coords(event.x, event.y)
            
            # En son eklenen yüzden (en üstte görünen) başlayarak kontrol et
            for i in range(len(self.face_locations) - 1, -1, -1):
                x1, y1, x2, y2 = self.face_locations[i]
                
                # Tolerans ekleyelim (seçimi kolaylaştırmak için - zoom'a göre ayarla)
                padding = 10 / self.display_scale
                if (x1 - padding <= img_x <= x2 + padding and 
                    y1 - padding <= img_y <= y2 + padding):
                    
                    # Seçim durumunu tersine çevir (BooleanVar üzerinden)
                    if i < len(self.face_checkbox_vars):
                        current_val = self.face_checkbox_vars[i].get()
                        self.face_checkbox_vars[i].set(not current_val)
                        # on_face_selection_change() otomatik çağrılmayabilir, manuel çağıralım
                        self.on_face_selection_change()
                        return # Sadece tek bir yüzü seç/deselect et
            return

        self.draw_start_x = event.x
        self.draw_start_y = event.y
        
        # Önceki çizimi sil
        if self.current_rect:
            self.canvas.delete(self.current_rect)
    
    def on_canvas_drag(self, event):
        """Canvas'ta sürüklendiğinde"""
        if not self.drawing_mode or self.draw_start_x is None:
            return
        
        # Önceki çizimi sil
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Yeni dikdörtgen çiz (Görsel geri bildirim için elips)
        self.current_rect = self.canvas.create_oval(
            self.draw_start_x, self.draw_start_y,
            event.x, event.y,
            outline="#00FF00",  # Parlak yeşil
            width=2,
            dash=(5, 3)
        )
    
    def on_canvas_release(self, event):
        """Canvas'tan el çekildiğinde"""
        if not self.drawing_mode or self.draw_start_x is None:
            return
        
        # Minimum boyut kontrolü
        width_canvas = abs(event.x - self.draw_start_x)
        height_canvas = abs(event.y - self.draw_start_y)
        
        if width_canvas < 10 or height_canvas < 10:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.current_rect = None
            return
        
        # Koordinat dönüşümü
        ix1, iy1 = self.canvas_to_img_coords(self.draw_start_x, self.draw_start_y)
        ix2, iy2 = self.canvas_to_img_coords(event.x, event.y)
        
        # Normalize et
        xmin, xmax = sorted([ix1, ix2])
        ymin, ymax = sorted([iy1, iy2])
        
        # Sınırları kontrol et
        img_w, img_h = self.original_image.size
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)
        
        if (xmax - xmin) > 5 and (ymax - ymin) > 5:
            self._save_state()  # Yüz eklemeden önce durumu kaydet
            self.face_locations.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            self.selected_faces.append(True)

            self.update_face_checkboxes()
            self.update_preview_with_selection()
        
        # Çizimi kaldır ve moddan çık
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = None
        self.exit_drawing_mode()

        
        # Yüz listesine ekle
        self.face_locations.append((img_x1, img_y1, img_x2, img_y2))
        self.selected_faces.append(True)
        
        # Geçici çizimi sil
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = None
        
        # UI güncelle
        self.update_face_checkboxes()
        self.update_preview_with_selection()
        
        self.status_label.configure(text=f"✅ Manuel yüz eklendi")
        self.face_count_label.configure(text=f"🎭 {len(self.face_locations)} yüz")
        
        # Koordinatları sıfırla
        self.draw_start_x = None
        self.draw_start_y = None
    
    def update_face_checkboxes(self):
        """Yüz seçim checkbox'larını güncelle"""
        # Eski widget'ları temizle
        for cb in self.face_checkboxes:
            cb.destroy()
        for btn in self.face_delete_buttons:
            btn.destroy()
        self.face_checkboxes.clear()
        self.face_delete_buttons.clear()
        self.face_checkbox_vars.clear()
        
        # Yeni checkbox'lar oluştur
        for i, (x1, y1, x2, y2) in enumerate(self.face_locations):
            frame = ctk.CTkFrame(self.face_checkboxes_frame, fg_color="transparent")
            frame.pack(fill="x", pady=2)
            
            var = ctk.BooleanVar(value=self.selected_faces[i] if i < len(self.selected_faces) else True)
            self.face_checkbox_vars.append(var)
            
            cb = ctk.CTkCheckBox(
                frame,
                text=f"Yüz #{i+1}",
                variable=var,
                font=ctk.CTkFont(size=12),
                width=120,
                command=self.on_face_selection_change
            )
            cb.pack(side="left", anchor="w")
            self.face_checkboxes.append(cb)
            
            # Silme butonu
            del_btn = ctk.CTkButton(
                frame,
                text="🗑️",
                width=30,
                height=24,
                fg_color="#E74C3C",
                hover_color="#C0392B",
                command=lambda idx=i: self.delete_face(idx)
            )
            del_btn.pack(side="right", padx=2)
            self.face_delete_buttons.append(del_btn)
        
        # Seçili yüzler listesini güncelle
        while len(self.selected_faces) < len(self.face_locations):
            self.selected_faces.append(True)
    
    def delete_face(self, index):
        """Belirtilen yüzü sil"""
        if 0 <= index < len(self.face_locations):
            self._save_state()  # Silmeden önce kaydet
            del self.face_locations[index]

            del self.selected_faces[index]
            self.update_face_checkboxes()
            self.update_preview_with_selection()
            self.face_count_label.configure(text=f"🎭 {len(self.face_locations)} yüz")
    
    def on_face_selection_change(self):
        """Yüz seçimi değiştiğinde önizlemeyi güncelle"""
        self.selected_faces = [var.get() for var in self.face_checkbox_vars]
        self.update_preview_with_selection()
    
    def select_all_faces(self):
        """Tüm yüzleri seç"""
        if self.face_checkbox_vars:
            self._save_state()
            for var in self.face_checkbox_vars:
                var.set(True)
            self.on_face_selection_change()
    
    def deselect_all_faces(self):
        """Tüm yüzlerin seçimini kaldır"""
        if self.face_checkbox_vars:
            self._save_state()
            for var in self.face_checkbox_vars:
                var.set(False)
            self.on_face_selection_change()

    
    def update_preview_with_selection(self):
        """Seçili yüzleri farklı renkte göster"""
        if self.original_image is None:
            return
        
        preview_image = self.original_image.copy()
        draw = ImageDraw.Draw(preview_image)
        
        margin_percent = self.face_margin.get() / 100.0
        
        for i, (x1, y1, x2, y2) in enumerate(self.face_locations):
            # Seçili mi kontrol et
            is_selected = i < len(self.selected_faces) and self.selected_faces[i]
            
            if is_selected:
                # Seçili ise margin hesapla
                w = x2 - x1
                h = y2 - y1
                mx = w * margin_percent
                my = h * margin_percent
                
                # Yeni koordinatlar
                img_w, img_h = self.original_image.size
                nx1 = max(0, x1 - mx)
                ny1 = max(0, y1 - my)
                nx2 = min(img_w, x2 + mx)
                ny2 = min(img_h, y2 + my)
                color = "#00FF00"  # Yeşil - seçili
            else:
                # Seçili değilse orijinal koordinatları kullan
                nx1, ny1, nx2, ny2 = x1, y1, x2, y2
                color = "#FF6B6B"  # Kırmızı - seçili değil
            
            # Elips çiz
            padding = 5
            for j in range(4):
                draw.ellipse(
                    [nx1-padding-j, ny1-padding-j, nx2+padding+j, ny2+padding+j], 
                    outline=color
                )
            
            # Numara etiketi
            text = f"#{i+1}"
            text_x = nx1 - 5
            text_y = ny1 - 25
            if text_y < 5:
                text_y = ny2 + 5
            draw.rectangle([text_x, text_y, text_x + 35, text_y + 20], fill=color)
            draw.text((text_x + 5, text_y + 2), text, fill="black")


        
        self.display_image(preview_image)
        
    def on_blur_change(self, value):
        """Bulanıklaştırma değeri değiştiğinde"""
        self.blur_value_label.configure(text=f"{int(value)}")
    
    def on_margin_change(self, value):
        """Margin değeri değiştiğinde"""
        self.margin_value_label.configure(text=f"{int(value)}%")
        if self.face_locations:
            self.update_preview_with_selection()
            # Önerileri de margin genişliğine göre güncelle
            self._update_smart_suggestions()

    def change_appearance_mode(self, new_appearance_mode: str):
        """Açık/Koyu tema değişimi"""
        ctk.set_appearance_mode(new_appearance_mode)
        self._save_app_settings()

    def change_color_theme(self, new_color_theme: str):
        """Renk teması değişimi (Restart gerekebilir uyarısı ile)"""
        # CTK'da default color theme çalışma anında tam değişmeyebilir
        # o yüzden kaydedip restart istiyoruz.
        self._save_app_settings()
        messagebox.showinfo("Yeniden Başlatma Gerekli", 
                          f"Renk paleti '{new_color_theme}' olarak kaydedildi.\n\n"
                          "Değişikliklerin tüm bileşenlerde tam olarak uygulanması için "
                          "lütfen uygulamayı kapatıp tekrar açın.")

    def change_scaling(self, new_scaling: str):
        """UI Ölçeklendirme (Yazı boyutu)"""
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
        ctk.set_window_scaling(new_scaling_float)
        self._save_app_settings()

    def _save_app_settings(self):
        """Mevcut görünüm ayarlarını settings.json dosyasına kaydet"""
        current_settings = {
            "appearance_mode": self.appearance_mode.get(),
            "color_theme": self.color_theme.get(),
            "ui_scaling": self.ui_scaling.get()
        }
        save_settings(current_settings)

    def handle_error(self, e, context="İşlem"):
        """Merkezi hata yönetimi ve kullanıcı bilgilendirme"""
        error_msg = str(e)
        tip = ""
        
        # Spesifik hata tiplerine göre ipuçları
        if "MemoryError" in error_msg:
            tip = "\n\n💡 İpucu: Sistem belleği yetersiz. Daha küçük resimlerle denemeyi veya diğer uygulamaları kapatmayı deneyin."
        elif "Permission denied" in error_msg:
            tip = "\n\n💡 İpucu: Dosyaya erişim izni yok. Dosyanın başka bir programda açık olmadığından emin olun."
        elif "Invalid image" in error_msg or "cannot identify image" in error_msg:
            tip = "\n\n💡 İpucu: Bu dosya formatı desteklenmiyor veya dosya bozuk."
            
        full_msg = f"{context} sırasında bir hata oluştu:\n{error_msg}{tip}"
        print(f"HATA [{context}]: {error_msg}")
        
        # UI üzerinden bildir
        self.status_label.configure(text=f"⚠️ {context} Hatası!", text_color="#E74C3C")
        messagebox.showerror("Hata", full_msg)



    def load_image(self):
        """Dosya seçici ile görüntü yükle"""
        file_path = filedialog.askopenfilename(
            title="Fotoğraf Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_image_from_path(self, file_path):
        """Belirtilen yoldan görüntü yükle"""
        try:
            # PIL ile yükle
            self.original_image = Image.open(file_path)
            
            # RGBA ise RGB'ye çevir
            if self.original_image.mode == 'RGBA':
                background = Image.new('RGB', self.original_image.size, (255, 255, 255))
                background.paste(self.original_image, mask=self.original_image.split()[3])
                self.original_image = background
            elif self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            
            self.processed_image = self.original_image.copy()
            
            # NumPy array olarak sakla (MediaPipe için)
            self.cv_image = np.array(self.original_image)
            
            # Yüz konumlarını ve geçmişi sıfırla
            self.face_locations = []
            self.selected_faces = []
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_face_checkboxes()

            
            # Çizim modunu kapat
            if self.drawing_mode:
                self.toggle_drawing_mode()
            
            # Görüntüyü göster
            self.display_image(self.original_image)
            
            # Placeholder'ı gizle
            self.canvas.delete(self.placeholder_text_id)
            
            # Durum güncelle
            file_name = Path(file_path).name
            self.status_label.configure(text=f"📷 {file_name}")
            self.face_count_label.configure(text="")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu:\n{e}")
    
    # --- ZOOM & PAN METHODS ---
    def on_mouse_wheel(self, event):
        """Mouse tekerleği ile zoom"""
        if self.original_image is None: return
        
        # Windows/Linux/Mac farkı
        if event.num == 4 or event.delta > 0: # Yukarı
            self.zoom_in()
        elif event.num == 5 or event.delta < 0: # Aşağı
            self.zoom_out()
            
    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.2, 10.0) # Maks 10x
        self.refresh_display()
        
    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.2, 0.1) # Min 0.1x
        self.refresh_display()
        
    def reset_zoom(self):
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.refresh_display()
        
    def start_pan(self, event):
        """Kaydırmayı başlat"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
        
    def do_pan(self, event):
        """Kaydır"""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.refresh_display()
            
    def stop_pan(self, event):
        """Kaydırmayı bitir"""
        self.is_panning = False
        self.canvas.config(cursor="")

    def refresh_display(self):
        """Mevcut durumu (orijinal veya işlenmiş) yeniden çiz"""
        current_img = self.processed_image if self.processed_image else self.original_image
        if current_img:
            # Eğer algılama yapılmışsa ve işlenmiş görüntü yoksa seçim halkalarını göster
            if not self.processed_image and self.face_locations:
                self.update_preview_with_selection()
            else:
                self.display_image(current_img)

    def display_image(self, pil_image):
        """Görüntüyü canvas'ta göster (Zoom ve Pan destekli)"""
        if pil_image is None:
            return
        
        self.canvas.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1: canvas_width = 800
        if canvas_height <= 1: canvas_height = 600
        
        img_width, img_height = pil_image.size
        
        # Temel fit scale (resmin canvas'a sığması için gereken scale)
        base_scale = min(canvas_width / img_width, canvas_height / img_height) * 0.95
        self.display_scale = base_scale * self.zoom_level
        
        new_width = int(img_width * self.display_scale)
        new_height = int(img_height * self.display_scale)
        
        # Merkezleme + Pan ofseti
        self.display_offset_x = (canvas_width - new_width) // 2 + self.pan_offset_x
        self.display_offset_y = (canvas_height - new_height) // 2 + self.pan_offset_y
        
        # Performans için sadece görünür alanı resize etsek iyi olurdu ama 
        # şimdilik basit tutup tüm resmi resize ediyoruz (PIL Lanczos kalitelidir)
        try:
            # Çok küçük veya çok büyük resize hatalarını engelle
            if new_width < 1 or new_height < 1: return
            
            resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            self.canvas_image = ImageTk.PhotoImage(resized_image)
            
            if self.canvas_image_id:
                self.canvas.delete(self.canvas_image_id)
            
            self.canvas_image_id = self.canvas.create_image(
                self.display_offset_x, self.display_offset_y,
                anchor="nw",
                image=self.canvas_image
            )
        except Exception as e:
            print(f"Görüntüleme hatası: {e}")

    
    def detect_faces(self):
        """Yüzleri algıla"""
        if self.cv_image is None:
            messagebox.showwarning("Uyarı", "Önce bir fotoğraf yükleyin!")
            return
        
        self.status_label.configure(text="🔍 Yüzler algılanıyor...")
        self.update()
        
        # Çizim modunu kapat
        if self.drawing_mode:
            self.toggle_drawing_mode()
        
        # Thread ile algılama yap
        thread = threading.Thread(target=self._detect_faces_thread)
        thread.start()
    
    def _detect_faces_thread(self):
        """Yüz algılama işlemi (arka plan thread'i)"""
        try:
            method = self.detection_method.get()
            print(f"Seçili yöntem: {method}")
            
            # Senkron algılama metodunu kullan (tutarlılık için)
            new_faces = self._detect_faces_sync(self.cv_image)
            
            if not new_faces and method != "hybrid" and not (self.face_detector or self.face_cascade):
                self.after(0, lambda: messagebox.showerror(
                    "Hata",
                    "Yüz algılama modeli yüklenemedi."
                ))
                return

            
            # Mevcut durumu geri alma için kaydet (yüzler eklenmeden önce)
            self.after(0, self._save_state)
            
            # Mevcut yüzlere ekle (çakışanları atla)
            for new_face in new_faces:

                if not self._is_duplicate_face(new_face):
                    self.face_locations.append(new_face)
                    self.selected_faces.append(True)
            
            # Sonuçları göster
            self.after(0, lambda m=method: self._show_detection_results(m))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Hata", f"Yüz algılama hatası:\n{e}"))
    
    def _is_duplicate_face(self, new_face, threshold=0.5):
        """Yeni yüzün mevcut yüzlerle çakışıp çakışmadığını kontrol et"""
        nx1, ny1, nx2, ny2 = new_face
        
        for (x1, y1, x2, y2) in self.face_locations:
            # IoU (Intersection over Union) hesapla
            inter_x1 = max(x1, nx1)
            inter_y1 = max(y1, ny1)
            inter_x2 = min(x2, nx2)
            inter_y2 = min(y2, ny2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (nx2 - nx1) * (ny2 - ny1)
                union_area = area1 + area2 - inter_area
                
                iou = inter_area / union_area
                if iou > threshold:
                    return True
        
        return False
    
    def _show_detection_results(self, method="unknown"):
        """Algılama sonuçlarını göster"""
        count = len(self.face_locations)
        if method == "mediapipe":
            method_name = "MediaPipe"
        elif method == "opencv_haar":
            method_name = "OpenCV"
        else:
            method_name = "Hibrit"

        
        # Checkbox'ları güncelle
        self.update_face_checkboxes()
        
        # Akıllı önerileri güncelle
        self._update_smart_suggestions()
        
        if count > 0:
            self.update_preview_with_selection()
            self.status_label.configure(text=f"✅ {method_name} ile algılandı")
            self.face_count_label.configure(text=f"🎭 {count} yüz bulundu")
        else:
            self.status_label.configure(text="❌ Yüz bulunamadı - Manuel çizin!")
            self.face_count_label.configure(text="")
            messagebox.showinfo(
                "Bilgi", 
                "Fotoğrafta yüz algılanamadı.\n\n"
                "💡 İpucu: 'Manuel Yüz Çiz' butonuna tıklayarak\n"
                "kendiniz yüz bölgesi çizebilirsiniz!"
            )
            
    def _update_smart_suggestions(self):
        """Yüz boyutuna ve görüntü kalitesine göre öneri oluştur"""
        if not self.face_locations or self.original_image is None:
            self.suggestion_label.configure(text="✨ Akıllı Öneri: (Henüz yok)")
            self.apply_suggestion_btn.configure(state="disabled")
            return
            
        margin_percent = self.face_margin.get() / 100.0
        img_w, img_h = self.original_image.size
        
        # Ortalama efektif yüz boyutlarını hesapla (margin dahil piksel cinsinden)
        total_eff_w = 0
        count = 0
        
        for i, (x1, y1, x2, y2) in enumerate(self.face_locations):
            # Seçili olanları baz al (yoksa hepsini)
            is_selected = i < len(self.selected_faces) and self.selected_faces[i]
            if is_selected or not any(self.selected_faces):
                face_w = x2 - x1
                # Efektif genişlik (margin dahil)
                eff_w = face_w * (1 + 2 * margin_percent)
                total_eff_w += eff_w
                count += 1
        
        if count == 0: return
        avg_face_w = total_eff_w / count
        
        # --- GELİŞMİŞ ÖNERİ MANTIĞI ---
        
        # 1. Stil Seçimi: 
        # Sadece 60 pikselden küçük yüzlerde Pixelate öner (Çünkü detay azdır)
        # Daha büyük yüzlerde kalite için her zaman Gaussian öner.
        if avg_face_w < 60:
            suggested_style = "pixelate"
            style_name = "Pikselleştirme"
            reason = "Küçük yüzlerde pikselleştirme daha güvenlidir."
        else:
            suggested_style = "gaussian"
            style_name = "Gaussian Blur"
            reason = "Yeterli çözünürlük olduğu için doğal blur önerilir."
            
        # 2. Seviye Seçimi:
        # Alan genişledikçe veya yüz küçüldükçe blur ihtiyacı artar.
        if avg_face_w < 40:
            suggested_blur = 30 # Çok küçük yüzlere yoğun blur
        elif avg_face_w < 100:
            suggested_blur = 15 # Orta boy
        elif avg_face_w < 300:
            suggested_blur = 5  # Büyük yüzler
        else:
            suggested_blur = 2  # Çok büyük portreler (hafif blur yeterlidir)
            
        self.smart_params = {
            "blur_strength": suggested_blur,
            "blur_style": suggested_style
        }
        
        # UI Güncelle
        self.suggestion_label.configure(
            text=f"✨ Öneri: {style_name} (Sertlik: {suggested_blur})\n"
                 f"💡 {reason}"
        )
        self.apply_suggestion_btn.configure(state="normal")


    def apply_smart_suggestion(self):
        """Önerilen ayarları uygula"""
        if hasattr(self, 'smart_params'):
            self.blur_strength.set(self.smart_params["blur_strength"])
            self.blur_style.set(self.smart_params["blur_style"])
            self.on_blur_change(self.smart_params["blur_strength"])
            self.status_label.configure(text="🪄 Akıllı öneriler uygulandı!")
            # Önizlemeyi güncelle (stil değişmiş olabilir)
            if self.face_locations:
                self.update_preview_with_selection()

    
    def apply_blur(self):
        """Bulanıklaştırma uygula"""
        if self.original_image is None:
            messagebox.showwarning("Uyarı", "Önce bir fotoğraf yükleyin!")
            return
            
        if not self.face_locations:
            messagebox.showwarning("Uyarı", "Önce yüz algılayın veya manuel çizin!")
            return
        
        # Seçili yüz var mı kontrol et
        if not any(self.selected_faces):
            messagebox.showwarning("Uyarı", "Bulanıklaştırmak için en az bir yüz seçin!")
            return
        
        # Seçili stili al
        blur_style = self.blur_style.get()
        
        self.status_label.configure(text="✨ İşleniyor...")
        self.update()
        
        # İşlemden önce durumu kaydet
        self._save_state()
        
        try:

            # Orijinal görüntünün kopyasını al
            result_image = self.original_image.copy()
            blur_strength = int(self.blur_strength.get())
            
            margin_percent = self.face_margin.get() / 100.0
            img_w, img_h = self.original_image.size
            blurred_count = 0

            
            for i, (x1, y1, x2, y2) in enumerate(self.face_locations):
                # Sadece seçili yüzleri işle
                if i >= len(self.selected_faces) or not self.selected_faces[i]:
                    continue
                
                # Margin hesapla
                w = x2 - x1
                h = y2 - y1
                mx = w * margin_percent
                my = h * margin_percent
                
                nx1 = int(max(0, x1 - mx))
                ny1 = int(max(0, y1 - my))
                nx2 = int(min(img_w, x2 + mx))
                ny2 = int(min(img_h, y2 + my))
                
                blurred_count += 1
                
                # Seçili stile göre işlem yap
                if blur_style == "gaussian":
                    result_image = self._apply_gaussian_blur(result_image, nx1, ny1, nx2, ny2, blur_strength)
                elif blur_style == "pixelate":
                    result_image = self._apply_pixelate(result_image, nx1, ny1, nx2, ny2, blur_strength)
                elif blur_style == "black":
                    result_image = self._apply_black_box(result_image, nx1, ny1, nx2, ny2)
                elif blur_style == "color":
                    result_image = self._apply_color_fill(result_image, nx1, ny1, nx2, ny2)
                elif blur_style == "emoji":
                    result_image = self._apply_emoji(result_image, nx1, ny1, nx2, ny2)
                elif blur_style == "sticker":
                    result_image = self._apply_sticker(result_image, nx1, ny1, nx2, ny2)

            
            self.processed_image = result_image
            self.display_image(self.processed_image)
            
            style_names = {
                "gaussian": "Blur",
                "pixelate": "Pikselleştirme",
                "black": "Siyah Kutu",
                "color": "Renk Dolgusu",
                "emoji": "Emoji"
            }
            style_name = style_names.get(blur_style, "İşlem")
            self.status_label.configure(text=f"✅ {blurred_count} yüz - {style_name}")
            
        except Exception as e:
            messagebox.showerror("Hata", f"İşlem hatası:\n{e}")

    # --- UNDO / REDO METHODS (MEMORY OPTIMIZED) ---
    def _save_state(self):
        """Mevcut durumu geri alma yığınına kaydet (Bellek Dostu - JPEG Sıkıştırma)"""
        import io
        
        # Görüntüyü bellek içinde sıkıştırarak sakla (RAM tasarrufu)
        buffer = None
        if self.processed_image:
            buffer = io.BytesIO()
            self.processed_image.save(buffer, format="JPEG", quality=85)
            img_data = buffer.getvalue()
        else:
            img_data = None

        state = {
            "face_locations": list(self.face_locations),
            "selected_faces": list(self.selected_faces),
            "processed_image_data": img_data
        }
        self.undo_stack.append(state)
        self.redo_stack.clear()
        
        if len(self.undo_stack) > self.max_stack_size:
            self.undo_stack.pop(0)

    def undo(self, event=None):
        """Son işlemi geri al"""
        if not self.undo_stack:
            self.status_label.configure(text="ℹ️ Geri alınacak işlem yok")
            return
            
        current_state = self._get_current_state_serialized()
        self.redo_stack.append(current_state)
        
        last_state = self.undo_stack.pop()
        self._apply_state_serialized(last_state)
        self.status_label.configure(text="↩️ İşlem geri alındı")

    def redo(self, event=None):
        """Geri alınan işlemi yinele"""
        if not self.redo_stack:
            self.status_label.configure(text="ℹ️ İleri alınacak işlem yok")
            return
            
        current_state = self._get_current_state_serialized()
        self.undo_stack.append(current_state)
        
        next_state = self.redo_stack.pop()
        self._apply_state_serialized(next_state)
        self.status_label.configure(text="↪️ İşlem yinelendi")

    def _get_current_state_serialized(self):
        """Mevcut durumu sıkıştırılmış formatta al"""
        import io
        img_data = None
        if self.processed_image:
            buf = io.BytesIO()
            self.processed_image.save(buf, format="JPEG", quality=85)
            img_data = buf.getvalue()
            
        return {
            "face_locations": list(self.face_locations),
            "selected_faces": list(self.selected_faces),
            "processed_image_data": img_data
        }

    def _apply_state_serialized(self, state):
        """Sıkıştırılmış durumu çöz ve uygula"""
        import io
        self.face_locations = list(state["face_locations"])
        self.selected_faces = list(state["selected_faces"])
        
        if state["processed_image_data"]:
            self.processed_image = Image.open(io.BytesIO(state["processed_image_data"]))
        else:
            self.processed_image = self.original_image.copy() if self.original_image else None
            
        # UI Güncelle
        self.update_face_checkboxes()
        if self.processed_image:
            self.display_image(self.processed_image)
            
        if self.face_locations:
            self.update_preview_with_selection()
        self._update_smart_suggestions()


    
    def _apply_gaussian_blur(self, image, x1, y1, x2, y2, strength):
        """Gaussian blur uygula"""
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Yüz bölgesini kırp
        face_region = image.crop((x1, y1, x2, y2))
        
        # Gaussian blur uygula
        blurred_face = face_region.filter(
            ImageFilter.GaussianBlur(radius=strength)
        )
        
        # Elips maskesi oluştur
        mask = Image.new('L', (face_width, face_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, face_width, face_height], fill=255)
        
        # Maskeyi yumuşat
        mask = mask.filter(ImageFilter.GaussianBlur(radius=10))
        
        # Blurlanmış yüzü yapıştır
        image.paste(blurred_face, (x1, y1), mask)
        return image
    
    def _apply_pixelate(self, image, x1, y1, x2, y2, strength):
        """Pikselleştirme efekti uygula"""
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Yüz bölgesini kırp
        face_region = image.crop((x1, y1, x2, y2))
        
        # Piksel boyutu (1-100 arası strength değerine göre)
        pixel_size = max(4, min(50, int(face_width / (100 - strength + 10))))
        
        # Küçült ve tekrar büyüt (pikselleştirme efekti)
        small_size = (max(1, face_width // pixel_size), max(1, face_height // pixel_size))
        face_small = face_region.resize(small_size, Image.NEAREST)
        pixelated_face = face_small.resize((face_width, face_height), Image.NEAREST)
        
        # Elips maskesi
        mask = Image.new('L', (face_width, face_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.ellipse([0, 0, face_width, face_height], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=5))
        
        image.paste(pixelated_face, (x1, y1), mask)
        return image
    
    def _apply_black_box(self, image, x1, y1, x2, y2):
        """Siyah kutu uygula"""
        draw = ImageDraw.Draw(image)
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Elips çiz
        draw.ellipse([x1, y1, x2, y2], fill="black")
        return image
    
    def _apply_color_fill(self, image, x1, y1, x2, y2):
        """Renk dolgusu uygula"""
        draw = ImageDraw.Draw(image)
        
        # Varsayılan renk: koyu gri
        color = self.blur_color
        
        # Elips çiz
        draw.ellipse([x1, y1, x2, y2], fill=color)
        return image
    
    def _apply_emoji(self, image, x1, y1, x2, y2):
        """Emoji uygula"""
        draw = ImageDraw.Draw(image)
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Önce altın sarısı elips arka plan
        draw.ellipse([x1, y1, x2, y2], fill="#FFD700")  # Altın sarısı arka plan
        
        # Emoji metni
        emoji = "😊"
        font_size = int(min(face_width, face_height) * 0.65)  # Biraz daha küçük
        
        try:
            from PIL import ImageFont
            # Segoe UI Emoji fontunu kullanmayı dene
            font = ImageFont.truetype("seguiemj.ttf", font_size)
        except:
            # Font bulunamazsa varsayılan
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = None
        
        # Emoji'yi merkeze yerleştir
        if font:
            # Text boyutunu al (bbox kullanarak)
            bbox = draw.textbbox((0, 0), emoji, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Merkeze yerleştir
            text_x = x1 + (face_width - text_width) // 2
            text_y = y1 + (face_height - text_height) // 2
            
            draw.text((text_x, text_y), emoji, fill="black", font=font)
        else:
            # Font yoksa basit smiley daire
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = min(face_width, face_height) // 3
            
            # Gülümseyen yüz çiz
            draw.ellipse([center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius], fill="yellow", outline="black", width=2)
            
            # Gözler
            eye_radius = radius // 6
            left_eye_x = center_x - radius // 2
            right_eye_x = center_x + radius // 2
            eye_y = center_y - radius // 3
            
            draw.ellipse([left_eye_x - eye_radius, eye_y - eye_radius,
                         left_eye_x + eye_radius, eye_y + eye_radius], fill="black")
            draw.ellipse([right_eye_x - eye_radius, eye_y - eye_radius,
                         right_eye_x + eye_radius, eye_y + eye_radius], fill="black")
            
            # Gülümseme (yay)
            smile_y = center_y + radius // 4
            draw.arc([center_x - radius//2, smile_y - radius//3,
                     center_x + radius//2, smile_y + radius//3], 
                    start=0, end=180, fill="black", width=2)
        
        return image

    
    def save_image(self):
        """Görüntüyü kaydet"""
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "Kaydedilecek görüntü yok!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Farklı Kaydet",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            try:
                self._save_clean_image(self.processed_image, file_path)
                self.status_label.configure(text="💾 Kaydedildi")
                info = "Görüntü başarıyla kaydedildi"
                if self.exif_strip.get():
                    info += "\n🔒 EXIF metadata temizlendi"
                messagebox.showinfo("Başarılı", f"{info}:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Hata", f"Kaydetme hatası:\n{e}")

    def _save_clean_image(self, img: Image.Image, path: str):
        """Görüntüyü EXIF temizleyerek kaydet."""
        is_jpeg = path.lower().endswith(('.jpg', '.jpeg'))
        save_img = img.convert('RGB') if (is_jpeg and img.mode == 'RGBA') else img.copy()

        if self.exif_strip.get():
            # Temiz bir PIL Image oluştur (metadata taşımaz)
            clean = Image.new(save_img.mode, save_img.size)
            clean.paste(save_img)
            save_img = clean

        if is_jpeg:
            save_img.save(path, quality=95, optimize=True)
        else:
            save_img.save(path)
    
    def reset_image(self):
        """Görüntüyü sıfırla"""
        if self.original_image is not None:
            self._save_state()  # Sıfırlamadan önce kaydet
            self.processed_image = self.original_image.copy()
            self.face_locations = []
            self.selected_faces = []

            self.update_face_checkboxes()
            self.display_image(self.original_image)
            
            if self.drawing_mode:
                self.toggle_drawing_mode()
            
            self.status_label.configure(text="🔄 Sıfırlandı")
            self.face_count_label.configure(text="")
    
    # ------------------------------------------------------------------
    # DRAG & DROP
    # ------------------------------------------------------------------
    def _on_drop(self, event):
        """Sürükle-bırak ile dosya yükle."""
        raw = event.data.strip()
        # Windows'ta {} ile sarılı gelebilir
        if raw.startswith('{') and raw.endswith('}'):
            raw = raw[1:-1]
        paths = raw.split('} {') if '} {' in raw else [raw]
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif'}
        images = [p for p in paths if Path(p).suffix.lower() in valid_exts]
        videos = [p for p in paths if Path(p).suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov'}]

        if images:
            if len(images) == 1:
                self.load_image_from_path(images[0])
            else:
                BatchManagerWindow(self, images)
        elif videos:
            VideoProcessorWindow(self, videos[0])
        else:
            messagebox.showwarning("Uyarı", "Desteklenmeyen dosya türü sürüklendi.")

    # ------------------------------------------------------------------
    # STICKER / LOGO
    # ------------------------------------------------------------------
    def select_sticker(self):
        """Sticker/logo dosyası seç."""
        path = filedialog.askopenfilename(
            title="Logo / Sticker Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if path:
            try:
                self.sticker_image = Image.open(path).convert("RGBA")
                self.sticker_name_label.configure(
                    text=Path(path).name[:28],
                    text_color="#2ECC71"
                )
                self.blur_style.set("sticker")
            except Exception as e:
                messagebox.showerror("Hata", f"Logo yüklenemedi:\n{e}")

    def _apply_sticker(self, image: Image.Image, x1, y1, x2, y2) -> Image.Image:
        """Yüz alanına sticker/logo yapıştır."""
        if self.sticker_image is None:
            messagebox.showwarning("Uyarı", "Önce bir logo/sticker seçin!")
            return image

        face_w = x2 - x1
        face_h = y2 - y1
        if face_w < 1 or face_h < 1:
            return image

        sticker = self.sticker_image.copy()
        sticker = sticker.resize((face_w, face_h), Image.LANCZOS)

        # Elips maskesi uygula
        mask = Image.new('L', (face_w, face_h), 0)
        ImageDraw.Draw(mask).ellipse([0, 0, face_w, face_h], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=6))

        base = image.convert("RGBA")
        sticker_rgba = sticker  # already RGBA
        # Maskeyi alpha kanalı ile birleştir
        combined_mask = Image.new('L', (face_w, face_h), 0)
        combined_mask.paste(mask)
        sticker_rgba.putalpha(combined_mask)
        base.paste(sticker_rgba, (x1, y1), sticker_rgba)
        return base.convert("RGB")

    # ------------------------------------------------------------------
    # VIDEO İŞLEME
    # ------------------------------------------------------------------
    def open_video(self):
        """Video dosyası seç ve işleyiciyi aç."""
        path = filedialog.askopenfilename(
            title="Video Dosyası Seç",
            filetypes=[
                ("Video Dosyaları", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if path:
            VideoProcessorWindow(self, path)

    def batch_process(self):
        """Toplu işlem - yeni yönetici penceresi ile"""
        file_paths = filedialog.askopenfilenames(
            title="Toplu İşlem İçin Fotoğraflar Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        if not file_paths:
            return
        BatchManagerWindow(self, list(file_paths))

    
    # ---- Eski toplu işlem metodları (BatchManagerWindow ile değiştirildi) ----
    def show_batch_preview(self, file_paths, output_dir):
        """Toplu işlem öncesi önizleme göster"""
        if not file_paths:
            return
        
        # İlk dosyayı yükle ve işle
        first_file = file_paths[0]
        file_count = len(file_paths)
        
        try:
            # Görüntüyü yükle
            preview_image = Image.open(first_file)
            if preview_image.mode == 'RGBA':
                background = Image.new('RGB', preview_image.size, (255, 255, 255))
                background.paste(preview_image, mask=preview_image.split()[3])
                preview_image = background
            elif preview_image.mode != 'RGB':
                preview_image = preview_image.convert('RGB')
            
            cv_image = np.array(preview_image)
            
            # Yüz algıla
            face_locations = self._detect_faces_sync(cv_image)
            
            if not face_locations:
                messagebox.showwarning(
                    "Önizleme",
                    f"İlk dosyada ({Path(first_file).name}) yüz bulunamadı.\n\n"
                    f"Yine de devam etmek istiyor musunuz?"
                )
            
            # Önizleme penceresi oluştur
            preview_window = ctk.CTkToplevel(self)
            preview_window.title(f"Toplu İşlem Önizleme - {file_count} Dosya")
            preview_window.geometry("900x700")
            preview_window.transient(self)
            preview_window.grab_set()
            
            # Başlık
            title_label = ctk.CTkLabel(
                preview_window,
                text=f"📊 {file_count} fotoğraf işlenecek",
                font=ctk.CTkFont(size=18, weight="bold")
            )
            title_label.pack(pady=10)
            
            subtitle_label = ctk.CTkLabel(
                preview_window,
                text=f"İlk dosya önizlemesi: {Path(first_file).name}\n🎭 {len(face_locations)} yüz bulundu",
                font=ctk.CTkFont(size=12),
                text_color="gray"
            )
            subtitle_label.pack(pady=5)
            
            # Ana frame
            content_frame = ctk.CTkFrame(preview_window)
            content_frame.pack(fill="both", expand=True, padx=20, pady=10)
            
            # Sol panel - Önizleme
            left_frame = ctk.CTkFrame(content_frame)
            left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
            
            preview_label = ctk.CTkLabel(
                left_frame,
                text="Önizleme",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            preview_label.pack(pady=10)
            
            # Canvas için frame
            canvas_frame = ctk.CTkFrame(left_frame, fg_color="gray15")
            canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            preview_canvas = Canvas(
                canvas_frame,
                bg="#252525",
                highlightthickness=0
            )
            preview_canvas.pack(fill="both", expand=True, padx=5, pady=5)
            
            # Sağ panel - Ayarlar
            right_frame = ctk.CTkFrame(content_frame, width=300)
            right_frame.pack(side="right", fill="y")
            right_frame.pack_propagate(False)
            
            settings_label = ctk.CTkLabel(
                right_frame,
                text="⚙️ Ayarlar",
                font=ctk.CTkFont(size=14, weight="bold")
            )
            settings_label.pack(pady=10)
            
            # Ayar bilgisi
            settings_info = ctk.CTkLabel(
                right_frame,
                text="Ayarları değiştirin ve\nönizlemeyi güncelleyin:",
                font=ctk.CTkFont(size=11),
                text_color="gray"
            )
            settings_info.pack(pady=5)
            
            # Mevcut ayarlar göster
            current_method = self.detection_method.get()
            current_style = self.blur_style.get()
            current_strength = self.blur_strength.get()
            
            method_info = ctk.CTkLabel(
                right_frame,
                text=f"Algılama: {current_method}",
                font=ctk.CTkFont(size=11)
            )
            method_info.pack(pady=2)
            
            style_info = ctk.CTkLabel(
                right_frame,
                text=f"Stil: {current_style}",
                font=ctk.CTkFont(size=11)
            )
            style_info.pack(pady=2)
            
            strength_label = ctk.CTkLabel(
                right_frame,
                text=f"Seviye: {current_strength}",
                font=ctk.CTkFont(size=11)
            )
            strength_label.pack(pady=2)
            
            # Güncelleme butonu
            update_btn = ctk.CTkButton(
                right_frame,
                text="🔄 Önizlemeyi Güncelle",
                command=lambda: update_preview(),
                height=40,
                fg_color="#3498DB",
                hover_color="#2980B9"
            )
            update_btn.pack(pady=15, padx=10, fill="x")
            
            # Ayırıcı
            separator = ctk.CTkFrame(right_frame, height=2, fg_color="gray30")
            separator.pack(fill="x", padx=10, pady=10)
            
            # Bilgi
            info_label = ctk.CTkLabel(
                right_frame,
                text="💡 İpucu:\nAyarları ana pencereden\ndeğiştirebilirsiniz.",
                font=ctk.CTkFont(size=10),
                text_color="#FFA500",
                justify="left"
            )
            info_label.pack(pady=10)
            
            # Butonlar
            button_frame = ctk.CTkFrame(preview_window)
            button_frame.pack(fill="x", padx=20, pady=10)
            
            cancel_btn = ctk.CTkButton(
                button_frame,
                text="❌ İptal",
                command=preview_window.destroy,
                fg_color="red",
                hover_color="darkred",
                width=150
            )
            cancel_btn.pack(side="left", padx=5)
            
            start_btn = ctk.CTkButton(
                button_frame,
                text=f"✅ İşleme Başla ({file_count} dosya)",
                command=lambda: start_batch(preview_window, file_paths, output_dir),
                fg_color="#2D7D46",
                hover_color="#236B38",
                width=300,
                height=45,
                font=ctk.CTkFont(size=14, weight="bold")
            )
            start_btn.pack(side="right", padx=5)
            
            def update_preview():
                """Önizlemeyi güncelle"""
                # Ayarları güncelle
                method_info.configure(text=f"Algılama: {self.detection_method.get()}")
                style_info.configure(text=f"Stil: {self.blur_style.get()}")
                strength_label.configure(text=f"Seviye: {self.blur_strength.get()}")
                
                # Görüntüyü işle
                result_image = preview_image.copy()
                blur_strength = int(self.blur_strength.get())
                blur_style = self.blur_style.get()
                margin_percent = self.face_margin.get() / 100.0
                img_w, img_h = preview_image.size
                
                for (x1, y1, x2, y2) in face_locations:
                    # Margin hesapla
                    w = x2 - x1
                    h = y2 - y1
                    mx = w * margin_percent
                    my = h * margin_percent
                    
                    nx1 = int(max(0, x1 - mx))
                    ny1 = int(max(0, y1 - my))
                    nx2 = int(min(img_w, x2 + mx))
                    ny2 = int(min(img_h, y2 + my))

                    if blur_style == "gaussian":
                        result_image = self._apply_gaussian_blur(result_image, nx1, ny1, nx2, ny2, blur_strength)
                    elif blur_style == "pixelate":
                        result_image = self._apply_pixelate(result_image, nx1, ny1, nx2, ny2, blur_strength)
                    elif blur_style == "black":
                        result_image = self._apply_black_box(result_image, nx1, ny1, nx2, ny2)
                    elif blur_style == "color":
                        result_image = self._apply_color_fill(result_image, nx1, ny1, nx2, ny2)
                    elif blur_style == "emoji":
                        result_image = self._apply_emoji(result_image, nx1, ny1, nx2, ny2)

                
                # Canvas'a göster
                display_preview(result_image)
            
            def display_preview(img):
                """Önizlemeyi canvas'ta göster"""
                canvas_frame.update()
                canvas_width = preview_canvas.winfo_width()
                canvas_height = preview_canvas.winfo_height()
                
                if canvas_width <= 1:
                    canvas_width = 500
                if canvas_height <= 1:
                    canvas_height = 500
                
                # Ölçeklendir
                img_width, img_height = img.size
                scale_x = canvas_width / img_width
                scale_y = canvas_height / img_height
                scale = min(scale_x, scale_y) * 0.9
                
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                
                resized = img.resize((new_width, new_height), Image.LANCZOS)
                photo = ImageTk.PhotoImage(resized)
                
                preview_canvas.delete("all")
                preview_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=photo,
                    anchor="center"
                )
                preview_canvas.image = photo
            
            def start_batch(window, paths, output):
                """İşlemi başlat"""
                window.destroy()
                self._start_batch_processing(paths, output)
            
            # İlk önizlemeyi göster
            preview_window.after(100, update_preview)
            
        except Exception as e:
            messagebox.showerror("Önizleme Hatası", f"Önizleme oluşturulamadı:\n{e}")
    
    def _start_batch_processing(self, file_paths, output_dir):
        """Toplu işlemi başlat (eskiden batch_process içindeydi)"""
        file_count = len(file_paths)
        
        # İlerleme penceresi oluştur
        self.batch_window = ctk.CTkToplevel(self)
        self.batch_window.title("Toplu İşlem")
        self.batch_window.geometry("500x300")
        self.batch_window.transient(self)
        self.batch_window.grab_set()
        
        # İlerleme label
        self.batch_status_label = ctk.CTkLabel(
            self.batch_window,
            text="İşlem başlatılıyor...",
            font=ctk.CTkFont(size=14)
        )
        self.batch_status_label.pack(pady=20)
        
        # İlerleme çubuğu
        self.batch_progress = ctk.CTkProgressBar(
            self.batch_window,
            width=400,
            height=20
        )
        self.batch_progress.pack(pady=10)
        self.batch_progress.set(0)
        
        # İlerleme yüzdesi
        self.batch_percent_label = ctk.CTkLabel(
            self.batch_window,
            text="0%",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.batch_percent_label.pack(pady=10)
        
        # Detay label
        self.batch_detail_label = ctk.CTkLabel(
            self.batch_window,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.batch_detail_label.pack(pady=10)
        
        # İptal butonu
        self.batch_cancel_btn = ctk.CTkButton(
            self.batch_window,
            text="İptal",
            command=self.cancel_batch_process,
            fg_color="red",
            hover_color="darkred"
        )
        self.batch_cancel_btn.pack(pady=10)
        
        # İşlemi thread'de başlat
        self.batch_cancelled = False
        thread = threading.Thread(
            target=self._batch_process_thread,
            args=(file_paths, output_dir)
        )
        thread.start()
    
    def cancel_batch_process(self):
        """Toplu işlemi iptal et"""
        self.batch_cancelled = True
        self.batch_status_label.configure(text="İptal ediliyor...")
    
    def _batch_process_thread(self, file_paths, output_dir):
        """Toplu işlem thread'i"""
        total_files = len(file_paths)
        processed_count = 0
        success_count = 0
        failed_files = []
        total_faces = 0
        
        try:
            for i, file_path in enumerate(file_paths):
                if self.batch_cancelled:
                    self.after(0, lambda: self.batch_status_label.configure(text="❌ İptal edildi"))
                    return
                
                # Durum güncelle
                file_name = Path(file_path).name
                self.after(0, lambda fn=file_name: self.batch_status_label.configure(text=f"İşleniyor: {fn}"))
                self.after(0, lambda fn=file_name: self.batch_detail_label.configure(text=f"Dosya: {fn}"))
                
                try:
                    # Görüntüyü yükle
                    image = Image.open(file_path)
                    if image.mode == 'RGBA':
                        background = Image.new('RGB', image.size, (255, 255, 255))
                        background.paste(image, mask=image.split()[3])
                        image = background
                    elif image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    cv_image = np.array(image)
                    
                    # Yüz algılama
                    face_locations = self._detect_faces_sync(cv_image)
                    
                    if face_locations:
                        total_faces += len(face_locations)
                        
                        # Tüm yüzleri işle
                        result_image = image.copy()
                        blur_strength = int(self.blur_strength.get())
                        blur_style = self.blur_style.get()
                        margin_percent = self.face_margin.get() / 100.0
                        img_w, img_h = image.size
                        
                        for (x1, y1, x2, y2) in face_locations:
                            # Margin hesapla
                            w = x2 - x1
                            h = y2 - y1
                            mx = w * margin_percent
                            my = h * margin_percent
                            
                            nx1 = int(max(0, x1 - mx))
                            ny1 = int(max(0, y1 - my))
                            nx2 = int(min(img_w, x2 + mx))
                            ny2 = int(min(img_h, y2 + my))

                            if blur_style == "gaussian":
                                result_image = self._apply_gaussian_blur(result_image, nx1, ny1, nx2, ny2, blur_strength)
                            elif blur_style == "pixelate":
                                result_image = self._apply_pixelate(result_image, nx1, ny1, nx2, ny2, blur_strength)
                            elif blur_style == "black":
                                result_image = self._apply_black_box(result_image, nx1, ny1, nx2, ny2)
                            elif blur_style == "color":
                                result_image = self._apply_color_fill(result_image, nx1, ny1, nx2, ny2)
                            elif blur_style == "emoji":
                                result_image = self._apply_emoji(result_image, nx1, ny1, nx2, ny2)

                        
                        # Kaydet
                        output_path = os.path.join(output_dir, f"processed_{file_name}")
                        if output_path.lower().endswith(('.jpg', '.jpeg')):
                            result_image.save(output_path, quality=95)
                        else:
                            result_image.save(output_path)
                        
                        success_count += 1
                    else:
                        # Yüz bulunamadı, orijinali kopyala
                        output_path = os.path.join(output_dir, f"noface_{file_name}")
                        image.save(output_path)
                        failed_files.append((file_name, "Yüz bulunamadı"))
                    
                except Exception as e:
                    failed_files.append((file_name, str(e)))
                    print(f"Hata ({file_name}): {e}")
                
                processed_count += 1
                progress = processed_count / total_files
                
                # UI güncelle
                self.after(0, lambda p=progress: self.batch_progress.set(p))
                self.after(0, lambda p=int(progress*100): self.batch_percent_label.configure(text=f"{p}%"))
            
            # İşlem tamamlandı
            self.after(0, lambda: self._show_batch_results(
                total_files, success_count, len(failed_files), total_faces, failed_files, output_dir
            ))
            
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Toplu İşlem Hatası", f"Beklenmeyen hata:\n{e}"))
            self.after(0, lambda: self.batch_window.destroy())
    
    def _detect_faces_sync(self, cv_image):
        """Senkron yüz algılama (Hız için optimize edilmiş)"""
        method = self.detection_method.get()
        orig_h, orig_w = cv_image.shape[:2]
        
        # PERFORMANS OPTİMİZASYONU: Büyük resimleri algılama için ölçeklendir (Maks 1024px)
        max_dim = 1024
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            target_w = int(orig_w * scale)
            target_h = int(orig_h * scale)
            work_img = cv2.resize(cv_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work_img = cv_image
            target_w, target_h = orig_w, orig_h

        all_faces = []
        
        def get_mediapipe_faces(img):
            mp_faces = []
            if self.face_detector:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
                detection_result = self.face_detector.detect(mp_image)
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    # Koordinatları orijinal ölçeğe çevir
                    x1 = int(max(0, bbox.origin_x) / scale)
                    y1 = int(max(0, bbox.origin_y) / scale)
                    x2 = int(min(orig_w, (bbox.origin_x + bbox.width) / scale))
                    y2 = int(min(orig_h, (bbox.origin_y + bbox.height) / scale))
                    
                    # Pillow 'Coordinate lower is less than upper' hatasını önlemek için güvenlik kontrolü
                    if x2 > x1 and y2 > y1:
                        mp_faces.append((x1, y1, x2, y2))
            return mp_faces

        def get_opencv_faces(img):
            cv_faces = []
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Parametreler work_img boyutuna göre ayarlandı
            if self.face_cascade:
                detected = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                )
                for (x, y, fw, fh) in detected:
                    x1, y1 = int(x / scale), int(y / scale)
                    x2, y2 = int((x + fw) / scale), int((y + fh) / scale)
                    # Güvenlik Kontrolü
                    if x2 > x1 and y2 > y1:
                        cv_faces.append((x1, y1, x2, y2))
            
            if self.profile_cascade:
                detected_profile = self.profile_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
                )
                for (x, y, fw, fh) in detected_profile:
                    x1, y1 = int(x / scale), int(y / scale)
                    x2, y2 = int((x + fw) / scale), int((y + fh) / scale)
                    # Güvenlik Kontrolü
                    if x2 > x1 and y2 > y1:
                        cv_faces.append((x1, y1, x2, y2))
            return cv_faces

        def merge_faces(base_list, new_list, threshold=0.4):
            for n_face in new_list:
                nx1, ny1, nx2, ny2 = n_face
                is_duplicate = False
                for b_face in base_list:
                    bx1, by1, bx2, by2 = b_face
                    # IoU
                    ix1, iy1 = max(nx1, bx1), max(ny1, by1)
                    ix2, iy2 = min(nx2, bx2), min(ny2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        i_area = (ix2 - ix1) * (iy2 - iy1)
                        a1 = (nx2 - nx1) * (ny2 - ny1)
                        a2 = (bx2 - bx1) * (by2 - by1)
                        iou = i_area / (a1 + a2 - i_area)
                        if iou > threshold:
                            is_duplicate = True
                            break
                if not is_duplicate:
                    base_list.append(n_face)
            return base_list

        try:
            # Algılama her zaman küçültülmüş 'work_img' üzerinde yapılmalı (Performans için)
            if method == "mediapipe":
                all_faces = get_mediapipe_faces(work_img)
            elif method == "opencv_haar":
                all_faces = get_opencv_faces(work_img)
            elif method == "hybrid":
                mp_faces = get_mediapipe_faces(work_img)
                cv_faces = get_opencv_faces(work_img)
                all_faces = merge_faces(mp_faces, cv_faces)
        except Exception as e:
            print(f"Algılama hatası: {e}")
        
        return all_faces

    
    def _show_batch_results(self, total, success, failed, faces, failed_files, output_dir):
        """Toplu işlem sonuçlarını göster"""
        self.batch_window.destroy()
        
        # Rapor oluştur
        report = f"📊 TOPLU İŞLEM RAPORU\n\n"
        report += f"✅ İşlenen Dosya: {total}\n"
        report += f"🎭 Bulunan Yüz: {faces}\n"
        report += f"✔️ Başarılı: {success}\n"
        report += f"❌ Başarısız: {failed}\n\n"
        
        if failed_files:
            report += "Başarısız Dosyalar:\n"
            for file_name, reason in failed_files[:10]:  # İlk 10'u göster
                report += f"• {file_name}: {reason}\n"
            if len(failed_files) > 10:
                report += f"... ve {len(failed_files) - 10} dosya daha\n"
        
        report += f"\n📁 Çıktı Klasörü:\n{output_dir}"
        
        # Rapor penceresi
        messagebox.showinfo("Toplu İşlem Tamamlandı", report)
        
        # Klasörü açmayı öner
        open_folder = messagebox.askyesno(
            "Klasörü Aç",
            "İşlenmiş dosyaların olduğu klasörü açmak ister misiniz?"
        )
        
        if open_folder:
            import subprocess
            if self.system == "Windows":
                subprocess.run(['explorer', os.path.normpath(output_dir)])
            elif self.system == "Darwin": # macOS
                subprocess.run(['open', output_dir])
            else: # Linux
                subprocess.run(['xdg-open', output_dir])

        
        self.status_label.configure(text=f"✅ {total} dosya işlendi")



# ============================================================
# VIDEO İŞLEYİCİ
# ============================================================
class VideoProcessorWindow(ctk.CTkToplevel):
    """Kare kare yüz bulanıklaştırma yapan video işleyici."""

    def __init__(self, parent: "FaceBlurApp", video_path: str):
        super().__init__(parent)
        self.parent_app = parent
        self.video_path = video_path
        self.title(f"Video İşleyici — {Path(video_path).name}")
        self.geometry("900x650")
        self.minsize(700, 500)
        self.transient(parent)
        self.grab_set()

        self._cancelled = False
        self._cap = None
        self._total_frames = 0
        self._fps = 25.0
        self._out_path: str | None = None

        self._build_ui()
        self.after(200, self._load_video_info)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Başlık
        ctk.CTkLabel(
            self, text="🎬 Video İşleyici",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            self,
            text=f"📄 {Path(self.video_path).name}",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack()

        # Önizleme canvas
        preview_frame = ctk.CTkFrame(self, fg_color="gray15", corner_radius=10)
        preview_frame.pack(fill="both", expand=True, padx=20, pady=10)

        from tkinter import Canvas as TkCanvas
        self.preview_canvas = TkCanvas(preview_frame, bg="#1e1e1e", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # Bilgi satırı
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        info_frame.pack(fill="x", padx=20)

        self.info_lbl = ctk.CTkLabel(
            info_frame, text="Video yükleniyor…",
            font=ctk.CTkFont(size=12)
        )
        self.info_lbl.pack(side="left")

        self.face_count_lbl = ctk.CTkLabel(
            info_frame, text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#3B8ED0"
        )
        self.face_count_lbl.pack(side="right")

        # İlerleme çubuğu
        self.progress = ctk.CTkProgressBar(self, height=16)
        self.progress.pack(fill="x", padx=20, pady=(8, 4))
        self.progress.set(0)

        self.pct_lbl = ctk.CTkLabel(
            self, text="0%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3B8ED0"
        )
        self.pct_lbl.pack()

        # Ayarlar satırı
        opts_frame = ctk.CTkFrame(self, fg_color="transparent")
        opts_frame.pack(fill="x", padx=20, pady=6)

        ctk.CTkLabel(opts_frame, text="Ayarlar: Ana penceredeki Bulanıklaştırma Stili ve Seviyesi kullanılır.",
                     font=ctk.CTkFont(size=10), text_color="gray").pack(side="left")

        # Butonlar
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(fill="x", padx=20, pady=(0, 15))

        self.output_btn = ctk.CTkButton(
            btn_frame, text="📁 Çıktı Dosyası Seç", width=180, height=38,
            font=ctk.CTkFont(size=12),
            fg_color="#E67E22", hover_color="#D35400",
            command=self._select_output
        )
        self.output_btn.pack(side="left", padx=4)

        self.preview_btn = ctk.CTkButton(
            btn_frame, text="👁 İlk Kareyi Önizle", width=170, height=38,
            font=ctk.CTkFont(size=12),
            fg_color="#2D7D46", hover_color="#236B38",
            command=self._preview_first_frame
        )
        self.preview_btn.pack(side="left", padx=4)

        self.start_btn = ctk.CTkButton(
            btn_frame, text="▶ İşlemeye Başla", width=160, height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#9B59B6", hover_color="#7D3C98",
            command=self._start_processing
        )
        self.start_btn.pack(side="right", padx=4)

        self.cancel_btn = ctk.CTkButton(
            btn_frame, text="⏹ Durdur", width=110, height=38,
            font=ctk.CTkFont(size=12),
            fg_color="#E74C3C", hover_color="#C0392B",
            state="disabled",
            command=self._cancel
        )
        self.cancel_btn.pack(side="right", padx=4)

    # ------------------------------------------------------------------
    # VIDEO BİLGİSİ
    # ------------------------------------------------------------------
    def _load_video_info(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.info_lbl.configure(text="❌ Video açılamadı!", text_color="#E74C3C")
                return
            self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            dur = self._total_frames / self._fps
            minutes, seconds = divmod(int(dur), 60)
            self.info_lbl.configure(
                text=f"✅ {self._total_frames} kare  |  {width}×{height}  |  {self._fps:.1f} fps  |  {minutes}:{seconds:02d}",
                text_color="#2ECC71"
            )
            cap.release()
            self._preview_first_frame()
        except Exception as e:
            self.info_lbl.configure(text=f"❌ Hata: {e}", text_color="#E74C3C")

    def _select_output(self):
        ext = Path(self.video_path).suffix.lower() or ".mp4"
        path = filedialog.asksaveasfilename(
            title="İşlenmiş Videoyu Kaydet",
            defaultextension=ext,
            filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("Tüm Dosyalar", "*.*")]
        )
        if path:
            self._out_path = path
            self.output_btn.configure(text=f"📁 {Path(path).name}")

    # ------------------------------------------------------------------
    # ÖNİZLEME
    # ------------------------------------------------------------------
    def _preview_first_frame(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)
            # Yüzleri algıla ve çiz
            faces = self.parent_app._detect_faces_sync(rgb)
            drw = ImageDraw.Draw(pil_frame)
            for (x1, y1, x2, y2) in faces:
                for t in range(3):
                    drw.ellipse([x1-t, y1-t, x2+t, y2+t], outline="#00FF00")
            self.face_count_lbl.configure(
                text=f"🎭 {len(faces)} yüz (ilk kare)"
            )
            self._show_on_canvas(pil_frame)
        except Exception as e:
            self.info_lbl.configure(text=f"Önizleme hatası: {e}", text_color="#E74C3C")

    def _show_on_canvas(self, pil_img: Image.Image):
        self.preview_canvas.update()
        cw = self.preview_canvas.winfo_width() or 700
        ch = self.preview_canvas.winfo_height() or 350
        pil_img.thumbnail((cw - 10, ch - 10), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(cw // 2, ch // 2, image=photo, anchor="center")
        self.preview_canvas.image = photo

    # ------------------------------------------------------------------
    # İŞLEME
    # ------------------------------------------------------------------
    def _start_processing(self):
        if self._out_path is None:
            ext = Path(self.video_path).suffix.lower() or ".mp4"
            path = filedialog.asksaveasfilename(
                title="İşlenmiş Videoyu Kaydet",
                defaultextension=ext,
                filetypes=[("MP4 Video", "*.mp4"), ("AVI Video", "*.avi"), ("Tüm Dosyalar", "*.*")]
            )
            if not path:
                return
            self._out_path = path
            self.output_btn.configure(text=f"📁 {Path(path).name}")

        self._cancelled = False
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.preview_btn.configure(state="disabled")
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _cancel(self):
        self._cancelled = True
        self.cancel_btn.configure(state="disabled", text="⏳ Durduruluyor…")

    def _process_thread(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.after(0, lambda: messagebox.showerror("Hata", "Video açılamadı!"))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        out_path = self._out_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") if out_path.endswith(".mp4") else cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        blur_strength = int(self.parent_app.blur_strength.get())
        blur_style = self.parent_app.blur_style.get()
        margin_pct = self.parent_app.face_margin.get() / 100.0

        frame_idx = 0
        total_faces = 0
        preview_every = max(1, total // 30)  # Her 30 karede bir önizleme

        while True:
            if self._cancelled:
                break
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)

            faces = self.parent_app._detect_faces_sync(rgb)
            total_faces += len(faces)
            img_w, img_h = pil_frame.size

            for (x1, y1, x2, y2) in faces:
                fw, fh = x2 - x1, y2 - y1
                nx1 = int(max(0, x1 - fw * margin_pct))
                ny1 = int(max(0, y1 - fh * margin_pct))
                nx2 = int(min(img_w, x2 + fw * margin_pct))
                ny2 = int(min(img_h, y2 + fh * margin_pct))
                app = self.parent_app
                if blur_style == "gaussian":
                    pil_frame = app._apply_gaussian_blur(pil_frame, nx1, ny1, nx2, ny2, blur_strength)
                elif blur_style == "pixelate":
                    pil_frame = app._apply_pixelate(pil_frame, nx1, ny1, nx2, ny2, blur_strength)
                elif blur_style == "black":
                    pil_frame = app._apply_black_box(pil_frame, nx1, ny1, nx2, ny2)
                elif blur_style == "color":
                    pil_frame = app._apply_color_fill(pil_frame, nx1, ny1, nx2, ny2)
                elif blur_style == "emoji":
                    pil_frame = app._apply_emoji(pil_frame, nx1, ny1, nx2, ny2)
                elif blur_style == "sticker":
                    pil_frame = app._apply_sticker(pil_frame, nx1, ny1, nx2, ny2)

            # PIL → OpenCV BGR
            out_frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            out.write(out_frame)

            frame_idx += 1
            pct = frame_idx / total

            if frame_idx % preview_every == 0:
                snap = pil_frame.copy()
                self.after(0, lambda s=snap: self._show_on_canvas(s))

            self.after(0, lambda p=pct, f=frame_idx, tf=total_faces: [
                self.progress.set(p),
                self.pct_lbl.configure(text=f"{int(p*100)}%  ({f}/{total} kare)"),
                self.face_count_lbl.configure(text=f"🎭 {tf} yüz bulanıklaştırıldı")
            ])

        cap.release()
        out.release()

        if self._cancelled:
            self.after(0, lambda: [
                self.info_lbl.configure(text="⏹ İşlem durduruldu.", text_color="#E74C3C"),
                self.start_btn.configure(state="normal"),
                self.cancel_btn.configure(state="disabled", text="⏹ Durdur"),
                self.preview_btn.configure(state="normal")
            ])
        else:
            self.after(0, lambda: self._on_done(frame_idx, total_faces))

    def _on_done(self, frames: int, faces: int):
        self.progress.set(1.0)
        self.pct_lbl.configure(text="100% — Tamamlandı!")
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled", text="⏹ Durdur")
        self.preview_btn.configure(state="normal")
        self.info_lbl.configure(
            text=f"✅ {frames} kare işlendi  |  {faces} yüz bulanıklaştırıldı",
            text_color="#2ECC71"
        )
        msg = (
            f"🎉 Video işleme tamamlandı!\n\n"
            f"İşlenen kare: {frames}\n"
            f"Bulanıklaştırılan yüz: {faces}\n\n"
            f"📁 Kayıt yeri:\n{self._out_path}"
        )
        messagebox.showinfo("Tamamlandı", msg)


# ============================================================
# TOPLU İŞLEM YÖNETİCİSİ
# ============================================================
class BatchManagerWindow(ctk.CTkToplevel):
    """Her fotoğrafı ayrı kart olarak gösteren toplu işlem yöneticisi."""

    THUMB_W = 220
    THUMB_H = 160

    def __init__(self, parent: FaceBlurApp, file_paths: list):
        super().__init__(parent)
        self.parent_app = parent
        self.title(f"Toplu İşlem Yöneticisi  —  {len(file_paths)} Fotoğraf")
        self.geometry("1150x780")
        self.minsize(900, 600)
        self.transient(parent)
        self.grab_set()

        self.items: list[dict] = []   # Her fotoğraf için state
        self.output_dir: str | None = None
        self._cancelled = False

        self._build_ui()
        self.after(100, lambda: self._load_images(file_paths))

    # ------------------------------------------------------------------
    # UI KURULUMU
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Üst kontrol çubuğu ----
        top = ctk.CTkFrame(self, height=65, corner_radius=0)
        top.pack(fill="x")
        top.pack_propagate(False)

        ctk.CTkLabel(
            top, text="🖼️ Toplu İşlem Yöneticisi",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(side="left", padx=15)

        # Sağ butonlar (ters sırayla pack)
        self.start_btn = ctk.CTkButton(
            top, text="✅ İşleme Başla", width=160, height=38,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#9B59B6", hover_color="#7D3C98",
            command=self._start_processing
        )
        self.start_btn.pack(side="right", padx=8)

        self.output_btn = ctk.CTkButton(
            top, text="📁 Çıktı Klasörü Seç", width=165, height=38,
            font=ctk.CTkFont(size=12),
            fg_color="#E67E22", hover_color="#D35400",
            command=self._select_output
        )
        self.output_btn.pack(side="right", padx=4)

        self.detect_all_btn = ctk.CTkButton(
            top, text="🔍 Tüm Yüzleri Algıla", width=185, height=38,
            font=ctk.CTkFont(size=12),
            fg_color="#2D7D46", hover_color="#236B38",
            command=self._detect_all
        )
        self.detect_all_btn.pack(side="right", padx=4)

        # ---- Kaydırılabilir fotoğraf alanı ----
        self.scroll = ctk.CTkScrollableFrame(self, fg_color="gray10")
        self.scroll.pack(fill="both", expand=True, padx=8, pady=8)

        # ---- Alt durum çubuğu ----
        bottom = ctk.CTkFrame(self, height=48, corner_radius=0)
        bottom.pack(fill="x")
        bottom.pack_propagate(False)

        self.status_lbl = ctk.CTkLabel(
            bottom, text="Fotoğraflar yükleniyor…",
            font=ctk.CTkFont(size=12)
        )
        self.status_lbl.pack(side="left", padx=15)

        self.progress_bar = ctk.CTkProgressBar(bottom, width=320, height=14)
        self.progress_bar.pack(side="right", padx=15)
        self.progress_bar.set(0)

    # ------------------------------------------------------------------
    # FOTOĞRAF YÜKLEME
    # ------------------------------------------------------------------
    def _load_images(self, paths: list):
        for idx, path in enumerate(paths):
            self._create_card(idx, path)
        n = len(self.items)
        self.status_lbl.configure(
            text=f"{n} fotoğraf yüklendi. 'Tüm Yüzleri Algıla' butonuna tıklayın."
        )
        self.progress_bar.set(1.0)

    def _create_card(self, index: int, path: str):
        """Tek fotoğraf kartı oluştur."""
        card = ctk.CTkFrame(self.scroll, fg_color="gray20", corner_radius=10)
        card.pack(fill="x", padx=6, pady=5)

        # --- Sol: küçük resim ---
        thumb_frame = ctk.CTkFrame(card, fg_color="gray15",
                                   width=self.THUMB_W + 10,
                                   height=self.THUMB_H + 10,
                                   corner_radius=8)
        thumb_frame.pack(side="left", padx=10, pady=10)
        thumb_frame.pack_propagate(False)

        from tkinter import Canvas as TkCanvas
        canvas = TkCanvas(
            thumb_frame, bg="#1e1e1e",
            highlightthickness=0,
            width=self.THUMB_W,
            height=self.THUMB_H
        )
        canvas.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Orta: bilgi + checkbox'lar ---
        mid = ctk.CTkFrame(card, fg_color="transparent")
        mid.pack(side="left", fill="both", expand=True, padx=5, pady=10)

        fname = os.path.basename(path)
        ctk.CTkLabel(
            mid, text=fname,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        ).pack(anchor="w")

        status_lbl = ctk.CTkLabel(
            mid, text="⏳ Yükleniyor…",
            font=ctk.CTkFont(size=11),
            text_color="gray", anchor="w"
        )
        status_lbl.pack(anchor="w", pady=(2, 6))

        cb_frame = ctk.CTkFrame(mid, fg_color="transparent")
        cb_frame.pack(fill="x")

        # --- Sağ: tekil butonlar ---
        right = ctk.CTkFrame(card, fg_color="transparent", width=140)
        right.pack(side="right", padx=10, pady=10)
        right.pack_propagate(False)

        ctk.CTkButton(
            right, text="🔍 Algıla", width=128, height=30,
            font=ctk.CTkFont(size=11),
            fg_color="#2D7D46", hover_color="#236B38",
            command=lambda i=index: threading.Thread(
                target=self._detect_single_thread, args=(i,), daemon=True
            ).start()
        ).pack(pady=3)

        ctk.CTkButton(
            right, text="🔄 Sıfırla", width=128, height=30,
            font=ctk.CTkFont(size=11),
            fg_color="#555555", hover_color="#444444",
            command=lambda i=index: self._reset_card(i)
        ).pack(pady=3)

        ctk.CTkButton(
            right, text="🔎 Düzenle / Çiz", width=128, height=30,
            font=ctk.CTkFont(size=11),
            fg_color="#2980B9", hover_color="#2471A3",
            command=lambda i=index: self._open_editor(i)
        ).pack(pady=3)

        # --- State kaydı ---
        item = {
            "index": index,
            "path": path,
            "image": None,        # PIL Image
            "cv_image": None,     # np.ndarray
            "face_locations": [],
            "selected_faces": [],
            "canvas": canvas,
            "status_lbl": status_lbl,
            "cb_frame": cb_frame,
            "checkbox_vars": [],
        }
        self.items.append(item)

        # Görüntüyü yükle ve küçük resmi göster
        try:
            img = Image.open(path)
            if img.mode == "RGBA":
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3])
                img = bg
            elif img.mode != "RGB":
                img = img.convert("RGB")
            item["image"] = img
            item["cv_image"] = np.array(img)
            self._draw_thumb(item)
            status_lbl.configure(
                text=f"✅ Yüklendi  ({img.width}×{img.height})",
                text_color="#2ECC71"
            )
        except Exception as e:
            status_lbl.configure(
                text=f"❌ Yüklenemedi: {str(e)[:50]}",
                text_color="#E74C3C"
            )

    # ------------------------------------------------------------------
    # KÜÇÜK RESİM ÇİZİMİ
    # ------------------------------------------------------------------
    def _draw_thumb(self, item: dict):
        """Yüz kutucukları ile birlikte küçük resmi canvas'a çiz."""
        if item["image"] is None:
            return
        img_copy = item["image"].copy()
        if item["face_locations"]:
            drw = ImageDraw.Draw(img_copy)
            for i, (x1, y1, x2, y2) in enumerate(item["face_locations"]):
                sel = i < len(item["selected_faces"]) and item["selected_faces"][i]
                color = "#00FF00" if sel else "#FF6B6B"
                for t in range(3):
                    drw.ellipse([x1 - t, y1 - t, x2 + t, y2 + t], outline=color)
                # Numara etiketi
                lx, ly = x1, max(0, y1 - 18)
                drw.rectangle([lx, ly, lx + 32, ly + 16], fill=color)
                drw.text((lx + 4, ly + 1), f"#{i+1}", fill="black")
        thumb = img_copy.copy()
        thumb.thumbnail((self.THUMB_W, self.THUMB_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        canvas = item["canvas"]
        canvas.delete("all")
        canvas.create_image(
            self.THUMB_W // 2, self.THUMB_H // 2,
            image=photo, anchor="center"
        )
        canvas.image = photo  # referans koru

    # ------------------------------------------------------------------
    # YÜZ TESPİT
    # ------------------------------------------------------------------
    def _detect_all(self):
        self.detect_all_btn.configure(state="disabled", text="🔍 Algılanıyor…")
        self.status_lbl.configure(text="Tüm yüzler algılanıyor…")
        self.progress_bar.set(0)
        threading.Thread(target=self._detect_all_thread, daemon=True).start()

    def _detect_all_thread(self):
        total = len(self.items)
        for i, item in enumerate(self.items):
            if item["cv_image"] is None:
                continue
            self.after(0, lambda s=item["status_lbl"]: s.configure(
                text="🔍 Algılanıyor…", text_color="#3B8ED0"
            ))
            self._run_detection(i)
            self.after(0, lambda p=(i + 1) / total: self.progress_bar.set(p))

        detected_total = sum(len(it["face_locations"]) for it in self.items)
        self.after(0, lambda: [
            self.detect_all_btn.configure(state="normal", text="🔍 Tüm Yüzleri Algıla"),
            self.status_lbl.configure(
                text=f"✅ Algılama tamamlandı — toplam {detected_total} yüz bulundu."
            )
        ])

    def _detect_single_thread(self, index: int):
        item = self.items[index]
        if item["cv_image"] is None:
            return
        self.after(0, lambda: item["status_lbl"].configure(
            text="🔍 Algılanıyor…", text_color="#3B8ED0"
        ))
        self._run_detection(index)

    def _run_detection(self, index: int):
        """Yüz tespiti yap ve kartı güncelle."""
        item = self.items[index]
        faces = self.parent_app._detect_faces_sync(item["cv_image"])
        item["face_locations"] = list(faces)
        item["selected_faces"] = [True] * len(faces)
        self.after(0, lambda: self._update_card(index))

    def _update_card(self, index: int):
        item = self.items[index]
        n = len(item["face_locations"])
        if n > 0:
            item["status_lbl"].configure(
                text=f"🎭 {n} yüz bulundu — seçimleri düzenleyebilirsiniz",
                text_color="#2ECC71"
            )
        else:
            item["status_lbl"].configure(
                text="❌ Yüz bulunamadı", text_color="#E74C3C"
            )
        self._draw_thumb(item)
        self._rebuild_checkboxes(index)

    # ------------------------------------------------------------------
    # CHECKBOX YÖNETİMİ
    # ------------------------------------------------------------------
    def _rebuild_checkboxes(self, index: int):
        item = self.items[index]
        cb_frame = item["cb_frame"]
        for w in cb_frame.winfo_children():
            w.destroy()
        item["checkbox_vars"].clear()

        if not item["face_locations"]:
            ctk.CTkLabel(
                cb_frame, text="Yüz algılanamadı.",
                font=ctk.CTkFont(size=10), text_color="gray"
            ).pack(anchor="w")
            return

        for i in range(len(item["face_locations"])):
            var = ctk.BooleanVar(
                value=item["selected_faces"][i]
                if i < len(item["selected_faces"]) else True
            )
            item["checkbox_vars"].append(var)
            ctk.CTkCheckBox(
                cb_frame,
                text=f"Yüz #{i + 1}",
                variable=var,
                font=ctk.CTkFont(size=11),
                width=80,
                command=lambda idx=index: self._on_cb_change(idx)
            ).pack(side="left", padx=4)

    def _on_cb_change(self, index: int):
        item = self.items[index]
        item["selected_faces"] = [v.get() for v in item["checkbox_vars"]]
        self._draw_thumb(item)

    def _reset_card(self, index: int):
        item = self.items[index]
        item["face_locations"] = []
        item["selected_faces"] = []
        item["checkbox_vars"] = []
        for w in item["cb_frame"].winfo_children():
            w.destroy()
        item["status_lbl"].configure(text="🔄 Sıfırlandı", text_color="gray")
        self._draw_thumb(item)

    def _open_editor(self, index: int):
        """Fotoğrafı düzenleyici pencerede aç."""
        item = self.items[index]
        if item["image"] is None:
            messagebox.showwarning("Uyarı", "Önce fotoğraf yüklenmeli!")
            return
        BatchImageEditorWindow(self, item)

    # ------------------------------------------------------------------
    # ÇIKTI KLASÖRÜ
    # ------------------------------------------------------------------
    def _select_output(self):
        d = filedialog.askdirectory(title="Çıktı Klasörü Seç")
        if d:
            self.output_dir = d
            short = os.path.basename(d) or d
            self.output_btn.configure(text=f"📁 {short}")

    # ------------------------------------------------------------------
    # İŞLEME BAŞLA
    # ------------------------------------------------------------------
    def _start_processing(self):
        if self.output_dir is None:
            d = filedialog.askdirectory(title="Önce çıktı klasörü seçin")
            if not d:
                return
            self.output_dir = d
            self.output_btn.configure(text=f"📁 {os.path.basename(d) or d}")

        has_faces = any(
            any(item["selected_faces"]) for item in self.items
            if item["selected_faces"]
        )
        if not has_faces:
            messagebox.showwarning(
                "Uyarı",
                "Hiçbir fotoğrafta seçili yüz yok!\n"
                "Önce 'Tüm Yüzleri Algıla' butonuna tıklayın."
            )
            return

        self.detect_all_btn.configure(state="disabled")
        self.start_btn.configure(state="disabled", text="⚙️ İşleniyor…")
        self._cancelled = False
        threading.Thread(target=self._process_thread, daemon=True).start()

    def _process_thread(self):
        total = len(self.items)
        success = 0
        failed_names = []
        total_faces_blurred = 0

        blur_strength = int(self.parent_app.blur_strength.get())
        blur_style = self.parent_app.blur_style.get()
        margin_pct = self.parent_app.face_margin.get() / 100.0

        for i, item in enumerate(self.items):
            if self._cancelled:
                break
            if item["image"] is None:
                failed_names.append(os.path.basename(item["path"]))
                continue

            self.after(0, lambda s=item["status_lbl"]: s.configure(
                text="⚙️ İşleniyor…", text_color="#3B8ED0"
            ))
            self.after(0, lambda p=i / total: self.progress_bar.set(p))

            try:
                result = item["image"].copy()
                img_w, img_h = result.size
                n_blurred = 0

                for j, (x1, y1, x2, y2) in enumerate(item["face_locations"]):
                    sel = (
                        j < len(item["selected_faces"])
                        and item["selected_faces"][j]
                    )
                    if not sel:
                        continue

                    w, h = x2 - x1, y2 - y1
                    nx1 = int(max(0, x1 - w * margin_pct))
                    ny1 = int(max(0, y1 - h * margin_pct))
                    nx2 = int(min(img_w, x2 + w * margin_pct))
                    ny2 = int(min(img_h, y2 + h * margin_pct))

                    app = self.parent_app
                    if blur_style == "gaussian":
                        result = app._apply_gaussian_blur(result, nx1, ny1, nx2, ny2, blur_strength)
                    elif blur_style == "pixelate":
                        result = app._apply_pixelate(result, nx1, ny1, nx2, ny2, blur_strength)
                    elif blur_style == "black":
                        result = app._apply_black_box(result, nx1, ny1, nx2, ny2)
                    elif blur_style == "color":
                        result = app._apply_color_fill(result, nx1, ny1, nx2, ny2)
                    elif blur_style == "emoji":
                        result = app._apply_emoji(result, nx1, ny1, nx2, ny2)

                    n_blurred += 1
                    total_faces_blurred += 1

                # Kaydet
                fname = os.path.basename(item["path"])
                out = os.path.join(self.output_dir, f"blurred_{fname}")
                if out.lower().endswith((".jpg", ".jpeg")):
                    result.save(out, quality=95)
                else:
                    result.save(out)

                success += 1

                # Kartı işlenmiş hale getir
                def _show_done(itm=item, res=result, nb=n_blurred):
                    itm["status_lbl"].configure(
                        text=f"✅ Kaydedildi  ({nb} yüz bulanıklaştırıldı)",
                        text_color="#2ECC71"
                    )
                    thumb = res.copy()
                    thumb.thumbnail((self.THUMB_W, self.THUMB_H), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(thumb)
                    itm["canvas"].delete("all")
                    itm["canvas"].create_image(
                        self.THUMB_W // 2, self.THUMB_H // 2,
                        image=photo, anchor="center"
                    )
                    itm["canvas"].image = photo

                self.after(0, _show_done)

            except Exception as exc:
                failed_names.append(os.path.basename(item["path"]))
                err_txt = str(exc)[:45]
                self.after(0, lambda s=item["status_lbl"], e=err_txt: s.configure(
                    text=f"❌ Hata: {e}", text_color="#E74C3C"
                ))

        # Tamamlandı
        def _on_done():
            self.progress_bar.set(1.0)
            self.detect_all_btn.configure(state="normal")
            self.start_btn.configure(state="normal", text="✅ İşleme Başla")

            msg = (
                f"🎉 İşlem tamamlandı!\n\n"
                f"Başarılı: {success} / {total}\n"
                f"Bulanıklaştırılan yüz: {total_faces_blurred}\n"
            )
            if failed_names:
                msg += "\nBaşarısız dosyalar:\n" + "\n".join(f"• {n}" for n in failed_names)
            msg += f"\n\n📁 Çıktı klasörü:\n{self.output_dir}"

            self.status_lbl.configure(
                text=f"✅ Tamamlandı — {success}/{total} dosya işlendi."
            )
            messagebox.showinfo("Tamamlandı", msg)

            if messagebox.askyesno("Klasörü Aç", "Çıktı klasörünü açmak ister misiniz?"):
                import subprocess
                if self.parent_app.system == "Windows":
                    subprocess.run(["explorer", os.path.normpath(self.output_dir)])
                elif self.parent_app.system == "Darwin":
                    subprocess.run(["open", self.output_dir])
                else:
                    subprocess.run(["xdg-open", self.output_dir])

        self.after(0, _on_done)




# ============================================================
# TOPLU İŞLEM — FOTOĞRAF DÜZENLEYİCİ
# ============================================================
class BatchImageEditorWindow(ctk.CTkToplevel):
    """
    BatchManagerWindow'daki bir fotoğrafı büyütülmüş olarak gösterir.
    Yüz seçimi (tıklayarak) ve manuel elips çizimi desteklenir.
    Değişiklikler 'Kaydet & Kapat' ile ana karta yansıtılır.
    """

    def __init__(self, batch_manager: "BatchManagerWindow", item: dict):
        super().__init__(batch_manager)
        self.bm = batch_manager          # BatchManagerWindow referansı
        self.item = item                 # Ortak item dict (referans, kopya değil)
        self.parent_app = batch_manager.parent_app

        fname = Path(item["path"]).name
        self.title(f"Düzenleyici — {fname}")
        self.geometry("1100x750")
        self.minsize(800, 550)
        self.transient(batch_manager)
        self.grab_set()

        # --- Çalışma kopyaları (orijinal değişmesin) ---
        self._face_locations: list = list(item["face_locations"])
        self._selected_faces: list = list(item["selected_faces"])

        # --- Canvas state ---
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._display_scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._is_panning = False
        self._pan_sx = 0
        self._pan_sy = 0

        # --- Çizim modu ---
        self._drawing = False
        self._draw_sx: float | None = None
        self._draw_sy: float | None = None
        self._draw_rect_id = None

        self._canvas_photo = None
        self._canvas_img_id = None

        self._build_ui()
        self.after(100, self._refresh)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        # ---- Üst araç çubuğu ----
        toolbar = ctk.CTkFrame(self, height=55, corner_radius=0)
        toolbar.pack(fill="x")
        toolbar.pack_propagate(False)

        ctk.CTkLabel(
            toolbar, text=f"🔎 {Path(self.item['path']).name}",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(side="left", padx=12)

        # Sağ butonlar
        ctk.CTkButton(
            toolbar, text="💾 Kaydet & Kapat", width=160, height=36,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2D7D46", hover_color="#236B38",
            command=self._save_and_close
        ).pack(side="right", padx=8)

        ctk.CTkButton(
            toolbar, text="❌ İptal", width=90, height=36,
            font=ctk.CTkFont(size=12),
            fg_color="#E74C3C", hover_color="#C0392B",
            command=self.destroy
        ).pack(side="right", padx=4)

        # Çizim modu butonu
        self._draw_btn = ctk.CTkButton(
            toolbar, text="✏️ Manuel Çizim: KAPALI", width=200, height=36,
            font=ctk.CTkFont(size=12),
            fg_color="#555555", hover_color="#444444",
            command=self._toggle_draw_mode
        )
        self._draw_btn.pack(side="right", padx=8)

        # Zoom butonları
        for txt, cmd in [("➕", self._zoom_in), ("0", self._zoom_reset), ("➖", self._zoom_out)]:
            ctk.CTkButton(
                toolbar, text=txt, width=36, height=36,
                font=ctk.CTkFont(size=14),
                fg_color="gray40", hover_color="gray50",
                command=cmd
            ).pack(side="right", padx=1)

        ctk.CTkLabel(toolbar, text="Zoom:", font=ctk.CTkFont(size=11)).pack(side="right", padx=4)

        # ---- Ana içerik ----
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=8, pady=8)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # Canvas alanı
        canvas_frame = ctk.CTkFrame(content, fg_color="gray15", corner_radius=10)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

        from tkinter import Canvas as TkCanvas
        self._canvas = TkCanvas(canvas_frame, bg="#1e1e1e", highlightthickness=0, cursor="arrow")
        self._canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        # Canvas olayları
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<ButtonPress-3>", self._start_pan)
        self._canvas.bind("<B3-Motion>", self._do_pan)
        self._canvas.bind("<ButtonRelease-3>", self._stop_pan)
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", self._on_wheel)
        self._canvas.bind("<Button-5>", self._on_wheel)

        # ---- Sağ panel: yüz listesi ----
        right_panel = ctk.CTkFrame(content, width=220, fg_color="gray20", corner_radius=10)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_propagate(False)

        ctk.CTkLabel(
            right_panel, text="🎯 Yüz Listesi",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(12, 4))

        ctk.CTkLabel(
            right_panel,
            text="Tıklayarak seç/kaldır\nYeşil = Bulanıklaştırılacak",
            font=ctk.CTkFont(size=10), text_color="gray"
        ).pack(padx=8)

        # Tümünü Seç / Kaldır
        sel_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        sel_frame.pack(fill="x", padx=8, pady=6)

        ctk.CTkButton(
            sel_frame, text="Tümünü Seç", height=26, width=92,
            font=ctk.CTkFont(size=10),
            fg_color="gray40", hover_color="gray50",
            command=self._select_all
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            sel_frame, text="Tümünü Kaldır", height=26, width=92,
            font=ctk.CTkFont(size=10),
            fg_color="gray40", hover_color="gray50",
            command=self._deselect_all
        ).pack(side="left", padx=2)

        # Yüz checkbox frame (scrollable)
        self._cb_scroll = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        self._cb_scroll.pack(fill="both", expand=True, padx=4, pady=4)

        # Bilgi
        ctk.CTkLabel(
            right_panel,
            text="💡 İpuçları:\n• Sağ tık sürükle = Pan\n• Tekerlek = Zoom\n• 0 tuşu = Zoom sıfırla",
            font=ctk.CTkFont(size=9), text_color="gray", justify="left"
        ).pack(padx=8, pady=8)

        self._right_panel = right_panel
        self._rebuild_face_list()

        # ---- Alt durum çubuğu ----
        self._status = ctk.CTkLabel(
            self, text="Fotoğraf yükleniyor…",
            font=ctk.CTkFont(size=11)
        )
        self._status.pack(pady=(0, 6))

    # ------------------------------------------------------------------
    # YÜZ LİSTESİ
    # ------------------------------------------------------------------
    def _rebuild_face_list(self):
        for w in self._cb_scroll.winfo_children():
            w.destroy()

        if not self._face_locations:
            ctk.CTkLabel(
                self._cb_scroll,
                text="Henüz yüz yok.\n✏️ Manuel çizim ile\nyüz ekleyebilirsiniz.",
                font=ctk.CTkFont(size=11), text_color="gray", justify="center"
            ).pack(pady=20)
            return

        for i, (x1, y1, x2, y2) in enumerate(self._face_locations):
            row = ctk.CTkFrame(self._cb_scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)

            sel = i < len(self._selected_faces) and self._selected_faces[i]
            var = ctk.BooleanVar(value=sel)

            cb = ctk.CTkCheckBox(
                row, text=f"Yüz #{i+1}",
                variable=var,
                font=ctk.CTkFont(size=11),
                width=100,
                command=lambda idx=i, v=var: self._on_cb(idx, v)
            )
            cb.pack(side="left")

            ctk.CTkButton(
                row, text="🗑️", width=30, height=24,
                fg_color="#E74C3C", hover_color="#C0392B",
                command=lambda idx=i: self._delete_face(idx)
            ).pack(side="right", padx=2)

    def _on_cb(self, index: int, var: ctk.BooleanVar):
        while len(self._selected_faces) <= index:
            self._selected_faces.append(True)
        self._selected_faces[index] = var.get()
        self._refresh()

    def _delete_face(self, index: int):
        if 0 <= index < len(self._face_locations):
            del self._face_locations[index]
            if index < len(self._selected_faces):
                del self._selected_faces[index]
        self._rebuild_face_list()
        self._refresh()

    def _select_all(self):
        self._selected_faces = [True] * len(self._face_locations)
        self._rebuild_face_list()
        self._refresh()

    def _deselect_all(self):
        self._selected_faces = [False] * len(self._face_locations)
        self._rebuild_face_list()
        self._refresh()

    # ------------------------------------------------------------------
    # CANVAS ÇİZİMİ
    # ------------------------------------------------------------------
    def _refresh(self):
        """Canvas'ı yeniden çiz."""
        img = self.item["image"]
        if img is None:
            return

        # Üzerine yüz kutucukları çizilmiş kopyası
        preview = img.copy()
        drw = ImageDraw.Draw(preview)
        for i, (x1, y1, x2, y2) in enumerate(self._face_locations):
            sel = i < len(self._selected_faces) and self._selected_faces[i]
            color = "#00FF00" if sel else "#FF6B6B"
            for t in range(4):
                drw.ellipse([x1-t, y1-t, x2+t, y2+t], outline=color)
            lx, ly = x1, max(0, y1 - 20)
            drw.rectangle([lx, ly, lx + 36, ly + 18], fill=color)
            drw.text((lx + 5, ly + 2), f"#{i+1}", fill="black")

        self._display(preview)
        n = len(self._face_locations)
        sel_n = sum(1 for s in self._selected_faces if s)
        self._status.configure(
            text=f"🎭 {n} yüz  |  ✅ {sel_n} seçili  |  Zoom: {self._zoom:.1f}x  |  "
                 f"{'✏️ Çizim Modu AÇIK' if self._drawing else 'Sağ tık = pan  •  Tekerlek = zoom'}"
        )

    def _display(self, pil_img: Image.Image):
        self._canvas.update()
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 550
        iw, ih = pil_img.size

        base_scale = min(cw / iw, ch / ih) * 0.96
        self._display_scale = base_scale * self._zoom
        nw = int(iw * self._display_scale)
        nh = int(ih * self._display_scale)

        self._offset_x = (cw - nw) // 2 + self._pan_x
        self._offset_y = (ch - nh) // 2 + self._pan_y

        if nw < 1 or nh < 1:
            return
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        photo = ImageTk.PhotoImage(resized)
        self._canvas_photo = photo  # referans koru

        if self._canvas_img_id:
            self._canvas.delete(self._canvas_img_id)
        self._canvas_img_id = self._canvas.create_image(
            self._offset_x, self._offset_y, image=photo, anchor="nw"
        )

    def _canvas_to_img(self, cx: float, cy: float):
        """Canvas koordinatını orijinal görüntü koordinatına çevir."""
        ix = (cx - self._offset_x) / self._display_scale
        iy = (cy - self._offset_y) / self._display_scale
        return ix, iy

    # ------------------------------------------------------------------
    # ZOOM & PAN
    # ------------------------------------------------------------------
    def _zoom_in(self):
        self._zoom = min(self._zoom * 1.25, 10.0)
        self._refresh()

    def _zoom_out(self):
        self._zoom = max(self._zoom / 1.25, 0.1)
        self._refresh()

    def _zoom_reset(self):
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._refresh()

    def _on_wheel(self, event):
        if event.num == 4 or event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _start_pan(self, event):
        self._is_panning = True
        self._pan_sx = event.x
        self._pan_sy = event.y
        self._canvas.configure(cursor="fleur")

    def _do_pan(self, event):
        if self._is_panning:
            self._pan_x += event.x - self._pan_sx
            self._pan_y += event.y - self._pan_sy
            self._pan_sx = event.x
            self._pan_sy = event.y
            self._refresh()

    def _stop_pan(self, event):
        self._is_panning = False
        self._canvas.configure(cursor="crosshair" if self._drawing else "arrow")

    # ------------------------------------------------------------------
    # MANUEL ÇİZİM
    # ------------------------------------------------------------------
    def _toggle_draw_mode(self):
        self._drawing = not self._drawing
        if self._drawing:
            self._draw_btn.configure(
                text="✏️ Manuel Çizim: AÇIK",
                fg_color="#E74C3C", hover_color="#C0392B"
            )
            self._canvas.configure(cursor="crosshair")
            self._status.configure(text="✏️ Fotoğraf üzerinde sürükleyerek yüz alanı çizin")
        else:
            self._draw_btn.configure(
                text="✏️ Manuel Çizim: KAPALI",
                fg_color="#555555", hover_color="#444444"
            )
            self._canvas.configure(cursor="arrow")
            self._refresh()

    def _on_press(self, event):
        if not self._drawing:
            # Tıklanan yüzü seç/kaldır
            ix, iy = self._canvas_to_img(event.x, event.y)
            for i in range(len(self._face_locations) - 1, -1, -1):
                x1, y1, x2, y2 = self._face_locations[i]
                pad = 10 / self._display_scale
                if x1 - pad <= ix <= x2 + pad and y1 - pad <= iy <= y2 + pad:
                    while len(self._selected_faces) <= i:
                        self._selected_faces.append(True)
                    self._selected_faces[i] = not self._selected_faces[i]
                    self._rebuild_face_list()
                    self._refresh()
                    return
            return

        self._draw_sx = event.x
        self._draw_sy = event.y
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)

    def _on_drag(self, event):
        if not self._drawing or self._draw_sx is None:
            return
        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
        self._draw_rect_id = self._canvas.create_oval(
            self._draw_sx, self._draw_sy, event.x, event.y,
            outline="#00FF00", width=2, dash=(5, 3)
        )

    def _on_release(self, event):
        if not self._drawing or self._draw_sx is None:
            return

        dw = abs(event.x - self._draw_sx)
        dh = abs(event.y - self._draw_sy)

        if dw > 8 and dh > 8:
            ix1, iy1 = self._canvas_to_img(self._draw_sx, self._draw_sy)
            ix2, iy2 = self._canvas_to_img(event.x, event.y)
            img_w, img_h = self.item["image"].size

            xmin = int(max(0, min(ix1, ix2)))
            ymin = int(max(0, min(iy1, iy2)))
            xmax = int(min(img_w, max(ix1, ix2)))
            ymax = int(min(img_h, max(iy1, iy2)))

            if xmax > xmin and ymax > ymin:
                self._face_locations.append((xmin, ymin, xmax, ymax))
                self._selected_faces.append(True)
                self._rebuild_face_list()
                self._refresh()

        if self._draw_rect_id:
            self._canvas.delete(self._draw_rect_id)
            self._draw_rect_id = None
        self._draw_sx = None
        self._draw_sy = None
        # Çizim modundan otomatik çık (tek çizim = çık)
        # İstenmiyorsa aşağıyı yoruma al:
        # self._toggle_draw_mode()

    # ------------------------------------------------------------------
    # KAYDET & KAPAT
    # ------------------------------------------------------------------
    def _save_and_close(self):
        """Değişiklikleri BatchManagerWindow'daki item'a yaz ve kapat."""
        self.item["face_locations"] = list(self._face_locations)
        self.item["selected_faces"] = list(self._selected_faces)

        # Kartı güncelle
        idx = self.item["index"]
        self.bm._update_card(idx)

        n = len(self._face_locations)
        sel = sum(1 for s in self._selected_faces if s)
        self.item["status_lbl"].configure(
            text=f"🎭 {n} yüz  ({sel} seçili) — düzenlendi",
            text_color="#3B8ED0"
        )
        self.destroy()


def main():
    """Ana fonksiyon"""
    app = FaceBlurApp()
    app.mainloop()


if __name__ == "__main__":
    main()
