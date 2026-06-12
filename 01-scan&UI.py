"""
OBS Ayarı:
    OBS → Tools → Virtual Camera → Start Virtual Camera

AD7747 Ayarı:
    Arduino COM portunu AD7747_PORT değişkeninden ayarlayın.
    Arduino'da kütüphanenin debug_mode = false olduğundan emin olun.
"""

import cv2
import time
import os
import serial
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
from pydobot import Dobot
import base64

# ─── DOBOT ────────────────────────────────────────────────────────────────────
DOBOT_PORT = "COM3"

HOME_X = 215.0
HOME_Y = -40.0
HOME_Z = -115.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

DEFAULT_SCAN_WIDTH   = 10.0  
DEFAULT_SCAN_HEIGHT  = 10.0  
STEP_SIZE    = 0.5

VELOCITY     = 20
ACCELERATION = 20

# ─── KAMERA ───────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 1
CAPTURE_DELAY = 1.0

# ─── AD7747 ───────────────────────────────────────────────────────────────────
AD7747_PORT  = "COM4"
AD7747_BAUD  = 9600
CAP_SAMPLES  = 5

# ─── KAYIT ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "scan_images"


# ══════════════════════════════════════════════════════════════════════════════
#  AD7747 OKUYUCU
# ══════════════════════════════════════════════════════════════════════════════

class AD7747Reader:
    def __init__(self, port, baud):
        self._port    = port
        self._baud    = baud
        self._ser     = None
        self._latest  = None
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=2)
            time.sleep(2)
            self._running = True
            self._thread  = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except serial.SerialException as e:
            return False

    def stop(self):
        self._running = False
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _read_loop(self):
        while self._running:
            try:
                line = self._ser.readline().decode("utf-8").strip()
                if not line:
                    continue
                value = float(line)
                with self._lock:
                    self._latest = value
            except (ValueError, UnicodeDecodeError):
                pass
            except serial.SerialException:
                break

    def read_average(self, n=CAP_SAMPLES, timeout=3.0):
        samples   = []
        deadline  = time.time() + timeout
        last_seen = None
        while len(samples) < n and time.time() < deadline:
            with self._lock:
                current = self._latest
            if current is not None and current != last_seen:
                samples.append(current)
                last_seen = current
            time.sleep(0.05)
        if samples:
            return sum(samples) / len(samples)
        with self._lock:
            return self._latest


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def setup_output_dir(base_dir, type_, design, composition):
    folder_name = f"{type_}_{design}_{composition}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_current_position(robot):
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]
    except Exception:
        return None


def clear_robot_queue(robot):
    try:
        robot._set_queued_cmd_stop_exec()
        time.sleep(0.2)
        robot._set_queued_cmd_clear()
        time.sleep(0.2)
        robot._set_queued_cmd_start_exec()
        time.sleep(0.3)
    except Exception:
        pass


def flush_robot(robot):
    try:
        robot.ser.reset_input_buffer()
        robot.ser.reset_output_buffer()
        time.sleep(0.2)
        clear_robot_queue(robot)
        time.sleep(0.5)
    except Exception:
        pass


def move_and_wait(robot, x, y, z, r,
                  retries=3, settle_time=0.4,
                  timeout=15.0, position_tolerance=0.3):
    start_pos = get_current_position(robot)
    if start_pos:
        dist = ((start_pos[0]-x)**2 +
                (start_pos[1]-y)**2 +
                (start_pos[2]-z)**2) ** 0.5
        if dist <= position_tolerance:
            return True

    for attempt in range(1, retries + 1):
        result    = [False]
        stop_flag = [False]

        def _move():
            try:
                robot.move_to(x, y, z, r, wait=True)
                if not stop_flag[0]:
                    result[0] = True
            except Exception:
                pass

        t = threading.Thread(target=_move, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if result[0]:
            time.sleep(settle_time)
            return True
        else:
            stop_flag[0] = True
            flush_robot(robot)
            time.sleep(1.0)

    return False


def move_with_fallback(robot, x, y, z, r,
                       offsets=[0.0, -0.1, 0.1, -0.2, 0.2]):
    for offset in offsets:
        ty = round(y + offset, 3)
        success = move_and_wait(robot, x, ty, z, r)
        if success:
            return True, ty
    return False, y


def generate_scan_points(home_x, home_y, width, height, step):
    points = []
    x      = home_x
    row    = 0
    while round(x, 6) >= round(home_x - height, 6):
        if row % 2 == 0:
            y = home_y
            while round(y, 6) <= round(home_y + width, 6):
                points.append((round(x, 3), round(y, 3)))
                y = round(y + step, 6)
        else:
            y = home_y + width
            while round(y, 6) >= round(home_y, 6):
                points.append((round(x, 3), round(y, 3)))
                y = round(y - step, 6)
        x   = round(x - step, 6)
        row += 1
    return points


# ══════════════════════════════════════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════════════════════════════════════

class ScanApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tarama Kontrol Paneli")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")

        try:
            self.icon_img = tk.PhotoImage(width=16, height=16)
            X = "#e94560"
            T = ""
            
            pixels = [
                [T,T,T,T,T,X,X,X,X,X,X,T,T,T,T,T],
                [T,T,T,X,X,T,T,T,T,T,T,X,X,T,T,T],
                [T,T,X,T,T,T,T,X,X,T,T,T,T,X,T,T],
                [T,X,T,T,T,T,T,X,X,T,T,T,T,T,X,T],
                [T,X,T,T,T,T,T,T,T,T,T,T,T,T,X,T],
                [X,T,T,T,T,X,X,T,T,X,X,T,T,T,T,X],
                [X,T,T,T,X,T,T,T,T,T,T,X,T,T,T,X],
                [X,X,X,X,T,T,T,T,T,T,T,T,X,X,X,X],
                [X,X,X,X,T,T,T,T,T,T,T,T,X,X,X,X],
                [X,T,T,T,X,T,T,T,T,T,T,X,T,T,T,X],
                [X,T,T,T,T,X,X,T,T,X,X,T,T,T,T,X],
                [T,X,T,T,T,T,T,T,T,T,T,T,T,T,X,T],
                [T,X,T,T,T,T,T,X,X,T,T,T,T,T,X,T],
                [T,T,X,T,T,T,T,X,X,T,T,T,T,X,T,T],
                [T,T,T,X,X,T,T,T,T,T,T,X,X,T,T,T],
                [T,T,T,T,T,X,X,X,X,X,X,T,T,T,T,T]
            ]
            
            for row_idx, row in enumerate(pixels):
                for col_idx, color in enumerate(row):
                    if color: # Skip transparent pixels
                        self.icon_img.put(color, (col_idx, row_idx))
                        
            self.wm_iconphoto(True, self.icon_img)
        except Exception as e:
            print(f"İkon yüklenirken hata oluştu: {e}")
            
        self._scan_thread = None
        self._stop_event  = threading.Event()
        
        # Süre hesabı için değişkenler
        self._scanned_count = 0
        self._total_count = 0

        self._build_ui()

    # ── UI inşa ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        PAD = 12
        BG  = "#1a1a2e"
        CARD= "#16213e"
        ACC = "#e94560"
        FG  = "#eaeaea"
        DIM = "#8892a4"
        FONT_H  = ("Segoe UI", 11, "bold")
        FONT_N  = ("Segoe UI", 10)
        FONT_SM = ("Segoe UI", 9)
        FONT_MONO = ("Consolas", 9)

        self.configure(bg=BG)

        # ── Başlık ────────────────────────────────────────────────────────────
        header = tk.Frame(self, bg=ACC, padx=PAD, pady=8)
        header.pack(fill="x")
        tk.Label(header, text="TARAMA SİSTEMİ", font=("Segoe UI", 13, "bold"),
                 bg=ACC, fg="white").pack(side="left")

        # ── Ana çerçeve ───────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG, padx=PAD, pady=PAD)
        body.pack(fill="both")

        # Sol sütun: konfigürasyon + ilerleme
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="y", padx=(0, PAD))

        # Sağ sütun: log
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # ── Kart Yardımcı Fonksiyonu ──────────────────────────────────────────
        def card(parent, title):
            f = tk.Frame(parent, bg=CARD, padx=10, pady=10,
                         highlightbackground="#2a2a4e", highlightthickness=1)
            f.pack(fill="x", pady=(0, 10))
            tk.Label(f, text=title, font=FONT_H, bg=CARD, fg=ACC).pack(anchor="w", pady=(0, 6))
            return f

        # ── Dosya Adı Kartı ───────────────────────────────────────────────────
        fname_card = card(left, "Dosya Adı")

        fields = [("TYPE", "17K1025-00"), ("DESIGN", "KaynakGroup"), ("COMPOSITION", "100%BAMBU")]
        self._entries = {}
        for label, default in fields:
            row = tk.Frame(fname_card, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=FONT_SM, bg=CARD, fg=DIM, width=12, anchor="w").pack(side="left")
            e = tk.Entry(row, font=FONT_N, bg="#0f3460", fg=FG,
                         insertbackground=FG, relief="flat",
                         highlightbackground="#3a3a6e", highlightthickness=1)
            e.insert(0, default)
            e.pack(side="left", fill="x", expand=True, ipady=3, padx=(4, 0))
            self._entries[label] = e

        self._preview_var = tk.StringVar()
        tk.Label(fname_card, textvariable=self._preview_var,
                 font=FONT_SM, bg=CARD, fg=DIM, wraplength=260, anchor="w").pack(fill="x", pady=(6, 0))
        self._update_preview()
        for e in self._entries.values():
            e.bind("<KeyRelease>", lambda _: self._update_preview())

        # ── Tarama Alanı Kartı ────────────────────────────────────────────────
        area_card = card(left, "Tarama Alanı (mm)")
        
        self._width_var = tk.StringVar(value=str(DEFAULT_SCAN_WIDTH))
        self._height_var = tk.StringVar(value=str(DEFAULT_SCAN_HEIGHT))

        def add_area_setting(parent, label_text, var):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label_text, font=FONT_SM, bg=CARD, fg=DIM, width=15, anchor="w").pack(side="left")
            e = tk.Entry(row, textvariable=var, font=FONT_N, bg="#0f3460", fg=FG,
                         insertbackground=FG, relief="flat",
                         highlightbackground="#3a3a6e", highlightthickness=1)
            e.pack(side="left", fill="x", expand=True, ipady=3, padx=(4, 0))
            return e

        self._height_entry = add_area_setting(area_card, "X Uzunluğu (Yük.):", self._height_var)
        self._width_entry  = add_area_setting(area_card, "Y Genişliği (Gen.):", self._width_var)

        # ── İlerleme Kartı ────────────────────────────────────────────────────
        prog_card = card(left, "İlerleme")

        # Büyük sayaç
        counter_row = tk.Frame(prog_card, bg=CARD)
        counter_row.pack(fill="x", pady=(0, 8))
        self._done_var = tk.StringVar(value="0")
        self._total_var = tk.StringVar(value="—")
        tk.Label(counter_row, textvariable=self._done_var,
                 font=("Segoe UI", 32, "bold"), bg=CARD, fg=FG).pack(side="left")
        tk.Label(counter_row, text=" / ", font=("Segoe UI", 20),
                 bg=CARD, fg=DIM).pack(side="left")
        tk.Label(counter_row, textvariable=self._total_var,
                 font=("Segoe UI", 20), bg=CARD, fg=DIM).pack(side="left")

        # İlerleme çubuğu
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("Scan.Horizontal.TProgressbar",
                        troughcolor="#0f3460", background=ACC,
                        thickness=10, borderwidth=0)
        self._pb = ttk.Progressbar(prog_card, style="Scan.Horizontal.TProgressbar",
                                   orient="horizontal", mode="determinate", length=260)
        self._pb.pack(fill="x", pady=(0, 6))

        # Kalan / geçen
        stat_row1 = tk.Frame(prog_card, bg=CARD)
        stat_row1.pack(fill="x", pady=(0, 2))
        self._remaining_var = tk.StringVar(value="Kalan: —")
        self._elapsed_var   = tk.StringVar(value="Geçen: 0s")
        tk.Label(stat_row1, textvariable=self._remaining_var,
                 font=FONT_SM, bg=CARD, fg=DIM).pack(side="left")
        tk.Label(stat_row1, textvariable=self._elapsed_var,
                 font=FONT_SM, bg=CARD, fg=DIM).pack(side="right")

        # Kalan Süre
        stat_row2 = tk.Frame(prog_card, bg=CARD)
        stat_row2.pack(fill="x")
        self._eta_var = tk.StringVar(value="Kalan Süre: —")
        # Rengi biraz daha belirgin yapıyoruz
        tk.Label(stat_row2, textvariable=self._eta_var,
                 font=("Segoe UI", 9, "bold"), bg=CARD, fg="#5dbcd2").pack(side="right")

        # Durum etiketi
        self._status_var = tk.StringVar(value="Hazır")
        tk.Label(prog_card, textvariable=self._status_var,
                 font=FONT_N, bg=CARD, fg=FG).pack(anchor="w", pady=(8, 0))

        # ── Butonlar ─────────────────────────────────────────────────────────
        btn_row = tk.Frame(left, bg=BG)
        btn_row.pack(fill="x", pady=(4, 0))

        self._home_btn = tk.Button(
            btn_row, text="⌂  BAŞLANGICA GİT", font=FONT_H,
            bg="#0f3460", fg="white", relief="flat",
            activebackground="#16213e", activeforeground="white",
            cursor="hand2", padx=8, pady=8,
            command=self._go_home)
        self._home_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._start_btn = tk.Button(
            btn_row, text="▶  BAŞLAT", font=FONT_H,
            bg=ACC, fg="white", relief="flat",
            activebackground="#c73652", activeforeground="white",
            cursor="hand2", padx=8, pady=8,
            command=self._start_scan)
        self._start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._stop_btn = tk.Button(
            btn_row, text="■  DURDUR", font=FONT_H,
            bg="#2a2a4e", fg=DIM, relief="flat",
            activebackground="#3a3a6e", activeforeground="white",
            cursor="hand2", padx=8, pady=8,
            state="disabled",
            command=self._stop_scan)
        self._stop_btn.pack(side="left", fill="x", expand=True)

        # ── Log Alanı ─────────────────────────────────────────────────────────
        tk.Label(right, text="LOG", font=FONT_H, bg=BG, fg=ACC).pack(anchor="w", pady=(0, 4))
        self._log = scrolledtext.ScrolledText(
            right, font=FONT_MONO, bg="#0d0d1a", fg="#7ec8e3",
            insertbackground="white", relief="flat",
            highlightbackground="#2a2a4e", highlightthickness=1,
            width=52, height=28, wrap="none")
        self._log.pack(fill="both", expand=True)
        self._log.configure(state="disabled")

        self._start_time = None

    # ── Yardımcı ──────────────────────────────────────────────────────────────

    def _update_preview(self):
        t = self._entries["TYPE"].get() or "—"
        d = self._entries["DESIGN"].get() or "—"
        c = self._entries["COMPOSITION"].get() or "—"
        self._preview_var.set(f"Klasör: {t}_{d}_{c}")

    def _log_write(self, msg):
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n")
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _set_progress(self, done, total):
        self._scanned_count = done
        self._total_count = total
        def _do():
            self._done_var.set(str(done))
            self._total_var.set(str(total))
            self._pb["maximum"] = total
            self._pb["value"]   = done
            remaining = total - done
            self._remaining_var.set(f"Kalan: {remaining} görüntü")
        self.after(0, _do)

    def _set_status(self, msg):
        self.after(0, lambda: self._status_var.set(msg))

    # ── Başlangıca Git İşlemleri ──────────────────────────────────────────────

    def _go_home(self):
        self._home_btn.configure(state="disabled")
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="disabled")
        
        self._set_status("Başlangıca gidiliyor…")
        threading.Thread(target=self._go_home_worker, daemon=True).start()

    def _go_home_worker(self):
        self._log_write(f"Dobot'a bağlanılıyor: {DOBOT_PORT}")
        try:
            robot = Dobot(port=DOBOT_PORT, verbose=False)
            robot.speed(VELOCITY, ACCELERATION)
            self._log_write("Dobot bağlantısı başarılı. Başlangıç pozisyonuna hareket ediliyor...")
            
            if move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R):
                self._log_write("Başlangıç pozisyonuna başarıyla ulaşıldı.")
                self._set_status("Başlangıçta hazır.")
            else:
                self._log_write("HATA: Başlangıç pozisyonuna ulaşılamadı.")
                self._set_status("Hata: Başlangıca gidilemedi.")
            
            robot.close()
        except Exception as e:
            self._log_write(f"Dobot bağlantı hatası: {e}")
            self._set_status("Hata: Dobot yok")

        self.after(0, self._on_home_done)

    def _on_home_done(self):
        self._home_btn.configure(state="normal")
        self._start_btn.configure(state="normal")

    # ── Tarama İşlemleri ──────────────────────────────────────────────────────

    def _start_scan(self):
        try:
            float(self._width_var.get())
            float(self._height_var.get())
        except ValueError:
            self._log_write("HATA: X Uzunluğu ve Y Genişliği sadece sayı olmalıdır!")
            self._set_status("Hata: Geçersiz Boyut")
            return

        self._stop_event.clear()
        self._home_btn.configure(state="disabled")
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal", bg="#e94560", fg="white")
        
        for e in self._entries.values():
            e.configure(state="disabled")
        self._width_entry.configure(state="disabled")
        self._height_entry.configure(state="disabled")

        # Değişkenleri sıfırla
        self._scanned_count = 0
        self._total_count = 0
        self._eta_var.set("Kalan Süre: Hesaplanıyor...")

        self._start_time = time.time()
        self._scan_thread = threading.Thread(target=self._scan_worker, daemon=True)
        self._scan_thread.start()
        self._tick_elapsed()

    def _stop_scan(self):
        self._stop_event.set()
        self._set_status("Durduruluyor…")

    def _tick_elapsed(self):
        if self._start_time and not self._stop_event.is_set():
            elapsed = int(time.time() - self._start_time)
            
            # Geçen süreyi formatla
            m_el, s_el = divmod(elapsed, 60)
            if m_el > 0:
                el_str = f"Geçen: {m_el}dk {s_el}sn"
            else:
                el_str = f"Geçen: {s_el}sn"
            self._elapsed_var.set(el_str)

            # Kalan süreyi hesapla ve formatla
            if self._scanned_count > 0 and self._total_count > 0:
                time_per_item = elapsed / self._scanned_count
                rem_items = self._total_count - self._scanned_count
                
                if rem_items > 0:
                    rem_time = int(time_per_item * rem_items)
                    m, s = divmod(rem_time, 60)
                    h, m = divmod(m, 60)
                    
                    if h > 0:
                        eta_str = f"Kalan Süre: ~{h}sa {m}dk"
                    elif m > 0:
                        eta_str = f"Kalan Süre: ~{m}dk {s}sn"
                    else:
                        eta_str = f"Kalan Süre: ~{s}sn"
                else:
                    eta_str = "Kalan Süre: 0sn"
                    
                self._eta_var.set(eta_str)
            elif self._scanned_count == 0 and elapsed > 0:
                self._eta_var.set("Kalan Süre: Hesaplanıyor...")

            self.after(1000, self._tick_elapsed)

    def _on_scan_done(self):
        self._home_btn.configure(state="normal")
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled", bg="#2a2a4e", fg="#8892a4")
        for e in self._entries.values():
            e.configure(state="normal")
        self._width_entry.configure(state="normal")
        self._height_entry.configure(state="normal")
        self._start_time = None

    def _scan_worker(self):
        type_       = self._entries["TYPE"].get()
        design      = self._entries["DESIGN"].get()
        composition = self._entries["COMPOSITION"].get()

        scan_w = float(self._width_var.get())
        scan_h = float(self._height_var.get())

        save_dir = setup_output_dir(OUTPUT_DIR, type_, design, composition)
        self._log_write(f"Görüntüler kaydedilecek: {save_dir}")
        self._log_write(f"Tarama Alanı: X={scan_h}mm, Y={scan_w}mm")

        # AD7747
        ad7747 = AD7747Reader(AD7747_PORT, AD7747_BAUD)
        ad7747_ok = ad7747.start()
        if ad7747_ok:
            self._log_write(f"[AD7747] Bağlandı: {AD7747_PORT}")
        else:
            self._log_write("[UYARI] AD7747 bağlanamadı.")

        # Kamera
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._log_write("Kamera açılamadı!")
            ad7747.stop()
            self.after(0, self._on_scan_done)
            self._set_status("Hata: Kamera yok")
            return

        # Dobot
        self._log_write(f"Dobot'a bağlanılıyor: {DOBOT_PORT}")
        try:
            robot = Dobot(port=DOBOT_PORT, verbose=False)
        except Exception as e:
            self._log_write(f"Dobot bağlantı hatası: {e}")
            cap.release()
            ad7747.stop()
            self.after(0, self._on_scan_done)
            self._set_status("Hata: Dobot yok")
            return

        robot.speed(VELOCITY, ACCELERATION)
        self._log_write("Dobot bağlantısı başarılı.")

        self._set_status("Home'a gidiliyor…")
        if not move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R):
            self._log_write("Home pozisyonuna ulaşılamadı. İptal.")
            cap.release()
            robot.close()
            ad7747.stop()
            self.after(0, self._on_scan_done)
            self._set_status("Hata: Home başarısız")
            return

        self._log_write("Home pozisyonuna ulaşıldı. Tarama başlıyor…")

        points = generate_scan_points(HOME_X, HOME_Y, scan_w, scan_h, STEP_SIZE)
        total  = len(points)
        self._log_write(f"Toplam {total} nokta taranacak.")
        self._set_progress(0, total)
        self._set_status("Taranıyor…")

        log_path  = os.path.join(save_dir, "scan_log.csv")
        img_index = 0

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("index,timestamp,x,y,z,r,capacitance_pF,image_file,sharpness\n")

            for i, (x, y) in enumerate(points):

                if self._stop_event.is_set():
                    self._log_write("Tarama kullanıcı tarafından durduruldu.")
                    break

                if i > 0 and i % 50 == 0:
                    clear_robot_queue(robot)

                self._log_write(f"[{i+1:04d}/{total}] X={x:.2f}, Y={y:.2f}")

                moved, actual_y = move_with_fallback(robot, x, y, SCAN_Z, SCAN_R)
                time.sleep(CAPTURE_DELAY)

                if not moved:
                    self._log_write("  NOKTA ATLANDI")
                    log_file.write(f"FAIL,{datetime.now().isoformat()},{x},{y},{SCAN_Z},{SCAN_R},,SKIPPED,\n")
                    continue

                img_index += 1

                cap_value = None
                if ad7747_ok:
                    cap_value = ad7747.read_average(n=CAP_SAMPLES)
                    if cap_value is not None:
                        self._log_write(f"  Kapasitans: {cap_value:.6f} pF")
                    else:
                        self._log_write("  Kapasitans: ölçüm alınamadı")

                img_name = f"img_{img_index:04d}_x{x:.1f}_y{actual_y:.1f}.jpg"
                img_path = os.path.join(save_dir, img_name)

                for _ in range(5):
                    cap.grab()
                ret, frame = cap.read()
                sharpness  = 0.0
                img_status = "FAIL"
                if ret and frame is not None:
                    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    img_status = "OK"
                    blur_tag = "[BULANIK]" if sharpness < 30 else "[OK]"
                    self._log_write(f"  Netlik: {sharpness:.1f} {blur_tag}")
                else:
                    self._log_write(f"  Görüntü alınamadı: {img_name}")

                cap_str = f"{cap_value:.6f}" if cap_value is not None else ""
                log_file.write(
                    f"{img_index},{datetime.now().isoformat()},"
                    f"{x},{actual_y},{SCAN_Z},{SCAN_R},"
                    f"{cap_str},"
                    f"{img_name if img_status == 'OK' else 'FAIL'},"
                    f"{sharpness:.1f}\n"
                )
                log_file.flush()

                self._set_progress(img_index, total)

        cap.release()
        robot.close()
        ad7747.stop()

        self._log_write(f"\nTarama tamamlandı. {img_index} nokta işlendi.")
        self._log_write(f"Log: {log_path}")
        self._set_status(f"Tamamlandı — {img_index} görüntü")
        self._eta_var.set("Kalan Süre: 0sn")
        self.after(0, self._on_scan_done)


if __name__ == "__main__":
    app = ScanApp()
    app.mainloop()