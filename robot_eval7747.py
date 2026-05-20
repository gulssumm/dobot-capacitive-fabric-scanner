"""
AD7747 Kapasitans + Dobot Robot Kol — Canlı Tarama
====================================================
Dobot kolu hareket ettirirken her noktada AD7747'den
kapasitans verisi okur ve CSV'ye kaydeder.

Gereksinimler:
    pip install pyserial pydobot

Kullanım:
    python ad7747_dobot_scan.py
"""

import serial
import threading
import time
import csv
import os
from datetime import datetime
from pydobot import Dobot

# ─────────────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────────────

DOBOT_PORT   = "COM3"       # Dobot COM portu
AD7747_PORT  = "COM1"       # AD7747 Eval board COM portu
AD7747_BAUD  = 115200        # Eval software baud rate (9600 veya 115200)

# Home (başlangıç) pozisyonu — kendi değerlerinize göre güncelleyin
HOME_X = 215.0
HOME_Y = -39.0
HOME_Z = -115.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

# Tarama alanı
SCAN_WIDTH  = 10.0    # X ekseninde genişlik (mm)
SCAN_HEIGHT = 10.0    # Y ekseninde derinlik (mm)
STEP_SIZE   = 0.5     # Adım büyüklüğü (mm)

VELOCITY     = 20
ACCELERATION = 20

# Her noktada kaç kez kapasitans ölçümü yapılsın?
SAMPLES_PER_POINT = 3
CAPTURE_DELAY     = 0.5    # Noktaya varınca ölçüm öncesi bekleme (saniye)

OUTPUT_DIR = "scan_data"
LOG_FILE   = os.path.join(OUTPUT_DIR, f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")


# ─────────────────────────────────────────────────────
#  AD7747 — PAYLAŞILAN SON DEĞER (thread-safe)
# ─────────────────────────────────────────────────────
_cap_lock      = threading.Lock()
_latest_cap_pf = None          # Son okunan kapasitans (pF)
_ad7747_ready  = threading.Event()   # İlk veri gelince set olur


def parse_ad7747(line: str):
    """
    AD7747 Eval software'den gelen ASCII satırı parse eder.
    Desteklenen formatlar:
        "CH1=123.456pF"   →  123.456
        "123.456"          →  123.456
        "CAP: 123.456"     →  123.456
    Tanınamazsa None döner.
    """
    line = line.strip()
    if not line:
        return None

    # "CH1=123.456pF" veya "CH1=123.456"
    if "=" in line:
        try:
            val = line.split("=", 1)[1]
            val = val.lower().replace("pf", "").replace(",", ".").strip()
            return float(val)
        except ValueError:
            pass

    # Token bazlı arama (herhangi bir sayısal token)
    for token in line.split():
        clean = token.lower().replace("pf", "").replace(",", ".").strip(":()")
        try:
            return float(clean)
        except ValueError:
            continue

    return None


def ad7747_reader(stop_event: threading.Event):
    """
    Arka planda sürekli AD7747'den veri okur.
    Her geçerli satırda _latest_cap_pf güncellenir.
    """
    global _latest_cap_pf

    print(f"[AD7747] {AD7747_PORT} açılıyor ({AD7747_BAUD} baud)...")
    try:
        ser = serial.Serial(
            port=AD7747_PORT,
            baudrate=AD7747_BAUD,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0,
        )
        print("[AD7747] Bağlandı")
    except serial.SerialException as e:
        print(f"[AD7747] HATA: {e}")
        print("         → AD7747_PORT değişkenini kontrol edin.")
        return

    while not stop_event.is_set():
        try:
            raw  = ser.readline()
            line = raw.decode("ascii", errors="ignore")
            val  = parse_ad7747(line)
            if val is not None:
                with _cap_lock:
                    _latest_cap_pf = val
                _ad7747_ready.set()
        except serial.SerialException as e:
            print(f"[AD7747] Okuma hatası: {e}")
            break

    ser.close()
    print("[AD7747] Port kapatıldı.")


def read_capacitance(n_samples: int = 3, timeout: float = 3.0):
    """
    n_samples adet geçerli ölçüm toplar, ortalamasını döndürür.
    Veri gelmezse timeout sonunda None döner.
    """
    if not _ad7747_ready.wait(timeout=timeout):
        print("[AD7747] Uyarı: Veri bekleniyor...")
        return None

    samples = []
    deadline = time.time() + timeout

    while len(samples) < n_samples and time.time() < deadline:
        with _cap_lock:
            val = _latest_cap_pf
        if val is not None:
            samples.append(val)
        time.sleep(0.08)   # ~80 ms → 12 Hz örnekleme hızı

    return round(sum(samples) / len(samples), 5) if samples else None


# ─────────────────────────────────────────────────────
#  DOBOT YARDIMCI FONKSİYONLARI  (kodunuzdan alındı)
# ─────────────────────────────────────────────────────

def get_current_position(robot: Dobot):
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]
    except Exception as e:
        print(f"Pozisyon okunamadı: {e}")
        return None


def clear_robot_queue(robot: Dobot):
    try:
        robot._set_queued_cmd_stop_exec()
        time.sleep(0.2)
        robot._set_queued_cmd_clear()
        time.sleep(0.2)
        robot._set_queued_cmd_start_exec()
        time.sleep(0.3)
    except Exception as e:
        print(f"Kuyruk temizleme hatası: {e}")


def flush_robot(robot: Dobot):
    try:
        robot.ser.reset_input_buffer()
        robot.ser.reset_output_buffer()
        time.sleep(0.2)
        clear_robot_queue(robot)
        time.sleep(0.5)
    except Exception as e:
        print(f"Flush hatası: {e}")


def move_and_wait(robot: Dobot, x, y, z, r,
                  retries=3, settle_time=0.6,
                  timeout=15.0, position_tolerance=0.3):
    # Zaten hedefe yakın mı?
    pos = get_current_position(robot)
    if pos:
        dist = ((pos[0]-x)**2 + (pos[1]-y)**2 + (pos[2]-z)**2) ** 0.5
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
            except Exception as e:
                if not stop_flag[0]:
                    print(f"move_to hatası: {e}")

        t = threading.Thread(target=_move, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if result[0]:
            time.sleep(settle_time)
            return True

        stop_flag[0] = True
        print(f"[FAIL] Timeout (deneme {attempt}/{retries})")
        flush_robot(robot)
        time.sleep(1.0)

    return False


def move_with_fallback(robot: Dobot, x, y, z, r,
                       offsets=(0.0, -0.1, 0.1, -0.2, 0.2)):
    for offset in offsets:
        ty = round(y + offset, 3)
        if offset != 0.0:
            print(f"[RETRY] Y={y} → Y={ty} (offset={offset:+.1f})")
        if move_and_wait(robot, x, ty, z, r):
            return True, ty
    return False, y


def generate_scan_points(home_x, home_y, width, height, step):
    """Yılan tarama — kodunuzdaki ile aynı mantık."""
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


# ─────────────────────────────────────────────────────
#  ANA TARAMA DÖNGÜSÜ
# ─────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── AD7747 okuma thread'ini başlat ──────────────────
    stop_event = threading.Event()
    ad_thread  = threading.Thread(
        target=ad7747_reader, args=(stop_event,), daemon=True, name="AD7747"
    )
    ad_thread.start()

    # ── Dobot bağlantısı ────────────────────────────────
    print(f"\n[Dobot]  {DOBOT_PORT} açılıyor...")
    try:
        robot = Dobot(port=DOBOT_PORT, verbose=False)
    except Exception as e:
        print(f"[Dobot]  HATA: {e}")
        stop_event.set()
        return

    robot.speed(VELOCITY, ACCELERATION)
    print("[Dobot]  Bağlandı")

    # ── Home pozisyonu ───────────────────────────────────
    print(f"\n[Dobot]  Home pozisyonuna gidiliyor: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}")
    if not move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R):
        print("[Dobot]  Home'a ulaşılamadı. Çıkılıyor.")
        stop_event.set()
        robot.close()
        return
    print("[Dobot]  Home'a ulaşıldı.\n")

    # ── Tarama noktaları ────────────────────────────────
    points = generate_scan_points(HOME_X, HOME_Y, SCAN_WIDTH, SCAN_HEIGHT, STEP_SIZE)
    total  = len(points)
    print(f"Toplam {total} nokta | Alan: {SCAN_WIDTH}×{SCAN_HEIGHT} mm | Adım: {STEP_SIZE} mm")
    print(f"CSV dosyası: {LOG_FILE}\n")
    print("─" * 72)
    print(f"{'#':>5}  {'X':>7} {'Y':>7} {'Z':>7}  {'Kapasite(pF)':>14}  Durum")
    print("─" * 72)

    # ── CSV dosyası aç ──────────────────────────────────
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x_mm", "y_mm", "z_mm", "r_deg",
                         "capacitance_pf", "timestamp", "status"])

        for i, (x, y) in enumerate(points, start=1):

            # Her 50 noktada kuyruk temizle
            if i > 1 and i % 50 == 0:
                clear_robot_queue(robot)

            # Hareketi dene
            moved, actual_y = move_with_fallback(robot, x, y, SCAN_Z, SCAN_R)

            if not moved:
                print(f"[{i:04d}/{total}]  X={x:7.2f} Y={y:7.2f}  →  ATLANDI")
                writer.writerow([i, x, y, SCAN_Z, SCAN_R, "", datetime.now().isoformat(), "SKIP"])
                continue

            # Titreşimin durması için bekle
            time.sleep(CAPTURE_DELAY)

            # Kapasitans ölç
            cap_pf = read_capacitance(n_samples=SAMPLES_PER_POINT)
            ts     = datetime.now().isoformat(timespec="milliseconds")

            # Ekrana yaz
            cap_str    = f"{cap_pf:>14.5f}" if cap_pf is not None else f"{'—':>14}"
            status_str = "OK" if cap_pf is not None else "NO_DATA"
            print(f"[{i:04d}/{total}]  X={x:7.2f} Y={actual_y:7.2f} Z={SCAN_Z:7.2f}  {cap_str}  {status_str}")

            # CSV'ye yaz
            writer.writerow([i, x, actual_y, SCAN_Z, SCAN_R,
                             cap_pf if cap_pf is not None else "",
                             ts, status_str])
            f.flush()

    # ── Temizlik ────────────────────────────────────────
    stop_event.set()
    ad_thread.join(timeout=2.0)
    robot.close()
    print(f"\nTarama tamamlandı. {total} nokta işlendi.")
    print(f"Veriler kaydedildi: {LOG_FILE}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHATA: {e}")
        raise