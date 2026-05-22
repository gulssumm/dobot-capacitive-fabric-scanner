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
from datetime import datetime
from pydobot import Dobot

# ─── DOBOT ────────────────────────────────────────────────────────────────────
DOBOT_PORT = "COM3"

HOME_X = 215.0
HOME_Y = -40.0
HOME_Z = -115.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

SCAN_WIDTH   = 10.0
SCAN_HEIGHT  = 10.0
STEP_SIZE    = 0.5

VELOCITY     = 20
ACCELERATION = 20

# ─── KAMERA ───────────────────────────────────────────────────────────────────
CAMERA_INDEX  = 1
CAPTURE_DELAY = 1.0

# ─── AD7747 ───────────────────────────────────────────────────────────────────
AD7747_PORT  = "COM4"   
AD7747_BAUD  = 9600
CAP_SAMPLES  = 5        # Her noktada kaç ölçüm ortalaması alınsın

# ─── KAYIT ────────────────────────────────────────────────────────────────────
TYPE        = "18K7119-00"
DESIGN      = "Kaynak_Tekstil"
COMPOSITION = "97%COTTON_3EA_pin_mark"
OUTPUT_DIR  = "scan_images"


# ══════════════════════════════════════════════════════════════════════════════
#  AD7747 OKUYUCU
# ══════════════════════════════════════════════════════════════════════════════

class AD7747Reader:
    """Arka planda sürekli okuma yapan kapasitans okuyucu."""

    def __init__(self, port, baud):
        self._port    = port
        self._baud    = baud
        self._ser     = None
        self._latest  = None   # son geçerli değer (float pF)
        self._lock    = threading.Lock()
        self._running = False
        self._thread  = None

    def start(self):
        try:
            self._ser = serial.Serial(self._port, self._baud, timeout=2)
            time.sleep(2)  # Arduino reset
            self._running = True
            self._thread  = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            print(f"[AD7747] Bağlandı: {self._port}")
            return True
        except serial.SerialException as e:
            print(f"[AD7747] Port hatası: {e}")
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
                pass  # başlık satırı veya debug mesajı, atla
            except serial.SerialException:
                break

    def read_average(self, n=CAP_SAMPLES, timeout=3.0):
        """
        n adet yeni ölçüm topla, ortalamasını döndür.
        Zaman aşımında eldeki son değeri döndürür; hiç değer yoksa None.
        """
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
        # zaman aşımı — en azından son değeri döndür
        with self._lock:
            return self._latest


# ══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════════════════════════════════════════

def setup_output_dir(base_dir: str) -> str:
    folder_name = f"{TYPE}_{DESIGN}_{COMPOSITION}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"Görüntüler kaydedilecek: {path}")
    return path


def capture_image(cap: cv2.VideoCapture, save_path: str) -> bool:
    for _ in range(5):
        cap.grab()
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Görüntü alınamadı: {save_path}")
        return False
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"  Netlik skoru: {score:.1f}  {'[BULANIK]' if score < 30 else '[OK]'}")
    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def get_current_position(robot: Dobot):
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]
    except Exception as e:
        print(f"Pozisyon okunamadı: {e}")
        return None


def clear_robot_queue(robot):
    try:
        robot._set_queued_cmd_stop_exec()
        time.sleep(0.2)
        robot._set_queued_cmd_clear()
        time.sleep(0.2)
        robot._set_queued_cmd_start_exec()
        time.sleep(0.3)
        print("[QUEUE] Kuyruk temizlendi.")
    except Exception as e:
        print(f"Kuyruk temizleme hatası: {e}")


def flush_robot(robot):
    try:
        robot.ser.reset_input_buffer()
        robot.ser.reset_output_buffer()
        time.sleep(0.2)
        clear_robot_queue(robot)
        time.sleep(0.5)
    except Exception as e:
        print(f"Flush hatası: {e}")


def move_and_wait(robot, x, y, z, r,
                  retries=3, settle_time=0.6,
                  timeout=15.0, position_tolerance=0.3):

    start_pos = get_current_position(robot)
    if start_pos:
        dist = ((start_pos[0]-x)**2 +
                (start_pos[1]-y)**2 +
                (start_pos[2]-z)**2) ** 0.5
        if dist <= position_tolerance:
            print(f"  [OK] Zaten hedefe yakın (dist={dist:.2f} mm)")
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
                    print(f"  move_to hatası: {e}")

        t = threading.Thread(target=_move, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if result[0]:
            print(f"  [OK] Hedefe ulaşıldı (deneme {attempt})")
            time.sleep(settle_time)
            return True
        else:
            stop_flag[0] = True
            print(f"  [FAIL] Timeout (deneme {attempt}/{retries})")
            flush_robot(robot)
            time.sleep(1.0)

    return False


def move_with_fallback(robot, x, y, z, r,
                       offsets=[0.0, -0.1, 0.1, -0.2, 0.2]):
    for offset in offsets:
        ty = round(y + offset, 3)
        if offset != 0.0:
            print(f"  [RETRY] Y={y} → Y={ty} (offset={offset:+.1f})")
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
#  ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    save_dir = setup_output_dir(OUTPUT_DIR)

    # ── AD7747 başlat ─────────────────────────────────────────────────────────
    ad7747 = AD7747Reader(AD7747_PORT, AD7747_BAUD)
    ad7747_ok = ad7747.start()
    if not ad7747_ok:
        print("[UYARI] AD7747 bağlanamadı, kapasitans sütunu boş kalacak.")

    # ── Kamera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        ad7747.stop()
        return

    # ── Dobot ─────────────────────────────────────────────────────────────────
    print(f"Dobot'a bağlanılıyor: {DOBOT_PORT}")
    robot = Dobot(port=DOBOT_PORT, verbose=False)
    robot.speed(VELOCITY, ACCELERATION)
    print("Bağlantı başarılı.")

    print(f"Home pozisyonuna gidiliyor: X={HOME_X}, Y={HOME_Y}")
    if not move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R):
        cap.release()
        robot.close()
        ad7747.stop()
        raise RuntimeError("Home pozisyonuna ulaşılamadı. Tarama iptal edildi.")
    print("Home pozisyonuna ulaşıldı. Tarama başlıyor...\n")

    # ── Tarama noktaları ──────────────────────────────────────────────────────
    points = generate_scan_points(HOME_X, HOME_Y, SCAN_WIDTH, SCAN_HEIGHT, STEP_SIZE)
    total  = len(points)
    print(f"Toplam {total} nokta taranacak.")
    print(f"Alan: X {SCAN_WIDTH} mm x Y {SCAN_HEIGHT} mm, Adım: {STEP_SIZE} mm\n")

    log_path = os.path.join(save_dir, "scan_log.csv")
    img_index = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        # Başlık satırı — kapasitans sütunu eklendi
        log_file.write("index,timestamp,x,y,z,r,capacitance_pF,image_file,sharpness\n")

        for i, (x, y) in enumerate(points):

            if i > 0 and i % 50 == 0:
                clear_robot_queue(robot)

            print(f"[{i+1:04d}/{total}] Hareket: X={x:.2f}, Y={y:.2f}")

            moved, actual_y = move_with_fallback(robot, x, y, SCAN_Z, SCAN_R)
            time.sleep(CAPTURE_DELAY)

            if not moved:
                print("  NOKTA ATLANDI")
                log_file.write(f"FAIL,{datetime.now().isoformat()},{x},{y},{SCAN_Z},{SCAN_R},,SKIPPED,\n")
                continue

            img_index += 1

            # ── Kapasitans ölçümü ─────────────────────────────────────────
            cap_value = None
            if ad7747_ok:
                cap_value = ad7747.read_average(n=CAP_SAMPLES)
                if cap_value is not None:
                    print(f"  Kapasitans: {cap_value:.6f} pF")
                else:
                    print("  Kapasitans: ölçüm alınamadı")

            # ── Görüntü kaydet ────────────────────────────────────────────
            img_name = f"img_{img_index:04d}_x{x:.1f}_y{actual_y:.1f}.jpg"
            img_path = os.path.join(save_dir, img_name)

            # Netlik skoru için tekrar hesapla
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
                print(f"  Netlik: {sharpness:.1f}  {'[BULANIK]' if sharpness < 30 else '[OK]'}")
            else:
                print(f"  Görüntü alınamadı: {img_name}")

            # ── CSV satırı ────────────────────────────────────────────────
            cap_str = f"{cap_value:.6f}" if cap_value is not None else ""
            log_file.write(
                f"{img_index},"
                f"{datetime.now().isoformat()},"
                f"{x},{actual_y},{SCAN_Z},{SCAN_R},"
                f"{cap_str},"
                f"{img_name if img_status == 'OK' else 'FAIL'},"
                f"{sharpness:.1f}\n"
            )
            log_file.flush()  # her satırda diske yaz

    # ── Temizlik ──────────────────────────────────────────────────────────────
    cap.release()
    robot.close()
    ad7747.stop()

    print(f"\nTarama tamamlandı. {img_index} nokta işlendi.")
    print(f"Görüntüler : {save_dir}")
    print(f"Log dosyası: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHata: {e}")
        raise