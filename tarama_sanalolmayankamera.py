"""
Dobot + Direkt Kamera Tarama Sistemi
OBS Virtual Camera'ya gerek yok — fiziksel kamera doğrudan kullanılıyor.
"""

import cv2
import time
import os
from pydobot import Dobot

DOBOT_PORT = "COM3"

HOME_X = 215.0
HOME_Y = -39.0
HOME_Z = -118.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

SCAN_WIDTH  = 40.0
SCAN_HEIGHT = 40.0
STEP_SIZE   = 5.0

VELOCITY     = 50
ACCELERATION = 50

# Kamera ayarları — fiziksel kamera indeksi (genellikle 0=dahili, 1=harici USB)
CAMERA_INDEX = 0
CAPTURE_DELAY = 1.0

# Kamera görüntü ayarları
CAM_WIDTH      = 640   # İstenen çözünürlük (kamera desteklemiyorsa düşürür)
CAM_HEIGHT     = 480
CAM_AUTOFOCUS  = 1      # 0=Manuel, 1=Otomatik
CAM_FOCUS      = 50     # Manuel odak değeri (0-255, kamerana göre ayarla)
CAM_AUTO_EXP   = 1      # 1=Manuel pozlama (Windows'ta 0.25 yerine 1 kullan)
CAM_EXPOSURE   = -6     # Manuel pozlama değeri (kamerana göre ayarla)
CAM_AUTO_WB    = 0      # 0=Manuel beyaz denge
CAM_WB_TEMP    = 4500   # Beyaz denge sıcaklığı (Kelvin)

TYPE        = "14307"
DESIGN      = "11"
COMPOSITION = "%100COTTON"
OUTPUT_DIR  = "scan_images"


def setup_output_dir(base_dir: str) -> str:
    folder_name = f"{TYPE}_{DESIGN}_{COMPOSITION}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"Görüntüler kaydedilecek: {path}")
    return path


import cv2

def open_camera_by_name(name_keyword: str) -> cv2.VideoCapture:
    """Kamerayı indeks yerine ismiyle açar — daha güvenilir."""
    import subprocess, re

    # DirectShow cihaz listesini ffmpeg ile al
    result = subprocess.run(
        ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True, text=True
    )
    output = result.stderr

    print("Bulunan DirectShow cihazları:")
    devices = []
    for line in output.splitlines():
        if '"' in line and "video" in line.lower():
            match = re.search(r'"([^"]+)"', line)
            if match:
                devices.append(match.group(1))
                print(f"  [{len(devices)-1}] {match.group(1)}")

    # İsme göre eşleştir
    target = None
    for d in devices:
        if name_keyword.lower() in d.lower():
            target = d
            break

    if target is None:
        print(f"'{name_keyword}' bulunamadı. Bulunan cihazlar: {devices}")
        return None

    print(f"\nAçılıyor: {target}")
    cap = cv2.VideoCapture(f"video={target}", cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Kamera açılamadı.")
        return None

    ret, frame = cap.read()
    if ret:
        cv2.imwrite("test_isimle.jpg", frame)
        print("test_isimle.jpg kaydedildi — kontrol et.")
    else:
        print("Frame alınamadı.")

    return cap

cap = open_camera_by_name("USB2.0 Digital Camera")
if cap:
    cap.release()

def print_camera_properties(cap: cv2.VideoCapture):
    """Mevcut kamera ayarlarını ekrana yazdırır (debug için)."""
    props = {
        "Autofocus":    cv2.CAP_PROP_AUTOFOCUS,
        "Focus":        cv2.CAP_PROP_FOCUS,
        "Auto Exposure":cv2.CAP_PROP_AUTO_EXPOSURE,
        "Exposure":     cv2.CAP_PROP_EXPOSURE,
        "Auto WB":      cv2.CAP_PROP_AUTO_WB,
        "WB Temp":      cv2.CAP_PROP_WB_TEMPERATURE,
        "Width":        cv2.CAP_PROP_FRAME_WIDTH,
        "Height":       cv2.CAP_PROP_FRAME_HEIGHT,
    }
    print("\n--- Kamera Mevcut Ayarları ---")
    for label, prop_id in props.items():
        val = cap.get(prop_id)
        print(f"  {label:15s}: {val}")
    print("------------------------------\n")


def capture_image(cap: cv2.VideoCapture, save_path: str) -> bool:
    # Buffer'daki eski kareleri temizle (en az 3, tercihen 5)
    for _ in range(5):
        cap.grab()

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  Görüntü alınamadı: {save_path}")
        return False

    # Netlik skoru (Laplacian varyansı — düşükse bulanık)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_tag = "[BULANIK]" if score < 30 else "[OK]"
    print(f"  Netlik skoru: {score:.1f}  {blur_tag}")

    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def get_current_position(robot: Dobot):
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]
    except Exception as e:
        print(f"Pozisyon okunamadı: {e}")
        return None


def move_and_wait(robot, x, y, z, r,
                  retries=3,
                  command_delay=0.5,
                  settle_time=0.5,
                  timeout=12.0,
                  poll_interval=0.3,
                  position_tolerance=3.5):

    start_pos = get_current_position(robot)
    if start_pos is None:
        print("Başlangıç pozisyonu okunamadı.")
        return False

    dist_to_target = ((start_pos[0]-x)**2 +
                      (start_pos[1]-y)**2 +
                      (start_pos[2]-z)**2) ** 0.5
    if dist_to_target <= position_tolerance:
        print(f"  [OK] Zaten hedefe yakın (dist={dist_to_target:.2f} mm)")
        return True

    command_sent = False
    for attempt in range(1, retries + 1):
        try:
            robot.move_to(x, y, z, r, wait=False)
            command_sent = True
            time.sleep(command_delay)
            break
        except Exception as e:
            print(f"  move_to başarısız ({attempt}/{retries}): {e}")
            time.sleep(0.5)

    if not command_sent:
        print("  Komut gönderilemedi.")
        return False

    elapsed = 0.0
    while elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

        pos = get_current_position(robot)
        if pos is None:
            continue

        dist_target = ((pos[0]-x)**2 +
                       (pos[1]-y)**2 +
                       (pos[2]-z)**2) ** 0.5

        if dist_target <= position_tolerance:
            print(f"  [OK] Hedefe ulaşıldı (dist={dist_target:.2f} mm)")
            time.sleep(settle_time)
            return True

    print(f"  [FAIL] Timeout — hedefe ulaşılamadı")
    return False


def generate_scan_points(home_x, home_y, width, height, step):
    points = []
    x = home_x
    row = 0

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

        x = round(x - step, 6)
        row += 1

    return points


def main():
    save_dir = setup_output_dir(OUTPUT_DIR)

    cap = open_camera_by_name(CAMERA_INDEX)
    print_camera_properties(cap)   # Ayarları doğrulamak için logla

    print(f"\nDobot'a bağlanılıyor: {DOBOT_PORT}")
    robot = Dobot(port=DOBOT_PORT, verbose=False)
    robot.speed(VELOCITY, ACCELERATION)
    print("Bağlantı başarılı.")

    print(f"Home pozisyonuna gidiliyor: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}, R={HOME_R}")
    home_reached = move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R)

    if not home_reached:
        cap.release()
        robot.close()
        raise RuntimeError("Home pozisyonuna ulaşılamadı. Tarama iptal edildi.")

    print("Home pozisyonuna ulaşıldı. Tarama başlıyor...\n")

    points = generate_scan_points(HOME_X, HOME_Y, SCAN_WIDTH, SCAN_HEIGHT, STEP_SIZE)
    total  = len(points)
    print(f"Toplam {total} nokta taranacak.")
    print(f"Alan: X {SCAN_HEIGHT} mm x Y {SCAN_WIDTH} mm, Adım: {STEP_SIZE} mm\n")

    log_path  = os.path.join(save_dir, "scan_log.csv")
    img_index = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("index,x,y,z,r,image_file\n")

        for i, (x, y) in enumerate(points):
            print(f"[{i+1:04d}/{total}] Hareket: X={x:.2f}, Y={y:.2f}")

            moved = move_and_wait(robot, x, y, SCAN_Z, SCAN_R)

            if not moved:
                print("  NOKTA ATLANDI (hareket başarısız)")
                continue

            time.sleep(CAPTURE_DELAY)

            img_index += 1
            img_name  = f"img_{img_index:04d}_x{x:.1f}_y{y:.1f}.jpg"
            img_path  = os.path.join(save_dir, img_name)

            success = capture_image(cap, img_path)
            status  = "OK" if success else "FAIL"
            print(f"  Görüntü: {img_name} [{status}]")

            log_file.write(f"{img_index},{x},{y},{SCAN_Z},{SCAN_R},{img_name}\n")

    cap.release()
    robot.close()
    print(f"\nTarama tamamlandı. {img_index} görüntü kaydedildi.")
    print(f"Görüntüler: {save_dir}")
    print(f"Log dosyası: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\nHata: {e}")
        raise