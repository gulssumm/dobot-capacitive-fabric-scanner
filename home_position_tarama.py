"""
Dobot Magician - Raster Tarama + OBS Virtual Camera Görüntü Kayıt
==================================================================
Gereksinimler:
    pip install pydobot opencv-python

OBS Ayarı:
    OBS → Tools → Virtual Camera → Start Virtual Camera

Kullanım:
    python dobot_raster_scan.py
"""

import cv2
import time
import os
from datetime import datetime
from pydobot import Dobot

DOBOT_PORT = "COM3"

HOME_X = 215.0
HOME_Y = -39.0
HOME_Z = -118.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

# Tarama alanı (mm)
SCAN_WIDTH  = 80.0   # X ekseninde toplam genişlik (sağa doğru)
SCAN_HEIGHT = 80.0   # Y ekseninde toplam derinlik (aşağıya doğru, negatif Y)
STEP_SIZE   = 2.0    # Her satır arasındaki mesafe (mm)

# Hareket hızı ve ivmesi (1-100)
VELOCITY     = 50
ACCELERATION = 50

# Kamera (OBS Virtual Camera)
CAMERA_INDEX = 1    
CAPTURE_DELAY = 0.5   # Her noktada görüntü almadan önce bekleme süresi (saniye)

OUTPUT_DIR = "scan_images"


def setup_output_dir(base_dir: str) -> str:
    """Zaman damgalı alt klasör oluşturur."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"scan_{timestamp}")
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Görüntüler kaydedilecek: {path}")
    return path


def open_camera(index: int) -> cv2.VideoCapture:
    backends = [
        (cv2.CAP_MSMF,  "MSMF"),   # OBS Virtual Camera için en uyumlu
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_ANY,   "ANY"),
    ]

    for backend, name in backends:
        print(f"[INFO] Kamera {index} deneniyor (backend: {name})...")
        cap = cv2.VideoCapture(index, backend)
        if cap is not None and cap.isOpened():
            # Test frame — bazı backend'ler isOpened() True döner ama okuyamaz
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                print(f"[INFO] Kamera {index} açıldı (backend: {name}).")
                return cap
            else:
                cap.release()
                print(f"[WARN] Kamera {index} açıldı ama frame okunamadı (backend: {name}), sonraki deneniyor...")
        else:
            print(f"[WARN] Kamera {index} açılamadı (backend: {name}), sonraki deneniyor...")


def capture_image(cap: cv2.VideoCapture, save_path: str) -> bool:
    """Kameradan bir kare okur ve kaydeder."""
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[WARN] Görüntü alınamadı: {save_path}")
        return False
    cv2.imwrite(save_path, frame)
    return True


def get_current_position(robot: Dobot):
    """Returns current (x, y, z, r) or None on failure."""
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]  # x, y, z, r
    except Exception as e:
        print(f"[WARN] Pozisyon okunamadı: {e}")
        return None


def move_and_wait(robot: Dobot,
                  x: float, y: float, z: float, r: float,
                  retries: int = 3,
                  command_delay: float = 0.5,
                  settle_time: float = 0.3,
                  timeout: float = 8.0,
                  poll_interval: float = 0.4,
                  position_tolerance: float = 2.5) -> bool:

    start_pos = get_current_position(robot)
    if start_pos is None:
        print("[ERROR] Başlangıç pozisyonu okunamadı.")
        return False

    dist_to_target = ((start_pos[0]-x)**2 +
                      (start_pos[1]-y)**2 +
                      (start_pos[2]-z)**2) ** 0.5
    if dist_to_target <= position_tolerance:
        print(f"[OK] Zaten hedefe yakın (dist={dist_to_target:.2f} mm)")
        return True

    command_sent = False
    for attempt in range(1, retries + 1):
        try:
            robot.move_to(x, y, z, r, wait=False)
            command_sent = True
            time.sleep(command_delay)
            break
        except Exception as e:
            print(f"[WARN] move_to başarısız ({attempt}/{retries}): {e}")
            time.sleep(0.5)

    if not command_sent:
        print("[FAIL] Komut gönderilemedi.")
        return False
    
    time.sleep(command_delay)

    # Hedefe ulaşmayı bekle
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
            print(f"[OK] Hedefe ulaşıldı (dist={dist_target:.2f} mm)")
            time.sleep(settle_time)
            return True
    
    print(f"[FAIL] Timeout")
    return False

def generate_scan_points(home_x, home_y, width, height, step):
    """
    Koordinat:
        X+: ileri
        X-: geri
        Y+: sol
        Y-: sağ

    Başlangıç:
        Home = sağ üst

    Hareket:
        - İlk satır: sağ → sol (Y artar)
        - Sonraki satır: sol → sağ (Y azalır)
        - Her satır sonrası geri gider (X azalır)

    Alan:
        X: home_x → home_x - height
        Y: home_y → home_y + width
    """
    points = []
    x = home_x
    row = 0
 
    while round(x, 6) >= round(home_x - height, 6):
        if row % 2 == 0:
            # sağdan sola → Y artar
            y = home_y
            while round(y, 6) <= round(home_y + width, 6):
                points.append((round(x, 3), round(y, 3)))
                y = round(y + step, 6)
        else:
            # soldan sağa → Y azalır
            y = home_y + width
            while round(y, 6) >= round(home_y, 6):
                points.append((round(x, 3), round(y, 3)))
                y = round(y - step, 6)
 
        x = round(x - step, 6)   # geri git
        row += 1
 
    return points


def main():
    save_dir = setup_output_dir(OUTPUT_DIR)

    cap = open_camera(CAMERA_INDEX)

    print(f"[INFO] Dobot'a bağlanılıyor: {DOBOT_PORT}")
    robot = Dobot(port=DOBOT_PORT, verbose=False)
    robot.speed(VELOCITY, ACCELERATION)
    print("[INFO] Bağlantı başarılı.")

    print(f"[INFO] Home pozisyonuna gidiliyor: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}, R={HOME_R}")
    
    home_reached = move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R)
    
    if not home_reached:
        cap.release()
        robot.close()
        raise RuntimeError("Home pozisyonuna ulaşılamadı. Tarama iptal edildi.")
    
    print("[INFO] Home pozisyonuna ulaşıldı. Tarama başlıyor...\n")

    points = generate_scan_points(HOME_X, HOME_Y, SCAN_WIDTH, SCAN_HEIGHT, STEP_SIZE)
    # İlk nokta HOME olduğu için atla
    if points and points[0] == (round(HOME_X, 3), round(HOME_Y, 3)):
        points = points[1:]
    total = len(points)
    print(f"[INFO] Toplam {total} nokta taranacak.")
    print(f"[INFO] Alan: X(sola) {SCAN_WIDTH} mm x Y(aşağı) {SCAN_HEIGHT} mm, Adım: {STEP_SIZE} mm\n")

    log_path = os.path.join(save_dir, "scan_log.csv")
    img_index = 0 

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("index,x,y,z,r,image_file\n")

        for i, (x, y) in enumerate(points):  # ← with bloğunun İÇİNE alındı
            print(f"[{i+1:04d}/{total}] Hareket: X={x:.2f}, Y={y:.2f}")

            moved = move_and_wait(robot, x, y, SCAN_Z, SCAN_R)

            if not moved:
                print("         → NOKTA ATLADI (hareket başarısız)")
                continue

            time.sleep(CAPTURE_DELAY)

            img_index += 1

            img_name = f"img_{img_index:04d}_x{x:.1f}_y{y:.1f}.jpg"
            img_path = os.path.join(save_dir, img_name)

            success = capture_image(cap, img_path)

            status = "OK" if success else "FAIL"
            print(f"         → Görüntü: {img_name} [{status}]")

            log_file.write(f"{img_index},{x},{y},{SCAN_Z},{SCAN_R},{img_name}\n")

    cap.release()
    robot.close()
    print(f"\n[INFO] Tarama tamamlandı. {total} nokta tarandı.")
    print(f"[INFO] Görüntüler: {save_dir}")
    print(f"[INFO] Log dosyası: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise