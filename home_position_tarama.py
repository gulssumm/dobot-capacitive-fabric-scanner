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

# ─────────────────────────────────────────────
#  AYARLAR — buradan değiştirin
# ─────────────────────────────────────────────

# Dobot seri port (Windows: "COM3", Linux: "/dev/ttyUSB0")
DOBOT_PORT = "COM3"

# Home (başlangıç) pozisyonu — tarama buradan başlar
HOME_X = 199.0
HOME_Y = 0.05
HOME_Z = -110.0
HOME_R = 0.01

# Sabit Z ve R değerleri (tarama boyunca sabit kalır)
SCAN_Z = HOME_Z
SCAN_R = HOME_R

# Tarama alanı (mm)
SCAN_WIDTH  = 100.0   # X ekseninde toplam genişlik (sağa doğru)
SCAN_HEIGHT = 100.0   # Y ekseninde toplam derinlik (aşağıya doğru, negatif Y)
STEP_SIZE   = 2.0    # Her satır arasındaki mesafe (mm)

# Hareket hızı ve ivmesi (1-100)
VELOCITY     = 50
ACCELERATION = 50

# Kamera (OBS Virtual Camera)
CAMERA_INDEX = 1      # Birden fazla kamera varsa 1, 2 ... deneyin
CAPTURE_DELAY = 0.5   # Her noktada görüntü almadan önce bekleme süresi (saniye)

# Görüntü kayıt klasörü
OUTPUT_DIR = "scan_images"

# ─────────────────────────────────────────────
#  YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────

def setup_output_dir(base_dir: str) -> str:
    """Zaman damgalı alt klasör oluşturur."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"scan_{timestamp}")
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Görüntüler kaydedilecek: {path}")
    return path


def open_camera(index: int) -> cv2.VideoCapture:
    """
    OBS Virtual Camera'yı açar.
    Sırasıyla farklı backend'leri dener: MSMF → DSHOW → varsayılan.
    """
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

    # Hiçbir backend çalışmadı — mevcut kamera index'lerini listele
    print("\n[INFO] Mevcut kameralar taranıyor...")
    for i in range(5):
        test = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if test is not None and test.isOpened():
            print(f"  → Kamera index {i} mevcut")
            test.release()

    raise RuntimeError(
        f"Kamera {index} hiçbir backend ile açılamadı.\n"
        "Yukarıdaki mevcut kamera index'lerinden birini CAMERA_INDEX olarak ayarlayın.\n"
        "OBS'de Virtual Camera'nın başlatıldığından emin olun."
    )


def capture_image(cap: cv2.VideoCapture, save_path: str) -> bool:
    """Kameradan bir kare okur ve kaydeder."""
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"[WARN] Görüntü alınamadı: {save_path}")
        return False
    cv2.imwrite(save_path, frame)
    return True


def move_and_wait(robot: Dobot, x: float, y: float, z: float, r: float,
                  retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Robotu belirtilen konuma taşır.
    Limit hatası veya başka hata olursa False döner, program devam eder.
    """
    for attempt in range(1, retries + 1):
        try:
            robot.move_to(x, y, z, r, wait=True)
            return True
        except AttributeError as e:
            if attempt < retries:
                print(f"[WARN] move_to hatası (deneme {attempt}/{retries}): {e} — tekrar deneniyor...")
                time.sleep(retry_delay)
            else:
                print(f"[WARN] move_to {retries} denemede başarısız, wait=False ile devam.")
                try:
                    robot.move_to(x, y, z, r, wait=False)
                    time.sleep(1.5)
                    return True
                except Exception as e2:
                    print(f"[SKIP] Nokta atlanıyor X={x}, Y={y}: {e2}")
                    return False
        except Exception as e:
            print(f"[SKIP] Limit veya bilinmeyen hata, nokta atlanıyor X={x}, Y={y}: {e}")
            return False
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
        - Her satır sonrası ileri gider (X artar)

    Alan:
        X: home_x → home_x + height
        Y: home_y → home_y + width
    """

    points = []
    x = home_x
    row = 0

    while x <= home_x + height:
        if row % 2 == 0:
            # sağdan sola → Y artar
            y = home_y
            while y <= home_y + width:
                points.append((round(x, 3), round(y, 3)))
                y += step
        else:
            # soldan sağa → Y azalır
            y = home_y + width
            while y >= home_y:
                points.append((round(x, 3), round(y, 3)))
                y -= step

        x += step   # ileri git
        row += 1

    return points

# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────

def main():
    # --- Çıktı klasörü ---
    save_dir = setup_output_dir(OUTPUT_DIR)

    # --- Kamerayı aç ---
    cap = open_camera(CAMERA_INDEX)

    # --- Robotu bağla ---
    print(f"[INFO] Dobot'a bağlanılıyor: {DOBOT_PORT}")
    robot = Dobot(port=DOBOT_PORT, verbose=False)
    robot.speed(VELOCITY, ACCELERATION)
    print("[INFO] Bağlantı başarılı.")

    # --- Home pozisyonuna git (tarama buradan başlar) ---
    print(f"[INFO] Home pozisyonuna gidiliyor: X={HOME_X}, Y={HOME_Y}, Z={HOME_Z}, R={HOME_R}")
    move_and_wait(robot, HOME_X, HOME_Y, HOME_Z, HOME_R)
    print("[INFO] Home pozisyonuna ulaşıldı. Tarama başlıyor...\n")

    # --- Tarama noktalarını hesapla (home'dan başlar) ---
    points = generate_scan_points(HOME_X, HOME_Y, SCAN_WIDTH, SCAN_HEIGHT, STEP_SIZE)
    total = len(points)
    print(f"[INFO] Toplam {total} nokta taranacak.")
    print(f"[INFO] Alan: X(sola) {SCAN_WIDTH} mm x Y(aşağı) {SCAN_HEIGHT} mm, Adım: {STEP_SIZE} mm\n")

    # --- Tarama döngüsü ---
    log_path = os.path.join(save_dir, "scan_log.csv")
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("index,x,y,z,r,image_file\n")

        for i, (x, y) in enumerate(points):
            print(f"[{i+1:04d}/{total}] Hareket: X={x:.2f}, Y={y:.2f}")

            # Robotu taşı
            move_and_wait(robot, x, y, SCAN_Z, SCAN_R)

            # Yerleşme süresi
            time.sleep(CAPTURE_DELAY)

            # Görüntü kaydet
            img_name = f"img_{i+1:04d}_x{x:.1f}_y{y:.1f}.jpg"
            img_path = os.path.join(save_dir, img_name)
            success = capture_image(cap, img_path)

            status = "OK" if success else "FAIL"
            print(f"         → Görüntü: {img_name} [{status}]")

            # Log yaz
            log_file.write(f"{i+1},{x},{y},{SCAN_Z},{SCAN_R},{img_name}\n")

    # --- Temizlik ---
    cap.release()
    robot.close()
    print(f"\n[INFO] Tarama tamamlandı. {total} nokta tarandı.")
    print(f"[INFO] Görüntüler: {save_dir}")
    print(f"[INFO] Log dosyası: {log_path}")


# ─────────────────────────────────────────────
#  ÇALIŞTIR
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise