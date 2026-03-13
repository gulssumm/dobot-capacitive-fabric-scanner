"""
Dobot Magician - Snake Tarama + OBS Sanal Kamera
=================================================

KURULUM:
    pip install pydobotplus opencv-python

KULLANIM:
    Kamera indeksini bulmak icin: python dobot_snake_scan.py --find-camera
    Normal calistirmak icin:      python dobot_snake_scan.py
"""

import time
import os
import sys
import cv2
from serial.tools import list_ports

try:
    from pydobotplus import Dobot
except ImportError:
    Dobot = None
    print("[UYARI] pydobotplus bulunamadi - simulasyon modunda calisiyor.")
    print("        Kurmak icin: pip install pydobotplus\n")


# ==================================================================
# AYARLAR
# ==================================================================
DOBOT_PORT    = "COM3"   # Cikti COM3'u gosterdi, elle belirttik

CAMERA_INDEX  = 1        # OBS Virtual Camera indeksi
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480

OUTPUT_DIR    = "kareler"

# Robot hareket edip tam durduktan sonra kac ms bekleyip ceksin
SETTLE_MS     = 300
# ==================================================================

# ------------------------------------------------------------------
# Tarama alani
# ------------------------------------------------------------------
X_START     = 233.70
X_END       = 143.50

Y_LEFT_TOP  =  22.12
Y_RIGHT_TOP = -40.75
Y_LEFT_BOT  =  16.87
Y_RIGHT_BOT = -40.70

Z    = -101.07
R    = -0.90

STEP_X = 5.0
STEP_Y = 5.0

VELOCITY     = 50
ACCELERATION = 50


# ------------------------------------------------------------------
# Nokta listesi
# ------------------------------------------------------------------
def interpolate_y(x, x_top, x_bot, y_top, y_bot):
    if x_top == x_bot:
        return y_top
    t = (x - x_top) / (x_bot - x_top)
    return y_top + t * (y_bot - y_top)


def generate_scan_points():
    points = []
    row = 0
    x = X_START
    while x >= X_END - 1e-6:
        y_left  = interpolate_y(x, X_START, X_END, Y_LEFT_TOP,  Y_LEFT_BOT)
        y_right = interpolate_y(x, X_START, X_END, Y_RIGHT_TOP, Y_RIGHT_BOT)
        if row % 2 == 0:
            y = y_left
            while y >= y_right - 1e-6:
                points.append((round(x, 3), round(y, 3), Z, R))
                y -= STEP_Y
        else:
            y = y_right
            while y <= y_left + 1e-6:
                points.append((round(x, 3), round(y, 3), Z, R))
                y += STEP_Y
        x   -= STEP_X
        row += 1
    return points


# ------------------------------------------------------------------
# Kamera
# ------------------------------------------------------------------
def find_cameras():
    print("\nKameralar taranıyor...\n")
    print(f"{'Index':<8} Durum")
    print("-" * 35)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            ok = ret and frame is not None and frame.sum() > 0
            print(f"{i:<8} {'OK - goruntu var' if ok else 'Acik ama goruntu yok'}")
        cap.release()
    print("\nOBS Virtual Camera'nin indeksini CAMERA_INDEX'e yaz.")


def connect_camera():
    # CAP_MSMF uyarisi sorun degil, yine de deniyor
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[KAMERA HATA] Indeks {CAMERA_INDEX} acilamadi. --find-camera ile kontrol edin.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Ilk birkac kare genelde bos gelir, isitiyoruz
    for _ in range(5):
        cap.read()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[KAMERA] Baglanti basarili -> indeks {CAMERA_INDEX}  ({w}x{h})")
    return cap


def capture_image(cap, index, x, y):
    """Robota hareket komutu verildikten ve wait=True ile durulduktan
       sonra cagrilir. SETTLE_MS kadar bekler, sonra TEK kare alir."""
    filename = os.path.join(OUTPUT_DIR, f"frame_{index:04d}.png")

    if cap is None:
        print(f"  [SIMULASYON] Kare #{index:04d}  X={x:8.3f}  Y={y:8.3f}")
        return

    # Robotun tamamen durmasini bekle
    if SETTLE_MS > 0:
        time.sleep(SETTLE_MS / 1000.0)

    # Tampon karelerini temizle (eski kareler birikmis olabilir)
    for _ in range(2):
        cap.grab()

    # Tek kare al
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"  [KAMERA HATA] Kare #{index:04d} okunamadi!")
        return

    cv2.imwrite(filename, frame)
    print(f"  Kare #{index:04d}  X={x:8.3f}  Y={y:8.3f}  -> {filename}")


# ------------------------------------------------------------------
# Dobot
# ------------------------------------------------------------------
def connect_dobot():
    if Dobot is None:
        return None

    port = DOBOT_PORT
    if port is None:
        ports = [p.device for p in list_ports.comports()]
        if not ports:
            print("[DOBOT HATA] Seri port bulunamadi.")
            return None
        port = ports[0]
        print(f"[DOBOT] Otomatik port: {port}")
        if len(ports) > 1:
            print(f"        Diger portlar: {ports[1:]}  -> yanlis secildiyse DOBOT_PORT'u elle yaz")

    try:
        device = Dobot(port=port)          # verbose parametresi kaldirildi
        device.speed(VELOCITY, ACCELERATION)
        print(f"[DOBOT] Baglanti basarili -> {port}")
        return device
    except Exception as e:
        print(f"[DOBOT HATA] {e}")
        return None


def move_to(device, x, y, z, r):
    if device is None:
        print(f"  [SIMULASYON] X={x:8.3f}  Y={y:8.3f}  Z={z}  R={r}")
        return
    # wait=True: komut bitmeden bir sonraki satira gecmez
    # Bu sayede capture_image her zaman robot durduktan sonra cagrilir
    device.move_to(x=x, y=y, z=z, r=r, wait=True)


# ------------------------------------------------------------------
# Ana akis
# ------------------------------------------------------------------
def main():
    if "--find-camera" in sys.argv:
        find_cameras()
        return

    print("=" * 60)
    print("  Dobot Magician - Snake Tarama + OBS Sanal Kamera")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    points = generate_scan_points()
    print(f"Toplam nokta  : {len(points)}")
    print(f"Ilk nokta     : {points[0]}")
    print(f"Son nokta     : {points[-1]}")
    print(f"Cikti klasoru : {os.path.abspath(OUTPUT_DIR)}")
    print("-" * 60)

    device = connect_dobot()
    cap    = connect_camera()

    # Tarama baslmadan once robotu ilk konuma gotur
    # Oraya varmadan hic goruntu alinmaz
    start_x, start_y, start_z, start_r = points[0]
    print(f"[HAZIRLIK] Baslangic konumuna gidiliyor: X={start_x}  Y={start_y}  Z={start_z}")
    move_to(device, start_x, start_y, start_z, start_r)
    print("[HAZIRLIK] Baslangic konumuna ulasildi. Tarama basliyor...\n")

    try:
        for i, (x, y, z, r) in enumerate(points):
            # 1) Robotu hareket ettir (wait=True ile tamamen dur)
            move_to(device, x, y, z, r)
            # 2) Durdu, simdi tek kare cek
            capture_image(cap, i + 1, x, y)

    except KeyboardInterrupt:
        print("\n[!] Kullanici tarafindan durduruldu.")
    finally:
        if device:
            device.close()
            print("[DOBOT] Kapatildi.")
        if cap:
            cap.release()
            print("[KAMERA] Kapatildi.")

    print("-" * 60)
    print(f"Tamamlandi. Kareler: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()