"""
OBS Ayarı:
    OBS → Tools → Virtual Camera → Start Virtual Camera
"""

import cv2
import time
import os
from datetime import datetime
from pydobot import Dobot
import threading

DOBOT_PORT = "COM3"

HOME_X = 215.0
HOME_Y = -39.0
HOME_Z = -114.0
HOME_R = 0.01

SCAN_Z = HOME_Z
SCAN_R = HOME_R

# Tarama alanı (mm)
SCAN_WIDTH  = 10.0   # X ekseninde toplam genişlik (sağa doğru)
SCAN_HEIGHT = 10.0   # Y ekseninde toplam derinlik (aşağıya doğru, negatif Y)
STEP_SIZE   = 0.5    # Her satır arasındaki mesafe (mm)

# Hareket hızı ve ivmesi (1-100)
VELOCITY     = 20
ACCELERATION = 20

# Kamera (OBS Virtual Camera)
CAMERA_INDEX = 1    
CAPTURE_DELAY = 1.0   # Her noktada görüntü almadan önce bekleme süresi (saniye)

TYPE = "blue_striped" #blue-striped_01 #KAYNAK TEKSTIL
DESIGN = "01"  #0 #17S5292-05
COMPOSITION = "pin_mark" #unknown #%100 COTTON
OUTPUT_DIR = "scan_images" 

# Sabit liste yerine dinamik offset
PROBLEMATIC_Y = -29.5  # firmware'in reddettiği Y


def setup_output_dir(base_dir: str) -> str:
    folder_name = f"{TYPE}_{DESIGN}_{COMPOSITION}"
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    print(f"Görüntüler kaydedilecek: {path}")
    return path


def capture_image(cap: cv2.VideoCapture, save_path: str) -> bool:
    # Buffer'daki eski kareleri temizle
    for _ in range(5):
        cap.grab()

    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Görüntü alınamadı: {save_path}")
        return False

    # Netlik skoru (debug)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Netlik skoru: {score:.1f}  {'[BULANIK]' if score < 30 else '[OK]'}")

    cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def get_current_position(robot: Dobot):
    """Returns current (x, y, z, r) or None on failure."""
    try:
        pose = robot.pose()
        return pose[0], pose[1], pose[2], pose[3]  # x, y, z, r
    except Exception as e:
        print(f"Pozisyon okunamadı: {e}")
        return None

def move_with_fallback(robot, x, y, z, r, offsets=[0.0, -0.1, 0.1, -0.2, 0.2]):
    """Hedef koordinat firmware tarafından reddedilirse küçük offset'ler dene."""
    for offset in offsets:
        ty = round(y + offset, 3)
        if offset != 0.0:
            print(f"[RETRY] Y={y} → Y={ty} (offset={offset:+.1f})")
        
        success = move_and_wait(robot, x, ty, z, r)
        if success:
            return True, ty
    
    return False, y

def clear_robot_queue(robot):
    """Dobot'un iç komut kuyruğunu temizle."""
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
        clear_robot_queue(robot)   # ← ekle
        time.sleep(0.5)
    except Exception as e:
        print(f"Flush hatası: {e}")


def move_and_wait(robot, x, y, z, r,
                  retries=3,
                  settle_time=0.6,
                  timeout=15.0,
                  position_tolerance=0.3):

    # Zaten hedefe yakın mı?
    start_pos = get_current_position(robot)
    if start_pos:
        dist = ((start_pos[0]-x)**2 +
                (start_pos[1]-y)**2 +
                (start_pos[2]-z)**2) ** 0.5
        if dist <= position_tolerance:
            print(f"[OK] Zaten hedefe yakın (dist={dist:.2f} mm)")
            return True

    for attempt in range(1, retries + 1):
        result = [False]
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
            print(f"[OK] Hedefe ulaşıldı (deneme {attempt})")
            time.sleep(settle_time)
            return True
        else:
            stop_flag[0] = True  # thread'e "artık önemli değil" sinyali
            print(f"[FAIL] Timeout (deneme {attempt}/{retries})")
            flush_robot(robot)   # ← serial buffer temizle
            time.sleep(1.0)

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

    # Kamerayı başlat (0 varsayılan kameradır)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        exit()

    print(f"Dobot'a bağlanılıyor: {DOBOT_PORT}")
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
    total = len(points)
    print(f"Toplam {total} nokta taranacak.")
    print(f"Alan: X(sola) {SCAN_WIDTH} mm x Y(aşağı) {SCAN_HEIGHT} mm, Adım: {STEP_SIZE} mm\n")

    log_path = os.path.join(save_dir, "scan_log.csv")
    img_index = 0 

    

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("index,x,y,z,r,image_file\n")

        for i, (x, y) in enumerate(points):  
            # Her 50 noktada kuyruğu temizle
            if i > 0 and i % 50 == 0:
                clear_robot_queue(robot)

            print(f"[{i+1:04d}/{total}] Hareket: X={x:.2f}, Y={y:.2f}")

            moved, actual_y = move_with_fallback(robot, x, y, SCAN_Z, SCAN_R)
            time.sleep(CAPTURE_DELAY)

            if not moved:
                print("NOKTA ATLADI")
                log_file.write(f"FAIL,{x},{y},{SCAN_Z},{SCAN_R},SKIPPED\n")
                continue

            img_index += 1

            img_name = f"img_{img_index:04d}_x{x:.1f}_y{y:.1f}.jpg"
            img_path = os.path.join(save_dir, img_name)
        
            
            success = capture_image(cap, img_path)

            status = "OK" if success else "FAIL"
            print(f"Görüntü: {img_name} [{status}]")

            log_file.write(f"{img_index},{x},{actual_y},{SCAN_Z},{SCAN_R},{img_name}\n")

    cap.release()
    robot.close()
    print(f"\nTarama tamamlandı. {total} nokta tarandı.")
    print(f"Görüntüler: {save_dir}")
    print(f"Log dosyası: {log_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"\n{e}")
        raise