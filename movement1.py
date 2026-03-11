from pydobotplus import Dobot
from pydobotplus.dobotplus import MODE_PTP
import time

device = Dobot(port="COM3")
device.clear_alarms()
time.sleep(0.5)
device.speed(5, 5)

START_X = 258.0
START_Y = 58.2
Z       = -128.8
R       = 25.7

ROW_LENGTH = 100  # mm
TOTAL_ROWS = 100
STEP       = 1    # mm

def movej(x, y):
    device._set_ptp_cmd(x, y, Z, R, MODE_PTP.MOVJ_XYZ, wait=False)
    
    time.sleep(0.3)
    
    timeout = time.time() + 30
    while time.time() < timeout:
        pose = device.get_pose().position
        if abs(pose.x - x) < 0.5 and abs(pose.y - y) < 0.5:
            return
        time.sleep(0.2)
    print(f"Hedefe ulaşılamadı! Hedef=({x:.1f},{y:.1f}) Gerçek=({pose.x:.1f},{pose.y:.1f})")

print("Tarama başlıyor...")

for row in range(TOTAL_ROWS + 1):
    # Tamsayı çarpımı ile float birikimini önle
    y = START_Y + row * STEP
    

    direction = "→" if row % 2 == 0 else "←"
    print(f"Satır {row:3d}/{TOTAL_ROWS} {direction}  Y={y:.1f}")

    if row % 2 == 0:
        movej(START_X + ROW_LENGTH, y)
    else:
        movej(START_X, y)

    if row < TOTAL_ROWS:
        next_y = START_Y + (row + 1) * STEP
        cur_x  = START_X + ROW_LENGTH if row % 2 == 0 else START_X
        movej(cur_x, next_y)

print("\nTarama tamamlandı!")
movej(START_X, START_Y)