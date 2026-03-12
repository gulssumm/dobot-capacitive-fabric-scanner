from pydobotplus import Dobot
from pydobotplus.dobotplus import MODE_PTP
import time

device = Dobot(port="COM3")
device.clear_alarms()
time.sleep(0.5)

device.speed(5, 5)

START_X = 293.3
START_Y = -1.2
Z       = -38.7
R       = -0.9

AREA_SIZE = 100   # mm (100x100 alan)
STEP = 1          # mm çözünürlük

TOTAL_ROWS = int(AREA_SIZE / STEP)
ROW_LENGTH = AREA_SIZE

def movej(x, y):
    device._set_ptp_cmd(x, y, Z, R, MODE_PTP.MOVJ_XYZ, wait=True)

print("Tarama başlangıç noktasına gidiliyor...")
movej(START_X, START_Y)

print("Tarama başlıyor...\n")

for row in range(TOTAL_ROWS + 1):

    y = START_Y + row * STEP

    direction = "→" if row % 2 == 0 else "←"
    print(f"Satır {row:3d}/{TOTAL_ROWS} {direction}  Y={y:.1f}")

    # snake hareketi
    if row % 2 == 0:
        target_x = START_X - ROW_LENGTH
    else:
        target_x = START_X

    movej(target_x, y)

    # bir sonraki satıra geç
    if row < TOTAL_ROWS:
        next_y = START_Y + (row + 1) * STEP
        movej(target_x, next_y)

print("\nTarama tamamlandı!")

movej(START_X, START_Y)