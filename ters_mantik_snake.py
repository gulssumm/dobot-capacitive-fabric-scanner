from pydobotplus import Dobot
from pydobotplus.dobotplus import MODE_PTP
import time

device = Dobot(port="COM3")
device.clear_alarms()
time.sleep(0.5)

device.speed(5, 5)

START_X = 193.3  # soldan başlatıyoruz
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

    # Satır yönü: çift satır → sağa, tek satır → sola
    if row % 2 == 0:
        # soldan sağa
        x_start = START_X
        x_end = START_X + ROW_LENGTH
    else:
        # sağdan sola
        x_start = START_X + ROW_LENGTH
        x_end = START_X

    # satır boyunca X eksenini adım adım hareket ettiriyoruz
    steps = int(ROW_LENGTH / STEP)
    for i in range(steps + 1):
        if row % 2 == 0:
            x = x_start + i * STEP
        else:
            x = x_start - i * STEP
        movej(x, y)

print("\nTarama tamamlandı!")
movej(START_X, START_Y)