from pydobotplus import Dobot, CustomPosition
import time

def horizontal_move():
    device = Dobot(port="COM3")

    # Başlangıç ve bitiş koordinatları
    start_x, start_y = 247.7, 53.8
    end_x, end_y = 279.2, -66.2
    z, r = -134.8, 44.1

    step_mm = 1  # Her adım 1 mm
    steps = int(((end_x - start_x)**2 + (end_y - start_y)**2)**0.5 / step_mm)

    # Başlangıç noktasına git
    device.move_to(position=CustomPosition(start_x, start_y, z, r), wait=True)
    device.speed(5, 5)

    # X ve Y ekseninde lineer adımlar
    for i in range(steps + 1):
        x = start_x + (end_x - start_x) * i / steps
        y = start_y + (end_y - start_y) * i / steps
        print(f"X={x:.2f}, Y={y:.2f}")
        device.move_to(position=CustomPosition(x, y, z, r), wait=True)
        time.sleep(0.02)

horizontal_move()