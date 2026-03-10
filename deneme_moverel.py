from pydobotplus import Dobot
import inspect, time

device = Dobot(port="COM3")
device.clear_alarms()

# _send_message'i monkey-patch ile izle
original_send = device._send_message

def patched_send(msg):
    print(f"GÖNDERILEN: id={msg.id}, ctrl={msg.ctrl}, params={msg.params.hex()}")
    return original_send(msg)

device._send_message = patched_send

device.speed(5, 5)
time.sleep(0.5)

print("X=270'e hareket komutu gönderiliyor...")
device.move_to(x=270, y=58.2, z=-128.8, r=25.7, wait=True)