import subprocess
import cv2
import os

sonuc = subprocess.run([
    'ffmpeg',
    '-f', 'dshow',
    '-i', 'video=USB2.0 Digital Camera',
    '-vframes', '1',
    '-y',
    'test_ffmpeg.jpg'
], capture_output=True, text=True)

print(sonuc.stderr[-500:])  # ffmpeg çıktısı

if os.path.exists('test_ffmpeg.jpg'):
    frame = cv2.imread('test_ffmpeg.jpg')
    print("Ortalama:", frame.mean())
    print("Kaydedildi: test_ffmpeg.jpg")
else:
    print("Dosya oluşturulamadı")