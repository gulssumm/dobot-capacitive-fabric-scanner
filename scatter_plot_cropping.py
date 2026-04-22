import cv2
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
INPUT_PANORAMA = "C:/Users/Aidata/DOBOT/blue-striped-2x2cm/panorama_curved.jpg"
INPUT_CSV = "C:/Users/Aidata/DOBOT/blue-striped-2x2cm/trajectory_data.csv"
OUTPUT_CROPPED = "panorama_cropped.jpg"
PADDING = 50  # Kenarlara eklenecek piksel boşluğu

# --- CSV'den koordinatları oku ---
print("Reading trajectory data:", INPUT_CSV)
df = pd.read_csv(INPUT_CSV)

min_x = df['x'].min()
max_x = df['x'].max()
min_y = df['y'].min()
max_y = df['y'].max()

print(f"Scatter coordinate range:")
print(f"  X: {min_x:.1f} → {max_x:.1f}  (width: {max_x - min_x:.1f} px)")
print(f"  Y: {min_y:.1f} → {max_y:.1f}  (height: {max_y - min_y:.1f} px)")

# --- Panoramayı yükle ---
img = cv2.imread(INPUT_PANORAMA)
if img is None:
    print("ERROR: Panorama image not found!")
    exit()

img_h, img_w = img.shape[:2]
print(f"\nPanorama size: {img_w} x {img_h} px")

# --- Kırpma koordinatlarını hesapla (padding ekle, sınırlar içinde tut) ---
x1 = max(0,     int(np.floor(min_x)) - PADDING)
y1 = max(0,     int(np.floor(min_y)) - PADDING)
x2 = min(img_w, int(np.ceil(max_x))  + PADDING)
y2 = min(img_h, int(np.ceil(max_y))  + PADDING)

print(f"\nCrop region: ({x1}, {y1}) → ({x2}, {y2})")
print(f"Cropped size: {x2 - x1} x {y2 - y1} px")

# --- Kırp ve kaydet ---
cropped = img[y1:y2, x1:x2]
cv2.imwrite(OUTPUT_CROPPED, cropped)
print(f"\nCropped panorama saved: {OUTPUT_CROPPED}")