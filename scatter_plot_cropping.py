import cv2
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
INPUT_PANORAMA = "C:/Users/21180/PycharmProjects/DOBOT_magician/steel_sheet_1x1cm/combined_panorama_scatter.jpg"
INPUT_CSV      = "C:/Users/21180/PycharmProjects/DOBOT_magician/steel_sheet_1x1cm/trajectory_data.csv"
OUTPUT_CROPPED = "panorama_cropped_scatter_steel.png"  # PNG for transparency support

df = pd.read_csv(INPUT_CSV)
pts = df[['x', 'y']].values.astype(np.float32)

hull = cv2.convexHull(pts)
hull_pts = hull.reshape(-1, 2).astype(np.int32)

img = cv2.imread(INPUT_PANORAMA)
print(f"Original size: {img.shape[1]} x {img.shape[0]} px")

# --- Alpha kanallı (BGRA) görüntü oluştur ---
img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

# --- Hull dışını şeffaf yap ---
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [hull_pts], 255)
img_bgra[:, :, 3] = mask  # Alpha kanalına maskeni ata

# --- Bounding box'a kırp ---
x, y, w, h = cv2.boundingRect(hull_pts)
cropped = img_bgra[y:y+h, x:x+w]

print(f"Cropped size: {w} x {h} px")
cv2.imwrite(OUTPUT_CROPPED, cropped)
print(f"Saved: {OUTPUT_CROPPED}")