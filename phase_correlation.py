"""
Phase Correlation — 2 goruntu arasindaki kaymayı olc
"""

from pathlib import Path
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

# ─── PARAMETRELER ─────────────────────────────────────────────────────────────
IMG_A = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260401_161126_eski\img_0001_x215.0_y-39.0.jpg")
IMG_B = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260401_161126_eski\img_0002_x215.0_y-37.0.jpg")
# ──────────────────────────────────────────────────────────────────────────────

a = cv2.imread(str(IMG_A))
b = cv2.imread(str(IMG_B))

if a is None or b is None:
    raise FileNotFoundError("Goruntu okunamadi, yolu kontrol et.")

ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)

shift, error, _ = phase_cross_correlation(ga, gb, upsample_factor=10)

dy = float(shift[0])  # dikey kayma (piksel)
dx = float(shift[1])  # yatay kayma (piksel)

img_h, img_w = a.shape[:2]
overlap_y = img_h - abs(dy)
overlap_x = img_w - abs(dx)

print(f"Goruntu boyutu : {img_w} x {img_h} px")
print(f"Kayma          : dy={dy:.2f}px  dx={dx:.2f}px")
print(f"Hata skoru     : {error:.4f}  (0=mukemmel, 1=cok kotu)")
print(f"Dikey overlap  : {overlap_y:.0f}px  ({overlap_y/img_h*100:.1f}%)")
print(f"Yatay overlap  : {overlap_x:.0f}px  ({overlap_x/img_w*100:.1f}%)")
print(f"\n→ OVERLAP_PX olarak kullan: {int(overlap_y)}")