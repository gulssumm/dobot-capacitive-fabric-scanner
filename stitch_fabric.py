"""
Yatay stitch test — ilk satır (41 görüntü)
"""
from pathlib import Path
import cv2
import numpy as np

INPUT_DIR = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski")
OUTPUT    = Path(r"C:\Users\Aidata\DOBOT\output\test_row.jpg")
EXT       = "jpg"
OVERLAP_H = 480   # yatay overlap (px)

# Aynı X değerindeki ilk satırı al
import re
from collections import defaultdict

def parse_xy(path):
    m = re.search(r'_x(-?[\d.]+)_y(-?[\d.]+)', path.stem)
    if not m: return None, None
    return float(m.group(1)), float(m.group(2))

files = sorted(INPUT_DIR.glob(f"*.{EXT}"), key=lambda p: p.stem)
by_x = defaultdict(list)
for f in files:
    x, y = parse_xy(f)
    if x is not None:
        by_x[x].append({"path": f, "x": x, "y": y})

# İlk X değeri = ilk satır
first_x = sorted(by_x.keys())[0]
row = sorted(by_x[first_x], key=lambda d: d["y"])
print(f"İlk satır: x={first_x}, {len(row)} görüntü")

# Yatay birleştir — her görüntüden sadece yeni 160px alanı ekle
result = None
for i, meta in enumerate(row):
    img = cv2.imread(str(meta["path"]))
    if img is None: continue
    if result is None:
        result = img
    else:
        # Sadece yeni kısmı ekle (overlap sonrası)
        new_strip = img[:, OVERLAP_H:]   # sağdan 160px
        result = np.concatenate([result, new_strip], axis=1)
    if i % 10 == 0:
        print(f"  [{i+1}/{len(row)}]")

cv2.imwrite(str(OUTPUT), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nKaydedildi: {OUTPUT}")
print(f"Boyut: {result.shape[1]}x{result.shape[0]}px")