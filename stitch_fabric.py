"""
Yatay stitch — her geçişte phase correlation ile overlap hesapla
"""
from pathlib import Path
import cv2
import numpy as np
import re
from collections import defaultdict
from skimage.registration import phase_cross_correlation

INPUT_DIR = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski")
OUTPUT    = Path(r"C:\Users\Aidata\DOBOT\output\test_row.jpg")
EXT       = "jpg"
DEFAULT_OVERLAP = 480  # phase correlation başarısız olursa bu kullanılır

def parse_xy(path):
    m = re.search(r'_x(-?[\d.]+)_y(-?[\d.]+)', path.stem)
    if not m: return None, None
    return float(m.group(1)), float(m.group(2))

def calc_overlap(img_a, img_b):
    """İki görüntü arasındaki yatay overlap'i phase correlation ile hesapla."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shift, error, _ = phase_cross_correlation(ga, gb, upsample_factor=10)
    dx = float(shift[1])
    overlap = img_a.shape[1] - abs(dx)
    return int(round(overlap)), dx, error

# ─── Görüntüleri yükle ────────────────────────────────────────────────────────
files = sorted(INPUT_DIR.glob(f"*.{EXT}"), key=lambda p: p.stem)
by_x = defaultdict(list)
for f in files:
    x, y = parse_xy(f)
    if x is not None:
        by_x[x].append({"path": f, "x": x, "y": y})

first_x = sorted(by_x.keys())[0]
row = sorted(by_x[first_x], key=lambda d: d["y"])
print(f"İlk satır: x={first_x}, {len(row)} görüntü\n")

# ─── Her geçişte overlap hesapla, şeritleri oluştur ──────────────────────────
imgs = []
for meta in row:
    img = cv2.imread(str(meta["path"]))
    if img is None:
        print(f"  [UYARI] Okunamadi: {meta['path'].name}")
        imgs.append(None)
    else:
        imgs.append(img)

strips = []
for i, img in enumerate(imgs):
    if img is None:
        continue
    if i == 0:
        strips.append(img)
        print(f"  [1/{len(imgs)}] ilk görüntü tamamen alındı → {img.shape[1]}x{img.shape[0]}px")
        continue

    prev = imgs[i - 1]
    if prev is None:
        strips.append(img)
        continue

    overlap, dx, error = calc_overlap(prev, img)
    overlap = max(0, min(overlap, img.shape[1] - 1))  # sınır kontrolü
    new_px  = img.shape[1] - overlap

    print(f"  [{i+1}/{len(imgs)}] dx={dx:.1f}px  overlap={overlap}px  "
          f"yeni={new_px}px  hata={error:.3f}")

    if new_px <= 0:
        print(f"           [UYARI] Yeni alan yok, DEFAULT_OVERLAP={DEFAULT_OVERLAP} kullanılıyor")
        new_px  = img.shape[1] - DEFAULT_OVERLAP
        overlap = DEFAULT_OVERLAP

    strips.append(img[:, overlap:])  # yeni alanı ekle

# ─── Birleştir ────────────────────────────────────────────────────────────────
result = np.concatenate(strips, axis=1)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT), result, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nKaydedildi : {OUTPUT}")
print(f"Boyut      : {result.shape[1]}x{result.shape[0]}px")