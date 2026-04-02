"""
İki görüntü arası ortak alan tespiti ve crop
"""
from pathlib import Path
import cv2
import numpy as np

IMG_A = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0001_....jpg")
IMG_B = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0002_....jpg")
OUTPUT_DIR = Path(r"C:\Users\Aidata\DOBOT\output\crop_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

a = cv2.imread(str(IMG_A))
b = cv2.imread(str(IMG_B))

# Farklı overlap değerlerini dene, her biri için karşılaştırma görseli kaydet
for overlap in range(100, 641, 20):
    new_px = 640 - overlap
    if new_px <= 0:
        break

    # A'nın sağ tarafı ile B'nin sol tarafı — bunlar ortak alan
    a_right = a[:, -overlap:]   # A'nın sağdan overlap kadarı
    b_left  = b[:, :overlap]    # B'nin soldan overlap kadarı

    # İkisini üst üste yüzde 50 blend et — ne kadar örtüştüğünü görmek için
    blended = ((a_right.astype(np.float32) * 0.5) +
               (b_left.astype(np.float32)  * 0.5)).astype(np.uint8)

    # A sağ | blend | B sol — yan yana koy
    compare = np.concatenate([a_right, blended, b_left], axis=1)

    out = OUTPUT_DIR / f"overlap_{overlap:03d}px.jpg"
    cv2.imwrite(str(out), compare, [cv2.IMWRITE_JPEG_QUALITY, 85])

print(f"Kaydedildi: {OUTPUT_DIR}")
print("overlap_XXXpx.jpg dosyalarına bak — blend en keskin görünen değer doğru overlap'tir")