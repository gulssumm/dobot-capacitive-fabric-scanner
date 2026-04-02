"""
Ortak alanı kes — A tamamen al, B'den sadece yeni 20px şerit al
"""
from pathlib import Path
import cv2
import numpy as np

IMG_A = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0001_....jpg")
IMG_B = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0002_....jpg")
OUTPUT_DIR = Path(r"C:\Users\Aidata\DOBOT\output\crop_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERLAP = 620
NEW_PX  = 640 - OVERLAP  # = 20px

a = cv2.imread(str(IMG_A))
b = cv2.imread(str(IMG_B))

# B'den sadece yeni alan (sağdan 20px)
b_new = b[:, OVERLAP:]   # 20x480px şerit

# A + B_new yan yana
result = np.concatenate([a, b_new], axis=1)  # 660x480px

# Karşılaştırma için kaydet
cv2.imwrite(str(OUTPUT_DIR / "a_tam.jpg"), a)
cv2.imwrite(str(OUTPUT_DIR / "b_yeni_serit.jpg"), b_new)
cv2.imwrite(str(OUTPUT_DIR / "ab_birlesmis.jpg"), result)

print(f"A boyutu      : {a.shape[1]}x{a.shape[0]}px")
print(f"B yeni şerit  : {b_new.shape[1]}x{b_new.shape[0]}px")
print(f"Birleşik boyut: {result.shape[1]}x{result.shape[0]}px")
print(f"Kaydedildi    : {OUTPUT_DIR}")