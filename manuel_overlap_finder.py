"""
Manuel overlap bulucu — iki görüntüyü üst üste koy, kaydır, kontrol et
"""
from pathlib import Path
import cv2
import numpy as np

IMG_A = Path(r"C:\...\img_0001_....jpg")  # 1. görüntü
IMG_B = Path(r"C:\...\img_0002_....jpg")  # 2. görüntü (hemen ardışık)

a = cv2.imread(str(IMG_A))
b = cv2.imread(str(IMG_B))

# Dene: a'nın altına b'yi overlap kadar üst üste koy
OVERLAP = 300  # buraya farklı değerler gir: 250, 300, 315, 350...

merged = np.concatenate([a[:-OVERLAP], 
                         ((a[-OVERLAP:].astype(np.float32) * 0.5) + 
                          (b[:OVERLAP].astype(np.float32) * 0.5)).astype(np.uint8),
                         b[OVERLAP:]], axis=0)

cv2.imwrite("test_overlap.jpg", merged)
print("test_overlap.jpg kaydedildi")