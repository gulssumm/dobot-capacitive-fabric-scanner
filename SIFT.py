"""
SIFT ile ortak alan tespiti — 2 görüntü arası
"""
from pathlib import Path
import cv2
import numpy as np

IMG_A = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0001_....jpg")
IMG_B = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski\img_0002_....jpg")
OUTPUT_DIR = Path(r"C:\Users\Aidata\DOBOT\output\sift_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

a = cv2.imread(str(IMG_A))
b = cv2.imread(str(IMG_B))

ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

# SIFT
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(ga, None)
kp_b, des_b = sift.detectAndCompute(gb, None)

print(f"A keypoint sayısı: {len(kp_a)}")
print(f"B keypoint sayısı: {len(kp_b)}")

if len(kp_a) < 2 or len(kp_b) < 2:
    print("[HATA] Yeterli keypoint bulunamadı — doku çok tekrarlı olabilir")
    exit()

# FLANN matcher
index_params  = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

print(f"Toplam eşleşme : {len(matches)}")
print(f"İyi eşleşme    : {len(good)}")

if len(good) < 4:
    print("[HATA] Yeterli iyi eşleşme yok")
    exit()

# Homography
src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print(f"\nHomography matrisi:\n{H}")

# Eşleşme görselini kaydet
match_img = cv2.drawMatches(a, kp_a, b, kp_b, good, None,
                             matchColor=(0, 255, 0),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(str(OUTPUT_DIR / "sift_matches.jpg"), match_img)
print(f"\nKaydedildi: {OUTPUT_DIR / 'sift_matches.jpg'}")