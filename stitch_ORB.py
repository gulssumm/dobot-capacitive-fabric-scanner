import cv2
import numpy as np
import glob

# ----------------------------
# 1. Görselleri yükle (frame_0001, frame_0002 …)
# ----------------------------
files = sorted(glob.glob("kareler/frame_*.png"))
images = [cv2.imread(f) for f in files]

# ----------------------------
# 2. ORB feature detector
# ----------------------------
orb = cv2.ORB_create(5000)  # feature sayısı artırılabilir

def stitch_two_images(img1, img2):
    # 2.1. Keypoint ve descriptor
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2.2. Feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 10:
        print("Yeterli eşleşme yok!")
        return None

    # 2.3. Noktalardan koordinat çıkar
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 2.4. Homography hesapla
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 2.5. Warping
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    panorama = cv2.warpPerspective(img1, H, (width1 + width2, max(height1, height2)))
    panorama[0:height2, 0:width2] = img2

    return panorama

# ----------------------------
# 3. Tüm görselleri sırayla stitch et
# ----------------------------
panorama = images[0]

for i in range(1, len(images)):
    print(f"Stitching: {files[i-1]} -> {files[i]}")
    result = stitch_two_images(panorama, images[i])
    if result is not None:
        panorama = result
    else:
        print(f"Frame {files[i]} atlandı.")

# ----------------------------
# 4. Sonucu kaydet ve göster
# ----------------------------
cv2.imwrite("stitched_result.png", panorama)
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()