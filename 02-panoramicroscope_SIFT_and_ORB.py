import cv2
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_FOLDER = r"C:\Users\ZEYNEP\Desktop\Dobot_yan_sanayi\steel_12_sheet\steel_12_sheet"

OUTPUT_PANORAMA = "panorama_curved_steel.jpg"
OUTPUT_CSV = "trajectory_data_stell.csv"
OUTPUT_SCATTER = "scatter_plot_steel.png"
OUTPUT_COMBINED = "combined_panorama_scatter_steel.jpg"

# Fiziksel birim → piksel ölçek faktörü (birim piksel ise 1.0 bırakın)
SCALE = 1.0
MAX_DIM = 30000  # Canvas maksimum boyutu (piksel)

# ============================================================
# DOSYA İSMİNDEN PARSE: img_0001_x215.0_y-39.0.jpg
# ============================================================
def parse_image_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'img_(\d+)_x([-\d.]+)_y([-\d.]+)', base)
    if match:
        return match.group(1), float(match.group(2)), float(match.group(3))
    return None, None, None

raw_files = sorted([
    f for f in os.listdir(TARGET_FOLDER)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

records = []
for f in raw_files:
    img_id, x, y = parse_image_filename(f)
    if img_id is not None:
        records.append({'image_id': img_id, 'x': x, 'y': y,
                        'path': os.path.join(TARGET_FOLDER, f)})
    else:
        print(f"WARNING: Could not parse filename: {f}")

records.sort(key=lambda r: int(r['image_id']))

valid_images = [r['path']      for r in records]
image_ids    = [r['image_id']  for r in records]
log_x        = [r['x']         for r in records]
log_y        = [r['y']         for r in records]

print(f"Found and parsed {len(records)} images.")
if records:
    print(f"  First : {os.path.basename(records[0]['path'])}  x={log_x[0]}, y={log_y[0]}")
    print(f"  Last  : {os.path.basename(records[-1]['path'])}  x={log_x[-1]}, y={log_y[-1]}")

if len(valid_images) < 2:
    print("ERROR: Not enough images.")
    exit()

# ============================================================
# DÜZELTME 1: CLAHE + SIFT (yüksek hassasiyet) + ORB fallback
# ============================================================
clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
sift     = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.02, edgeThreshold=5)
orb      = cv2.ORB_create(nfeatures=2000)
bf_sift  = cv2.BFMatcher(cv2.NORM_L2)
bf_orb   = cv2.BFMatcher(cv2.NORM_HAMMING)

def enhance_gray(gray):
    return clahe.apply(gray)

def match_images(gray1, gray2):
    g1, g2 = enhance_gray(gray1), enhance_gray(gray2)

    # -- SIFT --
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    M_rel = None

    if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
        matches = bf_sift.knnMatch(des2, des1, k=2)
        good = [m for m, n in matches if m.distance < 0.70 * n.distance]

        if len(good) >= 15:
            src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M_rel, inliers = cv2.estimateAffinePartial2D(
                src, dst, method=cv2.RANSAC,
                ransacReprojThreshold=2.0, confidence=0.99, maxIters=5000)

            # DÜZELTME 2: Inlier oranı kontrolü
            if M_rel is not None and inliers is not None:
                if inliers.sum() / len(good) < 0.4:
                    M_rel = None

    # -- ORB fallback --
    if M_rel is None:
        kp1o, des1o = orb.detectAndCompute(g1, None)
        kp2o, des2o = orb.detectAndCompute(g2, None)
        if des1o is not None and des2o is not None and len(des1o) >= 10 and len(des2o) >= 10:
            matches_o = bf_orb.knnMatch(des2o, des1o, k=2)
            good_o = [m for m, n in matches_o if m.distance < 0.70 * n.distance]
            if len(good_o) >= 15:
                src = np.float32([kp2o[m.queryIdx].pt for m in good_o]).reshape(-1, 1, 2)
                dst = np.float32([kp1o[m.trainIdx].pt for m in good_o]).reshape(-1, 1, 2)
                M_rel, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                if M_rel is not None and inliers is not None:
                    if inliers.sum() / len(good_o) < 0.4:
                        M_rel = None
    return M_rel

# --- STAGE 1: ROTASYON VE TRANSFORM HESAPLAMA ---
print("\nSTAGE 1: Calculating rotations and trajectory...")

first_img = cv2.imread(valid_images[0])
img_h, img_w = first_img.shape[:2]
prev_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

global_transforms = [np.eye(3)]
fail_count = 0

for i in range(1, len(valid_images)):
    if i % 10 == 0:
        print(f"  ({i}/{len(valid_images)}) | fails so far: {fail_count}")

    curr_gray = cv2.imread(valid_images[i], cv2.IMREAD_GRAYSCALE)
    M_rel = match_images(prev_gray, curr_gray)

    if M_rel is None:
        fail_count += 1
        # DÜZELTME 3+4: Dosya ismindeki fiziksel koordinatlardan akıllı fallback
        dx = (log_x[i] - log_x[i-1]) * SCALE
        dy = (log_y[i] - log_y[i-1]) * SCALE
        M_rel = np.array([[1, 0, dx],
                          [0, 1, dy]], dtype=np.float64)
        print(f"  [FALLBACK] Image {i}: log offset dx={dx:.1f}, dy={dy:.1f}")

    M_rel_3x3 = np.vstack([M_rel, [0, 0, 1]])
    M_global  = np.dot(global_transforms[-1], M_rel_3x3)
    global_transforms.append(M_global)
    prev_gray = curr_gray

print(f"\nTotal match failures: {fail_count}/{len(valid_images)-1} "
      f"({100*fail_count/max(len(valid_images)-1,1):.1f}%)")
if fail_count / max(len(valid_images)-1, 1) > 0.5:
    print("WARNING: >50% failures. Check image overlap and lighting.")

# --- STAGE 2: CANVAS BOYUTU ---
print("\nSTAGE 2: Calculating canvas dimensions...")
corners = np.array([[0,0,1],[img_w,0,1],[img_w,img_h,1],[0,img_h,1]]).T
all_corners = []
for M in global_transforms:
    tc = np.dot(M, corners)
    all_corners.append(tc[:2,:].T)
all_corners = np.vstack(all_corners)

min_x, min_y = np.int32(np.floor(all_corners.min(axis=0)))
max_x, max_y = np.int32(np.ceil(all_corners.max(axis=0)))
canvas_w = max_x - min_x
canvas_h = max_y - min_y

# DÜZELTME 5: Canvas boyutu sınırlaması
if canvas_w > MAX_DIM or canvas_h > MAX_DIM:
    print(f"WARNING: Canvas too large ({canvas_w}x{canvas_h}). Clamping to {MAX_DIM}.")
    canvas_w = min(canvas_w, MAX_DIM)
    canvas_h = min(canvas_h, MAX_DIM)

print(f"Canvas Size: {canvas_w} x {canvas_h} pixels")

T_offset = np.array([[1, 0, -min_x],
                     [0, 1, -min_y],
                     [0, 0,      1]])

# --- STAGE 3: WARP & BLEND ---
print("\nSTAGE 3: Warping and blending images onto the canvas...")
canvas         = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
canvas_weights = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
center_x_list, center_y_list, image_id_list = [], [], []
local_center = np.array([img_w / 2.0, img_h / 2.0, 1.0])

for i, img_path in enumerate(valid_images):
    if i % 10 == 0:
        print(f"  Blending ({i+1}/{len(valid_images)})...")

    img     = cv2.imread(img_path)
    M_final = np.dot(T_offset, global_transforms[i])
    M_warp  = M_final[:2, :]

    global_center = np.dot(M_final, local_center)
    center_x_list.append(global_center[0])
    center_y_list.append(global_center[1])
    image_id_list.append(image_ids[i])

    warped_img = cv2.warpAffine(img, M_warp, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # DÜZELTME 6: Gaussian ağırlık maskesi (kenar artefaktlarını azaltır)
    binary_mask  = (cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    weight_mask  = cv2.GaussianBlur(binary_mask, (51, 51), 0)
    weight_mask  = np.expand_dims(weight_mask, axis=-1)

    canvas         += warped_img.astype(np.float32) * weight_mask
    canvas_weights += weight_mask

print("\nFinalizing blending...")
np.divide(canvas, canvas_weights, out=canvas, where=canvas_weights > 0)
result_img = np.clip(canvas, 0, 255).astype(np.uint8)

# --- STAGE 4: OUTPUTS ---
cv2.imwrite(OUTPUT_PANORAMA, result_img)
print(f"[1/4] Panorama saved: {OUTPUT_PANORAMA}")

df_centers = pd.DataFrame({'image_id': image_id_list, 'x': center_x_list, 'y': center_y_list})
df_centers.to_csv(OUTPUT_CSV, index=False)
print(f"[2/4] CSV saved: {OUTPUT_CSV}")

plt.figure(figsize=(12, 8))
plt.scatter(center_x_list, center_y_list, color='red', s=15, zorder=2)
plt.plot(center_x_list, center_y_list, color='blue', linewidth=1, alpha=0.5, zorder=1)
for i, txt in enumerate(image_id_list):
    plt.annotate(txt, (center_x_list[i]+15, center_y_list[i]+15),
                 fontsize=5, alpha=0.8, rotation=45)
plt.gca().invert_yaxis()
plt.title("Trajectory Scatter Plot")
plt.xlabel("X (pixels)")
plt.ylabel("Y (pixels)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_SCATTER, dpi=300)
print(f"[3/4] Scatter saved: {OUTPUT_SCATTER}")

combined_img = result_img.copy()
for i in range(len(center_x_list)):
    pt_x, pt_y = int(center_x_list[i]), int(center_y_list[i])
    if i > 0:
        cv2.line(combined_img,
                 (int(center_x_list[i-1]), int(center_y_list[i-1])),
                 (pt_x, pt_y), (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(combined_img, (pt_x, pt_y), 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(combined_img, str(image_id_list[i]), (pt_x+15, pt_y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(OUTPUT_COMBINED, combined_img)
print(f"[4/4] Combined saved: {OUTPUT_COMBINED}")

print("\n--- ALL TASKS COMPLETED SUCCESSFULLY! ---")