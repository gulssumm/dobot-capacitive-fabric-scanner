import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_FOLDER = "blue-striped_01_0_unknown"
LOG_FILE = os.path.join(TARGET_FOLDER, "scan_log.csv")

OUTPUT_PANORAMA = "panorama_curved.jpg"
OUTPUT_CSV = "trajectory_data.csv"
OUTPUT_SCATTER = "scatter_plot.png"
OUTPUT_COMBINED = "combined_panorama_scatter.jpg"

print("Reading log file:", LOG_FILE)

if not os.path.exists(LOG_FILE):
    print("ERROR: Log file not found. Please check the path.")
    exit()

df = pd.read_csv(LOG_FILE)
valid_images = []
for idx, row in df.iterrows():
    img_path = os.path.join(TARGET_FOLDER, row['image_file'])
    if os.path.exists(img_path):
        valid_images.append(img_path)

if len(valid_images) < 2:
    print("ERROR: Not enough valid images found for stitching.")
    exit()

print(f"Total {len(valid_images)} images will be processed.")

# --- SIFT INITIALIZATION ---
sift = cv2.SIFT_create(nfeatures=2000)
bf = cv2.BFMatcher()

first_img = cv2.imread(valid_images[0])
img_h, img_w = first_img.shape[:2]
prev_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)

global_transforms = [np.eye(3)]

# --- STAGE 1: CALCULATING IMAGE ROTATIONS AND CURVED TRAJECTORY ---
print("\nSTAGE 1: Calculating image rotations and curved trajectory...")
for i in range(1, len(valid_images)):
    if i % 10 == 0:
        print(f"Rotation analysis ({i}/{len(valid_images)})...")
        
    curr_gray = cv2.imread(valid_images[i], cv2.IMREAD_GRAYSCALE)
    
    kp1, des1 = sift.detectAndCompute(prev_gray, None)
    kp2, des2 = sift.detectAndCompute(curr_gray, None)
    
    M_rel = None
    
    if des1 is not None and des2 is not None and len(des1) >= 10 and len(des2) >= 10:
        matches = bf.knnMatch(des2, des1, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) >= 10:
            src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M_rel, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    
    if M_rel is None:
        print(f"WARNING: No feature match found at image {i}! Estimating from previous step.")
        M_rel = global_transforms[-1][:2, :].copy()
        M_rel[0, 2] = 0
        M_rel[1, 2] = 0
        
    M_rel_3x3 = np.vstack([M_rel, [0, 0, 1]])
    M_global = np.dot(global_transforms[-1], M_rel_3x3)
    global_transforms.append(M_global)
    
    prev_gray = curr_gray

# --- STAGE 2: CALCULATING CANVAS DIMENSIONS ---
print("\nSTAGE 2: Calculating canvas dimensions...")
corners = np.array([[0, 0, 1], [img_w, 0, 1], [img_w, img_h, 1], [0, img_h, 1]]).T
all_corners = []

for M in global_transforms:
    transformed_corners = np.dot(M, corners)
    all_corners.append(transformed_corners[:2, :].T)
    
all_corners = np.vstack(all_corners)

min_x, min_y = np.int32(np.floor(all_corners.min(axis=0)))
max_x, max_y = np.int32(np.ceil(all_corners.max(axis=0)))

canvas_w = max_x - min_x
canvas_h = max_y - min_y

print(f"Canvas Size: {canvas_w} x {canvas_h} pixels")

T_offset = np.array([[1, 0, -min_x],
                     [0, 1, -min_y],
                     [0, 0, 1]])

# --- STAGE 3: WARPING AND BLENDING ---
print("\nSTAGE 3: Warping and blending images onto the canvas...")
canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
canvas_weights = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

center_x_list = []
center_y_list = []
image_id_list = []

local_center = np.array([img_w / 2.0, img_h / 2.0, 1.0])

for i, img_path in enumerate(valid_images):
    if i % 10 == 0:
        print(f"Blending image ({i+1}/{len(valid_images)})...")
        
    img = cv2.imread(img_path)
    
    M_final = np.dot(T_offset, global_transforms[i])
    M_warp = M_final[:2, :] 
    
    global_center = np.dot(M_final, local_center)
    center_x_list.append(global_center[0])
    center_y_list.append(global_center[1])
    
    base_name = os.path.basename(img_path)
    parts = base_name.split('_')
    img_id = parts[1] if len(parts) > 1 else str(i)
    image_id_list.append(img_id)
    
    warped_img = cv2.warpAffine(img, M_warp, (canvas_w, canvas_h))
    mask = (cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    
    canvas += warped_img
    canvas_weights += mask

print("\nFinalizing blending process...")
np.divide(canvas, canvas_weights, out=canvas, where=canvas_weights > 0)
result_img = np.clip(canvas, 0, 255).astype(np.uint8)

# 1. OUTPUT: Save Raw Panorama
cv2.imwrite(OUTPUT_PANORAMA, result_img)
print(f"[1/4] Raw panorama saved: {OUTPUT_PANORAMA}")

# --- STAGE 4: EXPORTING OUTPUTS ---
print("\nSTAGE 4: Generating additional outputs (CSV, Plot, Combined Image)...")

# 2. OUTPUT: Save CSV
df_centers = pd.DataFrame({
    'image_id': image_id_list,
    'x': center_x_list,
    'y': center_y_list
})
df_centers.to_csv(OUTPUT_CSV, index=False)
print(f"[2/4] Trajectory data saved: {OUTPUT_CSV}")

# 3. OUTPUT: Create Scatter Plot (Standalone)
plt.figure(figsize=(12, 8))
plt.scatter(center_x_list, center_y_list, color='red', marker='o', s=15, zorder=2)
plt.plot(center_x_list, center_y_list, color='blue', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

for i, txt in enumerate(image_id_list):
    plt.annotate(txt, (center_x_list[i] + 15, center_y_list[i] + 15), 
                 fontsize=5, alpha=0.8, rotation=45)

plt.gca().invert_yaxis() # Match image coordinate system
plt.title("Standalone Trajectory Scatter Plot")
plt.xlabel("X Coordinate (pixels)")
plt.ylabel("Y Coordinate (pixels)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_SCATTER, dpi=300)
print(f"[3/4] Standalone scatter plot saved: {OUTPUT_SCATTER}")

# 4. OUTPUT: Create Combined Image (Panorama + Scatter Overlay)
# We draw directly on a copy of the high-res panorama using OpenCV
combined_img = result_img.copy()

# Circle parameters
dot_color = (0, 0, 255)     # Red in BGR
line_color = (255, 0, 0)    # Blue in BGR
text_color = (0, 255, 255)  # Yellow in BGR

for i in range(len(center_x_list)):
    pt_x = int(center_x_list[i])
    pt_y = int(center_y_list[i])
    
    # Draw line connecting to the previous point
    if i > 0:
        prev_x = int(center_x_list[i-1])
        prev_y = int(center_y_list[i-1])
        cv2.line(combined_img, (prev_x, prev_y), (pt_x, pt_y), line_color, thickness=2, lineType=cv2.LINE_AA)
        
    # Draw the center dot
    cv2.circle(combined_img, (pt_x, pt_y), radius=6, color=dot_color, thickness=-1, lineType=cv2.LINE_AA)
    
    # Draw the Image ID label
    # Font scale is calculated relative to image size, but 1.0 is generally good for large images
    cv2.putText(combined_img, str(image_id_list[i]), (pt_x + 15, pt_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=text_color, thickness=2, lineType=cv2.LINE_AA)

cv2.imwrite(OUTPUT_COMBINED, combined_img)
print(f"[4/4] Combined image (Panorama + Scatter) saved: {OUTPUT_COMBINED}")

print("\n--- ALL TASKS COMPLETED SUCCESSFULLY! ---")