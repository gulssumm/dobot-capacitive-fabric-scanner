import cv2
import numpy as np
import pandas as pd
import os
import re
from numpy.linalg import lstsq

BASE_IMAGE_PATH = r"C:\Users\21180\PycharmProjects\DOBOT_magician\stitching_heatmap_sonuclari\results_18K7119-00_Kaynak_Tekstil_97%COTTON_3EA"
SCAN_CSV_PATH   = r"C:\Users\21180\PycharmProjects\scan_images\18K7119-00_Kaynak_Tekstil_97%COTTON_3EA\scan_log.csv"
TRAJECTORY_CSV  = r"C:\Users\21180\PycharmProjects\DOBOT_magician\stitching_heatmap_sonuclari\results_18K7119-00_Kaynak_Tekstil_97%COTTON_3EA\trajectory_data_18K7119_3EA.csv"
IMAGE_FOLDER    = r"C:\Users\21180\PycharmProjects\scan_images\18K7119-00_Kaynak_Tekstil_97%COTTON_3EA"
OUTPUT_HEATMAP  = "heatmap_capacitance_overlay_118K7119_3EA.jpg"

BLOB_RADIUS = 60
BLUR_SIGMA  = 80
ALPHA       = 0.6

def parse_image_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'img_(\d+)_x([-\d.]+)_y([-\d.]+)', base)
    if match:
        # Normalize to integer string so "0001" and "1" both become "1"
        return str(int(match.group(1))), float(match.group(2)), float(match.group(3))
    return None, None, None

def build_phys_to_pixel_transform(image_folder, trajectory_csv):
    raw_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    phys_records = []
    for f in raw_files:
        img_id, x, y = parse_image_filename(f)
        if img_id is not None:
            phys_records.append({'image_id': img_id, 'phys_x': x, 'phys_y': y})

    df_phys = pd.DataFrame(phys_records)

    df_traj = pd.read_csv(trajectory_csv)
    df_traj.columns = ['image_id', 'pixel_x', 'pixel_y']
    # Normalize both to integer string — fixes "0001" vs "1" mismatch
    df_traj['image_id'] = df_traj['image_id'].apply(lambda v: str(int(float(v))))
    df_phys['image_id'] = df_phys['image_id'].apply(lambda v: str(int(float(v))))

    df_map = df_phys.merge(df_traj, on='image_id')

    print(f"  df_phys rows : {len(df_phys)}")
    print(f"  df_traj rows : {len(df_traj)}")
    print(f"  Merged rows  : {len(df_map)}  ← must be > 0")

    if len(df_map) == 0:
        print("\n  Sample df_phys image_ids:", df_phys['image_id'].head(5).tolist())
        print("  Sample df_traj image_ids:", df_traj['image_id'].head(5).tolist())
        raise ValueError("Merge failed: image_id values don't match between filenames and trajectory CSV.")

    print(f"  Physical X range : {df_map['phys_x'].min():.1f} → {df_map['phys_x'].max():.1f}")
    print(f"  Physical Y range : {df_map['phys_y'].min():.1f} → {df_map['phys_y'].max():.1f}")
    print(f"  Pixel X range    : {df_map['pixel_x'].min():.0f} → {df_map['pixel_x'].max():.0f}")
    print(f"  Pixel Y range    : {df_map['pixel_y'].min():.0f} → {df_map['pixel_y'].max():.0f}")

    phys = df_map[['phys_x', 'phys_y']].values
    A    = np.hstack([phys, np.ones((len(phys), 1))])

    params_x, res_x, _, _ = lstsq(A, df_map['pixel_x'].values, rcond=None)
    params_y, res_y, _, _ = lstsq(A, df_map['pixel_y'].values, rcond=None)

    print(f"  Transform params_x (ax, bx, cx): {params_x}")
    print(f"  Transform params_y (ay, by, cy): {params_y}")

    return params_x, params_y

def apply_transform(scan_x, scan_y, params_x, params_y):
    A  = np.column_stack([scan_x, scan_y, np.ones(len(scan_x))])
    px = (A @ params_x).astype(int)
    py = (A @ params_y).astype(int)
    return px, py

def create_capacitance_heatmap():
    img = cv2.imread(BASE_IMAGE_PATH)
    if img is None:
        print(f"ERROR: Could not load {BASE_IMAGE_PATH}")
        return
    h, w = img.shape[:2]
    print(f"Panorama size: {w} x {h}")

    df = pd.read_csv(SCAN_CSV_PATH)
    print(f"Scan CSV columns: {df.columns.tolist()}")
    df['capacitance_pF'] = pd.to_numeric(df['capacitance_pF'], errors='coerce')
    df_clean = df.dropna(subset=['capacitance_pF', 'x', 'y']).copy()

    if len(df_clean) == 0:
        print("ERROR: No valid capacitance data.")
        return

    print(f"Valid readings  : {len(df_clean)}")
    print(f"Scan phys X range: {df_clean['x'].min():.1f} → {df_clean['x'].max():.1f}")
    print(f"Scan phys Y range: {df_clean['y'].min():.1f} → {df_clean['y'].max():.1f}")

    print("\nBuilding physical → pixel transform...")
    params_x, params_y = build_phys_to_pixel_transform(IMAGE_FOLDER, TRAJECTORY_CSV)

    px, py = apply_transform(df_clean['x'].values, df_clean['y'].values, params_x, params_y)

    print(f"\nMapped pixel X range: {px.min()} → {px.max()}")
    print(f"Mapped pixel Y range: {py.min()} → {py.max()}")

    in_bounds = np.sum((px >= 0) & (px < w) & (py >= 0) & (py < h))
    print(f"Points in bounds: {in_bounds}/{len(px)}")

    if in_bounds == 0:
        print("\nERROR: All points are out of bounds — the coordinate systems are incompatible.")
        print("Check that scan_log x,y are in the same physical units as the image filenames.")
        return

    scores      = df_clean['capacitance_pF'].values
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)

    heatmap_canvas = np.zeros((h, w), dtype=np.float32)
    for pt_x, pt_y, val in zip(px, py, scores_norm):
        if 0 <= pt_x < w and 0 <= pt_y < h:
            cv2.circle(heatmap_canvas, (pt_x, pt_y), radius=BLOB_RADIUS, color=float(val), thickness=-1)

    heatmap_blurred = cv2.GaussianBlur(heatmap_canvas, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA)
    heatmap_8u      = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored_heatmap = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)

    mask      = np.expand_dims(heatmap_8u.astype(np.float32) / 255.0, axis=-1)
    final_img = (img * (1 - mask * ALPHA) + colored_heatmap * (mask * ALPHA)).astype(np.uint8)

    cv2.imwrite(OUTPUT_HEATMAP, final_img)
    print(f"\nSaved: {OUTPUT_HEATMAP}")

if __name__ == "__main__":
    create_capacitance_heatmap()