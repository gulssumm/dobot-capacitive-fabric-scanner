"""
Multi-Modal Sensor Fusion: Capacitive Heatmap Registration & Overlay.

Maps physical Dobot coordinates (mm) to stitched panorama pixel coordinates via
least-squares affine regression and overlays a Gaussian-smoothed AD7747 capacitance
heatmap (pF) onto the stitched fabric surface panorama.
"""

import argparse
import os
import re
import cv2
import numpy as np
import pandas as pd
from numpy.linalg import lstsq

DEFAULT_IMAGE_FOLDER = os.path.join("scan_images", "sample_scan")
DEFAULT_SCAN_CSV = os.path.join(DEFAULT_IMAGE_FOLDER, "scan_log.csv")
DEFAULT_TRAJECTORY_CSV = "trajectory_data.csv"
DEFAULT_BASE_IMAGE = "panorama_stitched.jpg"
DEFAULT_OUTPUT_HEATMAP = "heatmap_capacitance_overlay.jpg"

BLOB_RADIUS = 60
BLUR_SIGMA = 80
ALPHA = 0.6


def parse_image_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r"img_(\d+)_x([-\d.]+)_y([-\d.]+)", base)
    if match:
        return str(int(match.group(1))), float(match.group(2)), float(match.group(3))
    return None, None, None


def build_phys_to_pixel_transform(image_folder, trajectory_csv):
    """Solves least-squares linear mapping from physical (mm) coordinates to panorama pixels."""
    raw_files = sorted([
        f for f in os.listdir(image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    phys_records = []
    for f in raw_files:
        img_id, x, y = parse_image_filename(f)
        if img_id is not None:
            phys_records.append({"image_id": img_id, "phys_x": x, "phys_y": y})

    df_phys = pd.DataFrame(phys_records)
    df_traj = pd.read_csv(trajectory_csv)
    df_traj.columns = ["image_id", "pixel_x", "pixel_y"]

    df_traj["image_id"] = df_traj["image_id"].apply(lambda v: str(int(float(v))))
    df_phys["image_id"] = df_phys["image_id"].apply(lambda v: str(int(float(v))))

    df_map = df_phys.merge(df_traj, on="image_id")
    if len(df_map) == 0:
        raise ValueError("Merge failed: image_id values do not match between filenames and trajectory CSV.")

    print(f"Matched {len(df_map)} coordinate pairs for physical-to-pixel calibration.")
    phys = df_map[["phys_x", "phys_y"]].values
    A = np.hstack([phys, np.ones((len(phys), 1))])

    params_x, _, _, _ = lstsq(A, df_map["pixel_x"].values, rcond=None)
    params_y, _, _, _ = lstsq(A, df_map["pixel_y"].values, rcond=None)

    return params_x, params_y


def apply_transform(scan_x, scan_y, params_x, params_y):
    A = np.column_stack([scan_x, scan_y, np.ones(len(scan_x))])
    px = (A @ params_x).astype(int)
    py = (A @ params_y).astype(int)
    return px, py


def create_capacitance_heatmap(base_image_path, scan_csv_path, trajectory_csv, image_folder, output_heatmap):
    img = cv2.imread(base_image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load base panorama image: {base_image_path}")

    h, w = img.shape[:2]
    print(f"Panorama dimensions: {w} x {h} px")

    df = pd.read_csv(scan_csv_path)
    df["capacitance_pF"] = pd.to_numeric(df["capacitance_pF"], errors="coerce")
    df_clean = df.dropna(subset=["capacitance_pF", "x", "y"]).copy()

    if len(df_clean) == 0:
        raise ValueError("No valid capacitance readings found in scan log CSV.")

    params_x, params_y = build_phys_to_pixel_transform(image_folder, trajectory_csv)
    px, py = apply_transform(df_clean["x"].values, df_clean["y"].values, params_x, params_y)

    in_bounds = np.sum((px >= 0) & (px < w) & (py >= 0) & (py < h))
    print(f"Projected capacitance points within canvas bounds: {in_bounds}/{len(px)}")
    if in_bounds == 0:
        raise ValueError("All projected points fell outside canvas bounds; check input files.")

    scores = df_clean["capacitance_pF"].values
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)

    heatmap_canvas = np.zeros((h, w), dtype=np.float32)
    for pt_x, pt_y, val in zip(px, py, scores_norm):
        if 0 <= pt_x < w and 0 <= pt_y < h:
            cv2.circle(heatmap_canvas, (pt_x, pt_y), radius=BLOB_RADIUS, color=float(val), thickness=-1)

    heatmap_blurred = cv2.GaussianBlur(heatmap_canvas, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA)
    heatmap_8u = cv2.normalize(heatmap_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored_heatmap = cv2.applyColorMap(heatmap_8u, cv2.COLORMAP_JET)

    mask = np.expand_dims(heatmap_8u.astype(np.float32) / 255.0, axis=-1)
    final_img = (img * (1 - mask * ALPHA) + colored_heatmap * (mask * ALPHA)).astype(np.uint8)

    cv2.imwrite(output_heatmap, final_img)
    print(f"Saved capacitance heatmap overlay: {output_heatmap}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay AD7747 capacitance heatmap onto stitched panorama")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE, help="Stitched panorama image file")
    parser.add_argument("--scan-csv", default=DEFAULT_SCAN_CSV, help="Path to scan_log.csv containing capacitance_pF")
    parser.add_argument("--trajectory-csv", default=DEFAULT_TRAJECTORY_CSV, help="Path to trajectory_data.csv")
    parser.add_argument("--image-folder", default=DEFAULT_IMAGE_FOLDER, help="Folder of raw scan images")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_HEATMAP, help="Output heatmap overlay image path")
    args = parser.parse_args()

    create_capacitance_heatmap(
        args.base_image,
        args.scan_csv,
        args.trajectory_csv,
        args.image_folder,
        args.output
    )