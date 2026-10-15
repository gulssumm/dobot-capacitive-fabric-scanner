"""
Convex-Hull ROI Cropping and Alpha Masking for Stitched Panoramas.

Computes the convex hull of the camera trajectory coordinates, masks out background
regions outside the scanned area using an alpha channel, and crops to the tight
bounding box.
"""

import argparse
import cv2
import numpy as np
import pandas as pd

DEFAULT_INPUT_PANORAMA = "panorama_stitched.jpg"
DEFAULT_INPUT_CSV = "trajectory_data.csv"
DEFAULT_OUTPUT_CROPPED = "panorama_cropped.png"


def crop_to_convex_hull(input_panorama, input_csv, output_cropped):
    df = pd.read_csv(input_csv)
    pts = df[["x", "y"]].values.astype(np.float32)

    hull = cv2.convexHull(pts)
    hull_pts = hull.reshape(-1, 2).astype(np.int32)

    img = cv2.imread(input_panorama)
    if img is None:
        raise FileNotFoundError(f"Could not read panorama image: {input_panorama}")

    print(f"Original size: {img.shape[1]} x {img.shape[0]} px")

    # Convert to BGRA and apply convex hull mask to the alpha channel
    img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [hull_pts], 255)
    img_bgra[:, :, 3] = mask

    # Crop to bounding rectangle of the convex hull
    x, y, w, h = cv2.boundingRect(hull_pts)
    cropped = img_bgra[y:y + h, x:x + w]

    print(f"Cropped size: {w} x {h} px")
    cv2.imwrite(output_cropped, cropped)
    print(f"Saved: {output_cropped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop stitched panorama to convex hull of trajectory")
    parser.add_argument("--panorama", default=DEFAULT_INPUT_PANORAMA, help="Input stitched panorama path")
    parser.add_argument("--csv", default=DEFAULT_INPUT_CSV, help="Input trajectory CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CROPPED, help="Output cropped PNG path")
    args = parser.parse_args()

    crop_to_convex_hull(args.panorama, args.csv, args.output)