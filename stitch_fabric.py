"""
Fabric Image Stitcher — Koordinatsiz
Ayni X degeri = ayni satir → yan yana birlestir
Satirlari alt alta diz → final
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ─── PARAMETRELER ─────────────────────────────────────────────────────────────
INPUT_DIR    = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260402_135416_eski")
OUTPUT_DIR   = Path(r"C:\Users\Aidata\DOBOT\output")
EXT          = "jpg"
OVERLAP_PX   = 50
BLEND_PX     = 20
JPEG_QUALITY = 92
# ──────────────────────────────────────────────────────────────────────────────


def parse_xy(path: Path):
    m = re.search(r'_x(-?[\d.]+)_y(-?[\d.]+)', path.stem)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def load_images(input_dir: Path, ext: str) -> list:
    files = sorted(input_dir.glob(f"*.{ext}"), key=lambda p: p.stem)
    if not files:
        files = sorted(input_dir.glob(f"*.{ext.upper()}"), key=lambda p: p.stem)
    if not files:
        sys.exit(f"[HATA] '{input_dir}' icinde .{ext} dosyasi bulunamadi.")
    images = []
    for f in files:
        x, y = parse_xy(f)
        if x is None:
            continue
        images.append({"path": f, "x": x, "y": y})
    print(f"[INFO] {len(images)} gorsel yuklendi.")
    return images


def blend_h(left: np.ndarray, right: np.ndarray, overlap: int) -> np.ndarray:
    """Yatay: left sagina right yapistir, overlap blend et."""
    if overlap <= 0:
        return np.concatenate([left, right], axis=1)
    ov = min(overlap, left.shape[1], right.shape[1])
    ramp = np.linspace(0, 1, ov, dtype=np.float32)[None, :, None]
    blended = (left[:, -ov:].astype(np.float32) * (1 - ramp)
             + right[:, :ov].astype(np.float32) * ramp).astype(np.uint8)
    return np.concatenate([left[:, :-ov], blended, right[:, ov:]], axis=1)


def blend_v(top: np.ndarray, bottom: np.ndarray, overlap: int) -> np.ndarray:
    """Dikey: top altina bottom yapistir, overlap blend et."""
    if overlap <= 0:
        return np.concatenate([top, bottom], axis=0)
    ov = min(overlap, top.shape[0], bottom.shape[0])
    ramp = np.linspace(0, 1, ov, dtype=np.float32)[:, None, None]
    blended = (top[-ov:].astype(np.float32) * (1 - ramp)
             + bottom[:ov].astype(np.float32) * ramp).astype(np.uint8)
    return np.concatenate([top[:-ov], blended, bottom[ov:]], axis=0)


def build_row(row_metas: list, overlap: int) -> np.ndarray:
    """Ayni X'teki goruntuleri Y'ye gore sirala, yan yana birlestir."""
    row_metas = sorted(row_metas, key=lambda d: d["y"])
    result = None
    for meta in row_metas:
        img = cv2.imread(str(meta["path"]))
        if img is None:
            print(f"  [UYARI] Okunamadi: {meta['path'].name}")
            continue
        result = img if result is None else blend_h(result, img, overlap)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows_dir = OUTPUT_DIR / "rows"
    rows_dir.mkdir(exist_ok=True)

    images = load_images(INPUT_DIR, EXT)
    if not images:
        sys.exit("[HATA] Gorsel parse edilemedi.")

    # X degerine gore satirlara grupla
    by_x = defaultdict(list)
    for meta in images:
        by_x[meta["x"]].append(meta)

    unique_xs = sorted(by_x.keys())
    print(f"[INFO] {len(unique_xs)} satir, OVERLAP_PX={OVERLAP_PX}\n")

    row_paths = []

    # Her satiri yatay olarak birlestir
    for i, x_val in enumerate(unique_xs):
        row = build_row(by_x[x_val], OVERLAP_PX)
        if row is None:
            continue
        label = f"{x_val:.1f}".replace("-", "m").replace(".", "_")
        out_path = rows_dir / f"row_{i:04d}_x{label}.jpg"
        cv2.imwrite(str(out_path), row, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        row_paths.append(out_path)
        if i % 10 == 0 or i == len(unique_xs) - 1:
            print(f"  [{i+1}/{len(unique_xs)}] x={x_val:.1f} -> {row.shape[1]}x{row.shape[0]}px")

    # Satirlari alt alta birlestir
    print(f"\n[Final] {len(row_paths)} satir alt alta birlestiriliyor...")
    final = None
    for idx, rp in enumerate(row_paths):
        row = cv2.imread(str(rp))
        if row is None:
            continue
        if final is None:
            final = row
        else:
            min_w = min(final.shape[1], row.shape[1])
            final = blend_v(final[:, :min_w], row[:, :min_w], OVERLAP_PX)
        if idx % 10 == 0 or idx == len(row_paths) - 1:
            print(f"  [{idx+1}/{len(row_paths)}] eklendi")

    if final is None:
        sys.exit("[HATA] Final goruntu olusturulamadi.")

    final_path = OUTPUT_DIR / "final_stitched.jpg"
    cv2.imwrite(str(final_path), final, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"\n[OK] Final: {final_path}")
    print(f"     Boyut : {final.shape[1]}x{final.shape[0]} px")


if __name__ == "__main__":
    main()