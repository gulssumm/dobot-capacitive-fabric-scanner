"""
Fabric Image Stitcher — Satir Bazli (Dusuk RAM)
Dosya formati: img_0001_x215.0_y-37.0.jpg
"""

import re
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# ─── PARAMETRELER ─────────────────────────────────────────────────────────────
INPUT_DIR    = Path(r"C:\Users\Aidata\DOBOT\scan_images\scan_20260401_134335")
OUTPUT_DIR   = Path(r"C:\Users\Aidata\DOBOT\output")
EXT          = "jpg"
BLEND_PX     = 10
JPEG_QUALITY = 92

# PX_PER_MM: otomatik hesaplanir (None birakinca)
# Elle girmek istersen: PX_PER_MM = 13.0
PX_PER_MM    = None
# ──────────────────────────────────────────────────────────────────────────────


def parse_filename(path: Path):
    m = re.search(r'_x(-?[\d.]+)_y(-?[\d.]+)', path.stem)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def load_metadata(input_dir: Path, ext: str) -> list:
    files = sorted(input_dir.glob(f"*.{ext}"), key=lambda p: p.stem)
    if not files:
        files = sorted(input_dir.glob(f"*.{ext.upper()}"), key=lambda p: p.stem)
    if not files:
        sys.exit(f"[HATA] '{input_dir}' icinde .{ext} dosyasi bulunamadi.")
    images, skipped = [], 0
    for f in files:
        coords = parse_filename(f)
        if coords is None:
            skipped += 1
            continue
        images.append({"path": f, "x": coords[0], "y": coords[1]})
    print(f"[INFO] {len(images)} gorsel yuklendi, {skipped} atlandi.")
    return images


def estimate_px_per_mm(images: list, sample_n: int = 20) -> float:
    """
    Ayni X kolonunda ardisik Y ciftlerinden phase correlation ile px/mm hesapla.
    """
    from skimage.registration import phase_cross_correlation

    by_x = defaultdict(list)
    for d in images:
        by_x[d["x"]].append(d)

    measurements = []
    for x_val, col in by_x.items():
        col = sorted(col, key=lambda d: d["y"])
        for i in range(len(col) - 1):
            if len(measurements) >= sample_n:
                break
            dy_mm = abs(col[i+1]["y"] - col[i]["y"])
            if dy_mm == 0:
                continue
            a = cv2.imread(str(col[i]["path"]))
            b = cv2.imread(str(col[i+1]["path"]))
            if a is None or b is None:
                continue
            ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            shift, err, _ = phase_cross_correlation(ga, gb, upsample_factor=10)
            dy_px = abs(float(shift[0]))
            # Sadece makul aralikta kabul et
            if err < 0.9 and 2 < dy_px < a.shape[0] * 0.95:
                measurements.append(dy_px / dy_mm)
                print(f"  Olcum: dy={dy_px:.1f}px / {dy_mm}mm = {dy_px/dy_mm:.2f} px/mm")
        if len(measurements) >= sample_n:
            break

    if not measurements:
        print("[UYARI] px/mm otomatik tespiti basarisiz — varsayilan 13.0 kullaniliyor.")
        print("        Elle ayarlamak icin: PX_PER_MM = <deger> (kodun ustunde)")
        return 13.0

    val = float(np.median(measurements))
    print(f"[INFO] px/mm = {val:.2f}  ({len(measurements)} olcumden medyan)")
    return val


def blend_patch(canvas, weights, patch, py, px, blend_px):
    h, w = patch.shape[:2]
    ch, cw = canvas.shape[:2]
    y1 = min(py + h, ch)
    x1 = min(px + w, cw)
    ih, iw = y1 - py, x1 - px
    if ih <= 0 or iw <= 0:
        return
    p = patch[:ih, :iw].astype(np.float32)
    fade = max(1, min(blend_px, ih // 2, iw // 2))
    mask = np.ones((ih, iw), dtype=np.float32)
    if fade > 1:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)
        mask[:fade,  :] *= ramp[:, None]
        mask[-fade:, :] *= ramp[::-1, None]
        mask[:,  :fade] *= ramp[None, :]
        mask[:, -fade:] *= ramp[::-1][None, :]
    m = mask[:, :, None]
    canvas[py:y1, px:x1]  += p * m
    weights[py:y1, px:x1] += m


def stitch_row(row_images, x_min, canvas_w, img_h, px_per_mm, blend_px):
    canvas  = np.zeros((img_h, canvas_w, 3), dtype=np.float32)
    weights = np.zeros((img_h, canvas_w, 1), dtype=np.float32)
    for meta in row_images:
        img = cv2.imread(str(meta["path"]))
        if img is None:
            continue
        px = int((meta["x"] - x_min) * px_per_mm)
        blend_patch(canvas, weights, img, 0, px, blend_px)
    weights = np.clip(weights, 1e-9, None)
    return np.clip(canvas / weights, 0, 255).astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strips_dir = OUTPUT_DIR / "strips"
    strips_dir.mkdir(exist_ok=True)

    images = load_metadata(INPUT_DIR, EXT)
    if not images:
        sys.exit("[HATA] Hic gorsel parse edilemedi.")

    # px/mm belirle
    px_per_mm = PX_PER_MM
    if px_per_mm is None:
        print("[INFO] px/mm otomatik hesaplaniyor...")
        px_per_mm = estimate_px_per_mm(images)

    # Y'ye gore grupla
    by_y = defaultdict(list)
    for meta in images:
        by_y[meta["y"]].append(meta)

    unique_ys = sorted(by_y.keys())
    x_vals    = [d["x"] for d in images]
    x_min     = min(x_vals)
    x_max     = max(x_vals)

    sample = cv2.imread(str(images[0]["path"]))
    if sample is None:
        sys.exit("[HATA] Ilk gorsel okunamadi.")
    img_h, img_w = sample.shape[:2]

    canvas_w  = int((x_max - x_min) * px_per_mm) + img_w + 4
    y_step_mm = unique_ys[1] - unique_ys[0] if len(unique_ys) > 1 else 2.0
    y_step_px = int(y_step_mm * px_per_mm)

    print(f"\n[INFO] px/mm={px_per_mm:.2f}  |  y_step={y_step_px}px ({y_step_mm}mm)")
    print(f"[INFO] Gorsel boyutu: {img_w}x{img_h}px")
    print(f"[INFO] Canvas genisligi: {canvas_w}px")
    print(f"[INFO] {len(unique_ys)} satir isleniyor...\n")

    strip_paths = []

    for i, y_val in enumerate(unique_ys):
        row = sorted(by_y[y_val], key=lambda d: d["x"])
        result = stitch_row(row, x_min, canvas_w, img_h, px_per_mm, BLEND_PX)
        label = f"{y_val:.1f}".replace("-", "m").replace(".", "_")
        out_path = strips_dir / f"strip_{i:04d}_y{label}.jpg"
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        strip_paths.append(out_path)
        if i % 10 == 0 or i == len(unique_ys) - 1:
            print(f"  [{i+1}/{len(unique_ys)}] y={y_val:.1f}mm -> {out_path.name}")

    # ─── Dikey birlesim ───────────────────────────────────────────────────────
    print(f"\n[Final] {len(strip_paths)} satir dikey birlestiriliyor...")
    total_h = y_step_px * (len(unique_ys) - 1) + img_h + 4

    final_canvas  = np.zeros((total_h, canvas_w, 3), dtype=np.float32)
    final_weights = np.zeros((total_h, canvas_w, 1), dtype=np.float32)

    for i, sp in enumerate(strip_paths):
        strip = cv2.imread(str(sp))
        if strip is None:
            continue
        py = i * y_step_px
        blend_patch(final_canvas, final_weights, strip, py, 0, BLEND_PX)
        if i % 10 == 0 or i == len(strip_paths) - 1:
            print(f"  [{i+1}/{len(strip_paths)}] eklendi")

    final_weights = np.clip(final_weights, 1e-9, None)
    final = np.clip(final_canvas / final_weights, 0, 255).astype(np.uint8)

    final_path = OUTPUT_DIR / "final_stitched.jpg"
    cv2.imwrite(str(final_path), final, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    print(f"\n[OK] Final: {final_path}")
    print(f"     Boyut: {final.shape[1]}x{final.shape[0]} px")


if __name__ == "__main__":
    main()