"""
Panoramik Satır Stitch — ECC Hizalama + Laplacian Multiband Blending
"""

import re
import cv2
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# AYARLAR — BURAYA KENDINIZINKINI YAZIN
# ─────────────────────────────────────────

INPUT_DIR       = "./images"       # görüntülerin bulunduğu klasör
X_VALUE         = 215.0            # birleştirilecek satırın x koordinatı
OUTPUT_PATH     = "./panorama.jpg" # çıktı dosyası
INITIAL_TX_RATIO = 0.8             # ECC başlangıç tx tahmini (0.0-1.0)
SIFT_FALLBACK   = True             # ECC başarısızsa SIFT dene
PYRAMID_LEVELS  = 4                # Laplacian pyramid derinliği (3-6)

# ─────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────

def parse_filename(filename):
    stem = Path(filename).stem
    # Format: img_0001_x215.0_y-39.0
    m = re.search(r'_x(-?[\d.]+)_y(-?[\d.]+)', stem)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


# ─────────────────────────────────────────
# HİZALAMA: ECC + SIFT FALLBACK
# ─────────────────────────────────────────

def align_ecc(img_ref, img_new, initial_tx_ratio=0.8):
    h, w = img_ref.shape[:2]
    g1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = w * initial_tx_ratio
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        _, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_TRANSLATION, crit)
        return float(warp[0, 2]), float(warp[1, 2])
    except cv2.error:
        return None


def align_sift(img_ref, img_new):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_ref, None)
    kp2, des2 = sift.detectAndCompute(img_new, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 10:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask.sum() < 8:
        return None
    return float(H[0, 2]), float(H[1, 2])


def find_alignment(canvas_crop, img_new):
    result = align_ecc(canvas_crop, img_new, INITIAL_TX_RATIO)
    if result is not None:
        return result, "ECC"
    if SIFT_FALLBACK:
        result = align_sift(canvas_crop, img_new)
        if result is not None:
            return result, "SIFT"
    w = canvas_crop.shape[1]
    return (w * INITIAL_TX_RATIO, 0.0), "DEFAULT"


# ─────────────────────────────────────────
# LAPLACIAN PYRAMID MULTIBAND BLENDING
# ─────────────────────────────────────────

def build_gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(img, levels):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        h, w = gp[i].shape[:2]
        up = cv2.pyrUp(gp[i + 1], dstsize=(w, h))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def multiband_blend(img_left, img_right, mask_left, levels=4):
    mask_right = 1.0 - mask_left
    lp_l  = build_laplacian_pyramid(img_left,  levels)
    lp_r  = build_laplacian_pyramid(img_right, levels)
    gp_ml = build_gaussian_pyramid(mask_left,  levels)
    gp_mr = build_gaussian_pyramid(mask_right, levels)
    blended = []
    for la, rb, ml, mr in zip(lp_l, lp_r, gp_ml, gp_mr):
        if ml.ndim == 2:
            ml = ml[:, :, np.newaxis]
            mr = mr[:, :, np.newaxis]
        blended.append(la * ml + rb * mr)
    result = blended[-1]
    for b in reversed(blended[:-1]):
        h, w = b.shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + b
    return result


# ─────────────────────────────────────────
# KANVASA BİRLEŞTİR
# ─────────────────────────────────────────

def merge_into_canvas(canvas, img_new, overlap_start, overlap_px, h_ref, w_ref):
    new_area_w = w_ref - overlap_px
    if new_area_w <= 0:
        return canvas

    # Kanvası genişlet
    ext = np.zeros((h_ref, new_area_w, 3), dtype=np.float32)
    canvas = np.concatenate([canvas, ext], axis=1)

    # Örtüşen bölge: multiband blend
    if overlap_px > 1:
        left_strip  = canvas[:, overlap_start:overlap_start + overlap_px].copy()
        right_strip = img_new[:, :overlap_px].astype(np.float32)
        alpha  = np.linspace(1.0, 0.0, overlap_px, dtype=np.float32)
        mask_l = np.repeat(np.tile(alpha, (h_ref, 1))[:, :, np.newaxis], 3, axis=2)
        lvl    = max(1, min(PYRAMID_LEVELS, int(np.log2(max(overlap_px, 2))) - 1))
        blended = multiband_blend(left_strip, right_strip, mask_l, levels=lvl)
        canvas[:, overlap_start:overlap_start + overlap_px] = blended.clip(0, 255)

    # Yeni bölge: direkt kopyala
    dst_start = overlap_start + overlap_px
    canvas[:, dst_start:dst_start + new_area_w] = \
        img_new[:, overlap_px:overlap_px + new_area_w].astype(np.float32)

    return canvas


# ─────────────────────────────────────────
# ANA STİTCH
# ─────────────────────────────────────────

def stitch_panoramic():
    entries = []
    for f in sorted(Path(INPUT_DIR).glob("*.jpg")):
        x, y = parse_filename(f.name)
        if x is not None and abs(x - X_VALUE) < 0.1:
            entries.append((y, str(f)))

    if not entries:
        print(f"❌ x={X_VALUE} için görüntü bulunamadı!")
        return

    entries.sort(key=lambda t: t[0])
    n = len(entries)
    print(f"✅ {n} görüntü  (x={X_VALUE}, y: {entries[0][0]:.1f} → {entries[-1][0]:.1f})")

    first = cv2.imread(entries[0][1])
    if first is None:
        print("❌ İlk görüntü okunamadı.")
        return

    h_ref, w_ref = first.shape[:2]
    print(f"Görüntü boyutu: {w_ref}x{h_ref} | Pyramid levels: {PYRAMID_LEVELS}\n")

    canvas  = first.astype(np.float32)
    skipped = 0

    for i, (y_val, fpath) in enumerate(entries[1:], start=1):
        img_new = cv2.imread(fpath)
        if img_new is None:
            print(f"  [{i:02d}/{n-1}] ⚠️  Atlandı (okunamadı)")
            skipped += 1
            continue

        canvas_w   = canvas.shape[1]
        crop_start = max(0, canvas_w - w_ref)
        canvas_crop = canvas[:, crop_start:].astype(np.uint8)

        (tx, ty), method = find_alignment(canvas_crop, img_new)

        overlap_start = crop_start + int(round(tx))
        overlap_px    = canvas_w - overlap_start
        new_area_w    = w_ref - overlap_px

        if overlap_px <= 0 or overlap_px >= w_ref or new_area_w <= 0:
            print(f"  [{i:02d}/{n-1}] ⚠️  Geçersiz örtüşme ({overlap_px}px), atlandı.")
            skipped += 1
            continue

        # Dikey hizalama
        ty_int = int(round(ty))
        if abs(ty_int) > 0:
            M = np.float32([[1, 0, 0], [0, 1, -ty_int]])
            img_new = cv2.warpAffine(img_new, M, (w_ref, h_ref),
                                     borderMode=cv2.BORDER_REPLICATE)

        canvas = merge_into_canvas(canvas, img_new.astype(np.float32),
                                   overlap_start, overlap_px, h_ref, w_ref)

        print(f"  [{i:02d}/{n-1}] y={y_val:7.1f}mm | {method:7s} | "
              f"tx={tx:7.1f}px  ty={ty:+6.1f}px | "
              f"örtüşme={overlap_px:4d}px | kanvas={canvas.shape[1]:6d}px")

    result = canvas.clip(0, 255).astype(np.uint8)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n✅ Kaydedildi : {OUTPUT_PATH}")
    print(f"   Final boyut : {result.shape[1]} x {result.shape[0]} px")
    print(f"   Atlanan     : {skipped}/{n-1}")


# ─────────────────────────────────────────
# ÇALIŞTIR
# ─────────────────────────────────────────

if __name__ == "__main__":
    stitch_panoramic()
