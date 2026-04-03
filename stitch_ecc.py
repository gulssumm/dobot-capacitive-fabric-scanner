"""
Panoramik Satır Stitch — ECC Hizalama + Laplacian Multiband Blending
=====================================================================
Tekrarlı doku gibi SIFT'in başarısız olduğu görüntülerde ECC daha güvenilir.
Birleştirme dikişlerini gizlemek için Laplacian pyramid (multiband) blending kullanılır.

Kullanım:
    python stitch_panoramic.py --input_dir ./images --x_value 215.0 --output panorama.jpg

    # ECC başarısız olan adımlar için SIFT fallback ile:
    python stitch_panoramic.py --input_dir ./images --x_value 215.0 --sift_fallback

Gereksinimler:
    pip install opencv-python numpy
"""

import re
import cv2
import numpy as np
import argparse
from pathlib import Path


# ─────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────

def parse_filename(filename):
    stem = Path(filename).stem
    m = re.search(r'_x(-?\d+)_(\d+)_y(-?\d+)_(\d+)', stem)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}"), float(f"{m.group(3)}.{m.group(4)}")
    return None, None


# ─────────────────────────────────────────
# HİZALAMA: ECC + SIFT FALLBACK
# ─────────────────────────────────────────

def align_ecc(img_ref, img_new, initial_tx_ratio=0.8):
    """ECC translation. Döndürür: (tx, ty) veya None."""
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
    """SIFT + homography. Sadece translation bileşenini döndürür: (tx, ty) veya None."""
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
    # Homography'den translation bileşeni al
    tx = float(H[0, 2])
    ty = float(H[1, 2])
    return tx, ty


def find_alignment(canvas_crop, img_new, initial_tx_ratio=0.8, use_sift_fallback=True):
    """ECC dene, başarısızsa SIFT fallback."""
    result = align_ecc(canvas_crop, img_new, initial_tx_ratio)
    if result is not None:
        return result, "ECC"
    if use_sift_fallback:
        result = align_sift(canvas_crop, img_new)
        if result is not None:
            return result, "SIFT"
    w = canvas_crop.shape[1]
    return (w * initial_tx_ratio, 0.0), "DEFAULT"


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
    """
    img_left, img_right : float32 BGR, aynı boyut
    mask_left           : float32 [0,1], img_left'in ağırlık maskesi
    levels              : piramit derinliği (örtüşme genişliğine göre 4-6 iyi)
    """
    mask_right = 1.0 - mask_left

    lp_l = build_laplacian_pyramid(img_left,  levels)
    lp_r = build_laplacian_pyramid(img_right, levels)
    gp_ml = build_gaussian_pyramid(mask_left,  levels)
    gp_mr = build_gaussian_pyramid(mask_right, levels)

    blended = []
    for la, rb, ml, mr in zip(lp_l, lp_r, gp_ml, gp_mr):
        if la.ndim == 2:
            ml = ml[:, :, np.newaxis]
            mr = mr[:, :, np.newaxis]
        else:
            if ml.ndim == 2:
                ml = ml[:, :, np.newaxis]
                mr = mr[:, :, np.newaxis]
        blended.append(la * ml + rb * mr)

    # Piramidi yeniden oluştur
    result = blended[-1]
    for b in reversed(blended[:-1]):
        h, w = b.shape[:2]
        result = cv2.pyrUp(result, dstsize=(w, h)) + b

    return result


# ─────────────────────────────────────────
# İKİ GÖRÜNTÜYÜ BİRLEŞTİR
# ─────────────────────────────────────────

def merge_into_canvas(canvas, img_new, overlap_start, overlap_px, h_ref, w_ref, levels=4):
    """
    canvas       : mevcut panorama (float32)
    img_new      : yeni görüntü (float32), dikey hizalama uygulanmış
    overlap_start: kanvasta örtüşmenin başladığı sütun
    overlap_px   : örtüşme genişliği (piksel)
    """
    canvas_w = canvas.shape[1]
    new_area_w = w_ref - overlap_px

    # Kanvası genişlet
    if new_area_w > 0:
        ext = np.zeros((h_ref, new_area_w, 3), dtype=np.float32)
        canvas = np.concatenate([canvas, ext], axis=1)

    # --- Örtüşen bölge: multiband blending ---
    if overlap_px > 1:
        # Kanvastaki örtüşen şerit
        left_strip  = canvas[:, overlap_start:overlap_start + overlap_px].copy()
        right_strip = img_new[:, :overlap_px].copy()

        # Lineer geçiş maskesi (sol=1 → sağ=0 canvas için)
        alpha = np.linspace(1.0, 0.0, overlap_px, dtype=np.float32)
        mask_left = np.tile(alpha, (h_ref, 1))[:, :, np.newaxis]
        mask_left = np.repeat(mask_left, 3, axis=2)

        # Piramit seviyesi: örtüşme genişliğine göre ayarla
        lvl = min(levels, int(np.log2(overlap_px)) - 1)
        lvl = max(1, lvl)

        blended_strip = multiband_blend(left_strip, right_strip, mask_left, levels=lvl)
        canvas[:, overlap_start:overlap_start + overlap_px] = blended_strip.clip(0, 255)

    # --- Yeni bölge: direkt kopyala ---
    if new_area_w > 0:
        new_start_canvas = overlap_start + overlap_px
        new_start_img    = overlap_px
        canvas[:, new_start_canvas:new_start_canvas + new_area_w] = \
            img_new[:, new_start_img:new_start_img + new_area_w]

    return canvas


# ─────────────────────────────────────────
# ANA STİTCH
# ─────────────────────────────────────────

def stitch_panoramic(input_dir, x_value, output_path,
                     initial_tx_ratio=0.8, sift_fallback=True, pyramid_levels=4):

    # Görüntüleri bul
    entries = []
    for f in sorted(Path(input_dir).glob("*.jpg")):
        x, y = parse_filename(f.name)
        if x is not None and abs(x - x_value) < 0.1:
            entries.append((y, str(f)))

    if not entries:
        print(f"❌ x={x_value} için görüntü bulunamadı!")
        return

    entries.sort(key=lambda t: t[0])
    n = len(entries)
    print(f"✅ {n} görüntü  (x={x_value}, y: {entries[0][0]:.1f} → {entries[-1][0]:.1f})")

    first = cv2.imread(entries[0][1])
    if first is None:
        print("❌ İlk görüntü okunamadı.")
        return

    h_ref, w_ref = first.shape[:2]
    print(f"Boyut: {w_ref}x{h_ref} | Pyramid levels: {pyramid_levels}\n")

    canvas = first.astype(np.float32)
    skipped = 0

    for i, (y_val, fpath) in enumerate(entries[1:], start=1):
        img_new = cv2.imread(fpath)
        if img_new is None:
            print(f"  [{i:02d}/{n-1}] ⚠️  Atlandı (okunamadı)")
            skipped += 1
            continue

        # Kanvasın sağ kenarından crop al
        canvas_w = canvas.shape[1]
        crop_start = max(0, canvas_w - w_ref)
        canvas_crop = canvas[:, crop_start:].astype(np.uint8)

        # Hizalama
        (tx, ty), method = find_alignment(canvas_crop, img_new,
                                          initial_tx_ratio, sift_fallback)

        img_new_start = crop_start + int(round(tx))
        overlap_start = img_new_start
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

        # Multiband blend ile birleştir
        canvas = merge_into_canvas(
            canvas, img_new.astype(np.float32),
            overlap_start, overlap_px, h_ref, w_ref, pyramid_levels
        )

        print(f"  [{i:02d}/{n-1}] y={y_val:7.1f}mm | {method:7s} | "
              f"tx={tx:7.1f}px ty={ty:+6.1f}px | "
              f"örtüşme={overlap_px:4d}px | kanvas={canvas.shape[1]:6d}px")

    # Kaydet
    result = canvas.clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_path, result)
    print(f"\n✅ Kaydedildi : {output_path}")
    print(f"   Final boyut : {result.shape[1]} x {result.shape[0]} px")
    print(f"   Atlanan     : {skipped}/{n-1}")
    return result


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Panoramik satır stitch — ECC/SIFT + Laplacian multiband blending")
    parser.add_argument("--input_dir",     required=True,  help="Görüntü klasörü")
    parser.add_argument("--x_value",       type=float, required=True,
                        help="Satırın x koordinatı (ör: 215.0)")
    parser.add_argument("--output",        default="panorama.jpg")
    parser.add_argument("--tx_ratio",      type=float, default=0.8,
                        help="ECC başlangıç tx oranı (varsayılan: 0.8)")
    parser.add_argument("--sift_fallback", action="store_true",
                        help="ECC başarısızsa SIFT dene")
    parser.add_argument("--levels",        type=int, default=4,
                        help="Laplacian pyramid seviyeleri (varsayılan: 4)")
    args = parser.parse_args()

    stitch_panoramic(
        input_dir=args.input_dir,
        x_value=args.x_value,
        output_path=args.output,
        initial_tx_ratio=args.tx_ratio,
        sift_fallback=args.sift_fallback,
        pyramid_levels=args.levels,
    )

if __name__ == "__main__":
    main()
