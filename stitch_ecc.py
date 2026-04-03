"""
Tek Satır Zincir Stitch
=======================
Aynı x değerine sahip tüm görüntüleri (y sırasıyla) soldan sağa zincir halinde birleştirir.
- Her adımda: mevcut kanvas'ın SAĞ KENARI + yeni görüntü → ECC → örtüşen kısım kırpılır
- 1. görüntünün tamamı korunur, sonraki her görüntüden sadece yeni alan eklenir

Kullanım:
    python stitch_row.py --input_dir ./images --x_value 215.0 --output row_215.jpg

Gereksinimler:
    pip install opencv-python numpy
"""

import re
import cv2
import numpy as np
import argparse
from pathlib import Path


# ─────────────────────────────────────────
# DOSYA ADI PARSER
# ─────────────────────────────────────────

def parse_filename(filename):
    """img_XXXX_xNNN_N_yMMM_M.jpg → (x, y)"""
    stem = Path(filename).stem
    match = re.search(r'_x(-?\d+)_(\d+)_y(-?\d+)_(\d+)', stem)
    if match:
        x = float(f"{match.group(1)}.{match.group(2)}")
        y = float(f"{match.group(3)}.{match.group(4)}")
        return x, y
    return None, None


# ─────────────────────────────────────────
# ECC OFSETİ BUL
# ─────────────────────────────────────────

def find_tx_ty(img_left, img_right, initial_tx_ratio=0.8):
    """
    img_right'ın img_left'e göre (tx, ty) piksel ofsetini ECC ile bulur.
    img_left olarak kanvasın sağ kenar crop'unu kullanırız.
    """
    h, w = img_left.shape[:2]
    g1 = cv2.cvtColor(img_left,  cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = w * initial_tx_ratio

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    try:
        _, warp = cv2.findTransformECC(g1, g2, warp, cv2.MOTION_TRANSLATION, criteria)
        return float(warp[0, 2]), float(warp[1, 2])
    except cv2.error as e:
        print(f"    ⚠️  ECC başarısız: {e}")
        return w * initial_tx_ratio, 0.0


# ─────────────────────────────────────────
# SATIR STİTCH
# ─────────────────────────────────────────

def stitch_row(input_dir, x_value, output_path, initial_tx_ratio=0.8):

    # 1. Bu x'e ait görüntüleri bul ve y'ye göre sırala
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
    print(f"✅ {n} görüntü bulundu  (x={x_value}, y: {entries[0][0]:.1f} → {entries[-1][0]:.1f})")

    # 2. İlk görüntüyü kanvas olarak başlat
    canvas = cv2.imread(entries[0][1])
    if canvas is None:
        print(f"❌ İlk görüntü okunamadı: {entries[0][1]}")
        return

    h_ref, w_ref = canvas.shape[:2]
    print(f"Görüntü boyutu: {w_ref} x {h_ref} px\n")

    # 3. Her yeni görüntüyü zincir halinde ekle
    for i, (y_val, fpath) in enumerate(entries[1:], start=1):
        img_new = cv2.imread(fpath)
        if img_new is None:
            print(f"  [{i}/{n-1}]   Atlandı (okunamadı): {fpath}")
            continue

        # ECC için kanvasın sağ kenarından 1 görüntü genişliği kadar crop al
        canvas_w = canvas.shape[1]
        crop_start = max(0, canvas_w - w_ref)
        canvas_right_crop = canvas[:, crop_start:]   # kanvasın sağ kısmı

        # img_new ile bu crop arasında ECC
        tx, ty = find_tx_ty(canvas_right_crop, img_new, initial_tx_ratio)

        # tx, crop içindeki ofseti veriyor.
        # Kanvas içindeki gerçek başlangıç pozisyonu:
        img_new_start_in_canvas = crop_start + int(round(tx))

        # img_new'den kırpılacak overlap miktarı
        new_part_start = canvas_w - img_new_start_in_canvas

        if new_part_start <= 0:
            print(f"  [{i}/{n-1}]   Örtüşme bulunamadı, görüntü atlandı.")
            continue
        if new_part_start >= w_ref:
            print(f"  [{i}/{n-1}]   Örtüşme tüm görüntüyü kaplıyor, atlandı.")
            continue

        # Dikey kaymayı düzelt
        ty_int = int(round(ty))
        if abs(ty_int) > 0:
            M = np.float32([[1, 0, 0], [0, 1, -ty_int]])
            img_new = cv2.warpAffine(img_new, M, (w_ref, h_ref),
                                     borderMode=cv2.BORDER_REPLICATE)

        # img_new'den sadece yeni kısım
        img_new_part = img_new[:, new_part_start:]
        new_cols = img_new_part.shape[1]

        # Kanvası genişlet
        extension = np.zeros((h_ref, new_cols, 3), dtype=np.uint8)
        canvas = np.concatenate([canvas, extension], axis=1)
        canvas[:, canvas_w:canvas_w + new_cols] = img_new_part

        print(f"  [{i:02d}/{n-1}] y={y_val:.1f}mm | tx={tx:.1f}px ty={ty:.1f}px | "
              f"örtüşme={new_part_start}px yeni={new_cols}px | "
              f"kanvas={canvas.shape[1]}px")

    # 4. Kaydet
    cv2.imwrite(output_path, canvas)
    print(f"\n✅ Kaydedildi: {output_path}")
    print(f"   Final boyut: {canvas.shape[1]} x {canvas.shape[0]} px")
    return canvas


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tek satır zincir stitch")
    parser.add_argument("--input_dir",  required=True,  help="Görüntü klasörü")
    parser.add_argument("--x_value",    type=float, required=True,
                        help="Satırın x koordinatı (ör: 215.0)")
    parser.add_argument("--output",     default="row_stitch.jpg",
                        help="Çıktı dosyası")
    parser.add_argument("--tx_ratio",   type=float, default=0.8,
                        help="ECC başlangıç tx tahmini oranı (varsayılan: 0.8)")
    args = parser.parse_args()

    stitch_row(
        input_dir=args.input_dir,
        x_value=args.x_value,
        output_path=args.output,
        initial_tx_ratio=args.tx_ratio,
    )

if __name__ == "__main__":
    main()