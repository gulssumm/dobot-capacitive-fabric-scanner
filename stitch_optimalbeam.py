import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
#  AYARLAR - Buradan düzenleyin
# ============================================================
IMAGE_FOLDER   = r"C:\verilerim\goruntüler"   # Görüntü klasörü
OUTPUT_PATH    = r"C:\verilerim\panorama.png"  # Çıktı dosyası
NUM_IMAGES     = 41                            # İlk satır görüntü sayısı
FILE_EXT       = ".png"                        # Uzantı (.png, .jpg, .tif)
OVERLAP_RATIO  = 0.15   # Beklenen örtüşme oranı (~%15)
ECC_ITERATIONS = 1000
ECC_EPSILON    = 1e-6
WARP_MODE      = cv2.MOTION_TRANSLATION
# ============================================================


def load_images(folder, n, ext):
    files = sorted(Path(folder).glob(f"*{ext}"))[:n]
    if len(files) < n:
        print(f"Uyarı: {len(files)} görüntü bulundu, {n} bekleniyor.")
    images = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Görüntü okunamadı: {f}")
        images.append(img)
    print(f"{len(images)} görüntü yüklendi.")
    return images


def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_translation_ecc(img_ref, img_mov, overlap_ratio=0.15,
                              warp_mode=cv2.MOTION_TRANSLATION,
                              n_iter=1000, epsilon=1e-6):
    """ECC ile sadece öteleme matrisini bul, (tx, ty) döndür."""
    h, w = img_ref.shape[:2]
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = w * (1.0 - overlap_ratio)

    gray_ref = to_gray32(img_ref)
    gray_mov = to_gray32(img_mov)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, epsilon)

    try:
        _, warp_matrix = cv2.findTransformECC(
            gray_ref, gray_mov,
            warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5
        )
    except cv2.error as e:
        print(f"  ECC uyarı: {e} — başlangıç tahmini kullanılıyor.")

    return float(warp_matrix[0, 2]), float(warp_matrix[1, 2])


# ------------------------------------------------------------------
#  MINIMUM COST SEAM  (dynamic programming)
# ------------------------------------------------------------------

def compute_seam_cost(overlap_left, overlap_right):
    """
    İki örtüşen bölge arasındaki fark görüntüsünden
    dikey minimum cost seam hesapla.

    overlap_left  : canvas'tan alınan örtüşme şeridi  (H x W x 3)
    overlap_right : yeni görüntüden alınan örtüşme şeridi (H x W x 3)
    Dönüş: seam_col (H,) — her satır için sütun indeksi
    """
    diff = (overlap_left.astype(np.float32) - overlap_right.astype(np.float32)) ** 2
    cost_map = diff.sum(axis=2)  # (H x W) — piksel başına maliyet

    H, W = cost_map.shape
    dp = cost_map.copy()

    # Yukarıdan aşağıya DP — her hücreden 3 komşuya geç
    for row in range(1, H):
        for col in range(W):
            left  = dp[row-1, max(col-1, 0)]
            mid   = dp[row-1, col]
            right = dp[row-1, min(col+1, W-1)]
            dp[row, col] += min(left, mid, right)

    # En düşük maliyetli sütunu son satırdan başlayarak geri iz sür
    seam_col = np.zeros(H, dtype=np.int32)
    seam_col[-1] = int(np.argmin(dp[-1]))

    for row in range(H-2, -1, -1):
        prev = seam_col[row+1]
        candidates = range(max(prev-1, 0), min(prev+2, W))
        seam_col[row] = min(candidates, key=lambda c: dp[row, c])

    return seam_col


def apply_seam_blend(canvas, next_img, tx, ty,
                     feather_width=15):
    """
    canvas ile next_img'i optimal seam ile birleştir.

    1. Örtüşen bölgeyi tespit et.
    2. DP ile minimum cost seam bul (organik dikey sınır).
    3. Seam boyunca soft feather maskesi uygula.
    4. Seam solunda canvas, sağında next_img kullan.
    5. Her iki tarafta da örtüşmeyen alanlar korunur / siyah kalır.
    """
    tx_int = int(round(tx))
    ty_int = int(round(ty))

    ch, cw = canvas.shape[:2]
    nh, nw = next_img.shape[:2]

    # Yeni canvas boyutları
    new_w = max(cw, tx_int + nw)
    new_h = max(ch, ty_int + nh)
    new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Mevcut canvas'ı yeni canvas'a kopyala
    new_canvas[:ch, :cw] = canvas

    # next_img'in canvas içindeki koordinatları
    x0 = tx_int          # next_img sol kenarı
    y0 = max(ty_int, 0)
    x1 = tx_int + nw     # next_img sağ kenarı
    y1 = ty_int + nh

    # Örtüşme bölgesi (x ekseninde)
    overlap_x_start = x0           # = tx_int
    overlap_x_end   = min(cw, x1)  # canvas sağ kenarı ile next_img sağı

    if overlap_x_end <= overlap_x_start:
        # Örtüşme yok — direkt yapıştır
        new_canvas[y0:y0+nh, x0:x0+nw] = next_img
        return new_canvas

    overlap_w = overlap_x_end - overlap_x_start  # piksel genişliği

    # Örtüşen şeritleri çıkar
    # Canvas tarafı
    ov_canvas = new_canvas[y0:y0+nh, overlap_x_start:overlap_x_end].copy()
    # next_img tarafı (next_img koordinatında overlap başlangıcı = 0)
    ov_next   = next_img[:nh, :overlap_w].copy()

    # Minimum cost seam bul
    seam_col = compute_seam_cost(ov_canvas, ov_next)  # (H,)

    # Seam maskesi: 0.0 = canvas kullan, 1.0 = next_img kullan
    # Seam solunda canvas, sağında next_img; feather ile yumuşat
    H_ov = ov_canvas.shape[0]
    mask = np.zeros((H_ov, overlap_w), dtype=np.float32)

    for row in range(H_ov):
        sc = seam_col[row]
        # Seam sağındaki pikseller next_img'e ait
        col_range = np.arange(overlap_w, dtype=np.float32)
        # Sert sınır
        hard = (col_range >= sc).astype(np.float32)
        # Feathering: seam etrafında ±feather_width piksel yumuşat
        dist = col_range - sc
        soft = np.clip(dist / max(feather_width, 1) * 0.5 + 0.5, 0.0, 1.0)
        mask[row] = soft

    mask3 = mask[:, :, np.newaxis]  # (H, W, 1) — broadcast için

    # Örtüşen bölgeyi blend et
    blended = (ov_canvas.astype(np.float32) * (1.0 - mask3) +
               ov_next.astype(np.float32)  * mask3).astype(np.uint8)

    new_canvas[y0:y0+H_ov, overlap_x_start:overlap_x_end] = blended

    # next_img'in örtüşme dışında kalan sağ kısmını yapıştır
    right_start_in_next   = overlap_w
    right_start_in_canvas = overlap_x_end
    right_w = nw - right_start_in_next

    if right_w > 0:
        new_canvas[y0:y0+nh,
                   right_start_in_canvas:right_start_in_canvas+right_w] = \
            next_img[:, right_start_in_next:]

    return new_canvas


# ------------------------------------------------------------------
#  ANA DÖNGÜ
# ------------------------------------------------------------------

def build_panorama(images, overlap_ratio, warp_mode, n_iter, epsilon):
    canvas = images[0].copy()

    for i in range(1, len(images)):
        print(f"  [{i}/{len(images)-1}] Görüntü {i} hizalanıyor...")

        tx, ty = estimate_translation_ecc(
            images[i-1], images[i],
            overlap_ratio=overlap_ratio,
            warp_mode=warp_mode,
            n_iter=n_iter,
            epsilon=epsilon
        )
        print(f"         tx={tx:.2f}  ty={ty:.2f}")

        canvas = apply_seam_blend(canvas, images[i], tx, ty,
                                   feather_width=20)
        print(f"         Canvas: {canvas.shape[1]}x{canvas.shape[0]}")

    return canvas


def main():
    print("=" * 55)
    print("Panoramik Birleştirme — Graph Cut / Optimal Seam")
    print("=" * 55)

    images = load_images(IMAGE_FOLDER, NUM_IMAGES, FILE_EXT)

    print("\nHizalama ve birleştirme başlıyor...")
    panorama = build_panorama(
        images,
        overlap_ratio=OVERLAP_RATIO,
        warp_mode=WARP_MODE,
        n_iter=ECC_ITERATIONS,
        epsilon=ECC_EPSILON
    )

    cv2.imwrite(OUTPUT_PATH, panorama)
    print(f"\n✓ Panorama kaydedildi: {OUTPUT_PATH}")
    print(f"  Boyut: {panorama.shape[1]} x {panorama.shape[0]} piksel")


if __name__ == "__main__":
    main()
