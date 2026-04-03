import cv2
import numpy as np
from pathlib import Path

# ============================================================
#  AYARLAR
# ============================================================
IMAGE_FOLDER  = r"C:\verilerim\goruntüler"
OUTPUT_PATH   = r"C:\verilerim\test_birlesim.png"
FILE_EXT      = ".png"
OVERLAP_RATIO = 0.15    # Başlangıç tahmini — görüntülerinize göre ayarlayın
ECC_ITERATIONS = 1000
ECC_EPSILON    = 1e-6
FEATHER_WIDTH  = 20     # Seam kenarı yumuşatma genişliği (piksel)
# ============================================================


def load_two_images(folder, ext):
    files = sorted(Path(folder).glob(f"*{ext}"))[:2]
    if len(files) < 2:
        raise ValueError(f"Klasörde en az 2 görüntü olmalı: {folder}")
    img0 = cv2.imread(str(files[0]))
    img1 = cv2.imread(str(files[1]))
    if img0 is None or img1 is None:
        raise ValueError("Görüntü okunamadı.")
    print(f"Yüklendi:\n  [0] {files[0].name}  {img0.shape}\n  [1] {files[1].name}  {img1.shape}")
    return img0, img1


def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_translation_ecc(img_ref, img_mov,
                              overlap_ratio=0.15,
                              n_iter=1000, epsilon=1e-6):
    h, w = img_ref.shape[:2]
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = w * (1.0 - overlap_ratio)   # tx başlangıç tahmini

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, epsilon)

    try:
        _, warp_matrix = cv2.findTransformECC(
            to_gray32(img_ref), to_gray32(img_mov),
            warp_matrix, cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5
        )
        print(f"ECC başarılı  →  tx={warp_matrix[0,2]:.2f}  ty={warp_matrix[1,2]:.2f}")
    except cv2.error as e:
        print(f"ECC hata: {e}\nBaşlangıç tahmini kullanılıyor.")

    return float(warp_matrix[0, 2]), float(warp_matrix[1, 2])


# ──────────────────────────────────────────────
#  MINIMUM COST SEAM  (DP)
# ──────────────────────────────────────────────

def find_min_cost_seam(overlap_left, overlap_right):
    """
    İki örtüşen şerit arasında renk farkının minimum olduğu
    dikey yolu DP ile bul.
    Dönüş: seam_col (H,) — her satır için sütun indeksi
    """
    diff = (overlap_left.astype(np.float32) -
            overlap_right.astype(np.float32)) ** 2
    cost = diff.sum(axis=2)          # (H, W)
    H, W = cost.shape

    dp   = cost.copy()
    back = np.zeros((H, W), dtype=np.int32)   # geri iz için

    for r in range(1, H):
        for c in range(W):
            c_l = max(c - 1, 0)
            c_r = min(c + 1, W - 1)
            choices = [dp[r-1, c_l], dp[r-1, c], dp[r-1, c_r]]
            best    = np.argmin(choices)
            dp[r, c]   += choices[best]
            back[r, c]  = c + (best - 1)   # -1, 0, +1 offset

    # Geri iz sür
    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        seam[r] = int(np.clip(back[r + 1, seam[r + 1]], 0, W - 1))

    return seam


def blend_with_seam(overlap_left, overlap_right, feather_width=20):
    """
    Seam boyunca feathering uygula.
    Seam solunda overlap_left, sağında overlap_right ağırlıklı.
    """
    H, W = overlap_left.shape[:2]
    seam = find_min_cost_seam(overlap_left, overlap_right)

    # Her satır için seam'e uzaklık bazlı yumuşak maske
    col_idx = np.tile(np.arange(W, dtype=np.float32), (H, 1))   # (H, W)
    seam_2d = seam[:, np.newaxis].astype(np.float32)             # (H, 1)

    dist = col_idx - seam_2d                                      # negatif=sol, pozitif=sağ
    # Sigmoid benzeri geçiş
    alpha = np.clip(dist / max(feather_width, 1) * 0.5 + 0.5,
                    0.0, 1.0)[:, :, np.newaxis]                  # (H, W, 1)

    blended = (overlap_left.astype(np.float32)  * (1.0 - alpha) +
               overlap_right.astype(np.float32) * alpha).astype(np.uint8)
    return blended, seam


def stitch_two(img_ref, img_mov, tx, ty, feather_width=20):
    """
    img_ref (canvas) + img_mov'u optimal seam ile birleştir.
    Boş alanlar siyah kalır.
    """
    tx_i = int(round(tx))
    ty_i = int(round(ty))

    rh, rw = img_ref.shape[:2]
    mh, mw = img_mov.shape[:2]

    # ── Yeni canvas boyutu ──────────────────────────────────
    new_w = max(rw, tx_i + mw)
    new_h = max(rh, ty_i + mh)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # img_ref'i yerleştir
    canvas[:rh, :rw] = img_ref

    # ── Örtüşme bölgesi (x ekseninde) ──────────────────────
    ov_x0 = tx_i           # örtüşme başlangıcı (canvas koordinatı)
    ov_x1 = min(rw, tx_i + mw)   # örtüşme bitişi
    ov_w  = ov_x1 - ov_x0

    # img_mov'un canvas içindeki y aralığı
    mov_y0 = max(ty_i, 0)
    mov_y1 = ty_i + mh

    print(f"Örtüşme genişliği: {ov_w} piksel  "
          f"(x: {ov_x0}–{ov_x1},  y: {mov_y0}–{mov_y1})")

    if ov_w <= 0:
        # Örtüşme yok — doğrudan yapıştır
        print("Uyarı: Örtüşme tespit edilemedi, görüntü doğrudan yapıştırılıyor.")
        canvas[mov_y0:mov_y1, tx_i:tx_i+mw] = img_mov
        return canvas

    # ── Örtüşen şeritleri çıkar ────────────────────────────
    # Canvas (img_ref) tarafındaki şerit
    ov_h          = mov_y1 - mov_y0
    strip_ref     = canvas[mov_y0:mov_y1, ov_x0:ov_x1].copy()   # (ov_h, ov_w, 3)

    # img_mov tarafındaki şerit (img_mov'un solundan ov_w kadar)
    strip_mov     = img_mov[:ov_h, :ov_w].copy()                 # (ov_h, ov_w, 3)

    # ── Seam blend ─────────────────────────────────────────
    blended, seam = blend_with_seam(strip_ref, strip_mov, feather_width)
    canvas[mov_y0:mov_y1, ov_x0:ov_x1] = blended

    # ── img_mov'un örtüşme dışı (sağ) kısmını yapıştır ────
    right_mov_x  = ov_w           # img_mov koordinatında
    right_can_x  = ov_x1          # canvas koordinatında
    right_w      = mw - right_mov_x

    if right_w > 0:
        canvas[mov_y0:mov_y1,
               right_can_x:right_can_x + right_w] = img_mov[:ov_h, right_mov_x:]

    # ── Seam çizgisini debug için kırmızıya boya (isterseniz kapatın) ──
    DEBUG_SEAM = True   # False yaparsanız seam çizgisi görünmez
    if DEBUG_SEAM:
        for r, c in enumerate(seam):
            cx = ov_x0 + c
            cy = mov_y0 + r
            if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1]:
                canvas[cy, cx] = [0, 0, 255]   # kırmızı

    return canvas


def main():
    print("=" * 55)
    print("Test: İlk 2 Görüntü — Optimal Seam Birleştirme")
    print("=" * 55)

    img0, img1 = load_two_images(IMAGE_FOLDER, FILE_EXT)

    tx, ty = estimate_translation_ecc(img0, img1,
                                       overlap_ratio=OVERLAP_RATIO,
                                       n_iter=ECC_ITERATIONS,
                                       epsilon=ECC_EPSILON)

    result = stitch_two(img0, img1, tx, ty, feather_width=FEATHER_WIDTH)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n✓ Kaydedildi: {OUTPUT_PATH}")
    print(f"  Boyut: {result.shape[1]} x {result.shape[0]} piksel")

    # Ekranda önizleme (isteğe bağlı)
    preview = cv2.resize(result, (min(1600, result.shape[1]),
                                   min(900,  result.shape[0])))
    cv2.imshow("Sonuc - ESC ile kapat", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
