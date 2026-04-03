import cv2
import numpy as np
from pathlib import Path

# ============================================================
#  AYARLAR
# ============================================================
IMAGE_FOLDER   = r"C:\verilerim\goruntüler"
OUTPUT_PATH    = r"C:\verilerim\test_birlesim.png"
FILE_EXT       = ".png"
ECC_ITERATIONS = 1000
ECC_EPSILON    = 1e-6
FEATHER_WIDTH  = 20

OVERLAP_MIN   = 0.05
OVERLAP_MAX   = 0.50
OVERLAP_STEPS = 20

# "ltr" → soldan sağa (1-2-3-4)
# "rtl" → sağdan sola (4-3-2-1)
SCAN_DIRECTION = "rtl"

DEBUG_SEAM = True   # False yaparsanız kırmızı çizgi görünmez
# ============================================================


def load_two_images(folder, ext, direction):
    files = sorted(Path(folder).glob(f"*{ext}"))[:2]
    if len(files) < 2:
        raise ValueError(f"En az 2 görüntü gerekli: {folder}")
    if direction == "rtl":
        files = list(reversed(files))
        print("Tarama yönü: SAĞDAN SOLA")
    else:
        print("Tarama yönü: SOLDAN SAĞA")
    img0 = cv2.imread(str(files[0]))
    img1 = cv2.imread(str(files[1]))
    print(f"  [0] {files[0].name}  {img0.shape}")
    print(f"  [1] {files[1].name}  {img1.shape}")
    return img0, img1


def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  NCC KABA ARAMA
# ──────────────────────────────────────────────────────────────

def ncc_score(strip_a, strip_b):
    """İki şerit arasında normalize çapraz korelasyon."""
    a = to_gray32(strip_a)
    b = to_gray32(strip_b)
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    an = a - a.mean()
    bn = b - b.mean()
    num   = (an * bn).sum()
    denom = np.sqrt((an**2).sum() * (bn**2).sum()) + 1e-10
    return float(num / denom)


def coarse_search(img_ref, img_mov, direction):
    """
    LTR: img_ref'in SAĞ şeridi ↔ img_mov'un SOL şeridi
    RTL: img_ref'in SOL şeridi ↔ img_mov'un SAĞ şeridi
    """
    h, w = img_ref.shape[:2]
    ov_range = np.linspace(int(w * OVERLAP_MIN),
                            int(w * OVERLAP_MAX),
                            OVERLAP_STEPS, dtype=int)
    scores = []
    print(f"\n[Kaba Arama - NCC ({direction.upper()})]")
    for ov in ov_range:
        if direction == "ltr":
            strip_a = img_ref[:, w - ov:]    # img_ref sağ kenarı
            strip_b = img_mov[:, :ov]         # img_mov sol kenarı
        else:
            strip_a = img_ref[:, :ov]         # img_ref sol kenarı
            strip_b = img_mov[:, w - ov:]     # img_mov sağ kenarı
        s = ncc_score(strip_a, strip_b)
        scores.append((ov, s))
        print(f"  overlap={ov:4d}px  NCC={s:.4f}")
    best_ov, best_s = max(scores, key=lambda x: x[1])
    print(f"→ En iyi overlap={best_ov}px  NCC={best_s:.4f}")
    return best_ov


# ──────────────────────────────────────────────────────────────
#  ECC İNCE AYAR
# ──────────────────────────────────────────────────────────────

def refine_with_ecc(img_ref, img_mov, overlap_init, direction,
                     search_range=30):
    h, w  = img_ref.shape[:2]
    mh, mw = img_mov.shape[:2]

    warp = np.eye(2, 3, dtype=np.float32)
    if direction == "ltr":
        tx_init = float(w - overlap_init)      # pozitif
    else:
        tx_init = float(-(mw - overlap_init))  # negatif

    warp[0, 2] = tx_init
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_ITERATIONS, ECC_EPSILON)
    try:
        _, warp_out = cv2.findTransformECC(
            to_gray32(img_ref), to_gray32(img_mov),
            warp.copy(), cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5
        )
        tx_r = float(warp_out[0, 2])
        ty_r = float(warp_out[1, 2])
        if abs(tx_r - tx_init) > search_range:
            print(f"ECC sınır dışı ({tx_r:.1f}) → kaba tahmin kullanılıyor.")
            return tx_init, 0.0
        print(f"ECC → tx={tx_r:.2f}  ty={ty_r:.2f}")
        return tx_r, ty_r
    except cv2.error as e:
        print(f"ECC hata: {e} → kaba tahmin kullanılıyor.")
        return tx_init, 0.0


def estimate_translation(img_ref, img_mov, direction):
    best_ov = coarse_search(img_ref, img_mov, direction)
    print(f"\n[İnce Ayar - ECC ({direction.upper()})]")
    tx, ty  = refine_with_ecc(img_ref, img_mov, best_ov, direction)

    mw = img_mov.shape[1]
    rw = img_ref.shape[1]
    if direction == "ltr":
        overlap_px = rw - int(round(tx))
    else:
        overlap_px = mw + int(round(tx))   # tx negatif

    print(f"\n✓ Final → tx={tx:.2f}  ty={ty:.2f}  overlap={overlap_px}px "
          f"({overlap_px / rw * 100:.1f}%)")
    return tx, ty


# ──────────────────────────────────────────────────────────────
#  MİNİMUM COST SEAM  (DP)
# ──────────────────────────────────────────────────────────────

def find_min_cost_seam(strip_left, strip_right):
    """
    strip_left  : örtüşme bölgesinin sol görüntüden gelen kısmı
    strip_right : örtüşme bölgesinin sağ görüntüden gelen kısmı
    Seam solu strip_left'e, sağı strip_right'a atar.
    """
    diff = (strip_left.astype(np.float32) -
            strip_right.astype(np.float32)) ** 2
    cost = diff.sum(axis=2)   # (H, W)
    H, W = cost.shape

    dp   = cost.copy()
    back = np.zeros((H, W), dtype=np.int32)

    for r in range(1, H):
        for c in range(W):
            cl = max(c - 1, 0)
            cr = min(c + 1, W - 1)
            opts = [dp[r-1, cl], dp[r-1, c], dp[r-1, cr]]
            best = int(np.argmin(opts))
            dp[r, c]  += opts[best]
            back[r, c] = c + (best - 1)   # -1, 0, +1

    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        seam[r] = int(np.clip(back[r + 1, seam[r + 1]], 0, W - 1))
    return seam


def blend_seam(strip_left, strip_right, feather_width=20):
    """Seam boyunca feathering ile blend."""
    seam  = find_min_cost_seam(strip_left, strip_right)
    H, W  = strip_left.shape[:2]
    cols  = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    s2d   = seam[:, np.newaxis].astype(np.float32)
    alpha = np.clip((cols - s2d) / max(feather_width, 1) * 0.5 + 0.5,
                    0.0, 1.0)[:, :, np.newaxis]
    blended = (strip_left.astype(np.float32)  * (1.0 - alpha) +
               strip_right.astype(np.float32) * alpha).astype(np.uint8)
    return blended, seam


# ──────────────────────────────────────────────────────────────
#  BİRLEŞTİRME — LTR ve RTL
# ──────────────────────────────────────────────────────────────

def stitch_ltr(img_ref, img_mov, tx, ty, feather_width, debug_seam):
    """img_mov, img_ref'in SAĞINA eklenir."""
    tx_i = int(round(tx))
    ty_i = int(round(ty))
    rh, rw = img_ref.shape[:2]
    mh, mw = img_mov.shape[:2]

    new_w  = max(rw, tx_i + mw)
    new_h  = max(rh, ty_i + mh)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    canvas[:rh, :rw] = img_ref

    ov_x0  = tx_i
    ov_x1  = min(rw, tx_i + mw)
    ov_w   = ov_x1 - ov_x0
    mov_y0 = max(ty_i, 0)
    mov_y1 = min(ty_i + mh, new_h)
    ov_h   = mov_y1 - mov_y0

    print(f"LTR örtüşme: x=[{ov_x0},{ov_x1}]  {ov_w}x{ov_h}px")

    if ov_w <= 0:
        canvas[mov_y0:mov_y1, tx_i:tx_i + mw] = img_mov
        return canvas

    strip_left  = canvas[mov_y0:mov_y1, ov_x0:ov_x1].copy()  # img_ref sağı
    strip_right = img_mov[:ov_h, :ov_w].copy()                # img_mov solu

    blended, seam = blend_seam(strip_left, strip_right, feather_width)
    canvas[mov_y0:mov_y1, ov_x0:ov_x1] = blended

    right_w = mw - ov_w
    if right_w > 0:
        canvas[mov_y0:mov_y1, ov_x1:ov_x1 + right_w] = img_mov[:ov_h, ov_w:]

    if debug_seam:
        for r, c in enumerate(seam):
            cx, cy = ov_x0 + c, mov_y0 + r
            if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1]:
                canvas[cy, cx] = [0, 0, 255]

    return canvas


def stitch_rtl(img_ref, img_mov, tx, ty, feather_width, debug_seam):
    """
    img_mov, img_ref'in SOLUNA eklenir.
    tx negatif → img_mov canvas'ın solunda, img_ref sağında.
    """
    tx_i = int(round(tx))   # negatif
    ty_i = int(round(ty))
    rh, rw = img_ref.shape[:2]
    mh, mw = img_mov.shape[:2]

    # Canvas düzeni:
    #  x=0                    x=ref_x0          x=new_w
    #  |<── img_mov ─────────>|<── img_ref ────>|
    ref_x0 = -tx_i            # img_ref canvas'ta buradan başlar (pozitif)
    new_w  = max(ref_x0 + rw, mw)
    new_h  = max(rh, mh)
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Önce her iki görüntüyü yerleştir
    canvas[:mh, :mw]              = img_mov           # sol
    canvas[:rh, ref_x0:ref_x0+rw] = img_ref           # sağ (üzerine yazar, örtüşme sonra blend edilir)

    # Örtüşme bölgesi
    ov_x0 = ref_x0             # örtüşme canvas'ta buradan başlar
    ov_x1 = min(mw, ref_x0 + rw)
    ov_w  = ov_x1 - ov_x0
    ov_h  = min(rh, mh)

    print(f"RTL örtüşme: x=[{ov_x0},{ov_x1}]  {ov_w}x{ov_h}px")

    if ov_w <= 0:
        return canvas

    # strip_left  = img_mov'un SAĞ kenarı (canvas sol tarafta)
    # strip_right = img_ref'in SOL kenarı (canvas sağ tarafta)
    strip_left  = img_mov[:ov_h, mw - ov_w:].copy()   # img_mov sağ ov_w pikseli
    strip_right = img_ref[:ov_h, :ov_w].copy()         # img_ref sol ov_w pikseli

    blended, seam = blend_seam(strip_left, strip_right, feather_width)
    canvas[:ov_h, ov_x0:ov_x1] = blended

    if debug_seam:
        for r, c in enumerate(seam):
            cx, cy = ov_x0 + c, r
            if 0 <= cy < canvas.shape[0] and 0 <= cx < canvas.shape[1]:
                canvas[cy, cx] = [0, 0, 255]

    return canvas


def stitch_two(img_ref, img_mov, tx, ty, direction, feather_width, debug_seam):
    if direction == "ltr":
        return stitch_ltr(img_ref, img_mov, tx, ty, feather_width, debug_seam)
    else:
        return stitch_rtl(img_ref, img_mov, tx, ty, feather_width, debug_seam)


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Minimum Cost Seam  |  "
          f"{'SAĞDAN SOLA (RTL)' if SCAN_DIRECTION == 'rtl' else 'SOLDAN SAĞA (LTR)'}")
    print("=" * 60)

    img0, img1 = load_two_images(IMAGE_FOLDER, FILE_EXT, SCAN_DIRECTION)
    tx, ty     = estimate_translation(img0, img1, SCAN_DIRECTION)

    result = stitch_two(img0, img1, tx, ty,
                         direction=SCAN_DIRECTION,
                         feather_width=FEATHER_WIDTH,
                         debug_seam=DEBUG_SEAM)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n✓ Kaydedildi: {OUTPUT_PATH}")
    print(f"  Boyut: {result.shape[1]} x {result.shape[0]} piksel")

    preview = cv2.resize(result, (min(1600, result.shape[1]),
                                   min(900,  result.shape[0])))
    cv2.imshow("Sonuc - ESC", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
