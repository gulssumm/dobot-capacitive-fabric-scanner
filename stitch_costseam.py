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

# "ltr" → dosyalar 1,2,3,4... sırasıyla, her biri sağa eklenir
# "rtl" → dosyalar ters sırayla (4,3,2,1) yüklenir, her biri yine sağa eklenir
#          çünkü tarama sağdan sola gittiği için son çekilen görüntü en soldadır
SCAN_DIRECTION = "rtl"

DEBUG_SEAM = True
# ============================================================


def load_images(folder, ext, direction, n=2):
    """
    LTR: [img1, img2, img3 ...]   → panoramada soldan sağa
    RTL: [img4, img3, img2, img1] → tarama tersi = panoramada soldan sağa
         (son çekilen = en sol, ilk çekilen = en sağ)
    """
    files = sorted(Path(folder).glob(f"*{ext}"))[:n]
    if len(files) < 2:
        raise ValueError(f"En az 2 görüntü gerekli: {folder}")

    if direction == "rtl":
        files = list(reversed(files))
        print("Tarama: SAĞDAN SOLA  →  dosyalar ters sırayla yüklendi")
        print("Panorama düzeni: son çekilen(sol) ... ilk çekilen(sağ)")
    else:
        print("Tarama: SOLDAN SAĞA")

    images = []
    for i, f in enumerate(files):
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Okunamadı: {f}")
        print(f"  [{i}] {f.name}  {img.shape}")
        images.append(img)
    return images


def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  NCC KABA ARAMA
#  Her iki yönde de: canvas'ın SAĞ kenarı ↔ yeni görüntünün SOL kenarı
# ──────────────────────────────────────────────────────────────

def ncc_score(strip_a, strip_b):
    a = to_gray32(strip_a)
    b = to_gray32(strip_b)
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    an = a - a.mean()
    bn = b - b.mean()
    num   = (an * bn).sum()
    denom = np.sqrt((an**2).sum() * (bn**2).sum()) + 1e-10
    return float(num / denom)


def coarse_search(img_ref, img_mov):
    """
    img_ref (canvas sağ kenarı) ↔ img_mov (sol kenarı)
    RTL veya LTR fark etmez — dosyalar zaten doğru sırayla geldi.
    """
    h, w = img_ref.shape[:2]
    ov_range = np.linspace(int(w * OVERLAP_MIN),
                            int(w * OVERLAP_MAX),
                            OVERLAP_STEPS, dtype=int)
    scores = []
    print("\n[Kaba Arama - NCC]")
    for ov in ov_range:
        strip_a = img_ref[:, w - ov:]   # img_ref'in SAĞ ov pikseli
        strip_b = img_mov[:, :ov]        # img_mov'un SOL ov pikseli
        s = ncc_score(strip_a, strip_b)
        scores.append((ov, s))
        print(f"  overlap={ov:4d}px  NCC={s:.4f}")
    best_ov, best_s = max(scores, key=lambda x: x[1])
    print(f"→ En iyi overlap={best_ov}px  NCC={best_s:.4f}")
    return best_ov


# ──────────────────────────────────────────────────────────────
#  ECC İNCE AYAR
# ──────────────────────────────────────────────────────────────

def refine_with_ecc(img_ref, img_mov, overlap_init, search_range=30):
    h, w   = img_ref.shape[:2]
    tx_init = float(w - overlap_init)   # pozitif — img_mov sağa

    warp = np.eye(2, 3, dtype=np.float32)
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


def estimate_translation(img_ref, img_mov):
    best_ov    = coarse_search(img_ref, img_mov)
    print("\n[İnce Ayar - ECC]")
    tx, ty     = refine_with_ecc(img_ref, img_mov, best_ov)
    overlap_px = img_ref.shape[1] - int(round(tx))
    print(f"\n✓ tx={tx:.2f}  ty={ty:.2f}  "
          f"overlap={overlap_px}px ({overlap_px/img_ref.shape[1]*100:.1f}%)")
    return tx, ty


# ──────────────────────────────────────────────────────────────
#  MİNİMUM COST SEAM  (DP)
# ──────────────────────────────────────────────────────────────

def find_min_cost_seam(strip_left, strip_right):
    diff = (strip_left.astype(np.float32) -
            strip_right.astype(np.float32)) ** 2
    cost = diff.sum(axis=2)
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
            back[r, c] = c + (best - 1)

    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        seam[r] = int(np.clip(back[r + 1, seam[r + 1]], 0, W - 1))
    return seam


def blend_seam(strip_left, strip_right, feather_width=20):
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
#  BİRLEŞTİRME  — her zaman sağa ekleme
# ──────────────────────────────────────────────────────────────

def stitch_pair(canvas, img_mov, tx, ty, feather_width, debug_seam):
    """
    canvas'ın SAĞ kenarı ↔ img_mov'un SOL kenarı
    LTR ve RTL için aynı mantık — dosya sırası load_images'da halledildi.
    """
    tx_i = int(round(tx))
    ty_i = int(round(ty))
    ch, cw = canvas.shape[:2]
    mh, mw = img_mov.shape[:2]

    new_w  = max(cw, tx_i + mw)
    new_h  = max(ch, ty_i + mh)
    out    = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    out[:ch, :cw] = canvas

    ov_x0  = tx_i
    ov_x1  = min(cw, tx_i + mw)
    ov_w   = ov_x1 - ov_x0
    mov_y0 = max(ty_i, 0)
    mov_y1 = min(ty_i + mh, new_h)
    ov_h   = mov_y1 - mov_y0

    print(f"  Örtüşme: x=[{ov_x0},{ov_x1}]  {ov_w}x{ov_h}px")

    if ov_w <= 0:
        out[mov_y0:mov_y1, tx_i:tx_i + mw] = img_mov
        return out

    strip_left  = out[mov_y0:mov_y1, ov_x0:ov_x1].copy()  # canvas sağ kenarı
    strip_right = img_mov[:ov_h, :ov_w].copy()              # img_mov sol kenarı

    blended, seam = blend_seam(strip_left, strip_right, feather_width)
    out[mov_y0:mov_y1, ov_x0:ov_x1] = blended

    right_w = mw - ov_w
    if right_w > 0:
        out[mov_y0:mov_y1, ov_x1:ov_x1 + right_w] = img_mov[:ov_h, ov_w:]

    if debug_seam:
        for r, c in enumerate(seam):
            cx, cy = ov_x0 + c, mov_y0 + r
            if 0 <= cy < out.shape[0] and 0 <= cx < out.shape[1]:
                out[cy, cx] = [0, 0, 255]

    return out


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Minimum Cost Seam  |  "
          f"{'SAĞDAN SOLA (RTL)' if SCAN_DIRECTION == 'rtl' else 'SOLDAN SAĞA (LTR)'}")
    print("=" * 60)

    images = load_images(IMAGE_FOLDER, FILE_EXT, SCAN_DIRECTION, n=2)

    canvas = images[0].copy()
    for i in range(1, len(images)):
        print(f"\n[Görüntü {i}/{len(images)-1} ekleniyor]")
        tx, ty = estimate_translation(canvas, images[i])
        canvas = stitch_pair(canvas, images[i], tx, ty,
                              FEATHER_WIDTH, DEBUG_SEAM)
        print(f"  Canvas: {canvas.shape[1]}x{canvas.shape[0]}px")

    cv2.imwrite(OUTPUT_PATH, canvas)
    print(f"\n✓ Kaydedildi: {OUTPUT_PATH}")
    print(f"  Boyut: {canvas.shape[1]} x {canvas.shape[0]} piksel")

    preview = cv2.resize(canvas, (min(1600, canvas.shape[1]),
                                   min(900,  canvas.shape[0])))
    cv2.imshow("Sonuc - ESC", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
