import cv2
import numpy as np
from pathlib import Path

# ============================================================
#  AYARLAR
# ============================================================
IMAGE_FOLDER   = r"C:\verilerim\goruntüler"
OUTPUT_PATH    = r"C:\verilerim\test_birlesim_rtl.png"
FILE_EXT       = ".png"
ECC_ITERATIONS = 1000
ECC_EPSILON    = 1e-6
FEATHER_WIDTH  = 25

OVERLAP_MIN    = 0.05
OVERLAP_MAX    = 0.50
OVERLAP_STEPS  = 20

GC_DATA_WEIGHT   = 1.0
GC_SMOOTH_WEIGHT = 5.0
GC_USE_GRAPHCUT  = True

# ============================================================
#  TARAMA YÖNİ
#  "ltr" → soldan sağa (1-2-3-4)
#  "rtl" → sağdan sola (4-3-2-1)
# ============================================================
SCAN_DIRECTION = "rtl"
# ============================================================


def load_two_images(folder, ext, direction="rtl"):
    files = sorted(Path(folder).glob(f"*{ext}"))[:2]
    if len(files) < 2:
        raise ValueError(f"En az 2 görüntü gerekli: {folder}")

    if direction == "rtl":
        # Sağdan sola: dosya sırası tersine çevrilir
        # img0 = sağdaki görüntü (taramada önce çekilen)
        # img1 = soldaki görüntü (taramada sonra çekilen)
        files = list(reversed(files))
        print("Tarama yönü: SAĞDAN SOLA (dosyalar ters sırayla yüklendi)")
    else:
        print("Tarama yönü: SOLDAN SAĞA")

    img0 = cv2.imread(str(files[0]))
    img1 = cv2.imread(str(files[1]))
    print(f"  [0] {files[0].name}  {img0.shape}  ← referans (sağ taraf)")
    print(f"  [1] {files[1].name}  {img1.shape}  ← eklenecek (sol taraf)")
    return img0, img1


def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ──────────────────────────────────────────────────────────────
#  NCC: sağdan sola için örtüşme tarafı değişir
# ──────────────────────────────────────────────────────────────

def ncc_score_rtl(img_ref, img_mov, overlap_w):
    """
    RTL: img_ref'in SOL şeridi ile img_mov'un SAĞ şeridini karşılaştır.
    overlap_w: örtüşme genişliği (piksel)
    """
    if overlap_w <= 0:
        return -1.0

    strip_ref = to_gray32(img_ref[:, :overlap_w])          # img_ref'in sol şeridi
    strip_mov = to_gray32(img_mov[:, -overlap_w:])          # img_mov'un sağ şeridi

    h_min = min(strip_ref.shape[0], strip_mov.shape[0])
    strip_ref = strip_ref[:h_min]
    strip_mov = strip_mov[:h_min]

    ref_n = strip_ref - strip_ref.mean()
    mov_n = strip_mov - strip_mov.mean()
    num   = (ref_n * mov_n).sum()
    denom = np.sqrt((ref_n**2).sum() * (mov_n**2).sum()) + 1e-10
    return float(num / denom)


def ncc_score_ltr(img_ref, img_mov, tx):
    """LTR: img_ref'in SAĞ şeridi ile img_mov'un SOL şeridini karşılaştır."""
    h, w = img_ref.shape[:2]
    overlap_w = w - tx
    if overlap_w <= 0:
        return -1.0
    strip_ref = to_gray32(img_ref[:, tx:])
    strip_mov = to_gray32(img_mov[:, :overlap_w])
    h_min = min(strip_ref.shape[0], strip_mov.shape[0])
    strip_ref, strip_mov = strip_ref[:h_min], strip_mov[:h_min]
    ref_n = strip_ref - strip_ref.mean()
    mov_n = strip_mov - strip_mov.mean()
    return float((ref_n * mov_n).sum() /
                 (np.sqrt((ref_n**2).sum() * (mov_n**2).sum()) + 1e-10))


def coarse_search(img_ref, img_mov, overlap_min, overlap_max, steps, direction):
    h, w = img_ref.shape[:2]
    ov_list = np.linspace(int(w * overlap_min), int(w * overlap_max),
                           steps, dtype=int)
    scores = []
    print(f"\n[Kaba Arama - NCC  ({direction.upper()})]")

    for ov in ov_list:
        if direction == "rtl":
            s = ncc_score_rtl(img_ref, img_mov, ov)
            label = f"overlap={ov:4d}px"
        else:
            tx = w - ov
            s  = ncc_score_ltr(img_ref, img_mov, tx)
            label = f"tx={tx:4d}  overlap={ov:4d}px"
        scores.append((ov, s))
        print(f"  {label}  NCC={s:.4f}")

    best_ov, best_s = max(scores, key=lambda x: x[1])
    print(f"→ En iyi overlap={best_ov}px  NCC={best_s:.4f}")
    return best_ov


def refine_with_ecc_rtl(img_ref, img_mov, overlap_init, search_range=30,
                          n_iter=1000, epsilon=1e-6):
    """
    RTL için ECC: img_mov'u img_ref'in soluna hizala.
    img_mov'un sağ kenarı img_ref'in sol kenarıyla örtüşür.
    tx negatif olacak (img_mov sola kaydırılmış).
    """
    h, w = img_ref.shape[:2]

    # Başlangıç tahmini: img_mov, img_ref'in solunda
    # tx = -(img_mov genişliği - overlap)
    mw = img_mov.shape[1]
    tx_init = -(mw - overlap_init)   # negatif → img_mov sola

    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = float(tx_init)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, epsilon)
    try:
        _, warp_out = cv2.findTransformECC(
            to_gray32(img_ref), to_gray32(img_mov),
            warp.copy(), cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5
        )
        tx_r = float(warp_out[0, 2])
        ty_r = float(warp_out[1, 2])
        if abs(tx_r - tx_init) > search_range:
            print(f"ECC sınır dışı ({tx_r:.1f}), kaba tahmin kullanılıyor.")
            return float(tx_init), 0.0
        print(f"ECC → tx={tx_r:.2f}  ty={ty_r:.2f}")
        return tx_r, ty_r
    except cv2.error as e:
        print(f"ECC hata: {e}")
        return float(tx_init), 0.0


def refine_with_ecc_ltr(img_ref, img_mov, overlap_init, search_range=30,
                          n_iter=1000, epsilon=1e-6):
    h, w = img_ref.shape[:2]
    tx_init = w - overlap_init
    warp = np.eye(2, 3, dtype=np.float32)
    warp[0, 2] = float(tx_init)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, epsilon)
    try:
        _, warp_out = cv2.findTransformECC(
            to_gray32(img_ref), to_gray32(img_mov),
            warp.copy(), cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5
        )
        tx_r, ty_r = float(warp_out[0, 2]), float(warp_out[1, 2])
        if abs(tx_r - tx_init) > search_range:
            return float(tx_init), 0.0
        print(f"ECC → tx={tx_r:.2f}  ty={ty_r:.2f}")
        return tx_r, ty_r
    except cv2.error as e:
        print(f"ECC hata: {e}")
        return float(tx_init), 0.0


def estimate_translation_dynamic(img_ref, img_mov, direction="rtl"):
    best_ov = coarse_search(img_ref, img_mov,
                             OVERLAP_MIN, OVERLAP_MAX, OVERLAP_STEPS, direction)
    print(f"\n[İnce Ayar - ECC ({direction.upper()})]")

    if direction == "rtl":
        tx, ty = refine_with_ecc_rtl(img_ref, img_mov, best_ov,
                                       n_iter=ECC_ITERATIONS, epsilon=ECC_EPSILON)
        overlap_px = img_mov.shape[1] + int(round(tx))  # tx negatif olduğu için
    else:
        tx, ty = refine_with_ecc_ltr(img_ref, img_mov, best_ov,
                                       n_iter=ECC_ITERATIONS, epsilon=ECC_EPSILON)
        overlap_px = img_ref.shape[1] - int(round(tx))

    print(f"\n✓ Final → tx={tx:.2f}  ty={ty:.2f}  overlap={overlap_px}px")
    return tx, ty


# ──────────────────────────────────────────────────────────────
#  GRAPH CUT MASKE
# ──────────────────────────────────────────────────────────────

def compute_pixel_diff_mask(strip_left, strip_right):
    grad_L = cv2.Laplacian(cv2.cvtColor(strip_left,  cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    grad_R = cv2.Laplacian(cv2.cvtColor(strip_right, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    rgb_diff = ((strip_left.astype(np.float32) -
                 strip_right.astype(np.float32))**2).sum(axis=2)
    cost_left  = np.abs(grad_L) + 0.1 * rgb_diff
    cost_right = np.abs(grad_R) + 0.1 * rgb_diff
    return (cost_right < cost_left).astype(np.uint8)


def compute_graphcut_mask(strip_left, strip_right,
                           data_weight=1.0, smooth_weight=5.0):
    try:
        import maxflow
    except ImportError:
        print("PyMaxflow bulunamadı → piksel farkı bazlı maskeye geçiliyor.")
        return compute_pixel_diff_mask(strip_left, strip_right)

    H, W = strip_left.shape[:2]
    diff = ((strip_left.astype(np.float32) -
             strip_right.astype(np.float32))**2).sum(axis=2)

    edge_w = W // 5
    cost_source = data_weight * diff.copy()
    cost_sink   = data_weight * diff.copy()

    BIG = 1e6
    cost_sink  [:, :edge_w]  = BIG
    cost_source[:, -edge_w:] = BIG

    def edge_weight(img, axis):
        if axis == 0:
            d = ((img[1:].astype(np.float32) - img[:-1].astype(np.float32))**2).sum(axis=2)
        else:
            d = ((img[:, 1:].astype(np.float32) - img[:, :-1].astype(np.float32))**2).sum(axis=2)
        return (smooth_weight * np.exp(-d / (2 * 255**2))).astype(np.float32)

    g = maxflow.Graph[float](H * W, H * W * 2)
    nodes = g.add_grid_nodes((H, W))
    g.add_grid_tedges(nodes, cost_source, cost_sink)
    g.add_grid_edges(nodes[:-1, :], edge_weight(strip_left, 0), symmetric=True)
    g.add_grid_edges(nodes[:, :-1], edge_weight(strip_left, 1), symmetric=True)
    g.maxflow()

    return g.get_grid_segments(nodes).astype(np.uint8)


def apply_regional_mask(strip_left, strip_right, mask, feather_width=25):
    ksize = feather_width * 2 + 1
    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize),
                                  feather_width / 2)
    mask_blur = np.clip(mask_blur, 0.0, 1.0)[:, :, np.newaxis]
    blended = (strip_left.astype(np.float32)  * (1.0 - mask_blur) +
               strip_right.astype(np.float32) * mask_blur).astype(np.uint8)
    return blended, mask_blur[:, :, 0]


# ──────────────────────────────────────────────────────────────
#  ANA BİRLEŞTİRME — yön farkındalıklı
# ──────────────────────────────────────────────────────────────

def stitch_two(img_ref, img_mov, tx, ty,
               direction="rtl", feather_width=25, debug=True):
    """
    direction="rtl": img_mov, img_ref'in SOLUNA eklenir
                     tx negatif — img_mov canvas'ın solunda başlar
    direction="ltr": img_mov, img_ref'in SAĞINA eklenir
                     tx pozitif
    """
    tx_i = int(round(tx))
    ty_i = int(round(ty))

    rh, rw = img_ref.shape[:2]
    mh, mw = img_mov.shape[:2]

    if direction == "rtl":
        # img_mov canvas'ın soluna yerleşir
        # tx_i negatif → img_mov x=0'dan başlar, img_ref tx_i kadar sağda
        mov_x0 = 0                      # img_mov canvas'ta x=0'dan başlar
        ref_x0 = -tx_i                  # img_ref canvas'ta bu x'ten başlar
        new_w  = max(ref_x0 + rw, mw)
        new_h  = max(rh, mh)
        canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)

        # img_mov'u sola yerleştir
        canvas[:mh, mov_x0:mov_x0 + mw] = img_mov

        # Örtüşme bölgesi
        ov_x0  = ref_x0                          # canvas koordinatında
        ov_x1  = min(ref_x0 + rw, mw)
        ov_w   = ov_x1 - ov_x0

        print(f"\nÖrtüşme: x=[{ov_x0},{ov_x1}]  genişlik={ov_w}px")

        if ov_w <= 0:
            canvas[:rh, ref_x0:ref_x0 + rw] = img_ref
            return canvas

        # img_ref'i de canvas'a yerleştir (örtüşme dışı sol kısım zaten img_mov'dan geldi)
        canvas[:rh, ref_x0:ref_x0 + rw] = img_ref   # örtüşen bölgeyi geçici olarak yaz

        # Örtüşen şeritler:
        # strip_left  = img_mov'un SAĞ kenarı (canvas sol taraf)
        # strip_right = img_ref'in SOL kenarı (canvas sağ taraf)
        ov_h = min(rh, mh)
        strip_left  = img_mov[:ov_h, -ov_w:].copy()   # img_mov'un sağ ov_w pikseli
        strip_right = img_ref[:ov_h, :ov_w].copy()    # img_ref'in sol ov_w pikseli

        mask = (compute_graphcut_mask(strip_left, strip_right,
                                       GC_DATA_WEIGHT, GC_SMOOTH_WEIGHT)
                if GC_USE_GRAPHCUT else
                compute_pixel_diff_mask(strip_left, strip_right))

        blended, mask_soft = apply_regional_mask(strip_left, strip_right,
                                                   mask, feather_width)
        canvas[:ov_h, ov_x0:ov_x1] = blended

    else:
        # LTR — önceki kodla aynı mantık
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

        print(f"\nÖrtüşme: x=[{ov_x0},{ov_x1}]  genişlik={ov_w}px")

        if ov_w <= 0:
            canvas[mov_y0:mov_y1, tx_i:tx_i + mw] = img_mov
            return canvas

        strip_left  = canvas[mov_y0:mov_y1, ov_x0:ov_x1].copy()
        strip_right = img_mov[:ov_h, :ov_w].copy()

        mask = (compute_graphcut_mask(strip_left, strip_right,
                                       GC_DATA_WEIGHT, GC_SMOOTH_WEIGHT)
                if GC_USE_GRAPHCUT else
                compute_pixel_diff_mask(strip_left, strip_right))

        blended, mask_soft = apply_regional_mask(strip_left, strip_right,
                                                   mask, feather_width)
        canvas[mov_y0:mov_y1, ov_x0:ov_x1] = blended

        right_w = mw - ov_w
        if right_w > 0:
            canvas[mov_y0:mov_y1, ov_x1:ov_x1 + right_w] = img_mov[:ov_h, ov_w:]

    # Debug kayıt
    if debug:
        debug_dir = Path(OUTPUT_PATH).parent / "debug"
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / "mask_raw.png"),
                    (mask * 255).astype(np.uint8))
        cv2.imwrite(str(debug_dir / "mask_soft.png"),
                    (mask_soft * 255).astype(np.uint8))
        cv2.imwrite(str(debug_dir / "strip_left.png"),  strip_left)
        cv2.imwrite(str(debug_dir / "strip_right.png"), strip_right)
        cv2.imwrite(str(debug_dir / "strip_blend.png"), blended)
        print(f"Debug → {debug_dir}")

    return canvas


def main():
    print("=" * 60)
    print(f"Tarama Yönü: {'SAĞDAN SOLA (RTL)' if SCAN_DIRECTION=='rtl' else 'SOLDAN SAĞA (LTR)'}")
    print("=" * 60)

    img0, img1 = load_two_images(IMAGE_FOLDER, FILE_EXT, SCAN_DIRECTION)

    tx, ty = estimate_translation_dynamic(img0, img1, SCAN_DIRECTION)

    result = stitch_two(img0, img1, tx, ty,
                         direction=SCAN_DIRECTION,
                         feather_width=FEATHER_WIDTH,
                         debug=True)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\n✓ Kaydedildi: {OUTPUT_PATH}")

    preview = cv2.resize(result, (min(1600, result.shape[1]),
                                   min(900, result.shape[0])))
    cv2.imshow("Sonuc - ESC", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
