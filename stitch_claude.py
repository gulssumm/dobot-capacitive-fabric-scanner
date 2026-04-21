import cv2
import numpy as np
from pathlib import Path

# ============================================================
#  SETTINGS
# ============================================================
IMAGE_FOLDER    = r"C:\Users\21180\PycharmProjects\blue-striped_01_0_unknown"
OUTPUT_PATH     = r"C:\Users\21180\PycharmProjects\DOBOT_magician\new_stitch.png"
FILE_EXT        = ".jpg"

# Grid layout — adjust these to match your capture setup
IMAGES_PER_ROW  = 21       # How many images per row in the grid
# NUM_ROWS is auto-calculated from total image count

# Scan parameters
FIRST_ROW_DIR   = "rtl"   # "rtl" = first row goes Right→Left (top-right start)
                           # "ltr" = first row goes Left→Right

# NCC search range
OVERLAP_MIN     = 0.05
OVERLAP_MAX     = 0.50
OVERLAP_STEPS   = 20

# ECC fine-tuning
ECC_ITERATIONS  = 1000
ECC_EPSILON     = 1e-6

# Blending
FEATHER_WIDTH   = 25

# White background (255,255,255) — fabric is dark, this makes seams visible
BACKGROUND      = 255

# Debug: draw red seam lines
DEBUG_SEAM      = False
# ============================================================


def load_images(folder, ext, n=None):
    files = sorted(Path(folder).glob(f"*{ext}"))
    if n:
        files = files[:n]
    if len(files) < 2:
        raise ValueError(f"Need at least 2 images in: {folder}")
    images = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            raise ValueError(f"Could not read: {f}")
        images.append((f.name, img))
        print(f"  Loaded: {f.name}  {img.shape[1]}x{img.shape[0]}")
    return images


def build_snake_grid(images, images_per_row, first_row_dir="rtl"):
    """
    Re-orders images into a 2D grid matching the physical snake scan layout.

    Snake starting at top-right, first row RTL:
      Capture order:  1  2  3  4  | 5  6  7  8
      Spatial left→right row 0:  4  3  2  1  (reversed)
      Spatial left→right row 1:  5  6  7  8  (as-is)

    Returns grid[row][col] where [row][0] is the leftmost image spatially.
    """
    grid = []
    n = len(images)
    row_idx = 0
    i = 0
    while i < n:
        chunk = images[i : i + images_per_row]
        names_imgs = chunk

        # Determine direction for this row
        if first_row_dir == "rtl":
            reverse_this_row = (row_idx % 2 == 0)
        else:
            reverse_this_row = (row_idx % 2 == 1)

        if reverse_this_row:
            names_imgs = list(reversed(names_imgs))

        grid.append(names_imgs)
        print(f"  Row {row_idx} ({'RTL→reversed' if reverse_this_row else 'LTR→as-is'}): "
              f"{[nm for nm, _ in names_imgs]}")

        i += images_per_row
        row_idx += 1

    return grid


# ──────────────────────────────────────────────────────────────
#  UTILITY
# ──────────────────────────────────────────────────────────────

def to_gray32(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def white_canvas(h, w):
    return np.full((h, w, 3), BACKGROUND, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────
#  NCC COARSE SEARCH  (horizontal: right edge ↔ left edge)
# ──────────────────────────────────────────────────────────────

def ncc_score(strip_a, strip_b):
    a = to_gray32(strip_a)
    b = to_gray32(strip_b)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])   # ← add this
    a, b = a[:h, :w], b[:h, :w]       # ← clip both dims
    an = a - a.mean();  bn = b - b.mean()
    num   = (an * bn).sum()
    denom = np.sqrt((an**2).sum() * (bn**2).sum()) + 1e-10
    return float(num / denom)


def ncc_score_vertical(strip_a, strip_b):
    """Bottom strip of A vs top strip of B."""
    a = to_gray32(strip_a)
    b = to_gray32(strip_b)
    w = min(a.shape[1], b.shape[1])
    a, b = a[:, :w], b[:, :w]
    an = a - a.mean();  bn = b - b.mean()
    num   = (an * bn).sum()
    denom = np.sqrt((an**2).sum() * (bn**2).sum()) + 1e-10
    return float(num / denom)


def coarse_search_h(img_ref, img_mov):
    """Horizontal overlap: ref's right edge ↔ mov's left edge."""
    h, w_ref = img_ref.shape[:2]
    w_mov = img_mov.shape[1]
    max_ov = min(int(w_ref * OVERLAP_MAX), w_mov - 1)  # can't exceed mov width
    min_ov = int(w_ref * OVERLAP_MIN)
    ov_range = np.linspace(min_ov, max_ov, OVERLAP_STEPS, dtype=int)
    scores = [(ov, ncc_score(img_ref[:, w_ref - ov:], img_mov[:, :ov]))
              for ov in ov_range]
    best_ov, best_s = max(scores, key=lambda x: x[1])
    print(f"    H-overlap best: {best_ov}px  NCC={best_s:.4f}")
    return best_ov


def coarse_search_v(img_ref, img_mov):
    """Vertical overlap: ref's bottom edge ↔ mov's top edge."""
    h_ref, w = img_ref.shape[:2]
    h_mov = img_mov.shape[0]
    max_ov = min(int(h_ref * OVERLAP_MAX), h_mov - 1)  # can't exceed mov height
    min_ov = int(h_ref * OVERLAP_MIN)
    ov_range = np.linspace(min_ov, max_ov, OVERLAP_STEPS, dtype=int)
    scores = [(ov, ncc_score_vertical(img_ref[h_ref - ov:, :], img_mov[:ov, :]))
              for ov in ov_range]
    best_ov, best_s = max(scores, key=lambda x: x[1])
    print(f"    V-overlap best: {best_ov}px  NCC={best_s:.4f}")
    return best_ov


# ──────────────────────────────────────────────────────────────
#  ECC FINE ALIGNMENT
# ──────────────────────────────────────────────────────────────

def refine_ecc(img_ref, img_mov, tx_init, ty_init, search_range=40):
    warp = np.array([[1, 0, tx_init],
                     [0, 1, ty_init]], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                ECC_ITERATIONS, ECC_EPSILON)
    try:
        _, warp_out = cv2.findTransformECC(
            to_gray32(img_ref), to_gray32(img_mov),
            warp.copy(), cv2.MOTION_TRANSLATION, criteria,
            inputMask=None, gaussFiltSize=5
        )
        tx_r, ty_r = float(warp_out[0, 2]), float(warp_out[1, 2])
        if abs(tx_r - tx_init) > search_range or abs(ty_r - ty_init) > search_range:
            print(f"    ECC out of range ({tx_r:.1f},{ty_r:.1f}) → using coarse")
            return tx_init, ty_init
        print(f"    ECC refined: tx={tx_r:.2f}  ty={ty_r:.2f}")
        return tx_r, ty_r
    except cv2.error as e:
        print(f"    ECC error: {e} → using coarse")
        return tx_init, ty_init


def estimate_h_translation(canvas, img_mov):
    """canvas right ↔ img_mov left."""
    print("  [Horizontal alignment]")
    best_ov = coarse_search_h(canvas, img_mov)
    tx_init = float(canvas.shape[1] - best_ov)
    tx, ty  = refine_ecc(canvas, img_mov, tx_init, 0.0)
    ov_px   = canvas.shape[1] - int(round(tx))
    print(f"  → tx={tx:.1f}  ty={ty:.1f}  overlap={ov_px}px "
          f"({ov_px/canvas.shape[1]*100:.1f}%)")
    return tx, ty


def estimate_v_translation(row_above, row_below):
    """row_above bottom ↔ row_below top."""
    print("  [Vertical alignment]")
    best_ov = coarse_search_v(row_above, row_below)
    ty_init = float(row_above.shape[0] - best_ov)
    tx, ty  = refine_ecc(row_above, row_below, 0.0, ty_init)
    ov_px   = row_above.shape[0] - int(round(ty))
    print(f"  → tx={tx:.1f}  ty={ty:.1f}  overlap={ov_px}px "
          f"({ov_px/row_above.shape[0]*100:.1f}%)")
    return tx, ty


# ──────────────────────────────────────────────────────────────
#  MINIMUM COST SEAM  (dynamic programming)
# ──────────────────────────────────────────────────────────────

def find_seam_vertical(strip_left, strip_right):
    """Vertical seam cutting between two horizontally adjacent strips."""
    diff = ((strip_left.astype(np.float32) -
             strip_right.astype(np.float32)) ** 2).sum(axis=2)
    H, W = diff.shape
    dp   = diff.copy()
    back = np.zeros((H, W), dtype=np.int32)
    for r in range(1, H):
        for c in range(W):
            cl = max(c - 1, 0);  cr = min(c + 1, W - 1)
            opts  = [dp[r-1, cl], dp[r-1, c], dp[r-1, cr]]
            best  = int(np.argmin(opts))
            dp[r, c]  += opts[best]
            back[r, c] = c + (best - 1)
    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        seam[r] = int(np.clip(back[r+1, seam[r+1]], 0, W - 1))
    return seam


def find_seam_horizontal(strip_top, strip_bot):
    """Horizontal seam cutting between two vertically adjacent strips."""
    diff = ((strip_top.astype(np.float32) -
             strip_bot.astype(np.float32)) ** 2).sum(axis=2)
    # Transpose so DP walks along rows of the transposed image (= cols of original)
    diff = diff.T
    H, W = diff.shape
    dp   = diff.copy()
    back = np.zeros((H, W), dtype=np.int32)
    for r in range(1, H):
        for c in range(W):
            cl = max(c - 1, 0);  cr = min(c + 1, W - 1)
            opts  = [dp[r-1, cl], dp[r-1, c], dp[r-1, cr]]
            best  = int(np.argmin(opts))
            dp[r, c]  += opts[best]
            back[r, c] = c + (best - 1)
    seam = np.zeros(H, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for r in range(H - 2, -1, -1):
        seam[r] = int(np.clip(back[r+1, seam[r+1]], 0, W - 1))
    # seam[col] = row in original image
    return seam


def blend_h(strip_left, strip_right, fw=FEATHER_WIDTH):
    """Blend two horizontally adjacent strips with vertical seam + feathering."""
    seam  = find_seam_vertical(strip_left, strip_right)
    H, W  = strip_left.shape[:2]
    cols  = np.tile(np.arange(W, dtype=np.float32), (H, 1))
    s2d   = seam[:, np.newaxis].astype(np.float32)
    alpha = np.clip((cols - s2d) / max(fw, 1) * 0.5 + 0.5, 0., 1.)[:, :, np.newaxis]
    blended = (strip_left  * (1. - alpha) +
               strip_right * alpha).astype(np.uint8)
    return blended, seam


def blend_v(strip_top, strip_bot, fw=FEATHER_WIDTH):
    """Blend two vertically adjacent strips with horizontal seam + feathering."""
    seam  = find_seam_horizontal(strip_top, strip_bot)  # seam[col] = row
    H_top = strip_top.shape[0]
    H_bot = strip_bot.shape[0]
    W     = min(strip_top.shape[1], strip_bot.shape[1])
    rows  = np.tile(np.arange(H_top, dtype=np.float32)[:, np.newaxis], (1, W))
    s2d   = seam[np.newaxis, :W].astype(np.float32)
    alpha = np.clip((rows - s2d) / max(fw, 1) * 0.5 + 0.5, 0., 1.)[:, :, np.newaxis]
    top_c = strip_top[:, :W].astype(np.float32)
    bot_c = strip_bot[:H_top, :W].astype(np.float32)
    blended = (top_c * (1. - alpha) + bot_c * alpha).astype(np.uint8)
    return blended, seam


# ──────────────────────────────────────────────────────────────
#  STITCH ONE IMAGE TO THE RIGHT OF CANVAS  (horizontal)
# ──────────────────────────────────────────────────────────────

def stitch_right(canvas, img_mov, tx, ty):
    tx_i = int(round(tx));  ty_i = int(round(ty))
    ch, cw = canvas.shape[:2]
    mh, mw = img_mov.shape[:2]

    new_w = max(cw, tx_i + mw)
    new_h = max(ch + abs(min(ty_i, 0)), ty_i + mh)
    out   = white_canvas(new_h, new_w)

    # Place canvas
    cy_off = abs(min(ty_i, 0))
    out[cy_off:cy_off + ch, :cw] = canvas

    # Overlap region
    mov_y0 = cy_off + ty_i
    mov_y1 = min(mov_y0 + mh, new_h)
    ov_h   = mov_y1 - mov_y0
    ov_x0  = tx_i
    ov_x1  = min(cw, tx_i + mw)
    ov_w   = ov_x1 - ov_x0

    if ov_w <= 0:
        out[mov_y0:mov_y1, tx_i:tx_i + mw] = img_mov[:ov_h]
        return out

    strip_left  = out[mov_y0:mov_y1, ov_x0:ov_x1].copy()
    strip_right = img_mov[:ov_h, :ov_w].copy()
    blended, seam = blend_h(strip_left, strip_right)
    out[mov_y0:mov_y1, ov_x0:ov_x1] = blended

    right_w = mw - ov_w
    if right_w > 0:
        out[mov_y0:mov_y1, ov_x1:ov_x1 + right_w] = img_mov[:ov_h, ov_w:]

    if DEBUG_SEAM:
        for r, c in enumerate(seam):
            px, py = ov_x0 + c, mov_y0 + r
            if 0 <= py < out.shape[0] and 0 <= px < out.shape[1]:
                out[py, px] = [0, 0, 255]

    return out


# ──────────────────────────────────────────────────────────────
#  STITCH ONE ROW BELOW ANOTHER  (vertical)
# ──────────────────────────────────────────────────────────────

def stitch_below(canvas, row_img, tx, ty):
    tx_i = int(round(tx));  ty_i = int(round(ty))
    ch, cw = canvas.shape[:2]
    mh, mw = row_img.shape[:2]

    new_w = max(cw + abs(min(tx_i, 0)), tx_i + mw)
    new_h = max(ch, ty_i + mh)
    out   = white_canvas(new_h, new_w)

    # Place canvas on top
    cx_off = abs(min(tx_i, 0))
    out[:ch, cx_off:cx_off + cw] = canvas

    # Overlap region
    ov_y0  = ty_i
    ov_y1  = min(ch, ty_i + mh)
    ov_h   = ov_y1 - ov_y0
    mov_x0 = cx_off + tx_i
    mov_x1 = min(new_w, mov_x0 + mw)
    ov_w   = mov_x1 - mov_x0

    if ov_h <= 0 or ov_w <= 0:
        out[ty_i:ty_i + mh, mov_x0:mov_x0 + mw] = row_img[:mh, :ov_w]
        return out

    strip_top = out[ov_y0:ov_y1, mov_x0:mov_x1].copy()
    strip_bot = row_img[:ov_h, :ov_w].copy()
    blended, seam = blend_v(strip_top, strip_bot)
    out[ov_y0:ov_y1, mov_x0:mov_x1] = blended

    # Below the overlap: paste remaining rows of row_img
    below_h = mh - ov_h
    if below_h > 0:
        out[ov_y1:ov_y1 + below_h, mov_x0:mov_x1] = row_img[ov_h:ov_h + below_h, :ov_w]

    if DEBUG_SEAM:
        for c, r in enumerate(seam[:ov_w]):
            px, py = mov_x0 + c, ov_y0 + r
            if 0 <= py < out.shape[0] and 0 <= px < out.shape[1]:
                out[py, px] = [0, 255, 0]

    return out


# ──────────────────────────────────────────────────────────────
#  STITCH ONE FULL ROW  (horizontal panorama)
# ──────────────────────────────────────────────────────────────

def stitch_row(row_images):
    canvas = row_images[0].copy()
    for i in range(1, len(row_images)):
        print(f"  Adding image {i+1}/{len(row_images)} →")
        tx, ty = estimate_h_translation(canvas, row_images[i])
        canvas = stitch_right(canvas, row_images[i], tx, ty)
        print(f"  Row canvas: {canvas.shape[1]}x{canvas.shape[0]}px")
    return canvas


# ──────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Fabric Panorama Stitcher")
    print(f"Snake start: top-right  |  First row: {FIRST_ROW_DIR.upper()}")
    print(f"Images per row: {IMAGES_PER_ROW}  |  Background: WHITE")
    print("=" * 60)

    # 1. Load all images
    print("\n[Loading images]")
    named_images = load_images(IMAGE_FOLDER, FILE_EXT)
    print(f"Total: {len(named_images)} images")

    # 2. Organise into snake grid
    print("\n[Building snake grid]")
    grid = build_snake_grid(named_images, IMAGES_PER_ROW, FIRST_ROW_DIR)
    print(f"Grid: {len(grid)} rows × up to {IMAGES_PER_ROW} cols")

    # 3. Stitch each row horizontally
    row_panoramas = []
    for r, row in enumerate(grid):
        imgs = [img for _, img in row]
        print(f"\n[Row {r} — {len(imgs)} images]")
        if len(imgs) == 1:
            row_panoramas.append(imgs[0].copy())
        else:
            row_panoramas.append(stitch_row(imgs))

    # 4. Stitch rows vertically
    print("\n[Stacking rows vertically]")
    canvas = row_panoramas[0]
    for r in range(1, len(row_panoramas)):
        print(f"\n  Row {r-1} ↔ Row {r}")
        tx, ty = estimate_v_translation(canvas, row_panoramas[r])
        canvas = stitch_below(canvas, row_panoramas[r], tx, ty)
        print(f"  Full canvas: {canvas.shape[1]}x{canvas.shape[0]}px")

    # 5. Save
    cv2.imwrite(OUTPUT_PATH, canvas)
    print(f"\nSaved: {OUTPUT_PATH}")
    print(f"  Final size: {canvas.shape[1]} x {canvas.shape[0]} px")

    # 6. Preview
    max_w, max_h = 1600, 900
    scale = min(max_w / canvas.shape[1], max_h / canvas.shape[0], 1.0)
    preview = cv2.resize(canvas, (int(canvas.shape[1] * scale),
                                   int(canvas.shape[0] * scale)))
    cv2.imshow("Result — ESC to close", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()