import cv2
import numpy as np
import os
from pathlib import Path

# ============================================================
#  AYARLAR - Buradan düzenleyin
# ============================================================
IMAGE_FOLDER = r"C:\verilerim\goruntüler"   # Görüntülerin bulunduğu klasör
OUTPUT_PATH  = r"C:\verilerim\panorama.png" # Çıktı dosyası
NUM_IMAGES   = 41                           # İlk satır: 41 görüntü
FILE_EXT     = ".png"                       # Dosya uzantısı (.png, .jpg, .tif ...)
OVERLAP_RATIO = 0.15    # Beklenen örtüşme oranı (başlangıç tahmini için, ~%15)
ECC_ITERATIONS = 1000   # ECC iterasyon sayısı
ECC_EPSILON    = 1e-6   # ECC durdurma kriteri
WARP_MODE = cv2.MOTION_TRANSLATION  # Sadece öteleme (zig-zag tarama için yeterli)
# ============================================================


def load_images(folder, n, ext):
    """Klasördeki ilk n görüntüyü sıraya göre yükle."""
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


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def estimate_translation_ecc(img_ref, img_mov, overlap_ratio=0.15, warp_mode=cv2.MOTION_TRANSLATION,
                               n_iter=1000, epsilon=1e-6):
    """
    ECC ile img_mov'un img_ref'e göre öteleme matrisini bul.
    overlap_ratio: başlangıç tahmini için örtüşme bölgesi genişliği (görüntü genişliğinin oranı).
    Dönüş: (tx, ty) piksel cinsinden öteleme
    """
    h, w = img_ref.shape[:2]

    # Başlangıç tahmini: görüntü yan yana, tx = w * (1 - overlap_ratio)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = w * (1.0 - overlap_ratio)  # tx başlangıç tahmini

    gray_ref = to_gray(img_ref)
    gray_mov = to_gray(img_mov)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, n_iter, epsilon)

    try:
        _, warp_matrix = cv2.findTransformECC(
            gray_ref, gray_mov,
            warp_matrix, warp_mode, criteria,
            inputMask=None, gaussFiltSize=5
        )
    except cv2.error as e:
        print(f"  ECC başarısız, başlangıç tahmini kullanılıyor: {e}")

    tx = warp_matrix[0, 2]
    ty = warp_matrix[1, 2]
    return tx, ty


def stitch_pair(canvas, canvas_offset_x, next_img, tx, ty):
    """
    Mevcut canvas'a next_img'i ECC ötelemesine göre yapıştır.
    Ortak alan: next_img'in sol kısmı kırpılır (canvas ile örtüşen kısım atılır).
    Kalan boş alan siyah kalır.
    
    canvas_offset_x: canvas içindeki mevcut içeriğin başlangıç x konumu
    tx: next_img'in canvas koordinatında başladığı x
    Dönüş: (yeni_canvas, yeni_offset_x)
    """
    ch, cw = canvas.shape[:2]
    nh, nw = next_img.shape[:2]

    # next_img canvas'ta nerede başlıyor?
    # tx: img_ref -> img_mov ötelemesi; img_mov canvas'ta tx pikselinde başlar
    next_x_in_canvas = int(round(tx))
    next_y_in_canvas = int(round(ty))

    # Ortak alan: next_img'in kaç pikseli canvas ile örtüşüyor?
    overlap_pixels = cw - next_x_in_canvas
    if overlap_pixels < 0:
        overlap_pixels = 0

    # next_img'den kırpılacak sol kısım = overlap_pixels
    crop_x = overlap_pixels  # next_img'in bu kadar solunu at
    cropped = next_img[:, crop_x:, :]  # ortak alan kırpıldı

    # Yeni canvas genişliği
    new_width = cw + cropped.shape[1]
    new_canvas = np.zeros((max(ch, nh + abs(min(next_y_in_canvas, 0))),
                           new_width, 3), dtype=np.uint8)

    # Mevcut canvas'ı yeni canvas'a kopyala
    y_off = max(0, next_y_in_canvas) if next_y_in_canvas > 0 else 0
    # Basit durum: ty ≈ 0 (aynı satır taraması)
    new_canvas[:ch, :cw] = canvas

    # Kırpılmış next_img'i ekle
    paste_x = cw  # canvas'ın hemen sağına
    paste_y = max(0, next_y_in_canvas)
    h_to_paste = min(cropped.shape[0], new_canvas.shape[0] - paste_y)
    w_to_paste = cropped.shape[1]

    new_canvas[paste_y:paste_y + h_to_paste, paste_x:paste_x + w_to_paste] = \
        cropped[:h_to_paste, :w_to_paste]

    return new_canvas


def build_panorama(images, overlap_ratio, warp_mode, n_iter, epsilon):
    """Tüm görüntüleri sırayla birleştirerek panorama oluştur."""
    canvas = images[0].copy()
    
    for i in range(1, len(images)):
        print(f"  [{i}/{len(images)-1}] Görüntü {i} hizalanıyor...")
        
        img_ref = images[i - 1]
        img_mov = images[i]

        tx, ty = estimate_translation_ecc(
            img_ref, img_mov,
            overlap_ratio=overlap_ratio,
            warp_mode=warp_mode,
            n_iter=n_iter,
            epsilon=epsilon
        )

        print(f"         tx={tx:.2f}, ty={ty:.2f}")

        canvas = stitch_pair(canvas, 0, img_mov, tx, ty)
        print(f"         Canvas boyutu: {canvas.shape[1]}x{canvas.shape[0]}")

    return canvas


def main():
    print("=" * 50)
    print("Panoramik Birleştirme (ECC + Kırpma Yöntemi)")
    print("=" * 50)

    images = load_images(IMAGE_FOLDER, NUM_IMAGES, FILE_EXT)

    print("\nGörüntüler hizalanıyor ve birleştiriliyor...")
    panorama = build_panorama(
        images,
        overlap_ratio=OVERLAP_RATIO,
        warp_mode=WARP_MODE,
        n_iter=ECC_ITERATIONS,
        epsilon=ECC_EPSILON
    )

    cv2.imwrite(OUTPUT_PATH, panorama)
    print(f"\nPanorama kaydedildi: {OUTPUT_PATH}")
    print(f"Son boyut: {panorama.shape[1]} x {panorama.shape[0]} piksel")


if __name__ == "__main__":
    main()
