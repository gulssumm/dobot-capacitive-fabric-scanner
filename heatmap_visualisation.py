import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

# ── Ayarlar ───────────────────────────────────────────────────────────────────
CSV_FILE   = "C:/Users/Aidata\DOBOT/scan_images/18K7119-00_Kaynak_Tekstil_97%COTTON_3EA/scan_log.csv"       # CSV dosyasının yolu
OUTPUT_DIR = "."                   # Heatmap'in kaydedileceği klasör
COLORMAP   = "plasma"              # viridis, plasma, inferno, coolwarm, jet
INTERP     = "cubic"               # nearest, linear, cubic
GRID_RES   = 200                   # ızgara çözünürlüğü (piksel)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── CSV yükle ─────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE)
    df = df[df["capacitance_pF"].notna()]          # boş kapasitans satırlarını çıkar
    df = df[df["image_file"] != "FAIL"]            # başarısız noktaları çıkar

    x   = df["x"].values
    y   = df["y"].values
    cap = df["capacitance_pF"].values

    print(f"Toplam nokta     : {len(df)}")
    print(f"X aralığı        : {x.min():.1f} → {x.max():.1f} mm")
    print(f"Y aralığı        : {y.min():.1f} → {y.max():.1f} mm")
    print(f"Kapasitans (pF)  : {cap.min():.6f} → {cap.max():.6f}")
    print(f"Ortalama         : {cap.mean():.6f} pF\n")

    # ── Izgara oluştur ────────────────────────────────────────────────────────
    xi = np.linspace(x.min(), x.max(), GRID_RES)
    yi = np.linspace(y.min(), y.max(), GRID_RES)
    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata((x, y), cap, (Xi, Yi), method=INTERP)

    # ── Heatmap çiz ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Kapasitans Heatmap — EVAL AD7747", fontsize=14, fontweight="bold")

    # Sol: interpolasyonlu heatmap
    ax1 = axes[0]
    im1 = ax1.pcolormesh(Xi, Yi, Zi, cmap=COLORMAP, shading="auto")
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("Kapasitans (pF)", fontsize=10)
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("İnterpolasyonlu Heatmap")
    ax1.set_aspect("equal")

    # Sağ: ham nokta scatter
    ax2 = axes[1]
    sc  = ax2.scatter(x, y, c=cap, cmap=COLORMAP, s=40, edgecolors="none")
    cb2 = fig.colorbar(sc, ax=ax2)
    cb2.set_label("Kapasitans (pF)", fontsize=10)
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Ham Veri Noktaları")
    ax2.set_aspect("equal")

    plt.tight_layout()

    # ── Kaydet ────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "capacitance_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Heatmap kaydedildi: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()