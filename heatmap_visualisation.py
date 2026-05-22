import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import mplcursors
import os

# ── Ayarlar ───────────────────────────────────────────────────────────────────
CSV_FILE   = r"C:/Users/Aidata/DOBOT/scan_images/18K7119-00_Kaynak_Tekstil_97%COTTON_3EA/scan_log.csv"
OUTPUT_DIR = "."

COLORMAP = "plasma"     # viridis, plasma, inferno, jet
INTERP   = "cubic"      # nearest, linear, cubic
GRID_RES = 200
# ─────────────────────────────────────────────────────────────────────────────


def main():

    # ── CSV yükle ────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE)

    # Geçersiz satırları çıkar
    df = df[df["capacitance_pF"].notna()]
    df = df[df["image_file"] != "FAIL"]

    x   = df["x"].values
    y   = df["y"].values
    cap = df["capacitance_pF"].values

    # ── Bilgi yazdır ─────────────────────────────────────────────────────────
    print(f"\nToplam nokta     : {len(df)}")
    print(f"X aralığı        : {x.min():.1f} → {x.max():.1f} mm")
    print(f"Y aralığı        : {y.min():.1f} → {y.max():.1f} mm")
    print(f"Kapasitans (pF)  : {cap.min():.6f} → {cap.max():.6f}")
    print(f"Ortalama         : {cap.mean():.6f} pF\n")

    # ── Interpolasyon grid'i ────────────────────────────────────────────────
    xi = np.linspace(x.min(), x.max(), GRID_RES)
    yi = np.linspace(y.min(), y.max(), GRID_RES)

    Xi, Yi = np.meshgrid(xi, yi)

    Zi = griddata(
        (x, y),
        cap,
        (Xi, Yi),
        method=INTERP
    )

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    fig.suptitle(
        "Kapasitans Heatmap — EVAL AD7747",
        fontsize=16,
        fontweight="bold"
    )

    # =========================================================================
    # SOL: Heatmap
    # =========================================================================
    ax1 = axes[0]

    im1 = ax1.pcolormesh(
        Xi,
        Yi,
        Zi,
        cmap=COLORMAP,
        shading="auto"
    )

    # Gerçek ölçüm noktalarını heatmap üstüne koy
    ax1.scatter(
        x,
        y,
        c="white",
        s=20,
        edgecolors="black",
        linewidths=0.5
    )

    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("Kapasitans (pF)")

    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_title("Interpolasyonlu Heatmap")

    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # =========================================================================
    # SAĞ: Scatter
    # =========================================================================
    ax2 = axes[1]

    sc = ax2.scatter(
        x,
        y,
        c=cap,
        cmap=COLORMAP,
        s=120,
        edgecolors="black"
    )

    cb2 = fig.colorbar(sc, ax=ax2)
    cb2.set_label("Kapasitans (pF)")

    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_title("Ham Veri Noktaları")

    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    # ── Her 10 noktada bir koordinat yaz ────────────────────────────────────
    for i, (xi_, yi_) in enumerate(zip(x, y)):

        if i % 10 == 0:

            ax2.annotate(
                f"({xi_:.0f},{yi_:.0f})",
                (xi_, yi_),

                textcoords="offset points",
                xytext=(5, 5),

                fontsize=8,
                color="black",

                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc="white",
                    alpha=0.8
                )
            )

    # =========================================================================
    # INTERAKTİF HOVER
    # =========================================================================
    cursor = mplcursors.cursor(sc, hover=True)

    @cursor.connect("add")
    def on_add(sel):

        i = sel.index

        sel.annotation.set_text(
            f"Index : {i}\n"
            f"X     : {x[i]:.1f} mm\n"
            f"Y     : {y[i]:.1f} mm\n"
            f"Cap   : {cap[i]:.6f} pF"
        )

        sel.annotation.get_bbox_patch().set(
            fc="white",
            alpha=0.95
        )

    # ── Layout ───────────────────────────────────────────────────────────────
    plt.tight_layout()

    # ── Kaydet ───────────────────────────────────────────────────────────────
    out_path = os.path.join(
        OUTPUT_DIR,
        "capacitance_heatmap_interactive.png"
    )

    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Heatmap kaydedildi:\n{out_path}")

    # ── Göster ───────────────────────────────────────────────────────────────
    plt.show()

if __name__ == "__main__":
    main()