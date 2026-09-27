# DOBOT Magician — Robotic Fabric Scanning & Capacitance Mapping

Bachelor's thesis project. A Dobot Magician robotic arm performs a serpentine
scan over a fabric sample, capturing synchronized camera images (via an OBS
virtual camera) and capacitance readings (via an AD7747 sensor on Arduino) at
each scan point. The captured images are then stitched into a panorama and
overlaid with a capacitance heatmap.

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `01-scan&UI.py` | Tkinter GUI to control the Dobot, run the serpentine scan, capture camera frames + capacitance readings at each point, and log positions |
| 2 | `02-panoramicroscope_SIFT_and_ORB.py` | Stitches the per-point images into a panorama using SIFT (falls back to ORB) feature matching; outputs the panorama, a trajectory CSV, and a scatter plot |
| 3 | `03-scatter_plot_cropping.py` | Crops the panorama to the convex hull of the scanned area and adds alpha transparency outside it |
| 4 | `04-overlay.py` | Maps physical scan coordinates to panorama pixel coordinates and overlays a capacitance heatmap on the cropped panorama |

`home_position_tarama_v2.py` is an earlier, simpler scan script (no GUI, no
capacitance sensor) — kept for reference / fallback use.

## Hardware setup

- **Dobot Magician** — connect and note its serial port (default `COM3` in the scripts)
- **Camera** — start OBS → Tools → Start Virtual Camera before running the scan
- **AD7747 capacitance sensor** — connected via Arduino; set the Arduino's serial
  port in the `AD7747_PORT` variable (default `COM4`), and make sure `debug_mode = false`
  in the Arduino sketch

## Requirements

```bash
pip install -r requirements.txt
```

Windows only (`pydobot`/`pyserial` talk to COM ports).

## Usage

1. Update the config constants at the top of each script (serial ports, home
   position, scan area, and — for scripts 02–04 — the input/output paths,
   which are currently hardcoded to a local machine and **must be changed**
   before running elsewhere).
2. Run `01-scan&UI.py` to perform a scan. Images are saved under `scan_images/<type>_<design>_<composition>/`.
3. Run `02-panoramicroscope_SIFT_and_ORB.py` on the resulting folder to build the panorama.
4. Run `03-scatter_plot_cropping.py` to crop it to the scanned region.
5. Run `04-overlay.py` to generate the capacitance heatmap overlay.

## Repository layout

```
DOBOT_magician/
├── 01-scan&UI.py
├── 02-panoramicroscope_SIFT_and_ORB.py
├── 03-scatter_plot_cropping.py
├── 04-overlay.py
├── home_position_tarama_v2.py
├── requirements.txt
└── stitching_heatmap_sonuclari/   # sample outputs per fabric sample
```

## Known limitations / TODO

- [ ] Replace hardcoded absolute paths in scripts 02–04 with CLI args or a config file
- [ ] Remove `.idea/` from version control (add to `.gitignore`)
- [ ] Consider moving/excluding `stitching_heatmap_sonuclari/` (large binary results) from git, or documenting it as sample output only
- [ ] Decide whether `home_position_tarama_v2.py` is still needed or can be archived