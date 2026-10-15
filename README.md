# Capacitive Imaging-Based Robotic Quality Control System (Dobot Magician & SIFT/ORB Stitching)

Developed as the robotics, sensor-fusion, and image-stitching pipeline of my B.Sc. thesis, this project automates high-resolution fabric surface inspection. A **Dobot Magician** robotic arm executes a precision serpentine (boustrophedon) scan over textile samples, capturing synchronized microscopic camera frames and **AD7747 capacitive sensor** readings (in picofarads, `pF`) via Arduino at every coordinate. The captured frames are registered into a unified panorama using a **hybrid SIFT + ORB** feature-matching algorithm and overlaid with a spatially calibrated **capacitance heatmap** to localize surface and subsurface anomalies.

*(For the deep-learning pixel-level defect segmentation module of this thesis using **Attention U-Net** and **HRNet**, see the companion repository on the AITEX dataset.)*

## End-to-End Pipeline

| Step | Script | Description |
| :--- | :--- | :--- |
| **1. Robotic Scan & Data Acquisition** | `01-scan&UI.py` | Multi-threaded **Tkinter** control panel that drives the Dobot Magician along a 0.5 mm serpentine grid, reads averaged capacitance values (`pF`) from an **AD7747** sensor over serial (`pyserial`), evaluates frame focus via **Laplacian variance sharpness**, and logs synchronized coordinates to `scan_log.csv`. |
| **2. Hybrid Panorama Stitching** | `02-panoramicroscope_SIFT_and_ORB.py` | Applies **CLAHE** contrast enhancement and registers sequential frames via **SIFT** (`RANSAC` partial affine estimation with inlier-ratio validation), falling back to **ORB** and physical encoder offsets when texture is uniform. Blends frames seamlessly using a **Gaussian feather mask**. |
| **3. Convex-Hull ROI Cropping** | `03-scatter_plot_cropping.py` | Computes the convex hull of the stitched camera trajectory, masks out exterior borders via an alpha channel (`BGRA`), and crops to the active inspection region. |
| **4. Capacitive Sensor Fusion & Heatmap Overlay** | `04-overlay.py` | Calibrates physical robot coordinates (`mm`) to stitched panorama coordinates (`px`) using **least-squares affine regression (`numpy.linalg.lstsq`)** and renders a Gaussian-smoothed `COLORMAP_JET` capacitance heatmap over the stitched fabric surface. |

## Hardware Setup

- **Robotic Manipulator:** Dobot Magician connected via USB serial (default `COM3`, step size `0.5 mm`, velocity/acceleration `20`).
- **Microscope Camera:** Connected via OBS Virtual Camera (`CAMERA_INDEX = 1`) with automatic buffer flushing and Laplacian blur detection (`sharpness < 30`).
- **Capacitive Sensing Unit:** Analog Devices **AD7747** 24-bit capacitance-to-digital converter interfaced via Arduino (`COM4`, `9600 baud`), averaging 5 samples per grid point.

## Installation & Usage

```bash
pip install -r requirements.txt
```

### 1. Run the Robotic Scanner GUI
```bash
python "01-scan&UI.py"
```
Configure the sample metadata (`TYPE`, `DESIGN`, `COMPOSITION`) and scan dimensions (`X`, `Y` in `mm`) in the GUI. Output images and `scan_log.csv` are saved under `scan_images/<TYPE>_<DESIGN>_<COMPOSITION>/`.

### 2. Stitch Microscopic Frames into a Panorama
```bash
python 02-panoramicroscope_SIFT_and_ORB.py \
    --input-dir "scan_images/sample_scan" \
    --output-panorama "panorama_stitched.jpg" \
    --output-csv "trajectory_data.csv"
```

### 3. Crop Panorama to Scanned Convex Hull
```bash
python 03-scatter_plot_cropping.py \
    --panorama "panorama_stitched.jpg" \
    --csv "trajectory_data.csv" \
    --output "panorama_cropped.png"
```

### 4. Generate Multi-Modal Capacitance Heatmap Overlay
```bash
python 04-overlay.py \
    --base-image "panorama_stitched.jpg" \
    --scan-csv "scan_images/sample_scan/scan_log.csv" \
    --trajectory-csv "trajectory_data.csv" \
    --image-folder "scan_images/sample_scan" \
    --output "heatmap_capacitance_overlay.jpg"
```

## Repository Structure

```text
├── 01-scan&UI.py                         # Tkinter GUI, Dobot serpentine controller & AD7747 serial logger
├── 02-panoramicroscope_SIFT_and_ORB.py   # CLAHE + SIFT/ORB + RANSAC + Gaussian feather panorama stitcher
├── 03-scatter_plot_cropping.py           # Convex-hull alpha masking and ROI cropping
├── 04-overlay.py                         # Least-squares coordinate calibration & capacitance heatmap fusion
├── requirements.txt                      # Python dependencies
└── README.md
```
