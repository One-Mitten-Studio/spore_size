"""
End-to-end spore sizing for a single micrograph:
  1. auto-calibrate the ocular-micrometer scale bar (calibration.py)
  2. segment candidate spore silhouettes (segment.py)
  3. convert to microns, filter to a plausible spore size range, and
     report length x width (+ Q ratio) per spore
  4. write an annotated overlay + a CSV of measurements
"""
import csv
import os

import cv2
import numpy as np

from .calibration import calibrate_image
from .segment import segment_spores


def filter_spores(raw_spores, um_per_px, min_length_um=3.0, max_length_um=25.0,
                  min_width_um=1.5, max_width_um=18.0):
    """Convert spores to microns and keep only those within the size limits."""
    kept = []
    for s in raw_spores:
        length_um = s["length_px"] * um_per_px
        width_um = s["width_px"] * um_per_px
        if not (min_length_um <= length_um <= max_length_um):
            continue
        if not (min_width_um <= width_um <= max_width_um):
            continue
        kept.append({
            **s,
            "length_um": length_um,
            "width_um": width_um,
            "Q_ratio": length_um / width_um if width_um > 0 else np.nan,
        })
    return kept


def summarize(kept, um_per_px):
    """Mean / SD / range statistics for a list of kept spores."""
    n = len(kept)
    lengths = np.array([s["length_um"] for s in kept])
    widths = np.array([s["width_um"] for s in kept])
    qs = np.array([s["Q_ratio"] for s in kept])
    return {
        "n": n,
        "length_mean": lengths.mean() if n else np.nan,
        "length_sd": lengths.std(ddof=1) if n > 1 else np.nan,
        "length_range": (lengths.min(), lengths.max()) if n else (np.nan, np.nan),
        "width_mean": widths.mean() if n else np.nan,
        "width_sd": widths.std(ddof=1) if n > 1 else np.nan,
        "width_range": (widths.min(), widths.max()) if n else (np.nan, np.nan),
        "Q_mean": qs.mean() if n else np.nan,
        "um_per_px": um_per_px,
    }


def write_measurements_csv(kept, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spore_id", "length_um", "width_um", "Q_ratio",
                    "length_px", "width_px", "center_x_px", "center_y_px", "solidity"])
        for i, s in enumerate(kept, 1):
            w.writerow([i, f"{s['length_um']:.2f}", f"{s['width_um']:.2f}",
                        f"{s['Q_ratio']:.2f}", f"{s['length_px']:.1f}", f"{s['width_px']:.1f}",
                        f"{s['center_px'][0]:.0f}", f"{s['center_px'][1]:.0f}",
                        f"{s['solidity']:.2f}"])


def draw_overlay(img_bgr, kept, um_per_px):
    """Annotated copy of the image with a red ellipse around each kept spore."""
    overlay = img_bgr.copy()
    for s in kept:
        cv2.ellipse(overlay, s["ellipse"], (0, 0, 255), 2)
    cv2.putText(overlay, f"n={len(kept)}  calib={um_per_px:.4f} um/px",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 140, 255), 3)
    return overlay


def process_image(path, out_dir, um_per_tick=1.0,
                  min_length_um=3.0, max_length_um=25.0,
                  min_width_um=1.5, max_width_um=18.0,
                  save_debug=True):
    """
    Calibrate, segment and measure one image, writing its CSV and overlay
    into out_dir. Returns (summary_dict, kept_spores).
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    debug_prefix = os.path.join(out_dir, stem) if save_debug else None

    calib = calibrate_image(img, microns_per_tick=um_per_tick, debug_prefix=debug_prefix)
    um_per_px = calib["microns_per_px"]

    raw_spores = segment_spores(img)  # final overlay is drawn below, kept spores only
    kept = filter_spores(raw_spores, um_per_px,
                         min_length_um=min_length_um, max_length_um=max_length_um,
                         min_width_um=min_width_um, max_width_um=max_width_um)

    csv_path = os.path.join(out_dir, f"{stem}_spore_measurements.csv")
    write_measurements_csv(kept, csv_path)

    overlay_path = os.path.join(out_dir, f"{stem}_spore_overlay.png")
    cv2.imwrite(overlay_path, draw_overlay(img, kept, um_per_px))

    summary = summarize(kept, um_per_px)
    summary["csv_path"] = csv_path
    summary["overlay_path"] = overlay_path
    return summary, kept
