"""End-to-end spore sizing for a single micrograph.

Steps:
1. Reject photos that are not usable spore micrographs: no microscope
   field of view, no reliable scale bar, or a cluttered field (tissue).
2. Auto-calibrate the ocular-micrometer scale bar (calibration.py).
3. Segment clean, in-focus spores (segment.py).
4. Convert to microns; drop spores on the scale bar, outside the plausible
   size range, or oddly sized compared with the other spores of the photo
   (or, from the command line, of the whole directory).
5. Write an annotated overlay and a CSV of measurements.
"""

from __future__ import annotations

import csv
import os
from typing import Any

import cv2
import numpy as np

from ._types import Calibration, Image, Spore
from .calibration import calibrate_image
from .errors import PhotoRejected
from .fov import check_micrograph, find_fov_circle, fov_mask
from .segment import segment_spores, spore_silhouettes


def clutter_fraction(img_bgr: Image, delta_e: float = 10.0) -> float:
    """Return the fraction of the field of view covered by non-spore clutter.

    Clutter is any area whose (smoothed) colour differs clearly from the
    background and that is not a spore-coloured object. A spore mount -- even
    a dense one -- is mostly background plus spores (~0-25%); a tissue squash
    is not (30%+).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    center, r = find_fov_circle(gray)
    inner = fov_mask(gray.shape, center, r, shrink=0.85) > 0
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    silhouettes, _ = spore_silhouettes(hsv, fov_mask(gray.shape, center, r, shrink=0.97))
    near_spore = cv2.dilate(silhouettes, np.ones((15, 15), np.uint8))[inner] > 0
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (0, 0), 4), cv2.COLOR_BGR2LAB)
    lab_fov = lab[inner].astype(np.float32)
    differs = np.linalg.norm(lab_fov - np.median(lab_fov, axis=0), axis=1) > delta_e
    return float(np.mean(differs & ~near_spore))


def near_ruler(
    spore: Spore, calib: Calibration, margin_px: float = 60, half_len_ticks: float = 55
) -> bool:
    """Return True if the spore lies on or right next to the scale bar."""
    cx, cy = calib["center"]
    ang = np.radians(calib["angle_deg"])
    d = np.array([np.cos(ang), np.sin(ang)])
    v = np.array(spore["center_px"]) - (cx, cy)
    along = float(v @ d)
    across = abs(float(v @ (-d[1], d[0])))
    half_len = half_len_ticks * calib["px_per_tick"]
    return abs(along) < half_len + margin_px and across < spore["length_px"] / 2 + margin_px


def drop_size_outliers(
    spores: list[Spore], k: float = 3.0, min_rel_spread: float = 0.04, min_n: int = 5
) -> list[Spore]:
    """Drop spores whose length or width is far from the group's median.

    Far means more than k robust SDs, with the SD floored at min_rel_spread x
    median. Nothing is dropped when there are fewer than min_n spores.
    """
    if len(spores) < min_n:
        return spores
    keep = np.ones(len(spores), bool)
    for key in ("length_um", "width_um"):
        x = np.array([s[key] for s in spores])
        med = np.median(x)
        sd = max(1.4826 * np.median(np.abs(x - med)), min_rel_spread * med)
        keep &= np.abs(x - med) <= k * sd
    return [s for s, ok in zip(spores, keep, strict=True) if ok]


def drop_outliers_by_directory(
    spores_by_image: dict[str, list[Spore]],
) -> tuple[dict[str, list[Spore]], dict[str, int]]:
    """Drop size outliers with all spores of an image directory pooled.

    One directory is one collection. Judging each photo on its own lets a
    few bad detections (e.g. two overlapping spores measured as one) skew
    that photo's median when it has few spores; pooling the directory keeps
    them a small minority. Returns the kept spores per image (each tagged
    with "image") and the number removed per directory.
    """
    kept_by_image: dict[str, list[Spore]] = {path: [] for path in spores_by_image}
    removed: dict[str, int] = {}
    for d in dict.fromkeys(os.path.dirname(path) for path in spores_by_image):
        pooled = [
            {**s, "image": path}
            for path, spores in spores_by_image.items()
            if os.path.dirname(path) == d
            for s in spores
        ]
        kept = drop_size_outliers(pooled)
        for s in kept:
            kept_by_image[s["image"]].append(s)
        removed[d] = len(pooled) - len(kept)
    return kept_by_image, removed


def filter_spores(
    raw_spores: list[Spore],
    um_per_px: float,
    min_length_um: float = 3.0,
    max_length_um: float = 25.0,
    min_width_um: float = 1.5,
    max_width_um: float = 18.0,
) -> list[Spore]:
    """Convert spores to microns and keep only those within the size limits."""
    kept: list[Spore] = []
    for s in raw_spores:
        length_um = s["length_px"] * um_per_px
        width_um = s["width_px"] * um_per_px
        if not (min_length_um <= length_um <= max_length_um):
            continue
        if not (min_width_um <= width_um <= max_width_um):
            continue
        kept.append(
            {
                **s,
                "length_um": length_um,
                "width_um": width_um,
                "Q_ratio": length_um / width_um if width_um > 0 else np.nan,
            }
        )
    return kept


def summarize(kept: list[Spore], um_per_px: float) -> dict[str, Any]:
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


def write_measurements_csv(kept: list[Spore], csv_path: str) -> None:
    """Write one line per measured spore of a photo, in microns and pixels."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "spore_id",
                "length_um",
                "width_um",
                "Q_ratio",
                "length_px",
                "width_px",
                "center_x_px",
                "center_y_px",
                "solidity",
            ]
        )
        for i, s in enumerate(kept, 1):
            w.writerow(
                [
                    i,
                    f"{s['length_um']:.2f}",
                    f"{s['width_um']:.2f}",
                    f"{s['Q_ratio']:.2f}",
                    f"{s['length_px']:.1f}",
                    f"{s['width_px']:.1f}",
                    f"{s['center_px'][0]:.0f}",
                    f"{s['center_px'][1]:.0f}",
                    f"{s['solidity']:.2f}",
                ]
            )


def draw_overlay(img_bgr: Image, kept: list[Spore], um_per_px: float) -> Image:
    """Annotated copy of the image with a red ellipse around each kept spore."""
    overlay = img_bgr.copy()
    for s in kept:
        cv2.ellipse(overlay, s["ellipse"], (0, 0, 255), 2)
    cv2.putText(
        overlay,
        f"n={len(kept)}  calib={um_per_px:.4f} um/px",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 140, 255),
        3,
    )
    return overlay


def analyze_image(
    path: str,
    out_dir: str,
    um_per_tick: float = 1.0,
    min_length_um: float = 3.0,
    max_length_um: float = 25.0,
    min_width_um: float = 1.5,
    max_width_um: float = 18.0,
    include_touching: bool = False,
    max_clutter: float = 0.28,
    save_debug: bool = True,
) -> tuple[Calibration, list[Spore]]:
    """Check, calibrate and segment one image; return (calibration, spores).

    Spores on the scale bar or outside the size limits are dropped, but no
    size-outlier check is done, so that it can be done over more spores
    (see drop_size_outliers). Only the calibration diagnostics are written.
    Raises PhotoRejected if the photo is not a usable spore micrograph.
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    stem = os.path.splitext(os.path.basename(path))[0]
    debug_prefix = os.path.join(out_dir, stem) if save_debug else None

    check_micrograph(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    clutter = clutter_fraction(img)
    if clutter > max_clutter:
        raise PhotoRejected(
            f"cluttered field ({clutter:.0%} of view is neither background "
            f"nor spores; tissue rather than a spore mount?)"
        )

    calib = calibrate_image(img, microns_per_tick=um_per_tick, debug_prefix=debug_prefix)
    raw_spores = segment_spores(img, include_touching=include_touching)
    raw_spores = [s for s in raw_spores if not near_ruler(s, calib)]
    spores = filter_spores(
        raw_spores,
        calib["microns_per_px"],
        min_length_um=min_length_um,
        max_length_um=max_length_um,
        min_width_um=min_width_um,
        max_width_um=max_width_um,
    )
    return calib, spores


def write_image_outputs(
    path: str, out_dir: str, kept: list[Spore], um_per_px: float
) -> dict[str, Any]:
    """Write one image's measurements CSV and overlay; return its summary."""
    stem = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, f"{stem}_spore_measurements.csv")
    write_measurements_csv(kept, csv_path)

    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    overlay_path = os.path.join(out_dir, f"{stem}_spore_overlay.png")
    cv2.imwrite(overlay_path, draw_overlay(img, kept, um_per_px))

    summary = summarize(kept, um_per_px)
    summary["csv_path"] = csv_path
    summary["overlay_path"] = overlay_path
    return summary


def process_image(
    path: str,
    out_dir: str,
    um_per_tick: float = 1.0,
    min_length_um: float = 3.0,
    max_length_um: float = 25.0,
    min_width_um: float = 1.5,
    max_width_um: float = 18.0,
    include_touching: bool = False,
    max_clutter: float = 0.28,
    save_debug: bool = True,
) -> tuple[dict[str, Any], list[Spore]]:
    """Calibrate, segment and measure one image on its own.

    Size outliers are judged within this image only (the command line pools
    each directory instead). Writes the image's CSV and overlay into out_dir
    and returns (summary_dict, kept_spores). Raises PhotoRejected if the
    photo is not a usable spore micrograph.
    """
    calib, spores = analyze_image(
        path,
        out_dir,
        um_per_tick=um_per_tick,
        min_length_um=min_length_um,
        max_length_um=max_length_um,
        min_width_um=min_width_um,
        max_width_um=max_width_um,
        include_touching=include_touching,
        max_clutter=max_clutter,
        save_debug=save_debug,
    )
    kept = drop_size_outliers(spores)
    summary = write_image_outputs(path, out_dir, kept, calib["microns_per_px"])
    return summary, kept
