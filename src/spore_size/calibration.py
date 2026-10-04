"""
Automatic calibration of a diagonal ocular-micrometer scale bar overlaid on
mushroom spore micrographs (the "0....10....20....100" hatched ruler).

Pipeline:
 1. Locate the field-of-view circle, and the ruler as the densest blob of
    achromatic (low-saturation) edge pixels near the centre of the frame.
 2. Within that blob, find individual tick strokes with a probabilistic
    Hough transform and fit a line through their midpoints (PCA) to get an
    accurate angle + centre for the ruler.
 3. Rotate the image so the ruler is horizontal, take an intensity profile
    along it, and measure tick-to-tick spacing by peak detection. This is
    far more robust than trying to fit the whole line at once, because it
    only needs LOCAL periodicity, not a clean line across occluding spores.

Output: pixels-per-scale-unit for the image. Multiply by microns-per-unit
(1.0 for a standard 0-100 micron ocular scale -- ADJUST if your scale bar
uses a different convention) to get the microns-per-pixel calibration.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from ._types import Calibration, Image, Point
from .errors import PhotoRejected
from .fov import find_fov_circle, fov_mask


def _achromatic_edges(img_bgr: Image, fov_center: Point, fov_radius: float) -> Image:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    inner_fov = fov_mask(img_bgr.shape, fov_center, fov_radius, shrink=0.85)
    low_sat = cv2.bitwise_and((sat < 35).astype(np.uint8) * 255, inner_fov)
    edges = cv2.Canny(gray, 40, 120)
    return cv2.bitwise_and(edges, edges, mask=low_sat)


def _ruler_region_mask(edges_achromatic: Image) -> Image:
    density = cv2.boxFilter((edges_achromatic > 0).astype(np.float32), -1, (25, 25))
    dense_mask = cv2.morphologyEx(
        (density > 0.10).astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8)
    )
    cnts, _ = cv2.findContours(dense_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not locate ruler blob (no dense achromatic-edge region found)")
    best = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(dense_mask)
    cv2.drawContours(mask, [best], -1, 255, -1)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _fit_ruler_line(edges_in_region: Image) -> dict[str, Any]:
    lines = cv2.HoughLinesP(
        edges_in_region, 1, np.pi / 360, threshold=15, minLineLength=8, maxLineGap=2
    )
    if lines is None or len(lines) < 8:
        raise RuntimeError("Not enough tick segments detected to fit the ruler line")
    segs = lines.reshape(-1, 4)  # OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4)
    mids = np.stack([(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2], axis=1)

    mean = mids.mean(axis=0)
    centered = mids - mean
    cov = centered.T @ centered
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, np.argmax(evals)]
    direction = direction / np.linalg.norm(direction)

    # drop points far off the fitted line (perpendicular outliers), refit centre
    perp = centered @ np.array([-direction[1], direction[0]])
    inlier = np.abs(perp) < max(3.0, 2.5 * np.std(perp))
    mids_in = mids[inlier]
    mean = mids_in.mean(axis=0)

    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return {"angle_deg": angle_deg, "center": tuple(mean), "n_ticks": int(inlier.sum())}


def _longest_regular_run(peaks: NDArray[np.intp], px_per_tick: float, tol: float = 0.2) -> int:
    """Longest run of consecutive tick gaps within tol of the expected spacing."""
    ok = np.abs(np.diff(peaks) / px_per_tick - 1) < tol
    best = run = 0
    for o in ok:
        run = run + 1 if o else 0
        best = max(best, run)
    return best


def _measure_spacing(
    gray: Image, center: Point, angle_deg: float, strip_len: int = 1600, strip_half_h: int = 25
) -> tuple[float, int, NDArray[np.float32], NDArray[np.intp]]:
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h))
    cx, cy = center
    x0, x1 = int(cx - strip_len / 2), int(cx + strip_len / 2)
    y0, y1 = int(cy - strip_half_h), int(cy + strip_half_h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    strip = rot[y0:y1, x0:x1].astype(np.float32)

    profile = strip.min(axis=0)
    inv_s = gaussian_filter1d(255 - profile, sigma=1.5)
    peaks, _ = find_peaks(inv_s, distance=8, prominence=10)
    diffs = np.diff(peaks)
    if len(diffs) < 5:
        raise RuntimeError("Not enough ticks found along the fitted ruler line")
    med = np.median(diffs)
    good = diffs[(diffs > med * 0.6) & (diffs < med * 1.4)]
    px_per_tick = float(np.median(good))
    return px_per_tick, len(peaks), strip, peaks


def calibrate_image(
    img_bgr: Image,
    microns_per_tick: float = 1.0,
    min_regular_ticks: int = 8,
    debug_prefix: str | None = None,
) -> Calibration:
    """
    Detect the scale bar in a BGR image and return its calibration as a dict
    with px_per_tick, microns_per_px, angle_deg, center and n_peaks.

    If debug_prefix is given, writes <prefix>_calib_overlay.png (green line =
    detected ruler) and <prefix>_calib_strip.png (red lines = detected ticks).
    Raises PhotoRejected if no ruler with at least min_regular_ticks
    consecutive evenly spaced ticks is found (e.g. the line locked onto
    tissue texture instead of the scale bar).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov_center, fov_radius = find_fov_circle(gray)

    edges = _achromatic_edges(img_bgr, fov_center, fov_radius)
    try:
        region_mask = _ruler_region_mask(edges)
        edges_region = cv2.bitwise_and(edges, edges, mask=region_mask)
        line = _fit_ruler_line(edges_region)
        px_per_tick, n_peaks, strip, peaks = _measure_spacing(
            gray, line["center"], line["angle_deg"]
        )
    except RuntimeError as e:
        raise PhotoRejected(f"scale bar not found ({e})") from e
    regular_run = _longest_regular_run(peaks, px_per_tick)

    microns_per_px = microns_per_tick / px_per_tick

    if debug_prefix:
        overlay = img_bgr.copy()
        cx, cy = line["center"]
        ang = np.radians(line["angle_deg"])
        d = np.array([np.cos(ang), np.sin(ang)])
        p0 = (int(cx - d[0] * 700), int(cy - d[1] * 700))
        p1 = (int(cx + d[0] * 700), int(cy + d[1] * 700))
        cv2.line(overlay, p0, p1, (0, 255, 0), 3)
        cv2.putText(
            overlay,
            f"{px_per_tick:.2f} px/unit ({microns_per_px:.4f} um/px)",
            (int(cx) - 250, int(cy) - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 140, 255),
            3,
        )
        cv2.imwrite(f"{debug_prefix}_calib_overlay.png", overlay)
        dbg_strip = cv2.cvtColor(strip.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for p in peaks:
            cv2.line(dbg_strip, (p, 0), (p, dbg_strip.shape[0]), (0, 0, 255), 1)
        cv2.imwrite(f"{debug_prefix}_calib_strip.png", dbg_strip)

    if regular_run < min_regular_ticks:
        raise PhotoRejected(
            f"scale bar not found reliably (only {regular_run} evenly "
            f"spaced ticks in a row; need {min_regular_ticks})"
        )

    return {
        "px_per_tick": px_per_tick,
        "microns_per_px": microns_per_px,
        "angle_deg": line["angle_deg"],
        "center": line["center"],
        "n_peaks": n_peaks,
        "regular_run": regular_run,
    }
