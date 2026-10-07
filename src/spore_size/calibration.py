"""Automatic calibration of the ocular-micrometer scale bar in a micrograph.

The scale bar is the diagonal "0....10....20....100" hatched ruler overlaid
on mushroom spore micrographs.

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
from .colour import best_channel_image, has_colour_cast
from .errors import PhotoRejected
from .fov import find_fov_circle, fov_mask


def _achromatic_edges(img_bgr: Image, fov_center: Point, fov_radius: float) -> Image:
    """Return edge pixels that are colourless (like the ruler), inside the field of view."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    inner_fov = fov_mask(img_bgr.shape, fov_center, fov_radius, shrink=0.85)
    low_sat = cv2.bitwise_and((sat < 35).astype(np.uint8) * 255, inner_fov)
    edges = cv2.Canny(gray, 40, 120)
    return cv2.bitwise_and(edges, edges, mask=low_sat)


def _ruler_region_mask(edges_achromatic: Image) -> Image:
    """Return a mask of the ruler: the largest region dense in colourless edges.

    Raises RuntimeError if no such region is found.
    """
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
    """Fit the ruler's centre line through the midpoints of its tick strokes.

    Returns a dict with angle_deg, center and n_ticks. Raises RuntimeError if
    too few tick segments are found.
    """
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


def _regular_runs(peaks: NDArray[np.intp], px_per_tick: float, tol: float = 0.2) -> list[slice]:
    """Return slices of peaks covering each run of evenly spaced ticks."""
    ok = np.abs(np.diff(peaks) / px_per_tick - 1) < tol
    runs = []
    start = None
    for i, o in enumerate(ok):
        if o and start is None:
            start = i
        elif not o and start is not None:
            runs.append(slice(start, i + 1))
            start = None
    if start is not None:
        runs.append(slice(start, len(peaks)))
    return runs


def _longest_regular_run(peaks: NDArray[np.intp], px_per_tick: float, tol: float = 0.2) -> int:
    """Longest run of consecutive tick gaps within tol of the expected spacing."""
    runs = _regular_runs(peaks, px_per_tick, tol)
    return max((r.stop - r.start - 1 for r in runs), default=0)


def _fit_tick_spacing(peaks: NDArray[np.intp], px_per_tick: float) -> float:
    """Refine the tick spacing to sub-pixel precision.

    Tick positions are whole pixels, so the median gap is a whole number and
    can be off by up to half a pixel per tick (several percent). Instead, fit
    a straight line through the positions of the longest run of evenly spaced
    ticks against their tick number; its slope is the spacing.
    """
    runs = _regular_runs(peaks, px_per_tick)
    if not runs:
        return px_per_tick
    longest = max(runs, key=lambda r: r.stop - r.start)
    pos = peaks[longest].astype(float)
    if len(pos) < 3:
        return px_per_tick
    index = np.arange(len(pos))
    slope = np.polyfit(index, pos, 1)[0]
    return float(slope)


def _strip_ticks(
    rot: Image, cx: float, cy: float, strip_len: int, strip_half_h: int
) -> tuple[float, NDArray[np.float32], NDArray[np.intp]] | None:
    """Find ticks in one horizontal strip of the rotated image.

    Returns (px_per_tick, strip, tick positions), or None if too few ticks.
    """
    h, w = rot.shape[:2]
    x0, x1 = max(0, int(cx - strip_len / 2)), min(w, int(cx + strip_len / 2))
    y0, y1 = max(0, int(cy - strip_half_h)), min(h, int(cy + strip_half_h))
    if y1 <= y0 or x1 <= x0:
        return None
    strip = rot[y0:y1, x0:x1].astype(np.float32)
    profile = strip.min(axis=0)
    inv_s = gaussian_filter1d(255 - profile, sigma=1.5)
    peaks, _ = find_peaks(inv_s, distance=8, prominence=10)
    diffs = np.diff(peaks)
    if len(diffs) < 5:
        return None
    med = np.median(diffs)
    good = diffs[(diffs > med * 0.6) & (diffs < med * 1.4)]
    if len(good) == 0:
        return None
    return float(np.median(good)), strip, peaks


def _measure_spacing(
    gray: Image,
    center: Point,
    angle_deg: float,
    strip_len: int = 1600,
    strip_half_h: int = 25,
    search_px: int = 60,
    search_step: int = 4,
) -> tuple[float, int, NDArray[np.float32], NDArray[np.intp], Point]:
    """Measure the tick spacing along the ruler line.

    Rotates the image so the ruler is horizontal, takes the darkest value
    across a strip along it, and finds the ticks as peaks in that profile.
    The fitted line can sit beside the ticks (the printed numbers pull it
    sideways), so strips up to search_px either side are tried and the one
    with the longest run of evenly spaced ticks is used. The spacing is then
    refined to sub-pixel precision from that run (see _fit_tick_spacing).

    Returns (px_per_tick, number of ticks, strip, tick positions, centre of
    the chosen strip in image coordinates). Raises RuntimeError if too few
    ticks are found.
    """
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rot = cv2.warpAffine(gray, M, (w, h))
    cx, cy = center

    best = None
    best_score = (-1, -1)
    for dy in range(-search_px, search_px + 1, search_step):
        found = _strip_ticks(rot, cx, cy + dy, strip_len, strip_half_h)
        if found is None:
            continue
        px_per_tick, strip, peaks = found
        score = (_longest_regular_run(peaks, px_per_tick), len(peaks))
        if score > best_score:
            best, best_score = (px_per_tick, strip, peaks, dy), score
    if best is None:
        raise RuntimeError("Not enough ticks found along the fitted ruler line")

    px_per_tick, strip, peaks, dy = best
    px_per_tick = _fit_tick_spacing(peaks, px_per_tick)
    # centre of the chosen strip, mapped back from the rotated image
    m_inv = cv2.invertAffineTransform(M)
    sx, sy = m_inv @ np.array([cx, cy + dy, 1.0])
    return px_per_tick, len(peaks), strip, peaks, (float(sx), float(sy))


def calibrate_image(
    img_bgr: Image,
    microns_per_tick: float = 1.0,
    min_regular_ticks: int = 8,
    debug_prefix: str | None = None,
) -> Calibration:
    """Detect the scale bar in a BGR image and return its calibration.

    The result has px_per_tick, microns_per_px, angle_deg, center, n_peaks
    and regular_run.

    If debug_prefix is given, writes <prefix>_calib_overlay.png (green line =
    detected ruler) and <prefix>_calib_strip.png (red lines = detected ticks).
    Raises PhotoRejected if no ruler with at least min_regular_ticks
    consecutive evenly spaced ticks is found (e.g. the line locked onto
    tissue texture instead of the scale bar).

    In a photo with a strong colour cast nothing is colourless, so the ruler
    is looked for in its most informative channel instead (see colour.py).
    """
    if has_colour_cast(img_bgr):
        img_bgr = best_channel_image(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov_center, fov_radius = find_fov_circle(gray)

    edges = _achromatic_edges(img_bgr, fov_center, fov_radius)
    try:
        region_mask = _ruler_region_mask(edges)
        edges_region = cv2.bitwise_and(edges, edges, mask=region_mask)
        line = _fit_ruler_line(edges_region)
        px_per_tick, n_peaks, strip, peaks, ruler_center = _measure_spacing(
            gray, line["center"], line["angle_deg"]
        )
    except RuntimeError as e:
        raise PhotoRejected(f"scale bar not found ({e})") from e
    regular_run = _longest_regular_run(peaks, px_per_tick)

    microns_per_px = microns_per_tick / px_per_tick

    if debug_prefix:
        overlay = img_bgr.copy()
        cx, cy = ruler_center
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
        "center": ruler_center,
        "n_peaks": n_peaks,
        "regular_run": regular_run,
    }
