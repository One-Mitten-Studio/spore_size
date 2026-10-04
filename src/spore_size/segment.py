"""Segment fungal spores in a brightfield micrograph and measure them.

Length and width come from an ellipse fitted to each spore's silhouette.

Spores here are yellow-brown ovals with a dark red-brown wall, against a
near-neutral (often slightly pink) background. Air bubbles and oil droplets
are the main confusers: their interiors are greenish and their rims are red
but only weakly saturated. So a pixel counts as "spore" if it is either
brown and moderately saturated (spore interior) or red and strongly
saturated (spore wall). OpenCV hue runs 0-180, with red wrapping at 0/180.

Touching spores are separated by shape: each spore silhouette (with its
interior filled, so oil drops inside don't matter) has one "thickest" core,
while two touching spores have two cores joined by a narrow neck. The cores
(where the distance to the outline is at least half the clump's maximum) are
used as seeds and every pixel of a clump is assigned to its nearest core.
Clumps with no brown interior at all (debris, bubble fragments) are dropped.

Finally each fitted spore must pass quality checks, so that only clean,
in-focus, isolated spores are measured (missing some is preferred over
measuring bad ones). Compared with the photo's own background it must be:
- clearly more saturated (rejects pale tissue / mountant tint),
- not much darker (rejects dirt and smudges),
- outlined by a sharp, dark wall (rejects out-of-focus and ghost spores).
By default spores touching another spore, or near the edge of the field of
view, are also left out.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from ._types import CvEllipse, Image, Spore
from .fov import find_fov_circle, fov_mask


def spore_color_masks(
    hsv: Image,
    interior_hue: tuple[int, int] = (8, 30),
    interior_sat: int = 35,
    rim_sat: int = 60,
) -> tuple[Image, Image]:
    """Binary masks (interior, wall) of pixels coloured like spore interior / wall."""
    hue, sat, _ = cv2.split(hsv)
    lo, hi = interior_hue
    interior = (hue >= lo) & (hue <= hi) & (sat > interior_sat)
    wall = ((hue < lo) | (hue >= 165)) & (sat > rim_sat)
    return interior.astype(np.uint8) * 255, wall.astype(np.uint8) * 255


def _fill_holes(mask: Image) -> Image:
    """Return the mask with every enclosed hole filled in."""
    h, w = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def spore_silhouettes(
    hsv: Image,
    inner_fov: Image,
    interior_hue: tuple[int, int] = (8, 30),
    interior_sat: int = 35,
    rim_sat: int = 60,
) -> tuple[Image, Image]:
    """Return masks of spore-coloured objects inside the field of view.

    Returns (filled, interior): the objects with their interiors filled in,
    and the raw interior-colour mask.
    """
    interior, wall = spore_color_masks(hsv, interior_hue, interior_sat, rim_sat)
    mask = cv2.bitwise_and(cv2.bitwise_or(interior, wall), inner_fov)
    # clean up: close small gaps in outlines, remove specks, fill interiors
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _fill_holes(mask), interior


def _shape_seeds(comp: Image, core_frac: float = 0.5) -> Image:
    """Cores of a filled clump: pixels at least core_frac x max distance from its edge."""
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    return (dist >= core_frac * np.max(dist)).astype(np.uint8) * 255


def _split_by_seeds(comp: Image, seeds: Image) -> list[Image]:
    """Split a component mask into one region per seed blob (nearest seed wins)."""
    n_seeds, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    if n_seeds <= 2:  # background + at most one seed: nothing to split
        return [comp] if n_seeds == 2 else []
    # nearest seed for every pixel; labels refer to the zero pixels' components
    _, nearest = cv2.distanceTransformWithLabels(
        cv2.bitwise_not(seeds), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP
    )
    regions: list[Image] = []
    for s in range(1, n_seeds):
        lab = nearest[seed_labels == s][0]
        region = ((nearest == lab) & (comp > 0)).astype(np.uint8) * 255
        # keep only the part connected to this seed
        n, lbl = cv2.connectedComponents(region, connectivity=8)
        if n > 2:
            keep = np.argmax(np.bincount(lbl[seed_labels == s]))
            region = (lbl == keep).astype(np.uint8) * 255
        regions.append(region)
    return regions


def _measure_region(
    region: Image,
    offset: tuple[int, int],
    min_area_px: int,
    max_area_px: int,
    min_solidity: float,
    min_axis_ratio: float,
    max_axis_ratio: float,
) -> Spore | None:
    """Fit an ellipse to one candidate region; None if it fails a quality check."""
    area = int(np.count_nonzero(region))
    if area < min_area_px or area > max_area_px:
        return None
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=offset)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    solidity = area / hull_area if hull_area > 0 else 0
    if solidity < min_solidity:
        return None  # likely overlapping spores or a debris cluster

    ellipse = cv2.fitEllipse(cnt)
    (ecx, ecy), axes, angle = ellipse
    # fitEllipse does not order its axes, so sort them explicitly
    minor_ax, major_ax = sorted(axes)
    if minor_ax <= 0:
        return None
    axis_ratio = major_ax / minor_ax
    if axis_ratio < min_axis_ratio or axis_ratio > max_axis_ratio:
        return None  # too round (likely debris/bubble) or too elongated (likely fused spores)

    # ellipse-fit quality: compare ellipse area to actual region area
    ellipse_area = np.pi * (major_ax / 2) * (minor_ax / 2)
    area_ratio = area / ellipse_area if ellipse_area > 0 else 0
    if area_ratio < 0.85 or area_ratio > 1.15:
        return None  # poor ellipse fit, likely not a clean single spore

    return {
        "center_px": (ecx, ecy),
        "length_px": major_ax,
        "width_px": minor_ax,
        "angle_deg": angle,
        "ellipse": ellipse,  # raw cv2 ellipse tuple, for drawing
        "area_px": area,
        "solidity": solidity,
    }


def _interior_hsv_median(hsv: Image, ellipse: CvEllipse, scale: float = 0.6) -> NDArray[Any]:
    """Median H, S, V over the central part of a fitted ellipse (avoids the wall)."""
    (cx, cy), (a, b), angle = ellipse
    r = int(max(a, b) / 2) + 2
    x0, y0 = max(int(cx) - r, 0), max(int(cy) - r, 0)
    crop = hsv[y0 : int(cy) + r + 1, x0 : int(cx) + r + 1]
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.ellipse(m, ((cx - x0, cy - y0), (a * scale, b * scale), angle), 255, -1)
    return np.median(crop[m > 0], axis=0)


def _wall_metrics(
    gray: Image,
    grad_mag: Image,
    ellipse: CvEllipse,
    n_angles: int = 72,
    band: int = 8,
) -> tuple[float, float]:
    """Sample the spore outline along radial lines through the fitted ellipse.

    Returns (sharpness, wall_level): the median over the outline of the
    strongest edge gradient and of the darkest grey level within +-band px.
    An in-focus spore has a crisp, dark wall: high sharpness, low wall_level.
    """
    (cx, cy), (a, b), angle = ellipse
    h, w = gray.shape
    th = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    t = np.radians(angle)
    ex, ey = a / 2 * np.cos(th), b / 2 * np.sin(th)
    px = cx + ex * np.cos(t) - ey * np.sin(t)
    py = cy + ex * np.sin(t) + ey * np.cos(t)
    norm = np.hypot(px - cx, py - cy)
    dx, dy = (px - cx) / norm, (py - cy) / norm
    steps = np.arange(-band, band + 1)
    xs = np.clip(np.round(px[:, None] + dx[:, None] * steps), 0, w - 1).astype(int)
    ys = np.clip(np.round(py[:, None] + dy[:, None] * steps), 0, h - 1).astype(int)
    sharpness = np.median(grad_mag[ys, xs].max(axis=1))
    wall_level = np.median(gray[ys, xs].min(axis=1))
    return float(sharpness), float(wall_level)


def segment_spores(
    img_bgr: Image,
    interior_hue: tuple[int, int] = (8, 30),
    interior_sat: int = 35,
    rim_sat: int = 60,
    min_area_px: int = 150,
    max_area_px: int = 20000,
    min_solidity: float = 0.85,
    min_axis_ratio: float = 1.05,
    max_axis_ratio: float = 3.0,
    min_sat_above_bg: float = 25,
    min_rel_brightness: float = 0.75,
    min_sharpness: float = 0.9,
    rel_sharpness: float = 0.5,
    sharpness_floor: float = 0.4,
    max_wall_level: float = 0.6,
    wall_margin: float = 0.12,
    wall_ceiling: float = 0.8,
    include_touching: bool = False,
    fov_margin: float = 0.95,
    debug_prefix: str | None = None,
) -> list[Spore]:
    """Detect clean spores in a BGR image and fit an ellipse to each.

    Returns a list of dicts, one per spore, each with the fitted ellipse
    (center, length/width in px, angle) and quality metrics.

    Focus checks, in units of the background grey level: a spore's wall
    edge gradient must reach min_sharpness and its darkest wall level must
    be at most max_wall_level. Wall contrast and edge softness depend on the
    microscope setup, so in a low-contrast photo these are relaxed relative
    to the photo's own spores: gradient at least rel_sharpness x the photo's
    75th percentile, wall level at most wall_margin above the photo's median.
    They are never relaxed past sharpness_floor and wall_ceiling (ghost and
    badly blurred spores), and never made stricter than the fixed limits.

    If debug_prefix is given, also writes <prefix>_spore_mask.png and
    <prefix>_spore_overlay.png.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov_center, fov_r = find_fov_circle(gray)
    inner_fov = fov_mask(gray.shape, fov_center, fov_r, shrink=0.97)
    bg_gray = float(np.median(gray[inner_fov > 0]))
    grad_mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    )

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # spores cover a small fraction of the field, so the median is background
    bg_sat, bg_val = np.median(hsv[inner_fov > 0][:, 1:], axis=0)
    filled, interior = spore_silhouettes(hsv, inner_fov, interior_hue, interior_sat, rim_sat)

    min_interior_px = min_area_px // 4

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    candidates: list[Spore] = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area_px:
            continue
        comp = (labels[y : y + bh, x : x + bw] == i).astype(np.uint8) * 255
        if np.count_nonzero(interior[y : y + bh, x : x + bw] & comp) < min_interior_px:
            continue  # no brown interior: debris or a bubble fragment
        regions = _split_by_seeds(comp, _shape_seeds(comp))
        touching = len(regions) > 1
        if touching and not include_touching:
            continue
        for region in regions:
            s = _measure_region(
                region,
                (int(x), int(y)),
                min_area_px,
                max_area_px,
                min_solidity,
                min_axis_ratio,
                max_axis_ratio,
            )
            if s is None:
                continue
            (cx, cy), r_out = s["center_px"], s["length_px"] / 2
            if np.hypot(cx - fov_center[0], cy - fov_center[1]) + r_out > fov_margin * fov_r:
                continue  # at the edge of the field: vignetted / distorted
            _, sat, val = _interior_hsv_median(hsv, s["ellipse"])
            if sat < bg_sat + min_sat_above_bg:
                continue  # too pale: tissue or background tint, not a pigmented spore
            if val < min_rel_brightness * bg_val:
                continue  # much darker than the field: dirt, smudge or debris
            sharpness, wall_level = _wall_metrics(gray, grad_mag, s["ellipse"])
            s.update(
                touching=touching, sharpness=sharpness / bg_gray, wall_level=wall_level / bg_gray
            )
            candidates.append(s)

    # focus checks relative to this photo's own spores
    spores: list[Spore] = []
    if candidates:
        typical_sharpness = np.percentile([c["sharpness"] for c in candidates], 75)
        typical_wall = np.median([c["wall_level"] for c in candidates])
        sharp_min = max(sharpness_floor, min(min_sharpness, rel_sharpness * typical_sharpness))
        wall_max = min(wall_ceiling, max(max_wall_level, typical_wall + wall_margin))
        # soft or faint outline: out of focus, or a ghost/immature spore
        spores = [
            c for c in candidates if c["sharpness"] >= sharp_min and c["wall_level"] <= wall_max
        ]

    if debug_prefix:
        overlay = img_bgr.copy()
        for s in spores:
            cv2.ellipse(overlay, s["ellipse"], (0, 0, 255), 2)
        cv2.imwrite(f"{debug_prefix}_spore_mask.png", filled)
        cv2.imwrite(f"{debug_prefix}_spore_overlay.png", overlay)

    return spores
