"""
Segment fungal spores in a brightfield micrograph and measure their length x
width by fitting ellipses to each spore's silhouette.

Spores here are yellow-brown ovals with a dark red-brown wall, against a
near-neutral (often slightly pink) background. Air bubbles and oil droplets
are the main confusers: their interiors are greenish and their rims are red
but only weakly saturated. So a pixel counts as "spore" if it is either
brown and moderately saturated (spore interior) or red and strongly
saturated (spore wall). OpenCV hue runs 0-180, with red wrapping at 0/180.

Touching spores are separated using their brown interiors as seeds: the
walls between neighbouring spores keep each interior a separate blob, so
every pixel of a clump is assigned to its nearest interior. Clumps with no
brown interior at all (debris, bubble fragments) are dropped.

Finally each fitted spore is compared with the photo's own background: it
must be clearly more saturated (rejects pale tissue / mountant tint) and not
much darker (rejects dirt and out-of-focus smudges).
"""
import cv2
import numpy as np

from .fov import find_fov_circle, fov_mask


def spore_color_masks(hsv, interior_hue=(8, 30), interior_sat=35, rim_sat=60):
    """Binary masks (interior, wall) of pixels coloured like spore interior / wall."""
    hue, sat, _ = cv2.split(hsv)
    lo, hi = interior_hue
    interior = (hue >= lo) & (hue <= hi) & (sat > interior_sat)
    wall = ((hue < lo) | (hue >= 165)) & (sat > rim_sat)
    return interior.astype(np.uint8) * 255, wall.astype(np.uint8) * 255


def _fill_holes(mask):
    h, w = mask.shape
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood))


def _split_by_seeds(comp, seeds):
    """Split a component mask into one region per seed blob (nearest seed wins)."""
    n_seeds, seed_labels = cv2.connectedComponents(seeds, connectivity=8)
    if n_seeds <= 2:  # background + at most one seed: nothing to split
        return [comp] if n_seeds == 2 else []
    # nearest seed for every pixel; labels refer to the zero pixels' components
    _, nearest = cv2.distanceTransformWithLabels(
        cv2.bitwise_not(seeds), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    regions = []
    for s in range(1, n_seeds):
        lab = nearest[seed_labels == s][0]
        region = ((nearest == lab) & (comp > 0)).astype(np.uint8) * 255
        # keep only the part connected to this seed
        n, lbl = cv2.connectedComponents(region, connectivity=8)
        if n > 2:
            keep = np.bincount(lbl[seed_labels == s]).argmax()
            region = (lbl == keep).astype(np.uint8) * 255
        regions.append(region)
    return regions


def _measure_region(region, offset, min_area_px, max_area_px, min_solidity,
                    min_axis_ratio, max_axis_ratio):
    """Fit an ellipse to one candidate region; None if it fails a quality check."""
    area = int(np.count_nonzero(region))
    if area < min_area_px or area > max_area_px:
        return None
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
                               offset=offset)
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


def _interior_hsv_median(hsv, ellipse, scale=0.6):
    """Median H, S, V over the central part of a fitted ellipse (avoids the wall)."""
    (cx, cy), (a, b), angle = ellipse
    r = int(max(a, b) / 2) + 2
    x0, y0 = max(int(cx) - r, 0), max(int(cy) - r, 0)
    crop = hsv[y0:int(cy) + r + 1, x0:int(cx) + r + 1]
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.ellipse(m, ((cx - x0, cy - y0), (a * scale, b * scale), angle), 255, -1)
    return np.median(crop[m > 0], axis=0)


def segment_spores(img_bgr, interior_hue=(8, 30), interior_sat=35, rim_sat=60,
                   min_area_px=150, max_area_px=20000,
                   min_solidity=0.85, min_axis_ratio=1.05, max_axis_ratio=3.0,
                   min_sat_above_bg=25, min_rel_brightness=0.75,
                   debug_prefix=None):
    """
    Returns a list of dicts, one per detected spore, each with the fitted
    ellipse (center, length/width in px, angle) and shape-quality metrics.
    If debug_prefix is given, also writes <prefix>_spore_mask.png and
    <prefix>_spore_overlay.png.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fov_center, fov_r = find_fov_circle(gray)
    inner_fov = fov_mask(gray.shape, fov_center, fov_r, shrink=0.97)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # spores cover a small fraction of the field, so the median is background
    bg_sat, bg_val = np.median(hsv[inner_fov > 0][:, 1:], axis=0)
    interior, wall = spore_color_masks(hsv, interior_hue, interior_sat, rim_sat)
    spore_mask = cv2.bitwise_and(cv2.bitwise_or(interior, wall), inner_fov)

    # clean up: close small gaps in outlines, remove specks, fill interiors
    spore_mask = cv2.morphologyEx(spore_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    spore_mask = cv2.morphologyEx(spore_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    filled = _fill_holes(spore_mask)

    # seeds: solid brown interiors, opened to drop specks and thin bridges
    seeds = cv2.bitwise_and(interior, inner_fov)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    seeds = _fill_holes(seeds)
    min_seed_area = min_area_px // 4

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    spores = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < min_area_px:
            continue
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8) * 255
        comp_seeds = cv2.bitwise_and(seeds[y:y + bh, x:x + bw], comp)
        # drop seed specks too small to be a spore interior
        n, lbl, sstats, _ = cv2.connectedComponentsWithStats(comp_seeds, connectivity=8)
        big = np.flatnonzero(sstats[1:, cv2.CC_STAT_AREA] >= min_seed_area) + 1
        comp_seeds = np.isin(lbl, big).astype(np.uint8) * 255

        for region in _split_by_seeds(comp, comp_seeds):
            s = _measure_region(region, (int(x), int(y)), min_area_px, max_area_px,
                                min_solidity, min_axis_ratio, max_axis_ratio)
            if s is None:
                continue
            _, sat, val = _interior_hsv_median(hsv, s["ellipse"])
            if sat < bg_sat + min_sat_above_bg:
                continue  # too pale: tissue or background tint, not a pigmented spore
            if val < min_rel_brightness * bg_val:
                continue  # much darker than the field: dirt, smudge or debris
            spores.append(s)

    if debug_prefix:
        overlay = img_bgr.copy()
        for s in spores:
            cv2.ellipse(overlay, s["ellipse"], (0, 0, 255), 2)
        cv2.imwrite(f"{debug_prefix}_spore_mask.png", filled)
        cv2.imwrite(f"{debug_prefix}_spore_overlay.png", overlay)

    return spores
