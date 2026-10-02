"""
Segment fungal spores in a brightfield micrograph and measure their length x
width by fitting ellipses to each spore's silhouette.

Spores here show up as chromatic (colored, often orange/brown/green-fringed)
ovals against a neutral gray background -- so we segment on HSV saturation,
which cleanly separates spores (and the ruler's antialiasing, excluded
separately) from the flat background.
"""
import cv2
import numpy as np

from .fov import find_fov_circle, fov_mask


def segment_spores(img_bgr, sat_thresh=25, min_area_px=150, max_area_px=20000,
                    min_solidity=0.85, min_axis_ratio=1.05, max_axis_ratio=3.0,
                    debug_prefix=None):
    """
    Returns a list of dicts, one per detected spore, each with the fitted
    ellipse (center, length/width in px, angle) and shape-quality metrics.
    If debug_prefix is given, also writes <prefix>_spore_mask.png and
    <prefix>_spore_overlay.png.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    fov_center, fov_r = find_fov_circle(gray)
    inner_fov = fov_mask(gray.shape, fov_center, fov_r, shrink=0.97)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    spore_mask = ((sat > sat_thresh).astype(np.uint8)) * 255
    spore_mask = cv2.bitwise_and(spore_mask, spore_mask, mask=inner_fov)

    # clean up: close small gaps in outlines, then remove specks
    spore_mask = cv2.morphologyEx(spore_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    spore_mask = cv2.morphologyEx(spore_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # fill holes (interiors) so fitEllipse sees the whole spore, not just its rim
    filled = spore_mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    flood = filled.copy()
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(spore_mask, holes)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, connectivity=8)

    spores = []
    overlay = img_bgr.copy() if debug_prefix else None

    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px or area > max_area_px:
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        if len(cnt) < 5:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue  # likely touching/overlapping spores or debris cluster

        ellipse = cv2.fitEllipse(cnt)
        (ecx, ecy), axes, angle = ellipse
        # fitEllipse does not order its axes, so sort them explicitly
        minor_ax, major_ax = sorted(axes)
        if minor_ax <= 0:
            continue
        axis_ratio = major_ax / minor_ax
        if axis_ratio < min_axis_ratio or axis_ratio > max_axis_ratio:
            continue  # too round (likely debris/bubble) or too elongated (likely fused spores)

        # ellipse-fit quality: compare ellipse area to actual contour area
        ellipse_area = np.pi * (major_ax / 2) * (minor_ax / 2)
        area_ratio = area / ellipse_area if ellipse_area > 0 else 0
        if area_ratio < 0.85 or area_ratio > 1.15:
            continue  # poor ellipse fit, likely not a clean single spore

        spores.append({
            "center_px": (ecx, ecy),
            "length_px": major_ax,
            "width_px": minor_ax,
            "angle_deg": angle,
            "ellipse": ellipse,  # raw cv2 ellipse tuple, for drawing
            "area_px": area,
            "solidity": solidity,
        })

        if overlay is not None:
            cv2.ellipse(overlay, ellipse, (0, 0, 255), 2)

    if overlay is not None:
        cv2.imwrite(f"{debug_prefix}_spore_mask.png", filled)
        cv2.imwrite(f"{debug_prefix}_spore_overlay.png", overlay)

    return spores

