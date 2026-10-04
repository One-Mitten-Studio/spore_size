"""Detection of the circular microscope field of view (FOV)."""

import cv2
import numpy as np

from .errors import PhotoRejected


def find_fov_circle(gray, threshold=20):
    """Return ((cx, cy), radius) of the bright circular field of view."""
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("Could not locate the field of view (image is entirely dark)")
    c = max(cnts, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(c)
    return (x, y), radius


def fov_mask(shape, center, radius, shrink=1.0):
    """Filled circular mask of the FOV, optionally shrunk by a factor."""
    out = np.zeros(shape[:2], np.uint8)
    cv2.circle(out, (int(center[0]), int(center[1])), int(radius * shrink), 255, -1)
    return out


def check_micrograph(gray, min_circle_fill=0.9, min_dark_fraction=0.1):
    """
    Raise PhotoRejected unless the image looks like a photo down a microscope:
    a bright, circular field of view surrounded by a dark border.
    """
    (cx, cy), r = find_fov_circle(gray)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    circle = fov_mask(gray.shape, (cx, cy), r)
    fill = np.count_nonzero(cv2.bitwise_and(mask, circle)) / max(np.count_nonzero(circle), 1)
    dark = float(np.mean(gray <= 20))
    if fill < min_circle_fill or dark < min_dark_fraction:
        raise PhotoRejected("not a micrograph (no circular field of view)")
