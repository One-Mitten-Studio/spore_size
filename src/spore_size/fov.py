"""Detection of the circular microscope field of view (FOV)."""
import cv2
import numpy as np


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
