"""Shared type aliases for annotations."""

from __future__ import annotations

from typing import Any, TypedDict

from cv2.typing import MatLike

#: image or mask as OpenCV handles it: BGR / HSV (h, w, 3) or single channel
#: (h, w); OpenCV's own type, since its functions return any int/float dtype
Image = MatLike
#: (x, y) in pixels
Point = tuple[float, float]
#: OpenCV rotated rectangle / ellipse: (centre, (axis, axis), angle in degrees)
CvEllipse = tuple[Point, tuple[float, float], float]
#: one measured spore; keys grow along the pipeline (px measures from
#: segmentation, then length_um / width_um / Q_ratio, then image in the CLI)
Spore = dict[str, Any]
#: one row of the run summary (see summary.summary_rows)
SummaryRow = dict[str, Any]


class Calibration(TypedDict):
    """Result of calibration.calibrate_image."""

    px_per_tick: float
    microns_per_px: float
    angle_deg: float
    center: Point
    n_peaks: int
    regular_run: int
