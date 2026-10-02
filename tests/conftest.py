"""Test fixtures.

- sample_ruler.jpg: a crop of a real micrograph around the ocular scale bar
  (known calibration: 12 px per tick, i.e. 0.0833 um/px).
- make_image(): synthetic micrograph with coloured elliptical "spores" of
  known size, for exercising segmentation.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

SAMPLE_RULER = Path(__file__).parent / "data" / "sample_ruler.jpg"
SAMPLE_PX_PER_TICK = 12.0
SAMPLE_RULER_ANGLE_DEG = 30.0  # modulo 180

SIZE = 1400
# (center, (length_px, width_px), angle_deg) -- varied angles exercise axis ordering
SPORES = [
    ((420, 400), (110, 70), 0),
    ((980, 420), (120, 75), 90),
    ((400, 1000), (100, 60), 45),
    ((1000, 1000), (130, 80), 135),
]


def make_image():
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    c = SIZE // 2
    cv2.circle(img, (c, c), int(SIZE * 0.48), (175, 175, 175), -1)

    for center, (length, width), angle in SPORES:
        cv2.ellipse(img, center, (length // 2, width // 2), angle, 0, 360,
                    (40, 110, 190), -1, cv2.LINE_AA)
    return img


@pytest.fixture
def synthetic_image():
    return make_image()


@pytest.fixture
def sample_ruler_image():
    return cv2.imread(str(SAMPLE_RULER))
