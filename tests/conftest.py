"""Test fixtures.

- sample_spores.jpg: the field of view of a real 1000x micrograph of
  Thaxterogaster spores with the ocular scale bar (known calibration:
  12 px per tick, i.e. 0.0833 um/px; ~11 clean isolated spores, ~6.9 x 5.0 um).
- make_image(): synthetic micrograph with brown elliptical "spores" of known
  size (including a touching pair and one with an oil drop inside) plus
  clear bubbles, for exercising
  segmentation.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

SAMPLE = Path(__file__).parent / "data" / "sample_spores.jpg"
SAMPLE_PX_PER_TICK = 12.0
SAMPLE_RULER_ANGLE_DEG = 62.0  # modulo 180

SIZE = 1400
# (center, (length_px, width_px), angle_deg) -- varied angles exercise axis ordering
SPORES = [
    ((420, 400), (110, 70), 0),
    ((980, 420), (120, 75), 90),
    ((400, 1000), (100, 60), 45),
    ((1000, 1000), (130, 80), 135),
    # a touching pair, end to end
    ((640, 700), (110, 70), 0),
    ((752, 700), (110, 70), 0),
]
# (center, radius): clear bubbles with a weak red rim and greenish interior,
# one of them touching the first spore
BUBBLES = [((515, 400), 40), ((1000, 700), 45)]

SPORE_WALL_BGR = (60, 40, 150)  # dark, strongly saturated red-brown
SPORE_INNER_BGR = (60, 130, 190)  # yellow-brown
BUBBLE_RIM_BGR = (160, 150, 185)  # weakly saturated pink-red (S~48)
BUBBLE_INNER_BGR = (170, 190, 160)  # pale green
OIL_DROP_BGR = (175, 195, 170)  # pale, nearly colourless


def make_image():
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    c = SIZE // 2
    cv2.circle(img, (c, c), int(SIZE * 0.48), (175, 175, 175), -1)

    for center, r in BUBBLES:
        cv2.circle(img, center, r, BUBBLE_INNER_BGR, -1, cv2.LINE_AA)
        cv2.circle(img, center, r, BUBBLE_RIM_BGR, 6, cv2.LINE_AA)
    for center, (length, width), angle in SPORES:
        axes = (length // 2, width // 2)
        cv2.ellipse(img, center, axes, angle, 0, 360, SPORE_WALL_BGR, -1, cv2.LINE_AA)
        inner = (axes[0] - 5, axes[1] - 5)
        cv2.ellipse(img, center, inner, angle, 0, 360, SPORE_INNER_BGR, -1, cv2.LINE_AA)
    # a large central oil drop in the first spore, spanning its whole width
    # so that it cuts the brown interior in two
    (cx, cy), (_, width), _ = SPORES[0]
    cv2.circle(img, (cx, cy), width // 2 - 4, OIL_DROP_BGR, -1, cv2.LINE_AA)
    return img


@pytest.fixture
def synthetic_image():
    return make_image()


@pytest.fixture
def sample_image():
    return cv2.imread(str(SAMPLE))
