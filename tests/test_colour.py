import cv2
import numpy as np
import pytest

from spore_size import calibrate_image, process_image
from spore_size.colour import background_colour, correct_colour_cast, has_colour_cast

from conftest import SAMPLE, SAMPLE_PX_PER_TICK


def _with_cast(img):
    """The sample photo as shot with the white balance far off: blue nearly gone,
    red clipped, like the yellow-orange photos this handles."""
    f = img.astype(np.float32) * np.array([0.03, 1.0, 1.6], np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def test_colour_cast_is_detected_only_when_strong(sample_image):
    assert not has_colour_cast(sample_image)
    assert has_colour_cast(_with_cast(sample_image))


def test_calibration_through_a_colour_cast(sample_image):
    r = calibrate_image(_with_cast(sample_image), microns_per_tick=1.0)
    assert r["px_per_tick"] == pytest.approx(SAMPLE_PX_PER_TICK, rel=0.005)


def test_corrected_background_is_neutral(sample_image):
    cast = _with_cast(sample_image)
    corrected = correct_colour_cast(cast, calibrate_image(cast))
    b, g, r = background_colour(corrected)
    assert max(b, g, r) - min(b, g, r) < 12  # near grey, like a normal photo


def test_spores_measured_through_a_colour_cast(tmp_path, sample_image):
    (tmp_path / "normal").mkdir()
    (tmp_path / "cast").mkdir()
    _, normal = process_image(str(SAMPLE), str(tmp_path / "normal"), save_debug=False)
    cast_path = str(tmp_path / "cast.jpg")
    cv2.imwrite(cast_path, _with_cast(sample_image), [cv2.IMWRITE_JPEG_QUALITY, 95])
    _, cast = process_image(cast_path, str(tmp_path / "cast"), save_debug=False)
    assert len(cast) >= len(normal) - 3
    mean = np.mean([s["length_um"] for s in normal])
    assert np.mean([s["length_um"] for s in cast]) == pytest.approx(mean, abs=0.3)
