import numpy as np
import pytest

from spore_size.calibration import _fit_tick_spacing, _longest_regular_run


@pytest.mark.parametrize("true_spacing", [12.0, 12.4, 13.3, 14.25])
def test_fit_tick_spacing_is_sub_pixel(true_spacing):
    # tick positions in an image are whole pixels
    peaks = np.round(100 + true_spacing * np.arange(100)).astype(np.intp)
    median_gap = float(np.median(np.diff(peaks)))  # a whole number
    assert _fit_tick_spacing(peaks, median_gap) == pytest.approx(true_spacing, abs=0.01)


def test_fit_uses_longest_regular_run():
    # 60 regular ticks, a gap with spurious peaks, then 20 more regular ticks
    good = np.round(50 + 12.4 * np.arange(60))
    junk = good[-1] + np.array([5.0, 9.0, 31.0])
    tail = np.round(junk[-1] + 12.4 * np.arange(1, 21))
    peaks = np.concatenate([good, junk, tail]).astype(np.intp)
    assert _longest_regular_run(peaks, 12.0) == 59
    assert _fit_tick_spacing(peaks, 12.0) == pytest.approx(12.4, abs=0.01)
