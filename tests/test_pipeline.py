import cv2
import numpy as np
import pytest

from spore_size import calibrate_image, process_image, segment_spores
from spore_size.cli import main
from conftest import (SAMPLE_PX_PER_TICK, SAMPLE_RULER, SAMPLE_RULER_ANGLE_DEG,
                      SPORES, make_image)


def test_calibration_on_real_ruler(sample_ruler_image):
    r = calibrate_image(sample_ruler_image, microns_per_tick=1.0)
    assert r["px_per_tick"] == pytest.approx(SAMPLE_PX_PER_TICK, rel=0.01)
    assert r["microns_per_px"] == pytest.approx(1.0 / SAMPLE_PX_PER_TICK, rel=0.01)
    angle_err = (r["angle_deg"] - SAMPLE_RULER_ANGLE_DEG + 90) % 180 - 90
    assert abs(angle_err) < 2


def test_calibration_scales_with_um_per_tick(sample_ruler_image):
    r = calibrate_image(sample_ruler_image, microns_per_tick=2.5)
    assert r["microns_per_px"] == pytest.approx(2.5 / SAMPLE_PX_PER_TICK, rel=0.01)


def test_segmentation_finds_all_spores_any_orientation(synthetic_image):
    spores = segment_spores(synthetic_image)
    assert len(spores) == len(SPORES)
    for s in spores:
        assert s["length_px"] >= s["width_px"]
        expected = min(SPORES, key=lambda e: np.hypot(e[0][0] - s["center_px"][0],
                                                     e[0][1] - s["center_px"][1]))
        assert s["length_px"] == pytest.approx(expected[1][0], rel=0.05)
        assert s["width_px"] == pytest.approx(expected[1][1], rel=0.05)


def test_process_image_writes_outputs(tmp_path):
    summary, kept = process_image(str(SAMPLE_RULER), str(tmp_path))
    assert summary["um_per_px"] == pytest.approx(1.0 / SAMPLE_PX_PER_TICK, rel=0.01)
    assert summary["n"] == len(kept)
    for name in ("sample_ruler_spore_measurements.csv", "sample_ruler_spore_overlay.png",
                 "sample_ruler_calib_overlay.png"):
        assert (tmp_path / name).exists()
    for s in kept:
        assert s["Q_ratio"] >= 1.0


def test_cli_reports_failure_without_ruler(tmp_path, capsys):
    img_path = tmp_path / "noruler.png"
    cv2.imwrite(str(img_path), make_image())
    assert main([str(img_path), "--out-dir", str(tmp_path / "out")]) == 1
    assert "[FAILED]" in capsys.readouterr().err


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "out"
    assert main([str(SAMPLE_RULER), "--out-dir", str(out)]) == 0
    assert "sample_ruler.jpg: n=" in capsys.readouterr().out
