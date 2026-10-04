import csv
import os
import shutil

import cv2
import numpy as np
import pytest

from spore_size import calibrate_image, process_image, segment_spores
from spore_size.cli import directory_labels, main
from spore_size.errors import PhotoRejected
from spore_size.measure import drop_outliers_by_directory

from conftest import SAMPLE, SAMPLE_PX_PER_TICK, SAMPLE_RULER_ANGLE_DEG, SPORES, make_image

ISOLATED = SPORES[:4]  # the last two synthetic spores touch each other


def _nearest(spore, candidates):
    cx, cy = spore["center_px"]
    return min(candidates, key=lambda e: np.hypot(e[0][0] - cx, e[0][1] - cy))


def test_calibration_on_real_ruler(sample_image):
    r = calibrate_image(sample_image, microns_per_tick=1.0)
    assert r["px_per_tick"] == pytest.approx(SAMPLE_PX_PER_TICK, rel=0.003)
    assert r["microns_per_px"] == pytest.approx(1.0 / SAMPLE_PX_PER_TICK, rel=0.01)
    angle_err = (r["angle_deg"] - SAMPLE_RULER_ANGLE_DEG + 90) % 180 - 90
    assert abs(angle_err) < 2


def test_calibration_scales_with_um_per_tick(sample_image):
    r = calibrate_image(sample_image, microns_per_tick=2.5)
    assert r["microns_per_px"] == pytest.approx(2.5 / SAMPLE_PX_PER_TICK, rel=0.01)


def test_segmentation_isolated_spores_only_by_default(synthetic_image):
    """Bubbles ignored (even one touching a spore); touching pair left out."""
    spores = segment_spores(synthetic_image)
    assert len(spores) == len(ISOLATED)
    for s in spores:
        expected = _nearest(s, ISOLATED)
        assert s["length_px"] >= s["width_px"]
        assert s["length_px"] == pytest.approx(expected[1][0], rel=0.05)
        assert s["width_px"] == pytest.approx(expected[1][1], rel=0.05)


def test_segmentation_splits_touching_spores(synthetic_image):
    spores = segment_spores(synthetic_image, include_touching=True)
    assert len(spores) == len(SPORES)
    for s in spores:
        expected = _nearest(s, SPORES)
        assert s["length_px"] == pytest.approx(expected[1][0], rel=0.05)
        assert s["width_px"] == pytest.approx(expected[1][1], rel=0.05)


def test_segmentation_ignores_bubbles_alone():
    img = make_image()
    # paint over the spores with background, leaving only bubbles
    mask = np.zeros(img.shape[:2], np.uint8)
    for center, (length, width), angle in SPORES:
        cv2.ellipse(mask, center, (length // 2 + 3, width // 2 + 3), angle, 0, 360, 255, -1)
    img[mask > 0] = (175, 175, 175)
    assert segment_spores(img) == []


def test_process_image_on_real_micrograph(tmp_path):
    summary, kept = process_image(str(SAMPLE), str(tmp_path))
    assert summary["um_per_px"] == pytest.approx(1.0 / SAMPLE_PX_PER_TICK, rel=0.01)
    assert 8 <= summary["n"] <= 14
    assert summary["length_mean"] == pytest.approx(6.9, abs=0.3)
    assert summary["width_mean"] == pytest.approx(5.0, abs=0.3)
    for name in (
        "sample_spores_spore_measurements.csv",
        "sample_spores_spore_overlay.png",
        "sample_spores_calib_overlay.png",
    ):
        assert (tmp_path / name).exists()


def test_rejects_non_micrograph(tmp_path):
    # a bright, non-circular scene with no dark surround (like a specimen photo)
    img = np.full((800, 1200, 3), 170, np.uint8)
    cv2.rectangle(img, (100, 100), (700, 500), (60, 80, 120), -1)
    path = tmp_path / "box.png"
    cv2.imwrite(str(path), img)
    with pytest.raises(PhotoRejected, match="not a micrograph"):
        process_image(str(path), str(tmp_path))


def test_rejects_cluttered_field(tmp_path, sample_image):
    # cover most of the field of view (but not the dark surround) with "tissue"
    img = sample_image.copy()
    h, w = img.shape[:2]
    tissue = img.copy()
    rng = np.random.default_rng(0)
    for _ in range(400):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        colour = tuple(int(v) for v in rng.integers(60, 200, 3))
        cv2.circle(tissue, c, int(rng.integers(40, 120)), colour, -1)
    in_view = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 20
    img[in_view] = tissue[in_view]
    path = tmp_path / "tissue.png"
    cv2.imwrite(str(path), img)
    with pytest.raises(PhotoRejected, match="cluttered"):
        process_image(str(path), str(tmp_path))


def test_cli_skips_photo_without_ruler(tmp_path, capsys):
    img_path = tmp_path / "noruler.png"
    cv2.imwrite(str(img_path), make_image())
    assert main([str(img_path), "--out-dir", str(tmp_path / "out")]) == 1
    assert "[SKIPPED] noruler.png" in capsys.readouterr().err


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "out"
    assert main([str(SAMPLE), "--out-dir", str(out)]) == 0
    assert "sample_spores.jpg: n=" in capsys.readouterr().out
    assert (out / "all_spores_combined.csv").exists()


def test_cli_combined_csvs_per_directory(tmp_path):
    for d in ("collA", "collB"):
        (tmp_path / d).mkdir()
        shutil.copy(SAMPLE, tmp_path / d / f"{d}_photo.jpg")
    out = tmp_path / "out"
    images = ["collA/collA_photo.jpg", "collB/collB_photo.jpg"]
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert main(images + ["--out-dir", str(out), "--no-debug"]) == 0
    finally:
        os.chdir(cwd)

    with open(out / "all_spores_combined.csv") as f:
        all_images = {r["image"] for r in csv.DictReader(f)}
    assert all_images == set(images)
    for d, img in zip(("collA", "collB"), images, strict=True):
        with open(out / f"{d}_spores_combined.csv") as f:
            assert {r["image"] for r in csv.DictReader(f)} == {img}

    # one plot per image, per directory and for all spores
    for name in (
        "collA_photo_spore_plot.png",
        "collB_photo_spore_plot.png",
        "collA_spore_plot.png",
        "collB_spore_plot.png",
        "all_spores_plot.png",
    ):
        assert (out / name).stat().st_size > 0


def test_directory_labels_disambiguate_same_name():
    labels = directory_labels(["a/x", "b/x", "c/y"])
    assert labels == {"a/x": "a_x", "b/x": "b_x", "c/y": "y"}


def _spore(length, width):
    return {"length_um": length, "width_um": width, "Q_ratio": length / width}


def test_outliers_judged_per_directory():
    # photo a: half its detections are merged pairs (~2x length), which skews
    # its own median; pooled with photo b of the same directory they stand out
    rng = np.random.default_rng(0)
    normal = [_spore(7.0 + d, 5.0 + d / 2) for d in rng.normal(0, 0.3, 23)]
    merged = [_spore(11.0, 7.5), _spore(11.3, 7.2), _spore(10.4, 4.8)]
    spores = {"coll/a.jpg": normal[:3] + merged, "coll/b.jpg": normal[3:], "other/c.jpg": merged}
    kept, removed = drop_outliers_by_directory(spores)
    assert removed == {"coll": 3, "other": 0}  # "other" is too small to judge
    assert len(kept["coll/a.jpg"]) == 3 and len(kept["coll/b.jpg"]) == 20
    assert all(s["length_um"] < 9 for s in kept["coll/a.jpg"])
    assert all(s["image"] == "coll/a.jpg" for s in kept["coll/a.jpg"])


def test_low_contrast_spores_are_found():
    """Pale, softer walls (another scope setup) pass the photo-relative focus checks."""
    img = make_image(low_contrast=True)
    assert len(segment_spores(img)) == len(ISOLATED)
    # the fixed limits alone would reject every one of them
    assert segment_spores(img, sharpness_floor=0.9, wall_ceiling=0.6) == []


def test_blurred_spore_among_sharp_ones_is_rejected(synthetic_image):
    img = synthetic_image.copy()
    (cx, cy), (length, _), _ = ISOLATED[1]
    r = length // 2 + 25
    patch = img[cy - r : cy + r, cx - r : cx + r]
    img[cy - r : cy + r, cx - r : cx + r] = cv2.GaussianBlur(patch, (0, 0), 6)
    found = segment_spores(img)
    assert len(found) == len(ISOLATED) - 1
    assert all(np.hypot(s["center_px"][0] - cx, s["center_px"][1] - cy) > 20 for s in found)
