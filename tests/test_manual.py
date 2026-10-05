import csv

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent  # noqa: E402

from spore_size import manual  # noqa: E402
from spore_size.manual import (  # noqa: E402
    ZOOM_STEP,
    ImageMeasurement,
    build_window,
    expected_calibration,
    flag_saved_calibration,
    instructions,
    keep_within,
    main,
    manual_csv_path,
    open_measurement,
    zoom_view,
)

from conftest import SAMPLE, SAMPLE_PX_PER_TICK  # noqa: E402


def _measure_one(m, length_ends, width_ends):
    events = [m.click(*p) for p in (*length_ends, *width_ends)]
    assert events == ["point", "point", "point", "spore"]


def test_clicks_make_a_spore_in_microns():
    m = ImageMeasurement("a.jpg", um_per_px=0.1, calibration="auto")
    _measure_one(m, [(0, 0), (70, 0)], [(30, -25), (30, 25)])
    (row,) = m.spore_rows()
    assert row["length_um"] == pytest.approx(7.0)
    assert row["width_um"] == pytest.approx(5.0)
    assert row["Q_ratio"] == pytest.approx(1.4)


def test_length_and_width_clicked_in_either_order():
    m = ImageMeasurement("a.jpg", um_per_px=0.1)
    _measure_one(m, [(30, -25), (30, 25)], [(0, 0), (70, 0)])  # width first
    (row,) = m.spore_rows()
    assert (row["length_um"], row["width_um"]) == pytest.approx((7.0, 5.0))


def test_undo_removes_point_then_spore():
    m = ImageMeasurement("a.jpg", um_per_px=0.1)
    _measure_one(m, [(0, 0), (70, 0)], [(30, -25), (30, 25)])
    m.click(100, 100)
    m.undo()
    assert m.pending == [] and len(m.spores) == 1
    m.undo()
    assert m.spores == []


def test_manual_calibration_from_scale_ends():
    m = ImageMeasurement("a.jpg", um_per_tick=1.0, ruler_ticks=100)
    assert m.calibrating
    assert m.click(100, 100) == "point"
    assert m.click(1300, 100) == "calibrated"  # 100 ticks over 1200 px
    assert m.um_per_px == pytest.approx(1 / 12)
    assert m.calibration == "manual" and not m.calibrating


def test_csv_round_trip(tmp_path):
    m = ImageMeasurement("dir/a.jpg", um_per_px=0.0833, calibration="auto")
    _measure_one(m, [(10, 20), (95, 50)], [(40, 10), (60, 70)])
    _measure_one(m, [(200, 200), (280, 200)], [(240, 170), (240, 230)])
    path = tmp_path / "a_manual_measurements.csv"
    m.save_csv(str(path))
    back = ImageMeasurement.load_csv("dir/a.jpg", str(path))
    assert back.um_per_px == pytest.approx(0.0833)
    assert back.calibration == "auto"
    assert len(back.spores) == 2
    for a, b in zip(m.spore_rows(), back.spore_rows(), strict=True):
        assert a["length_um"] == pytest.approx(b["length_um"], abs=0.01)
        assert a["width_um"] == pytest.approx(b["width_um"], abs=0.01)


def test_open_measurement_uses_auto_calibration(tmp_path):
    m = open_measurement(str(SAMPLE), str(tmp_path), um_per_tick=1.0, ruler_ticks=100)
    assert m.calibration == "auto" and not m.calibrating
    assert m.um_per_px == pytest.approx(1 / SAMPLE_PX_PER_TICK, rel=0.003)


def _click(fig, ax, x, y):
    px, py = ax.transData.transform((x, y))
    event = MouseEvent("button_press_event", fig.canvas, px, py, button=MouseButton.LEFT)
    fig.canvas.callbacks.process("button_press_event", event)


def _key(fig, key):
    fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", fig.canvas, key))


def test_window_clicks_keys_and_saving(tmp_path):
    m = ImageMeasurement(str(SAMPLE), um_per_px=0.1, calibration="auto")
    csv_path = str(tmp_path / "s_manual_measurements.csv")
    fig, state = build_window(m, csv_path)
    ax = fig.axes[0]
    for p in [(500, 500), (570, 500), (530, 475), (530, 525)]:
        _click(fig, ax, *p)
    assert len(m.spores) == 1
    with open(csv_path) as f:  # saved after the spore
        assert len(list(csv.DictReader(f))) == 1
    _click(fig, ax, 800, 800)
    _key(fig, "u")
    assert m.pending == []
    _key(fig, "c")
    assert m.calibrating
    _key(fig, "q")
    assert state["quit"]
    plt.close("all")


def test_report_only_builds_reports(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    m = ImageMeasurement(str(SAMPLE), um_per_px=0.1, calibration="auto")
    _measure_one(m, [(0, 0), (70, 0)], [(30, -25), (30, 25)])
    _measure_one(m, [(0, 0), (66, 0)], [(30, -23), (30, 23)])
    m.save_csv(manual_csv_path(str(SAMPLE), str(out)))
    assert main([str(SAMPLE), "--out-dir", str(out), "--report-only"]) == 0
    assert "2 spores measured by hand" in capsys.readouterr().out
    for name in ("spore_summary.csv", "all_spores_combined.csv", "all_spores_plot.png"):
        assert (out / name).exists()


def test_qt_display_problem_reports_missing_xcb_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """On X11 without libxcb-cursor, explain the fix instead of letting Qt abort."""
    monkeypatch.setattr(manual.sys, "platform", "linux")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(manual.ctypes.util, "find_library", lambda name: None)
    problem = manual.qt_display_problem()
    assert problem is not None and "libxcb-cursor0" in problem
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert manual.qt_display_problem() is None
    monkeypatch.setattr(manual.ctypes.util, "find_library", lambda name: "libxcb-cursor.so.0")
    monkeypatch.delenv("WAYLAND_DISPLAY")
    assert manual.qt_display_problem() is None


FULL_X, FULL_Y = (-0.5, 999.5), (749.5, -0.5)  # an image's view: y runs top to bottom


def test_zoom_keeps_point_under_pointer_fixed():
    xlim, ylim = zoom_view(FULL_X, FULL_Y, 300, 200, 0.5, FULL_X, FULL_Y)
    assert xlim == pytest.approx((149.75, 649.75))
    assert ylim == pytest.approx((474.75, 99.75))  # still inverted


def test_zoom_out_stops_at_whole_photo_and_in_at_min_view():
    xlim, ylim = zoom_view((100, 200), (200, 100), 150, 150, 50, FULL_X, FULL_Y)
    assert (xlim, ylim) == (FULL_X, FULL_Y)
    assert zoom_view((100, 125), (125, 100), 110, 110, 0.5, FULL_X, FULL_Y) == (
        (100, 125),
        (125, 100),
    )


def test_keep_within_shifts_view_back_inside():
    assert keep_within((-50.5, 49.5), FULL_X) == (-0.5, 99.5)
    assert keep_within((780, 680), FULL_Y) == (749.5, 649.5)
    assert keep_within((10, 20), FULL_X) == (10, 20)


def test_window_wheel_zoom_and_drag_pan(tmp_path):
    m = ImageMeasurement(str(SAMPLE), um_per_px=0.1, calibration="auto")
    fig, _ = build_window(m, str(tmp_path / "s_manual_measurements.csv"))
    ax = fig.axes[0]
    full = ax.get_xlim(), ax.get_ylim()
    assert "Spore 1" in fig.texts[0].get_text()  # status lines sit above the photo
    assert fig.texts[0].get_position()[1] > ax.get_position().y1

    px, py = ax.transData.transform((800, 600))
    for _ in range(3):
        event = MouseEvent("scroll_event", fig.canvas, px, py, step=1)
        fig.canvas.callbacks.process("scroll_event", event)
    xlim = ax.get_xlim()
    assert xlim[1] - xlim[0] == pytest.approx((full[0][1] - full[0][0]) / ZOOM_STEP**3)
    # the photo point under the pointer has not moved
    assert ax.transData.inverted().transform((px, py)) == pytest.approx((800, 600))

    def mouse(name, x, y, button=MouseButton.RIGHT):
        fig.canvas.callbacks.process(name, MouseEvent(name, fig.canvas, x, y, button=button))

    mouse("button_press_event", px, py)
    mouse("motion_notify_event", px + 50, py)
    mouse("button_release_event", px + 50, py)
    assert ax.get_xlim()[0] < xlim[0]  # dragged right: the view moved left
    assert m.pending == []  # panning adds no measurement points

    _key(fig, "-")
    _key(fig, "-")
    _key(fig, "-")
    assert (ax.get_xlim(), ax.get_ylim()) == full
    plt.close("all")


def test_hand_calibration_far_from_other_photos_is_held_for_checking():
    # the mistake that happened: a spore's two ends (77 px) clicked as the 0-100 scale
    m = ImageMeasurement("a.jpg", expected_um_per_px=0.085)
    assert m.click(1990, 1730) == "point"
    assert m.click(2060, 1698) == "suspect"
    assert m.um_per_px is None and m.calibrating
    assert "CHECK CALIBRATION" in instructions(m)
    m.click(1400, 2040)  # now the real 0 and 100 ticks
    assert m.click(2300, 1240) == "calibrated"
    assert m.um_per_px == pytest.approx(100 / 1204.2, rel=1e-3)
    assert m.suspect_um_per_px is None and not m.calibrating


def test_suspect_calibration_can_be_accepted():
    m = ImageMeasurement("a.jpg", expected_um_per_px=0.085)
    m.click(0, 0)
    m.click(1000, 0)  # 0.1 µm/px: 18% off
    assert m.accept_suspect_calibration()
    assert m.um_per_px == pytest.approx(0.1) and m.calibration == "manual"
    assert not m.accept_suspect_calibration()


def test_without_other_photos_a_tiny_scale_span_is_held():
    m = ImageMeasurement("a.jpg", diagonal_px=5000)
    m.click(0, 0)
    assert m.click(77, 0) == "suspect"
    m.click(0, 0)
    assert m.click(1200, 0) == "calibrated"


def test_saved_bad_calibration_is_flagged_and_spores_rescale():
    m = ImageMeasurement("a.jpg", um_per_px=1.307, calibration="manual")
    _measure_one(m, [(0, 0), (77, 0)], [(38, -26), (38, 26)])
    m.expected_um_per_px = 0.085
    assert "15x" in flag_saved_calibration(m)
    assert m.calibrating and m.suspect_um_per_px == 1.307
    m.click(0, 0)
    m.click(1176, 0)
    (row,) = m.spore_rows()  # the saved clicks are kept, now in the right scale
    assert row["length_um"] == pytest.approx(77 * 100 / 1176)


def test_status_shows_last_spore_size():
    m = ImageMeasurement("a.jpg", um_per_px=0.1, calibration="auto")
    _measure_one(m, [(0, 0), (70, 0)], [(30, -25), (30, 25)])
    assert "last spore 7.00 x 5.00 µm" in instructions(m)


def test_report_warns_about_odd_calibration(tmp_path, capsys):
    paths = [str(tmp_path / f"p{i}.jpg") for i in range(3)]
    for path, um_per_px in zip(paths, [0.085, 1.307, 0.084], strict=True):
        m = ImageMeasurement(path, um_per_px=um_per_px, calibration="manual")
        _measure_one(m, [(0, 0), (70, 0)], [(30, -25), (30, 25)])
        m.save_csv(manual_csv_path(path, str(tmp_path)))
    assert main([*paths, "--out-dir", str(tmp_path), "--report-only", "--no-plots"]) == 0
    out = capsys.readouterr().out
    assert out.count("[WARNING]") == 1 and "1.3070 µm/px" in out


def test_expected_calibration_trusts_automatic_ones():
    calibs = {"a": (0.085, "auto"), "b": (1.307, "manual"), "c": (0.084, "manual")}
    assert expected_calibration(calibs, "c") == 0.085  # the auto one, not the bad one
    assert expected_calibration(calibs, "a") == 0.085  # median of all, a bad one outvoted
    assert expected_calibration({}, "a") is None
