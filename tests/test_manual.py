import csv

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent  # noqa: E402

from spore_size import manual  # noqa: E402
from spore_size.manual import (  # noqa: E402
    ImageMeasurement,
    build_window,
    main,
    manual_csv_path,
    open_measurement,
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
