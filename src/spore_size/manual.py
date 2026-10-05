"""Check automatic spore measurements and add more by hand.

First spore-size's automatic analysis runs on all the photos (calibration,
spore detection, size outliers judged per directory), unless --no-auto.
Then each photo opens in a window with the automatic spores outlined, to
check them: delete a wrong one (d with the pointer on it), and measure more
by hand by clicking the two ends of a spore's length, then the two sides of
its width. If the scale bar can't be found automatically, click its 0 and
100 ticks instead. Measurements, automatic and by hand, are saved after
every change (<photo>_manual_measurements.csv, with a source column) and
restored when a photo is reopened, without rerunning the automatic
analysis. The run's reports (combined CSVs, spore_summary.csv, plots) have
the same format as spore-size's.

Mouse wheel zooms about the pointer; drag with the right or middle button to pan.
Keys: u / backspace undo, d delete the spore at the pointer, c calibrate by
hand, n next photo, q save and quit.
"""

from __future__ import annotations

import argparse
import csv
import ctypes.util
import os
import sys
from collections.abc import Sequence
from typing import Any, Final

import cv2
import numpy as np

from ._types import CvEllipse, Point, Spore
from .calibration import calibrate_image
from .errors import PhotoRejected
from .measure import analyze_image, drop_outliers_by_directory
from .summary import DEFAULT_SYS_ERROR_UM

ZOOM_STEP: Final = 1.25  # view shrinks or grows by this factor per wheel step or +/- key
MIN_VIEW_PX: Final = 20.0  # never zoom in further than this many image pixels across
# a hand calibration this far from the other photos' (as a fraction) must be confirmed
CALIB_TOLERANCE: Final = 0.15
# without other photos to compare with, the clicked full scale (ruler_ticks) must span at
# least this fraction of the photo's diagonal; a spore clicked by mistake spans far less
MIN_SCALE_FRACTION: Final = 0.05

Limits = tuple[float, float]
#: a spore's length ends, then its width sides
SporePoints = tuple[Point, Point, Point, Point]

CSV_FIELDS = [
    "spore_id",
    "length_um",
    "width_um",
    "Q_ratio",
    "length_px",
    "width_px",
    "length_x1",
    "length_y1",
    "length_x2",
    "length_y2",
    "width_x1",
    "width_y1",
    "width_x2",
    "width_y2",
    "um_per_px",
    "calibration",
    "source",
]


def _distance(a: Point, b: Point) -> float:
    """Return the distance between two points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def ellipse_points(ellipse: CvEllipse) -> SporePoints:
    """Return an OpenCV ellipse's major axis ends, then its minor axis ends.

    These are the points a hand measurement would click, so an automatically
    measured spore has the same length and width either way.
    """
    (cx, cy), (a, b), angle = ellipse
    t = np.radians(angle)
    ux, uy = np.cos(t), np.sin(t)  # direction of the first axis, a
    ends_a = ((cx - ux * a / 2, cy - uy * a / 2), (cx + ux * a / 2, cy + uy * a / 2))
    ends_b = ((cx + uy * b / 2, cy - ux * b / 2), (cx - uy * b / 2, cy + ux * b / 2))
    (l1, l2), (w1, w2) = (ends_a, ends_b) if a >= b else (ends_b, ends_a)
    pts = [(float(x), float(y)) for x, y in (l1, l2, w1, w2)]
    return pts[0], pts[1], pts[2], pts[3]


class ImageMeasurement:
    """Measurements of one photo: calibration, spores and pending clicks.

    Each spore has a source, "auto" (from spore-size's detection) or "manual".
    Clicks build up a manual spore: two points for the length, two for the width.
    In calibration mode the next two clicks are the 0 and full-scale ticks
    of the scale bar instead. A calibration far from expected_um_per_px (the
    other photos' calibration), or spanning implausibly few pixels, is held as
    suspect_um_per_px until accepted or clicked again.
    """

    def __init__(
        self,
        path: str,
        um_per_px: float | None = None,
        calibration: str = "",
        um_per_tick: float = 1.0,
        ruler_ticks: float = 100.0,
        expected_um_per_px: float | None = None,
        diagonal_px: float | None = None,
    ) -> None:
        """Start measuring a photo, with a known calibration or none yet."""
        self.path = path
        self.um_per_px = um_per_px
        self.calibration = calibration
        self.um_per_tick = um_per_tick
        self.ruler_ticks = ruler_ticks
        self.spores: list[SporePoints] = []
        self.sources: list[str] = []  # "auto" or "manual", one per spore
        # what undo reverses, newest last: ("add", index) or ("delete", index, points, source)
        self.history: list[tuple[Any, ...]] = []
        self.pending: list[Point] = []
        self.calibrating = um_per_px is None
        self.expected_um_per_px = expected_um_per_px
        self.diagonal_px = diagonal_px
        self.suspect_um_per_px: float | None = None
        self.suspect_reason = ""

    def start_calibration(self) -> None:
        """Make the next two clicks the scale bar's 0 and full-scale ticks."""
        self.calibrating = True
        self.pending = []
        self.suspect_um_per_px = None

    def calibration_problem(self, um_per_px: float, span_px: float) -> str:
        """Return why a hand calibration looks wrong, or "" if it looks right."""
        expected = self.expected_um_per_px
        if expected:
            if abs(um_per_px / expected - 1) > CALIB_TOLERANCE:
                return (
                    f"{um_per_px:.4f} µm/px is {um_per_px / expected:.2g}x "
                    f"the other photos' {expected:.4f} µm/px"
                )
            return ""
        if self.diagonal_px and span_px < MIN_SCALE_FRACTION * self.diagonal_px:
            return f"the two clicks are only {span_px:.0f} px apart"
        return ""

    def accept_suspect_calibration(self) -> bool:
        """Accept a held calibration as it is; return False if none is held."""
        if self.suspect_um_per_px is None:
            return False
        self.um_per_px = self.suspect_um_per_px
        self.calibration = "manual"
        self.calibrating = False
        self.suspect_um_per_px = None
        return True

    def click(self, x: float, y: float) -> str:
        """Add a clicked point; return "point", "spore", "calibrated" or "suspect"."""
        self.pending.append((x, y))
        if self.calibrating:
            if len(self.pending) == 1:
                self.suspect_um_per_px = None
            if len(self.pending) < 2:
                return "point"
            span_px = _distance(self.pending[0], self.pending[1])
            self.pending = []
            if span_px <= 0:
                return "point"
            um_per_px = self.ruler_ticks * self.um_per_tick / span_px
            problem = self.calibration_problem(um_per_px, span_px)
            if problem:
                self.suspect_um_per_px, self.suspect_reason = um_per_px, problem
                return "suspect"
            self.suspect_um_per_px = um_per_px
            self.accept_suspect_calibration()
            return "calibrated"
        if len(self.pending) < 4:
            return "point"
        p = self.pending
        self.add_spore((p[0], p[1], p[2], p[3]), "manual")
        self.history.append(("add", len(self.spores) - 1))
        self.pending = []
        return "spore"

    def add_spore(self, points: SporePoints, source: str) -> None:
        """Add a finished spore."""
        self.spores.append(points)
        self.sources.append(source)

    def spore_at(self, x: float, y: float) -> int | None:
        """Return the index of the spore under (x, y), or None.

        That is the spore whose centre is nearest, if (x, y) is within half its
        length (at least 10 px) of the centre.
        """
        best, best_d = None, np.inf
        for i, (l1, l2, _, _) in enumerate(self.spores):
            centre = ((l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2)
            d = _distance(centre, (x, y))
            if d <= max(_distance(l1, l2) / 2, 10.0) and d < best_d:
                best, best_d = i, d
        return best

    def delete_at(self, x: float, y: float) -> bool:
        """Delete the spore under (x, y); return False if there is none."""
        i = self.spore_at(x, y)
        if i is None:
            return False
        self.history.append(("delete", i, self.spores.pop(i), self.sources.pop(i)))
        return True

    def undo(self) -> None:
        """Undo the last pending point, else the last spore added or deleted.

        Without anything to undo from this session, a reopened photo's last
        manual spore is removed.
        """
        if self.pending:
            self.pending.pop()
        elif self.calibrating:
            return
        elif self.history:
            action = self.history.pop()
            if action[0] == "add":
                del self.spores[action[1]], self.sources[action[1]]
            else:
                _, i, points, source = action
                self.spores.insert(i, points)
                self.sources.insert(i, source)
        elif self.sources and self.sources[-1] == "manual":
            self.spores.pop()
            self.sources.pop()

    def counts(self) -> tuple[int, int]:
        """Return the number of (automatic, manual) spores."""
        n_auto = self.sources.count("auto")
        return n_auto, len(self.sources) - n_auto

    def spore_rows(self) -> list[Spore]:
        """Return the spores as rows in microns (length >= width), like spore-size."""
        if self.um_per_px is None:
            return []
        rows = []
        for (l1, l2, w1, w2), source in zip(self.spores, self.sources, strict=True):
            a, b = _distance(l1, l2), _distance(w1, w2)
            length_px, width_px = max(a, b), min(a, b)
            length_um, width_um = length_px * self.um_per_px, width_px * self.um_per_px
            rows.append(
                {
                    "image": self.path,
                    "length_px": length_px,
                    "width_px": width_px,
                    "length_um": length_um,
                    "width_um": width_um,
                    "Q_ratio": length_um / width_um if width_um > 0 else np.nan,
                    "points": (l1, l2, w1, w2),
                    "source": source,
                }
            )
        return rows

    def save_csv(self, csv_path: str) -> None:
        """Write the spores, with their clicked points and calibration."""
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_FIELDS)
            for i, r in enumerate(self.spore_rows(), 1):
                l1, l2, w1, w2 = r["points"]
                w.writerow(
                    [
                        i,
                        f"{r['length_um']:.2f}",
                        f"{r['width_um']:.2f}",
                        f"{r['Q_ratio']:.2f}",
                        f"{r['length_px']:.1f}",
                        f"{r['width_px']:.1f}",
                        *(f"{v:.2f}" for pt in (l1, l2, w1, w2) for v in pt),
                        f"{self.um_per_px:.6f}",
                        self.calibration,
                        r["source"],
                    ]
                )

    @classmethod
    def load_csv(
        cls, path: str, csv_path: str, um_per_tick: float = 1.0, ruler_ticks: float = 100.0
    ) -> ImageMeasurement:
        """Restore a photo's saved measurements (files without a source are all manual)."""
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        m = cls(path, um_per_tick=um_per_tick, ruler_ticks=ruler_ticks)
        if rows:
            m.um_per_px = float(rows[0]["um_per_px"])
            m.calibration = rows[0]["calibration"]
            m.calibrating = False
        for r in rows:
            pts = [
                (float(r[f"{k}_x{i}"]), float(r[f"{k}_y{i}"]))
                for k in ("length", "width")
                for i in (1, 2)
            ]
            m.add_spore((pts[0], pts[1], pts[2], pts[3]), r.get("source") or "manual")
        return m


def manual_csv_path(path: str, out_dir: str) -> str:
    """Return where a photo's hand measurements are saved."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(out_dir, f"{stem}_manual_measurements.csv")


AutoResult = tuple[float, list[Spore]]  # µm/px and kept spores of one photo


def run_auto(paths: Sequence[str], out_dir: str, um_per_tick: float) -> dict[str, AutoResult]:
    """Run spore-size's automatic analysis on photos, as spore-size does.

    Size outliers are judged with each directory's spores pooled. Photos the
    analysis rejects are reported and left out; they can still be measured by hand.
    """
    analyzed: dict[str, AutoResult] = {}
    for path in paths:
        try:
            calib, spores = analyze_image(path, out_dir, um_per_tick=um_per_tick, save_debug=False)
            analyzed[path] = (calib["microns_per_px"], spores)
        except PhotoRejected as e:
            print(f"{os.path.basename(path)}: no automatic spores ({e})")
        except Exception as e:
            print(f"[FAILED] automatic analysis of {path}: {e}", file=sys.stderr)
    kept, removed = drop_outliers_by_directory({p: spores for p, (_, spores) in analyzed.items()})
    for d, n_removed in removed.items():
        if n_removed:
            print(f"{d or '.'}: {n_removed} automatic size outlier(s) removed")
    for path, (um_per_px, _) in analyzed.items():
        analyzed[path] = (um_per_px, kept[path])
        print(f"{os.path.basename(path)}: {len(kept[path])} automatic spores")
    return analyzed


def open_measurement(
    path: str,
    out_dir: str,
    um_per_tick: float,
    ruler_ticks: float,
    expected_um_per_px: float | None = None,
    auto: AutoResult | None = None,
) -> ImageMeasurement:
    """Load a photo's saved measurements, or start new ones.

    New ones start from auto (the automatic calibration and spores) if given,
    else with the scale bar calibrated automatically and no spores. A photo
    with a saved file never gets automatic spores again, so deleted ones stay
    deleted. expected_um_per_px (the other photos' calibration) is what a hand
    calibration of this photo is checked against.
    """
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    diagonal_px = float(np.hypot(*img.shape[:2]))
    csv_path = manual_csv_path(path, out_dir)
    saved = os.path.exists(csv_path)
    if saved:
        m = ImageMeasurement.load_csv(path, csv_path, um_per_tick, ruler_ticks)
        if m.um_per_px is not None:
            m.expected_um_per_px, m.diagonal_px = expected_um_per_px, diagonal_px
            return m
    m = ImageMeasurement(
        path,
        um_per_tick=um_per_tick,
        ruler_ticks=ruler_ticks,
        expected_um_per_px=expected_um_per_px,
        diagonal_px=diagonal_px,
    )
    if auto is not None:
        m.um_per_px, spores = auto
        m.calibration = "auto"
        m.calibrating = False
        if not saved:
            for spore in spores:
                m.add_spore(ellipse_points(spore["ellipse"]), "auto")
        return m
    try:
        calib = calibrate_image(img, microns_per_tick=um_per_tick)
        m.um_per_px = calib["microns_per_px"]
        m.calibration = "auto"
        m.calibrating = False
    except PhotoRejected as e:
        print(f"{os.path.basename(path)}: {e}; click the scale bar's 0 and 100 ticks")
    return m


def instructions(m: ImageMeasurement) -> str:
    """Return the status lines shown above the photo: the next step, then the controls."""
    controls = (
        "wheel zoom   right/middle drag pan   h home view   |   "
        "u undo   d delete spore at pointer   c calibrate   n next photo   q save & quit"
    )
    if m.calibrating and m.suspect_um_per_px is not None and not m.pending:
        return (
            f"CHECK CALIBRATION: {m.suspect_reason}\n"
            f"Click the scale bar's 0 and {m.ruler_ticks:g} ticks again, "
            f"or press a to accept it\n{controls}"
        )
    if m.calibrating:
        step = "the 0 tick" if not m.pending else f"the {m.ruler_ticks:g} tick"
        return f"CALIBRATE FIRST: click {step} of the scale bar (not a spore)\n{controls}"
    steps = ["length: one end", "length: other end", "width: one side", "width: other side"]
    n_auto, n_manual = m.counts()
    calib = f"{m.um_per_px:.4f} µm/px ({m.calibration})"
    rows = [r for r in m.spore_rows() if r["source"] == "manual"]
    last = (
        f"      last spore {rows[-1]['length_um']:.2f} x {rows[-1]['width_um']:.2f} µm"
        if m.history and m.history[-1][0] == "add" and rows
        else ""
    )
    return (
        f"{n_auto} automatic (orange) + {n_manual} by hand      {calib}{last}\n"
        f"Next spore by hand, {steps[len(m.pending)]}\n{controls}"
    )


def keep_within(lim: Limits, full: Limits) -> Limits:
    """Shift an axis range, without resizing it, so it lies inside the full range.

    Works for inverted ranges too (an image's y axis runs top to bottom).
    """
    lo, hi = sorted(lim)
    full_lo, full_hi = sorted(full)
    shift = full_lo - lo if lo < full_lo else (full_hi - hi if hi > full_hi else 0.0)
    return lim[0] + shift, lim[1] + shift


def zoom_view(
    xlim: Limits, ylim: Limits, x: float, y: float, factor: float, full_x: Limits, full_y: Limits
) -> tuple[Limits, Limits]:
    """Return the view zoomed about (x, y); factor < 1 zooms in.

    The point under the pointer stays put. Zooming out stops at the whole photo,
    zooming in at MIN_VIEW_PX image pixels across.
    """
    new_x = (x + (xlim[0] - x) * factor, x + (xlim[1] - x) * factor)
    new_y = (y + (ylim[0] - y) * factor, y + (ylim[1] - y) * factor)
    span_x, span_y = abs(new_x[1] - new_x[0]), abs(new_y[1] - new_y[0])
    if span_x >= abs(full_x[1] - full_x[0]) or span_y >= abs(full_y[1] - full_y[0]):
        return full_x, full_y
    if min(span_x, span_y) < MIN_VIEW_PX:
        return xlim, ylim
    return keep_within(new_x, full_x), keep_within(new_y, full_y)


def build_window(m: ImageMeasurement, csv_path: str) -> tuple[Any, dict[str, bool]]:
    """Build the measuring window for one photo; return (figure, state).

    state["quit"] becomes True when the user presses q.
    """
    import matplotlib.pyplot as plt

    # free c, backspace and q from matplotlib's default keys for our own use, and
    # turn off keys that would put a log scale or grid on the photo
    plt.rcParams["keymap.back"] = ["left"]
    plt.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]
    for keymap in ("keymap.xscale", "keymap.yscale", "keymap.grid", "keymap.grid_minor"):
        plt.rcParams[keymap] = []

    bgr = cv2.imread(m.path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {m.path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.canvas.manager.set_window_title(f"spore-measure: {m.path}")  # type: ignore[union-attr]
    ax.imshow(img)
    ax.set_axis_off()
    # leave a band at the top for the status lines, which tight_layout would not
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.9)
    status = fig.text(0.01, 0.995, "", fontsize=10, va="top", ha="left", linespacing=1.6)
    full_x, full_y = ax.get_xlim(), ax.get_ylim()
    artists: list[Any] = []
    state = {"quit": False}
    drag: dict[str, Any] = {}  # pan in progress: start pixel and start view

    def redraw() -> None:
        """Redraw all spores and pending points."""
        for a in artists:
            a.remove()
        artists.clear()
        t = np.linspace(0, 2 * np.pi, 60)
        for i, ((l1, l2, w1, w2), source) in enumerate(zip(m.spores, m.sources, strict=True), 1):
            cx, cy = (l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2
            if source == "auto":  # the fitted ellipse, through its axis ends
                ux, uy = (l2[0] - l1[0]) / 2, (l2[1] - l1[1]) / 2
                vx, vy = (w2[0] - w1[0]) / 2, (w2[1] - w1[1]) / 2
                ex = cx + ux * np.cos(t) + vx * np.sin(t)
                ey = cy + uy * np.cos(t) + vy * np.sin(t)
                artists.extend(ax.plot(ex, ey, "-", color="orange", lw=1.2))
            else:
                artists.extend(ax.plot(*zip(l1, l2, strict=True), "-", color="red", lw=1.2))
                artists.extend(ax.plot(*zip(w1, w2, strict=True), "-", color="#1f4fff", lw=1.2))
            label = ax.text(cx + 15, cy - 15, str(i), color="black", fontsize=9)
            label.set_bbox({"facecolor": "white", "alpha": 0.75, "linewidth": 0, "pad": 1})
            artists.append(label)
        if m.pending:
            xs, ys = zip(*m.pending, strict=True)
            colour = "lime" if m.calibrating else ("red" if len(m.pending) <= 2 else "#1f4fff")
            artists.extend(ax.plot(xs, ys, "+", color=colour, ms=12, mew=1.5))
        status.set_text(instructions(m))
        status.set_color("red" if m.calibrating else "black")
        status.set_fontweight("bold" if m.calibrating else "normal")
        fig.canvas.draw_idle()

    def zoom(x: float, y: float, factor: float) -> None:
        """Zoom the view about (x, y)."""
        xlim, ylim = zoom_view(ax.get_xlim(), ax.get_ylim(), x, y, factor, full_x, full_y)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.canvas.draw_idle()

    def on_scroll(event: Any) -> None:
        """Zoom in on wheel up, out on wheel down, about the pointer."""
        if event.inaxes is ax:
            zoom(event.xdata, event.ydata, 1 / ZOOM_STEP if event.step > 0 else ZOOM_STEP)

    def on_click(event: Any) -> None:
        """Start a pan on a right/middle press; add a point on a plain left click."""
        toolbar = fig.canvas.toolbar
        if event.inaxes is not ax or (toolbar and toolbar.mode != ""):
            return
        if event.button in (2, 3):
            drag.update(x=event.x, y=event.y, xlim=ax.get_xlim(), ylim=ax.get_ylim())
            return
        if event.button != 1:
            return
        if m.click(event.xdata, event.ydata) in ("spore", "calibrated"):
            m.save_csv(csv_path)
        redraw()

    def on_motion(event: Any) -> None:
        """Move the view with the pointer while panning."""
        if not drag:
            return
        bbox = ax.bbox
        (x0, x1), (y0, y1) = drag["xlim"], drag["ylim"]
        dx = (event.x - drag["x"]) * (x1 - x0) / bbox.width
        dy = (event.y - drag["y"]) * (y1 - y0) / bbox.height
        ax.set_xlim(keep_within((x0 - dx, x1 - dx), full_x))
        ax.set_ylim(keep_within((y0 - dy, y1 - dy), full_y))
        fig.canvas.draw_idle()

    def on_release(event: Any) -> None:
        """End a pan."""
        drag.clear()

    def on_key(event: Any) -> None:
        """Handle zoom, undo, calibrate, next and quit keys."""
        if event.key in ("+", "=", "-"):
            if event.inaxes is ax:
                x, y = event.xdata, event.ydata
            else:
                x, y = sum(ax.get_xlim()) / 2, sum(ax.get_ylim()) / 2
            zoom(x, y, ZOOM_STEP if event.key == "-" else 1 / ZOOM_STEP)
            return
        if event.key in ("u", "backspace"):
            m.undo()
            m.save_csv(csv_path)
        elif event.key == "d":
            if event.inaxes is ax and m.delete_at(event.xdata, event.ydata):
                m.save_csv(csv_path)
        elif event.key == "c":
            m.start_calibration()
        elif event.key == "a":
            if m.accept_suspect_calibration():
                m.save_csv(csv_path)
        elif event.key in ("n", "q"):
            state["quit"] = event.key == "q"
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    redraw()
    return fig, state


def measure_window(m: ImageMeasurement, csv_path: str) -> bool:
    """Show one photo for hand measuring; return False if the user quit."""
    import matplotlib.pyplot as plt

    _, state = build_window(m, csv_path)
    plt.show()
    if m.um_per_px is not None:
        m.save_csv(csv_path)
    return not state["quit"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    ap = argparse.ArgumentParser(
        prog="spore-measure",
        description="Check automatic spore measurements and measure more by hand "
        "(length x width, um) on micrographs with an "
        "ocular-micrometer scale bar.",
    )
    ap.add_argument("images", nargs="+", help="image files to measure")
    ap.add_argument(
        "--out-dir",
        default="./spore_manual_results",
        help="directory for measurements and reports (default: %(default)s)",
    )
    ap.add_argument(
        "--um-per-tick",
        type=float,
        default=1.0,
        help="microns per scale-bar tick (default: %(default)s)",
    )
    ap.add_argument(
        "--ruler-ticks",
        type=float,
        default=100.0,
        help="ticks between the two marks clicked for manual calibration (default: %(default)s)",
    )
    ap.add_argument(
        "--sys-error-um",
        type=float,
        default=DEFAULT_SYS_ERROR_UM,
        help="systematic error of a length measurement in um (default: %(default)s)",
    )
    ap.add_argument(
        "--no-auto",
        action="store_true",
        help="measure by hand only, without spore-size's automatic spores",
    )
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="only rebuild the reports from saved measurements, without opening windows",
    )
    ap.add_argument("--no-plots", action="store_true", help="skip the length vs width plots")
    return ap


def qt_display_problem() -> str | None:
    """Return why Qt can't open a window on this X11 display, or None if it should.

    Qt 6.5+ aborts the whole process (no Python exception) when libxcb-cursor is
    missing, so check for it before Qt starts.
    """
    platform = os.environ.get("QT_QPA_PLATFORM", "")
    if not sys.platform.startswith("linux") or platform not in ("", "xcb"):
        return None
    if not platform and os.environ.get("WAYLAND_DISPLAY"):
        return None
    if ctypes.util.find_library("xcb-cursor"):
        return None
    return (
        "spore-measure can't open a window: Qt needs the libxcb-cursor library.\n"
        "Install it with: sudo apt install libxcb-cursor0"
    )


def saved_calibrations(paths: Sequence[str], out_dir: str) -> dict[str, tuple[float, str]]:
    """Return the saved (µm/px, "auto" or "manual") of each photo with hand measurements."""
    calibs = {}
    for path in paths:
        csv_path = manual_csv_path(path, out_dir)
        if os.path.exists(csv_path):
            m = ImageMeasurement.load_csv(path, csv_path)
            if m.um_per_px is not None:
                calibs[path] = (m.um_per_px, m.calibration)
    return calibs


def expected_calibration(calibs: dict[str, tuple[float, str]], path: str) -> float | None:
    """Return the µm/px that path's calibration should be close to, or None if unknown.

    That is the median of the other photos' automatic calibrations, which come from
    the detected tick spacing and can be trusted. Without any, it is the median of
    all hand calibrations, path's own included, so that one bad photo out of three
    stands out instead of pulling the other two's reference.
    """
    autos = [v for p, (v, kind) in calibs.items() if p != path and kind == "auto"]
    values = autos or [v for v, _ in calibs.values()]
    return float(np.median(values)) if values else None


def flag_saved_calibration(m: ImageMeasurement) -> str:
    """Hold a saved hand calibration that disagrees with the other photos for checking.

    Returns the reason, or "" if the calibration looks right.
    """
    if m.calibration != "manual" or m.um_per_px is None:
        return ""
    problem = m.calibration_problem(m.um_per_px, float("inf"))
    if problem:
        saved = m.um_per_px
        m.start_calibration()
        m.suspect_um_per_px, m.suspect_reason = saved, f"saved calibration {problem}"
    return problem


def main(argv: Sequence[str] | None = None) -> int:
    """Run spore-measure on the given images; return the process exit code."""
    from .cli import write_run_reports

    args = build_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.report_only:
        import matplotlib

        try:
            matplotlib.use("QtAgg")
        except ImportError:
            print(
                "spore-measure needs a GUI toolkit: pip install -e '.[gui]'",
                file=sys.stderr,
            )
            return 1
        problem = qt_display_problem()
        if problem:
            print(problem, file=sys.stderr)
            return 1
        calibs = saved_calibrations(args.images, args.out_dir)
        auto: dict[str, AutoResult] = {}
        if not args.no_auto:
            # photos with saved measurements keep them; the rest are analysed together,
            # so that size outliers are judged per directory as spore-size does
            new = [p for p in args.images if not os.path.exists(manual_csv_path(p, args.out_dir))]
            if new:
                print(f"Automatic analysis of {len(new)} photo(s)...")
                auto = run_auto(new, args.out_dir, args.um_per_tick)
            for path, (um_per_px, _) in auto.items():
                calibs.setdefault(path, (um_per_px, "auto"))
        for path in args.images:
            expected = expected_calibration(calibs, path)
            try:
                m = open_measurement(
                    path,
                    args.out_dir,
                    args.um_per_tick,
                    args.ruler_ticks,
                    expected,
                    auto.get(path),
                )
            except Exception as e:
                print(f"[FAILED] {path}: {e}", file=sys.stderr)
                continue
            problem = flag_saved_calibration(m)
            if problem:
                print(f"{os.path.basename(path)}: saved calibration {problem}; check it")
            keep_going = measure_window(m, manual_csv_path(path, args.out_dir))
            if m.um_per_px is not None:
                calibs[path] = (m.um_per_px, m.calibration)
            if not keep_going:
                break

    calibs = saved_calibrations(args.images, args.out_dir)
    all_rows: list[Spore] = []
    measured: list[str] = []
    for path in args.images:
        csv_path = manual_csv_path(path, args.out_dir)
        if not os.path.exists(csv_path):
            continue
        m = ImageMeasurement.load_csv(path, csv_path)
        rows = m.spore_rows()
        measured.append(path)
        all_rows.extend(rows)
        n_auto, n_manual = m.counts()
        print(f"{os.path.basename(path)}: {n_auto} automatic + {n_manual} by hand")
        if path not in calibs:
            continue
        um_per_px = calibs[path][0]
        expected = expected_calibration(calibs, path)
        if expected and abs(um_per_px / expected - 1) > CALIB_TOLERANCE:
            print(
                f"  [WARNING] its calibration, {um_per_px:.4f} µm/px, is "
                f"{um_per_px / expected:.2g}x the other photos' {expected:.4f} µm/px"
            )
    if not measured:
        print("No measurements saved yet.", file=sys.stderr)
        return 1
    write_run_reports(all_rows, measured, args.out_dir, args.sys_error_um, args.no_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
