"""Measure spores by hand: click the ends of each spore's length and width.

Each photo opens in a window (zoom and pan with the toolbar). For every
spore, click the two ends of its length, then the two sides of its width.
The scale bar is calibrated automatically as in spore-size; if that fails,
click its 0 and 100 ticks instead. Measurements are saved after every spore
(<photo>_manual_measurements.csv) and restored when a photo is reopened.
The run's reports (combined CSVs, spore_summary.csv, plots) have the same
format as spore-size's, so hand and automatic measurements compare
directly.

Keys: u / backspace undo, c calibrate by hand, n next photo, q save and quit.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ._types import Point, Spore
from .calibration import calibrate_image
from .errors import PhotoRejected
from .summary import DEFAULT_SYS_ERROR_UM

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
]


def _distance(a: Point, b: Point) -> float:
    """Return the distance between two points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class ImageMeasurement:
    """Hand measurements of one photo: calibration, spores and pending clicks.

    Clicks build up a spore: two points for the length, two for the width.
    In calibration mode the next two clicks are the 0 and full-scale ticks
    of the scale bar instead.
    """

    def __init__(
        self,
        path: str,
        um_per_px: float | None = None,
        calibration: str = "",
        um_per_tick: float = 1.0,
        ruler_ticks: float = 100.0,
    ) -> None:
        """Start measuring a photo, with a known calibration or none yet."""
        self.path = path
        self.um_per_px = um_per_px
        self.calibration = calibration
        self.um_per_tick = um_per_tick
        self.ruler_ticks = ruler_ticks
        self.spores: list[tuple[Point, Point, Point, Point]] = []
        self.pending: list[Point] = []
        self.calibrating = um_per_px is None

    def start_calibration(self) -> None:
        """Make the next two clicks the scale bar's 0 and full-scale ticks."""
        self.calibrating = True
        self.pending = []

    def click(self, x: float, y: float) -> str:
        """Add a clicked point; return "point", "spore" or "calibrated"."""
        self.pending.append((x, y))
        if self.calibrating:
            if len(self.pending) < 2:
                return "point"
            span_px = _distance(self.pending[0], self.pending[1])
            self.pending = []
            if span_px <= 0:
                return "point"
            self.um_per_px = self.ruler_ticks * self.um_per_tick / span_px
            self.calibration = "manual"
            self.calibrating = False
            return "calibrated"
        if len(self.pending) < 4:
            return "point"
        p = self.pending
        self.spores.append((p[0], p[1], p[2], p[3]))
        self.pending = []
        return "spore"

    def undo(self) -> None:
        """Remove the last pending point, or else the last finished spore."""
        if self.pending:
            self.pending.pop()
        elif self.spores and not self.calibrating:
            self.spores.pop()

    def spore_rows(self) -> list[Spore]:
        """Return the spores as rows in microns (length >= width), like spore-size."""
        if self.um_per_px is None:
            return []
        rows = []
        for l1, l2, w1, w2 in self.spores:
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
                        *(f"{v:.1f}" for pt in (l1, l2, w1, w2) for v in pt),
                        f"{self.um_per_px:.6f}",
                        self.calibration,
                    ]
                )

    @classmethod
    def load_csv(
        cls, path: str, csv_path: str, um_per_tick: float = 1.0, ruler_ticks: float = 100.0
    ) -> ImageMeasurement:
        """Restore a photo's saved hand measurements."""
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
            m.spores.append((pts[0], pts[1], pts[2], pts[3]))
        return m


def manual_csv_path(path: str, out_dir: str) -> str:
    """Return where a photo's hand measurements are saved."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(out_dir, f"{stem}_manual_measurements.csv")


def open_measurement(
    path: str, out_dir: str, um_per_tick: float, ruler_ticks: float
) -> ImageMeasurement:
    """Load a photo's saved measurements, or start new ones with auto calibration."""
    csv_path = manual_csv_path(path, out_dir)
    if os.path.exists(csv_path):
        m = ImageMeasurement.load_csv(path, csv_path, um_per_tick, ruler_ticks)
        if m.um_per_px is not None:
            return m
    m = ImageMeasurement(path, um_per_tick=um_per_tick, ruler_ticks=ruler_ticks)
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    try:
        calib = calibrate_image(img, microns_per_tick=um_per_tick)
        m.um_per_px = calib["microns_per_px"]
        m.calibration = "auto"
        m.calibrating = False
    except PhotoRejected as e:
        print(f"{os.path.basename(path)}: {e}; click the scale bar's 0 and 100 ticks")
    return m


def instructions(m: ImageMeasurement) -> str:
    """Return the status line shown above the photo."""
    keys = "u undo   c calibrate   n next photo   q save & quit"
    if m.calibrating:
        step = "the 0 tick" if not m.pending else f"the {m.ruler_ticks:g} tick"
        return f"Calibrate: click {step} of the scale bar      ({keys})"
    steps = ["length: one end", "length: other end", "width: one side", "width: other side"]
    n = len(m.spores) + 1
    calib = f"{m.um_per_px:.4f} µm/px ({m.calibration})"
    return f"Spore {n} {steps[len(m.pending)]}   |   {calib}   |   {keys}"


def build_window(m: ImageMeasurement, csv_path: str) -> tuple[Any, dict[str, bool]]:
    """Build the measuring window for one photo; return (figure, state).

    state["quit"] becomes True when the user presses q.
    """
    import matplotlib.pyplot as plt

    # free c, backspace and q from matplotlib's default keys for our own use
    plt.rcParams["keymap.back"] = ["left"]
    plt.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]

    bgr = cv2.imread(m.path)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {m.path}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.canvas.manager.set_window_title(f"spore-measure: {m.path}")  # type: ignore[union-attr]
    ax.imshow(img)
    ax.set_axis_off()
    fig.tight_layout()
    artists: list[Any] = []
    state = {"quit": False}

    def redraw() -> None:
        """Redraw all spores and pending points."""
        for a in artists:
            a.remove()
        artists.clear()
        for i, (l1, l2, w1, w2) in enumerate(m.spores, 1):
            artists.extend(ax.plot(*zip(l1, l2, strict=True), "-", color="red", lw=1.2))
            artists.extend(ax.plot(*zip(w1, w2, strict=True), "-", color="#1f4fff", lw=1.2))
            cx, cy = (l1[0] + l2[0]) / 2, (l1[1] + l2[1]) / 2
            label = ax.text(cx + 15, cy - 15, str(i), color="black", fontsize=9)
            label.set_bbox({"facecolor": "white", "alpha": 0.75, "linewidth": 0, "pad": 1})
            artists.append(label)
        if m.pending:
            xs, ys = zip(*m.pending, strict=True)
            colour = "lime" if m.calibrating else ("red" if len(m.pending) <= 2 else "#1f4fff")
            artists.extend(ax.plot(xs, ys, "+", color=colour, ms=12, mew=1.5))
        ax.set_title(instructions(m), fontsize=9, loc="left")
        fig.canvas.draw_idle()

    def on_click(event: Any) -> None:
        """Add a point on a plain left click (not while zooming or panning)."""
        toolbar = fig.canvas.toolbar
        if event.inaxes is not ax or event.button != 1 or (toolbar and toolbar.mode != ""):
            return
        if m.click(event.xdata, event.ydata) in ("spore", "calibrated"):
            m.save_csv(csv_path)
        redraw()

    def on_key(event: Any) -> None:
        """Handle undo, calibrate, next and quit keys."""
        if event.key in ("u", "backspace"):
            m.undo()
            m.save_csv(csv_path)
        elif event.key == "c":
            m.start_calibration()
        elif event.key in ("n", "q"):
            state["quit"] = event.key == "q"
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
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
        description="Measure spore length x width (um) by hand on micrographs with an "
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
        "--report-only",
        action="store_true",
        help="only rebuild the reports from saved measurements, without opening windows",
    )
    ap.add_argument("--no-plots", action="store_true", help="skip the length vs width plots")
    return ap


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
        for path in args.images:
            try:
                m = open_measurement(path, args.out_dir, args.um_per_tick, args.ruler_ticks)
            except Exception as e:
                print(f"[FAILED] {path}: {e}", file=sys.stderr)
                continue
            if not measure_window(m, manual_csv_path(path, args.out_dir)):
                break

    all_rows: list[Spore] = []
    measured: list[str] = []
    for path in args.images:
        csv_path = manual_csv_path(path, args.out_dir)
        if not os.path.exists(csv_path):
            continue
        rows = ImageMeasurement.load_csv(path, csv_path).spore_rows()
        measured.append(path)
        all_rows.extend(rows)
        print(f"{os.path.basename(path)}: {len(rows)} spores measured by hand")
    if not measured:
        print("No measurements saved yet.", file=sys.stderr)
        return 1
    write_run_reports(all_rows, measured, args.out_dir, args.sys_error_um, args.no_plots)
    return 0


if __name__ == "__main__":
    sys.exit(main())
