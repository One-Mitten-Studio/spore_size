"""Summary statistics for a run, written to a single CSV.

There is one row per image, per image directory and for all spores together.

For length, width and Q:
- mean
- SD, the spread of individual spores (natural variation, not an error).
- SEM, the statistical error of the mean: SD / sqrt(n), with the sample SD
  (n-1 denominator). How much the mean would vary if another random sample of
  n spores were measured from the same population; shrinks as n grows.
- sys, the systematic error: the light microscope's resolution limits every
  length measurement to ~0.1 um, and that does not average away with more
  spores. For Q = L/W it is propagated from the length and width errors.
- total, the total error of the mean: stat and sys added in quadrature,
  sqrt(sem^2 + sys^2).
Each row also has a ready-to-paste publication text:
  "<length> ± <SD> µm (err: <total>) by <width> ± <SD> µm (err: <total>)".
Directory and "all" rows pool the individual spores (not the image means).
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from typing import Any

import numpy as np

from ._types import Spore, SummaryRow

MEASURES = (("length", "length_um"), ("width", "width_um"), ("Q", "Q_ratio"))
STATS = ("mean", "sd", "sem", "sys", "total_err")
DEFAULT_SYS_ERROR_UM = 0.1


def describe(values: Iterable[float]) -> dict[str, Any]:
    """n, mean, sample SD (n-1) and SEM = SD / sqrt(n)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {"n": n, "mean": x.mean(), "sd": sd, "sem": sd / np.sqrt(n) if n > 1 else np.nan}


def q_systematic(length_mean: float, width_mean: float, sys_error_um: float) -> float:
    """Systematic error of Q = L/W from independent errors on L and W."""
    q = length_mean / width_mean
    return q * np.hypot(sys_error_um / length_mean, sys_error_um / width_mean)


def publication_text(r: SummaryRow) -> str:
    """'6.46 ± 0.51 µm (err: 0.11) by 4.56 ± 0.32 µm (err: 0.10)'; '' if no spores."""
    if r["n_spores"] == 0:
        return ""

    def num(v: float) -> str:
        """Format a value to 2 decimals, or "-" if missing."""
        return "-" if np.isnan(v) else f"{v:.2f}"

    def part(m: str) -> str:
        """Return "<mean> ± <SD> µm (err: <total>)" for one measure."""
        return f"{num(r[m + '_mean'])} ± {num(r[m + '_sd'])} µm (err: {num(r[m + '_total_err'])})"

    return f"{part('length')} by {part('width')}"


def summary_rows(
    spore_rows: list[Spore],
    images: Iterable[str] | None = None,
    sys_error_um: float = DEFAULT_SYS_ERROR_UM,
) -> list[SummaryRow]:
    """Build summary rows from per-spore rows.

    Each spore row is a dict with "image", "length_um", "width_um" and
    "Q_ratio". Rows are grouped as each directory's images followed by that
    directory's total, then a final row for all spores. Pass images (in
    processing order) to also list images where no spore was kept.
    """
    if images is None:
        images = [r["image"] for r in spore_rows]
    images = list(dict.fromkeys(images))
    dirs = list(dict.fromkeys(os.path.dirname(i) for i in images))
    has_dirs = any(dirs)

    def row(level: str, name: str, rows: list[Spore]) -> SummaryRow:
        """Return the summary row for one group of spores."""
        out: SummaryRow = {
            "level": level,
            "name": name,
            "n_images": len({r["image"] for r in rows}),
            "n_spores": len(rows),
        }
        for m, key in MEASURES:
            d = describe([r[key] for r in rows])
            out.update({f"{m}_mean": d["mean"], f"{m}_sem": d["sem"], f"{m}_sd": d["sd"]})
        has = bool(rows)
        out["length_sys"] = sys_error_um if has else np.nan
        out["width_sys"] = sys_error_um if has else np.nan
        out["Q_sys"] = (
            q_systematic(out["length_mean"], out["width_mean"], sys_error_um) if has else np.nan
        )
        for m, _ in MEASURES:  # total error: stat and sys in quadrature
            out[f"{m}_total_err"] = np.hypot(out[f"{m}_sem"], out[f"{m}_sys"])
        out["publication"] = publication_text(out)
        return out

    result = []
    for d in dirs:
        for img in (i for i in images if os.path.dirname(i) == d):
            result.append(row("image", img, [r for r in spore_rows if r["image"] == img]))
        if has_dirs:
            result.append(
                row(
                    "directory",
                    d or ".",
                    [r for r in spore_rows if os.path.dirname(r["image"]) == d],
                )
            )
    result.append(row("all", "all", spore_rows))
    return result


def write_summary_csv(rows: list[SummaryRow], csv_path: str) -> None:
    """Write the summary rows to a CSV, with blanks for missing values."""
    fields = ["level", "name", "n_images", "n_spores"]
    fields += [f"{m}_{s}" for m, _ in MEASURES for s in STATS]
    fields.append("publication")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    k: (
                        ("" if np.isnan(r[k]) else f"{r[k]:.3f}")
                        if isinstance(r[k], float)
                        else r[k]
                    )
                    for k in fields
                }
            )


def format_table(rows: list[SummaryRow]) -> str:
    """Plain-text version of the summary for the terminal."""

    def num(v: float, fmt: str) -> str:
        """Format a value with fmt, or "-" if missing."""
        return "-" if np.isnan(v) else format(v, fmt)

    def cells(r: SummaryRow, m: str, fmt: str, width: int) -> str:
        """Return one measure's mean, SD, stat, sys and total as aligned columns."""
        return "".join(f"{num(r[f'{m}_{s}'], fmt):>{width}}" for s in STATS)

    lw, qw = 6, 7  # column widths for length/width and for Q
    sub = "".join(f"{h:>{lw}}" for h in ("mean", "SD", "stat", "sys", "total"))
    qsub = "".join(f"{h:>{qw}}" for h in ("mean", "SD", "stat", "sys", "total"))
    lines = [
        f"{'':<34}{'n':>5}   {'length um':^{5 * lw}}   {'width um':^{5 * lw}}   {'Q':^{5 * qw}}",
        f"{'':<34}{'':>5}   {sub}   {sub}   {qsub}",
    ]
    rule = "-" * len(lines[1])
    for r in rows:
        if r["level"] != "image":
            lines.append(rule)
        name = r["name"] if r["level"] == "image" else f"{r['name']} (total)"
        if r["n_spores"] == 0:
            lines.append(f"{name:<34}{0:>5}")
            continue
        lines.append(
            f"{name:<34}{r['n_spores']:>5}   {cells(r, 'length', '.2f', lw)}   "
            f"{cells(r, 'width', '.2f', lw)}   {cells(r, 'Q', '.3f', qw)}"
        )
    return "\n".join(lines)
