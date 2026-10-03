"""
Summary statistics for a run: one row per image, per image directory and for
all spores together, written to a single CSV.

For length, width and Q: mean, sample SD (n-1 denominator) and the standard
error of the mean, SEM = SD / sqrt(n) -- how much the mean would vary if
another random sample of n spores were measured from the same population.
Directory and "all" rows pool the individual spores (not the image means).
"""
import csv
import os

import numpy as np

MEASURES = (("length", "length_um"), ("width", "width_um"), ("Q", "Q_ratio"))


def describe(values):
    """n, mean, sample SD, SEM, min, 5th and 95th percentile, max."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {
        "n": n,
        "mean": x.mean(),
        "sd": sd,
        "sem": sd / np.sqrt(n) if n > 1 else np.nan,
        "min": x.min(),
        "p5": np.percentile(x, 5),
        "p95": np.percentile(x, 95),
        "max": x.max(),
    }


def size_notation(length, width, q):
    """Mycological notation, e.g. '(5.9-)6.1-7.0(-7.4) x (4.2-)4.3-4.9(-5.1) um, Q = 1.42'."""
    def rng(d):
        return f"({d['min']:.1f}–){d['p5']:.1f}–{d['p95']:.1f}(–{d['max']:.1f})"
    return f"{rng(length)} × {rng(width)} µm, Q = {q['mean']:.2f}"


def summary_rows(spore_rows, images=None):
    """
    Build summary rows from per-spore rows (dicts with "image", "length_um",
    "width_um", "Q_ratio"). Grouped as: each directory's images followed by
    that directory's total, then a final row for all spores. Pass images
    (in processing order) to also list images where no spore was kept.
    """
    if images is None:
        images = [r["image"] for r in spore_rows]
    images = list(dict.fromkeys(images))
    dirs = list(dict.fromkeys(os.path.dirname(i) for i in images))
    has_dirs = any(dirs)

    def row(level, name, rows):
        stats = {m: describe([r[key] for r in rows]) for m, key in MEASURES}
        out = {"level": level, "name": name,
               "n_images": len({r["image"] for r in rows}), "n_spores": len(rows)}
        for m, _ in MEASURES:
            for stat in ("mean", "sd", "sem"):
                out[f"{m}_{stat}"] = stats[m].get(stat, np.nan)
        out["size"] = size_notation(stats["length"], stats["width"], stats["Q"]) if rows else ""
        return out

    result = []
    for d in dirs:
        dir_images = [i for i in images if os.path.dirname(i) == d]
        for img in dir_images:
            result.append(row("image", img, [r for r in spore_rows if r["image"] == img]))
        if has_dirs:
            result.append(row("directory", d or ".",
                              [r for r in spore_rows if os.path.dirname(r["image"]) == d]))
    result.append(row("all", "all", spore_rows))
    return result


def write_summary_csv(rows, csv_path):
    fields = ["level", "name", "n_images", "n_spores"]
    for m, _ in MEASURES:
        fields += [f"{m}_mean", f"{m}_sd", f"{m}_sem"]
    fields.append("size")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) and not np.isnan(v)
                            else "" if isinstance(v, float) else v)
                        for k, v in r.items()})


def format_table(rows):
    """Plain-text version of the summary for the terminal."""
    lines = [f"{'':<38} {'n':>4}  {'length um (SD, SEM)':<22} {'width um (SD, SEM)':<22} {'Q':>5}"]
    for r in rows:
        if r["level"] != "image":
            lines.append("-" * 96)
        name = r["name"] if r["level"] == "image" else f"{r['name']} (total)"
        if r["n_spores"] == 0:
            lines.append(f"{name:<38} {0:>4}")
            continue

        def cell(m):
            sd = "" if np.isnan(r[f"{m}_sd"]) else f"{r[f'{m}_sd']:.2f}"
            sem = "" if np.isnan(r[f"{m}_sem"]) else f"{r[f'{m}_sem']:.2f}"
            return f"{r[f'{m}_mean']:.2f} ({sd}, {sem})"
        lines.append(f"{name:<38} {r['n_spores']:>4}  {cell('length'):<22} {cell('width'):<22} "
                     f"{r['Q_mean']:>5.2f}")
    return "\n".join(lines)
