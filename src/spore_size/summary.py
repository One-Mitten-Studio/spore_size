"""
Summary statistics for a run: one row per image, per image directory and for
all spores together, written to a single CSV.

For length, width and Q:
- mean
- SEM, the statistical error of the mean: SD / sqrt(n), with the sample SD
  (n-1 denominator). How much the mean would vary if another random sample of
  n spores were measured from the same population; shrinks as n grows.
- sys, the systematic error: the light microscope's resolution limits every
  length measurement to ~0.1 um, and that does not average away with more
  spores. For Q = L/W it is propagated from the length and width errors.
- SD, the spread of individual spores (natural variation).
Directory and "all" rows pool the individual spores (not the image means).
"""
import csv
import os

import numpy as np

MEASURES = (("length", "length_um"), ("width", "width_um"), ("Q", "Q_ratio"))
STATS = ("mean", "sem", "sys", "sd")
DEFAULT_SYS_ERROR_UM = 0.1


def describe(values):
    """n, mean, sample SD (n-1) and SEM = SD / sqrt(n)."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "sem": np.nan}
    sd = x.std(ddof=1) if n > 1 else np.nan
    return {"n": n, "mean": x.mean(), "sd": sd, "sem": sd / np.sqrt(n) if n > 1 else np.nan}


def q_systematic(length_mean, width_mean, sys_error_um):
    """Systematic error of Q = L/W from independent errors on L and W."""
    q = length_mean / width_mean
    return q * np.hypot(sys_error_um / length_mean, sys_error_um / width_mean)


def summary_rows(spore_rows, images=None, sys_error_um=DEFAULT_SYS_ERROR_UM):
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
        out = {"level": level, "name": name,
               "n_images": len({r["image"] for r in rows}), "n_spores": len(rows)}
        for m, key in MEASURES:
            d = describe([r[key] for r in rows])
            out.update({f"{m}_mean": d["mean"], f"{m}_sem": d["sem"], f"{m}_sd": d["sd"]})
        has = bool(rows)
        out["length_sys"] = sys_error_um if has else np.nan
        out["width_sys"] = sys_error_um if has else np.nan
        out["Q_sys"] = (q_systematic(out["length_mean"], out["width_mean"], sys_error_um)
                        if has else np.nan)
        return out

    result = []
    for d in dirs:
        for img in (i for i in images if os.path.dirname(i) == d):
            result.append(row("image", img, [r for r in spore_rows if r["image"] == img]))
        if has_dirs:
            result.append(row("directory", d or ".",
                              [r for r in spore_rows if os.path.dirname(r["image"]) == d]))
    result.append(row("all", "all", spore_rows))
    return result


def write_summary_csv(rows, csv_path):
    fields = ["level", "name", "n_images", "n_spores"]
    fields += [f"{m}_{s}" for m, _ in MEASURES for s in STATS]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (("" if np.isnan(r[k]) else f"{r[k]:.3f}")
                            if isinstance(r[k], float) else r[k]) for k in fields})


def format_table(rows):
    """Plain-text version of the summary for the terminal: mean ± stat ± sys (SD)."""
    def num(v, fmt):
        return "-" if np.isnan(v) else format(v, fmt)

    def cell(r, m, fmt):
        return (f"{num(r[m + '_mean'], fmt)} ± {num(r[m + '_sem'], fmt)} "
                f"± {num(r[m + '_sys'], fmt)}  ({num(r[m + '_sd'], fmt)})")

    head = "mean ± stat ± sys  (SD)"
    lines = [f"{'':<34} {'n':>4}  {'length um':<27} {'width um':<27} {'Q':<27}",
             f"{'':<34} {'':>4}  {head:<27} {head:<27} {head:<27}"]
    for r in rows:
        if r["level"] != "image":
            lines.append("-" * 122)
        name = r["name"] if r["level"] == "image" else f"{r['name']} (total)"
        if r["n_spores"] == 0:
            lines.append(f"{name:<34} {0:>4}")
            continue
        lines.append(f"{name:<34} {r['n_spores']:>4}  {cell(r, 'length', '.2f'):<27} "
                     f"{cell(r, 'width', '.2f'):<27} {cell(r, 'Q', '.3f'):<27}")
    return "\n".join(lines)
