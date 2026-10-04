"""
Length vs width plots of measured spores: one per image, one per image
directory and one for all spores in the run.

Each spore is a semi-transparent circle with a thin black edge, centred on
its (length, width) in um, with a radius equal to the systematic measurement
error. Both axes use the same um scale (so circles stay round) and every plot
of a run shares the same axis ranges, so plots can be compared directly.
Each directory has its own colour, used for its images, its own plot and its
group in the all-spores plot (which has a legend).

A dotted ellipse centred on the mean (length, width) has semi-axes equal to
the length and width SDs (mean ± 1 SD). The all-spores plot also draws one
per directory, in the directory's colour.
"""
import os

import matplotlib

matplotlib.use("Agg")  # file output only, no display needed
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Ellipse  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import numpy as np  # noqa: E402

# fixed-order categorical palette (colour follows the directory, not its rank)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FILL_ALPHA = 0.35
TEXT = "#222222"
MUTED = "#666666"
GRID = "#e5e5e5"


def axis_limits(spore_rows, pad_um=0.4):
    """Shared (xlim, ylim) covering every spore in the run, padded."""
    ls = [r["length_um"] for r in spore_rows]
    ws = [r["width_um"] for r in spore_rows]
    return ((min(ls) - pad_um, max(ls) + pad_um), (min(ws) - pad_um, max(ws) + pad_um))


def sd_ellipse(ax, rows, colour, linewidth):
    """Dotted ellipse at the mean (length, width) with semi-axes = SDs, plus a + at the mean."""
    if len(rows) < 2:
        return  # no SD from a single spore
    ls = np.array([r["length_um"] for r in rows])
    ws = np.array([r["width_um"] for r in rows])
    centre = (ls.mean(), ws.mean())
    halo = [pe.withStroke(linewidth=linewidth + 2.5, foreground="white")]  # readable over circles
    ax.add_patch(Ellipse(centre, 2 * ls.std(ddof=1), 2 * ws.std(ddof=1), fill=False,
                         edgecolor=colour, linestyle=":", linewidth=linewidth, zorder=3,
                         path_effects=halo))
    ax.plot(*centre, marker="+", color=colour, markersize=9, markeredgewidth=linewidth,
            zorder=4, path_effects=halo)


def plot_spores(groups, path, title, subtitle, limits, radius_um):
    """
    groups: list of (label, colour, rows). A legend is drawn when there is
    more than one group.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)
    for _, colour, rows in groups:
        circles = [Circle((r["length_um"], r["width_um"]), radius_um) for r in rows]
        ax.add_collection(PatchCollection(circles, facecolor=_rgba(colour, FILL_ALPHA),
                                          edgecolor="black", linewidth=0.4))
    if len(groups) > 1:
        for _, colour, rows in groups:
            sd_ellipse(ax, rows, colour, linewidth=1.8)
    sd_ellipse(ax, [r for _, _, rows in groups for r in rows], "black", linewidth=1.4)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_aspect("equal")
    ax.set_xlabel("Length (µm)", color=TEXT)
    ax.set_ylabel("Width (µm)", color=TEXT)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_title(f"{title}\n", loc="left", color=TEXT, fontsize=11, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=MUTED, fontsize=8.5)
    if len(groups) > 1:
        handles = [Line2D([], [], marker="o", linestyle=":", markersize=8, color=c,
                          markerfacecolor=_rgba(c, FILL_ALPHA), markeredgecolor="black",
                          markeredgewidth=0.4, label=f"{label} (n={len(rows)})")
                   for label, c, rows in groups]
        handles.append(Line2D([], [], linestyle=":", color="black", linewidth=1.4,
                              marker="+", markersize=8, label="all spores"))
        ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=TEXT,
                  loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.text(1.0, -0.15, f"circle radius = systematic error ({radius_um:g} µm);  "
            f"dotted ellipse = mean ± 1 SD",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=7.5)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rgba(hex_colour, alpha):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return (r, g, b, alpha)


def write_plots(spore_rows, summary, out_dir, dir_labels, radius_um):
    """
    Write <image stem>_spore_plot.png per image, <dir label>_spore_plot.png
    per directory and all_spores_plot.png. summary is the list from
    summary.summary_rows (for subtitles). Returns the written paths.
    """
    if not spore_rows:
        return []
    limits = axis_limits(spore_rows)
    pub = {(r["level"], r["name"]): r for r in summary}
    dirs = list(dict.fromkeys(os.path.dirname(r["image"]) for r in spore_rows))
    colour_of = {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(dirs)}
    paths = []

    def subtitle(row):
        return f"n = {row['n_spores']}   {row['publication']}"

    for img in dict.fromkeys(r["image"] for r in spore_rows):
        rows = [r for r in spore_rows if r["image"] == img]
        stem = os.path.splitext(os.path.basename(img))[0]
        path = os.path.join(out_dir, f"{stem}_spore_plot.png")
        plot_spores([(img, colour_of[os.path.dirname(img)], rows)], path, img,
                    subtitle(pub[("image", img)]), limits, radius_um)
        paths.append(path)

    for d in dirs:
        if not d:
            continue
        rows = [r for r in spore_rows if os.path.dirname(r["image"]) == d]
        path = os.path.join(out_dir, f"{dir_labels[d]}_spore_plot.png")
        plot_spores([(d, colour_of[d], rows)], path, d,
                    subtitle(pub[("directory", d)]), limits, radius_um)
        paths.append(path)

    groups = [(d or ".", colour_of[d], [r for r in spore_rows if os.path.dirname(r["image"]) == d])
              for d in dirs]
    path = os.path.join(out_dir, "all_spores_plot.png")
    plot_spores(groups, path, "All spores", subtitle(pub[("all", "all")]), limits, radius_um)
    paths.append(path)
    return paths
