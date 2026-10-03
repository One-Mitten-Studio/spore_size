"""
Command-line entry point.

Usage:
    spore-size <image.jpg> [more images...] [--out-dir DIR] [--um-per-tick 1.0]
               [--min-length-um 3] [--max-length-um 25]
               [--min-width-um 1.5] [--max-width-um 18]

Writes spore_summary.csv (mean, SD and SEM of length, width and Q for each
image, each image directory and all spores), all_spores_combined.csv (every measured spore, with the image path as
given on the command line) and, for each directory the images came from,
<dir>_spores_combined.csv with just that directory's spores.
"""
import argparse
import csv
import os
import sys

from .errors import PhotoRejected
from .measure import process_image
from .summary import format_table, summary_rows, write_summary_csv


def build_parser():
    ap = argparse.ArgumentParser(
        prog="spore-size",
        description="Measure spore length x width (um) from micrographs with an "
                    "ocular-micrometer scale bar.")
    ap.add_argument("images", nargs="+", help="image files to process")
    ap.add_argument("--out-dir", default="./spore_results",
                    help="directory for CSVs and overlays (default: %(default)s)")
    ap.add_argument("--um-per-tick", type=float, default=1.0,
                    help="microns per scale-bar tick (default: %(default)s)")
    ap.add_argument("--min-length-um", type=float, default=3.0)
    ap.add_argument("--max-length-um", type=float, default=25.0)
    ap.add_argument("--min-width-um", type=float, default=1.5)
    ap.add_argument("--max-width-um", type=float, default=18.0)
    ap.add_argument("--include-touching", action="store_true",
                    help="also measure spores touching another spore (split apart); "
                         "by default only isolated spores are measured")
    ap.add_argument("--max-clutter", type=float, default=0.28,
                    help="skip photos where more than this fraction of the view is "
                         "neither background nor spores, e.g. tissue (default: %(default)s)")
    ap.add_argument("--no-debug", action="store_true",
                    help="skip writing the calibration diagnostic images")
    return ap


def write_combined_csv(rows, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "length_um", "width_um", "Q_ratio"])
        for r in rows:
            w.writerow([r["image"], f"{r['length_um']:.2f}", f"{r['width_um']:.2f}",
                        f"{r['Q_ratio']:.2f}"])


def directory_labels(dirs):
    """
    Short, unique file-name labels for image directories: the last path
    component (e.g. "inat255552721"), or the whole path joined with "_" when
    two directories share a last component.
    """
    def last(d):
        return os.path.basename(os.path.normpath(d))

    labels = {}
    for d in dirs:
        clash = any(last(o) == last(d) for o in dirs if o != d)
        label = os.path.normpath(d).strip(os.sep).replace(os.sep, "_") if clash else last(d)
        labels[d] = label or "root"
    return labels


def main(argv=None):
    args = build_parser().parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    processed = []
    n_ok = 0
    for path in args.images:
        try:
            summary, kept = process_image(
                path, args.out_dir, um_per_tick=args.um_per_tick,
                min_length_um=args.min_length_um, max_length_um=args.max_length_um,
                min_width_um=args.min_width_um, max_width_um=args.max_width_um,
                include_touching=args.include_touching, max_clutter=args.max_clutter,
                save_debug=not args.no_debug)
        except PhotoRejected as e:
            print(f"[SKIPPED] {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[FAILED] {path}: {e}", file=sys.stderr)
            continue
        n_ok += 1
        processed.append(path)
        print(f"{os.path.basename(path)}: n={summary['n']}  "
              f"L={summary['length_mean']:.2f}±{summary['length_sd']:.2f} um  "
              f"W={summary['width_mean']:.2f}±{summary['width_sd']:.2f} um  "
              f"Q={summary['Q_mean']:.2f}  ({summary['um_per_px']:.4f} um/px)")
        for s in kept:
            all_rows.append({"image": path, **s})

    if all_rows:
        combined_path = os.path.join(args.out_dir, "all_spores_combined.csv")
        write_combined_csv(all_rows, combined_path)
        print(f"\nCombined CSV: {combined_path}")

        # one combined CSV per image directory, if images came from directories
        dirs = sorted({os.path.dirname(r["image"]) for r in all_rows} - {""})
        for d, label in directory_labels(dirs).items():
            rows = [r for r in all_rows if os.path.dirname(r["image"]) == d]
            dir_path = os.path.join(args.out_dir, f"{label}_spores_combined.csv")
            write_combined_csv(rows, dir_path)
            print(f"  {d}: {dir_path}")

    if processed:
        stats = summary_rows(all_rows, images=processed)
        summary_path = os.path.join(args.out_dir, "spore_summary.csv")
        write_summary_csv(stats, summary_path)
        print(f"\nSummary (mean, SD, SEM = SD/sqrt(n)): {summary_path}\n")
        print(format_table(stats))

    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
