"""
Command-line entry point.

Usage:
    spore-size <image.jpg> [more images...] [--out-dir DIR] [--um-per-tick 1.0]
               [--min-length-um 3] [--max-length-um 25]
               [--min-width-um 1.5] [--max-width-um 18]
"""
import argparse
import csv
import os
import sys

from .errors import PhotoRejected
from .measure import process_image


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


def main(argv=None):
    args = build_parser().parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
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
        print(f"{os.path.basename(path)}: n={summary['n']}  "
              f"L={summary['length_mean']:.2f}±{summary['length_sd']:.2f} um  "
              f"W={summary['width_mean']:.2f}±{summary['width_sd']:.2f} um  "
              f"Q={summary['Q_mean']:.2f}  ({summary['um_per_px']:.4f} um/px)")
        for s in kept:
            all_rows.append({"image": os.path.basename(path), **s})

    if all_rows:
        combined_path = os.path.join(args.out_dir, "all_spores_combined.csv")
        with open(combined_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "length_um", "width_um", "Q_ratio"])
            for r in all_rows:
                w.writerow([r["image"], f"{r['length_um']:.2f}", f"{r['width_um']:.2f}",
                            f"{r['Q_ratio']:.2f}"])
        print(f"\nCombined CSV: {combined_path}")

    return 0 if n_ok else 1


if __name__ == "__main__":
    sys.exit(main())
