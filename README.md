# Automated Spore Sizing

Measures spore length x width (um) from micrographs that have a diagonal
0-100 ocular-micrometer scale bar overlay, like the sample photo tested here.

## Install
Requires Python 3.9+. From a clone of this repo:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

This installs the `spore-size` command along with its dependencies
(numpy, scipy, opencv-python-headless).

## Usage
Run on one or many photos at once:

    spore-size photo1.jpg photo2.jpg photo3.jpg --out-dir results

Or a whole folder:

    spore-size *.jpg --out-dir results

Run `spore-size --help` for all options. `python -m spore_size.cli` works too.

From Python:

```python
from spore_size import process_image
summary, spores = process_image("photo1.jpg", "results")
```

## What it does, per photo
1. Auto-detects the ruler and computes microns-per-pixel for THAT photo
   (each photo is calibrated independently, since the ruler's position/angle
   varies shot to shot).
2. Segments spore silhouettes and fits an ellipse to each one.
3. Filters out debris/bubbles (implausible size) and touching/overlapping
   spore clusters (irregular shape) -- kept spores are single, well-fit ovals.
4. Writes, into --out-dir:
   - `<photo>_spore_overlay.png`   - annotated image, red ellipse per spore
   - `<photo>_spore_measurements.csv` - length_um, width_um, Q ratio per spore
   - `<photo>_calib_overlay.png`   - diagnostic: green line = detected ruler
   - `<photo>_calib_strip.png`     - diagnostic: ruler profile, red = detected ticks
     (skip both diagnostics with `--no-debug`)
   - `all_spores_combined.csv`     - every spore from every photo, one file

It also prints a one-line summary (mean +/- SD) per photo to the terminal.

## Assumptions / things to check
- The scale bar is assumed to run 0-100 = 0-100 microns (1 tick = 1 um).
  Change with `--um-per-tick` if your scale bar uses a different convention.
- Default accepted spore size range is 3-25 um long, 1.5-18 um wide
  (`--min-length-um` etc.) -- widen/narrow these for your species.
- Touching or overlapping spores are currently EXCLUDED rather than split,
  so total counts are a conservative undercount in dense fields. The
  `*_spore_overlay.png` lets you visually spot-check what got skipped.
- Always check `*_calib_overlay.png` for a couple of photos in a new batch
  to confirm the green line is tracking the actual ruler -- if lighting or
  ruler style differs a lot from the sample, the auto-detection may need
  retuning (thresholds are in `src/spore_size/calibration.py` /
  `src/spore_size/segment.py`).

## Tuning
If a batch's spores are systematically missed or merged, the main knobs are
in `src/spore_size/segment.py`'s `segment_spores()`: `sat_thresh` (color threshold for
spore vs background), `min_solidity` (how strict about excluding
merged/irregular blobs), and `min_axis_ratio`/`max_axis_ratio` (roundness
bounds).

## Project layout
    src/spore_size/
        calibration.py   scale-bar detection -> microns per pixel
        segment.py       spore segmentation + ellipse fitting
        measure.py       per-image pipeline: filter, CSV, overlay
        fov.py           microscope field-of-view detection
        cli.py           `spore-size` command
    tests/               pytest suite (real ruler crop + synthetic spores)

## Development
    pip install -e ".[dev]"
    pytest
