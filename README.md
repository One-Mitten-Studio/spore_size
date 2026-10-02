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
2. Finds spores by colour: yellow-brown interiors and dark red-brown walls.
   Air bubbles and oil droplets (greenish centres, faint rims) are ignored.
3. Splits touching spores apart (each spore's brown interior is used as a
   seed) and fits an ellipse to each one.
4. Rejects anything that is not a clean, single, pigmented oval: implausible
   size, irregular shape, too pale compared with the background (tissue,
   mountant tint), or much darker than it (dirt, smudges).
5. Writes, into --out-dir:
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
- Detection is tuned for pigmented (yellow-brown) spores like Cortinarius /
  Thaxterogaster. Hyaline or differently coloured spores need the colour
  settings changed (see Tuning).
- Touching spores are split, but overlapping ones (one on top of another)
  and spores embedded in tissue are usually skipped, so counts are a
  conservative undercount. The `*_spore_overlay.png` lets you visually
  spot-check what was measured.
- Always check `*_calib_overlay.png` for a couple of photos in a new batch
  to confirm the green line is tracking the actual ruler -- if lighting or
  ruler style differs a lot from the sample, the auto-detection may need
  retuning (thresholds are in `src/spore_size/calibration.py` /
  `src/spore_size/segment.py`).

## Tuning
If a batch's spores are systematically missed or merged, the main knobs are
in `src/spore_size/segment.py`'s `segment_spores()` (OpenCV HSV: hue 0-180,
saturation/brightness 0-255):
- `interior_hue`, `interior_sat`: hue range and minimum saturation of the
  spore interior (default yellow-brown, 8-30 and >35).
- `rim_sat`: minimum saturation of the red-brown spore wall (default 60);
  raise it if bubble rims get merged into spores.
- `min_sat_above_bg`, `min_rel_brightness`: how much more saturated than the
  background a spore must be, and how dark it may be relative to it.
- `min_solidity`, `min_axis_ratio`/`max_axis_ratio`: shape strictness.

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

## License
MIT -- see [LICENSE](LICENSE).
