# Automated Spore Sizing

Measures spore length x width (um) from micrographs that have a diagonal
0-100 ocular-micrometer scale bar overlay, like the sample photo tested here.

## Install
Requires Python 3.10+. From a clone of this repo:

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
The goal is a trustworthy size range: it measures only clean, in-focus,
isolated spores and would rather skip a spore (or a whole photo) than
measure a doubtful one.

1. Skips photos that are not usable spore micrographs, printing
   `[SKIPPED] <photo>: <reason>`:
   - no circular microscope field of view (e.g. a photo of the collection),
   - a cluttered field: more than 28% of the view is neither plain
     background nor spores, as in tissue squashes (dense spore mounts are
     fine; `--max-clutter` to change),
   - no reliable scale bar (fewer than 8 evenly spaced ticks in a row).
2. Auto-detects the ruler and computes microns-per-pixel for THAT photo
   (each photo is calibrated independently, since the ruler's position/angle
   varies shot to shot).
3. Finds spores by colour: yellow-brown interiors and dark red-brown walls.
   Air bubbles and oil droplets (greenish centres, faint rims) are ignored.
4. Keeps only spores that are:
   - in focus: a sharp, dark wall all round (ghost, pale and blurred spores
     are dropped). Wall contrast depends on the microscope setup, so in a
     low-contrast photo the limits are relaxed relative to the photo's own
     spores (never past fixed minimums, and never made stricter),
   - isolated: not touching another spore (`--include-touching` splits and
     measures touching spores too),
   - clear of the scale bar and of the edge of the field of view,
   - single, well-fit ovals of plausible size, not much paler or darker than
     a pigmented spore should be,
   - not oddly sized: more than ~3 robust SDs from the median length or
     width of all spores from the same directory (collection), pooled over
     its photos. This mainly removes two overlapping spores measured as
     one, which are roughly 1.5-2x the normal size; pooling the directory
     keeps a few of them in one photo from skewing that photo's median. The
     terminal reports how many were removed per directory.
5. Writes, into --out-dir:
   - `<photo>_spore_overlay.png`   - annotated image, red ellipse per measured spore
   - `<photo>_spore_measurements.csv` - length_um, width_um, Q ratio per spore
   - `<photo>_calib_overlay.png`   - diagnostic: green line = detected ruler
   - `<photo>_calib_strip.png`     - diagnostic: ruler profile, red = detected ticks
     (skip both diagnostics with `--no-debug`)
   - `spore_summary.csv`           - statistics for the whole run (see below)
   - `*_spore_plot.png`            - length vs width plots: one per photo, one
     per directory and `all_spores_plot.png` (see below; `--no-plots` to skip)
   - `all_spores_combined.csv`     - every spore from every photo, one file;
     the `image` column is the path exactly as given on the command line
   - `<dir>_spores_combined.csv`   - the same, for each directory the images
     came from (e.g. `inat255552721_spores_combined.csv` for
     `inat255552721/P9102898.JPG ...`)

It also prints a one-line summary (mean +/- SD) per photo as it goes, and the
summary table at the end.

## Measuring by hand: `spore-measure`
For photos the automatic detection can't handle (e.g. crowded gill-face
mounts), measure spores by hand:

    pip install -e ".[gui]"      # once: installs the Qt window toolkit
    spore-measure photo1.jpg photo2.jpg --out-dir manual_results

On Linux with X11, Qt also needs a system library: `sudo apt install
libxcb-cursor0` (spore-measure tells you if it is missing).

Each photo opens in a window. Zoom in until the spore is large enough to
click its edges accurately: the mouse wheel (or `+` / `-`) zooms about the
pointer, dragging with the right or middle button pans, and `h` returns to
the whole photo. For each spore, click the two ends of its length (red line), then the two sides of its
width (blue line). The scale bar is calibrated automatically; if that fails
(or press `c`), click its 0 and 100 ticks (`--ruler-ticks` if you click
other marks). While the photo needs calibrating, the status lines are red;
clicks then set the scale, not a spore. A hand calibration more than 15%
off the run's automatic calibrations (or, without any, the other photos')
is held until you click the ticks again or press `a` to accept it, and a
saved one is flagged when the photo is reopened, so clicking a spore as the
scale by mistake can't silently inflate every measurement. The status line
shows each spore's size in µm as you finish it. Keys: `u` / Backspace undo,
`c` calibrate by hand, `a` accept a flagged calibration, `n` next photo,
`q` save and quit.

Measurements are saved after every spore to
`<photo>_manual_measurements.csv` (with the clicked points), and reopening a
photo restores them, so you can stop and continue later. When you finish,
the same reports as `spore-size` are written (`spore_summary.csv`, combined
CSVs, plots), so hand and automatic measurements compare directly.
`--report-only` rebuilds the reports from the saved measurements without
opening any windows.

Only measure spores lying flat in side view: focus up and down at the
microscope -- a tilted spore's outline goes in and out of focus around its
edge, and its length would be measured too short.

## Summary statistics
`spore_summary.csv` has one row per photo, a total row for each directory
(directly after its photos) and a final row for all spores. Directory and
"all" rows pool the individual spores rather than averaging photo means.
For length, width and Q it gives, in this order:
- `_mean`
- `_sd`: the spread of individual spores -- natural variation, not an error.
- `_sem`: the statistical error of the mean, SD / sqrt(n) (sample SD, n-1
  denominator) -- how much the mean would vary if another random sample of
  the same number of spores were measured. Shrinks as more spores are
  measured.
- `_sys`: the systematic error. A light microscope cannot resolve better
  than ~0.1 um, so every measurement carries at least that error, and it
  does not average away (`--sys-error-um` to change). For Q it is propagated
  from the length and width errors.
- `_total_err`: the total error of the mean, stat and sys added in quadrature,
  sqrt(sem² + sys²).

- `publication`: ready to paste, e.g.
  `6.46 ± 0.51 µm (err: 0.11) by 4.56 ± 0.32 µm (err: 0.10)` --
  mean ± SD, with the total error of the mean, for length by width.

Report a mean as `mean ± stat ± sys`, e.g. `6.46 ± 0.05 ± 0.10 µm`, or with
the total error, `6.46 ± 0.11 µm`. When
comparing collections measured with the same microscope and scale bar, the
systematic error is largely shared, so differences between them are judged
against the statistical errors.

The "all" row pools every photo on the command line; only read it as one
population if all photos are of the same collection.

## Plots
Each plot shows every measured spore as a semi-transparent circle with a thin
black edge, centred on its length and width, with a radius equal to the
systematic error (0.1 µm by default). Both axes use the same µm scale, and
all plots of a run share the same axis ranges, so photos and collections can
be compared by eye. Each directory has its own colour, used for its photos,
its own plot and its group in `all_spores_plot.png`.

A dotted ellipse marks mean ± 1 SD: centred on the mean length and width (+),
with the length SD and width SD as its semi-axes. `all_spores_plot.png`
shows one per directory in its colour, plus a black one for all spores.

## Assumptions / things to check
- **Magnification:** the scale bar is in the eyepiece, so its ticks are the
  same number of pixels apart at every objective -- the program cannot tell
  400x from 1000x. The default `--um-per-tick 1.0` is correct only for the
  1000x (100x oil) objective. Only pass 1000x spore photos, or set
  `--um-per-tick` to your micrometer's value for the objective used.
- Default accepted spore size range is 3-25 um long, 1.5-18 um wide
  (`--min-length-um` etc.) -- widen/narrow these for your species.
- Detection is tuned for pigmented (yellow-brown) spores like Cortinarius /
  Thaxterogaster. Hyaline or differently coloured spores need the colour
  settings changed (see Tuning).
- Counts are deliberately conservative: typically about half the spores in a
  field are measured. The `*_spore_overlay.png` shows exactly which ones.
- Check `*_calib_overlay.png` for a couple of photos in a new batch to
  confirm the green line is tracking the actual ruler.

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
- `min_sharpness`, `max_wall_level`: how crisp and dark the spore wall must
  be, relative to the background, to always pass (focus strictness);
  `rel_sharpness`, `wall_margin`: how far these relax in a low-contrast
  photo, relative to its own spores; `sharpness_floor`, `wall_ceiling`: the
  hard limits they never relax past.
- `min_solidity`, `min_axis_ratio`/`max_axis_ratio`: shape strictness.

## Project layout
    src/spore_size/
        calibration.py   scale-bar detection -> microns per pixel
        segment.py       spore segmentation + ellipse fitting
        measure.py       per-image pipeline: filter, CSV, overlay
        fov.py           microscope field-of-view detection / micrograph check
        errors.py        PhotoRejected (photo skipped, not an error)
        _types.py        shared type aliases for annotations
        plots.py         length vs width plots
        summary.py       run statistics (mean, SD, stat/sys/total errors) -> spore_summary.csv
        cli.py           `spore-size` command
        manual.py        `spore-measure`: measuring spores by hand
    tests/               pytest suite (real micrograph + synthetic spores)

## Development
    pip install -e ".[dev]"
    pytest
    black src tests        # code formatting (line length 100, set in pyproject.toml)
    ruff check src tests   # linting incl. PEP 257 docstrings (rules in pyproject.toml; --fix)
    mypy                   # type checking (settings in pyproject.toml)

All functions in the package have type annotations, and mypy requires them
for new code. Shared types (`Image`, `Spore`, `Calibration`, ...) are in
`src/spore_size/_types.py`.

## License
MIT -- see [LICENSE](LICENSE).
