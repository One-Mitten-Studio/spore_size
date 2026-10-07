"""Photos with a strong colour cast (camera white balance far off).

The rest of the pipeline expects a near-neutral background: the scale bar
is found as colourless edges, spores by their yellow-brown and red-brown
colours. With a strong cast (e.g. a deep yellow background) every pixel is
saturated, and one channel can be nearly empty (blue ~0) while another is
clipped (red at 255).

Such photos are detected from the background colour. The scale bar is then
found in the single most informative channel, and the photo is corrected to
a neutral background in optical density (OD = -log10(pixel / background),
i.e. how much light each object absorbs, per channel). The background is
a smooth surface fitted across the field of view, since the cast is often
uneven (yellower on one side, more orange on the other); a surface this
smooth follows the lighting but not spores or tissue. Green and red are
kept, blue is extrapolated from them (brown pigment absorbs more towards
blue: OD_blue = 2 OD_green - OD_red), and the image is rebuilt on a neutral
grey background. Red clipped at 255 in the background understates red
absorption, so the red OD is scaled so that the scale bar -- an opaque,
neutral object present in every photo -- absorbs equally in red and green,
the way a grey card is used to set white balance.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from ._types import Calibration, Image
from .fov import find_fov_circle, fov_mask

#: background saturation (OpenCV HSV, 0-255) above which a photo has a colour
#: cast; normal photos' backgrounds are near grey (0-25)
CAST_SATURATION: Final = 80
#: grey level of the background in a corrected photo, like a normal photo's
CORRECTED_BACKGROUND: Final = 170.0
#: bounds on the red OD gain taken from the scale bar
RED_GAIN_LIMITS: Final = (1.0, 4.0)


def background_colour(img_bgr: Image) -> NDArray[np.float64]:
    """Return the median BGR colour of the field of view (mostly background)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    center, radius = find_fov_circle(gray)
    inner = fov_mask(gray.shape, center, radius, shrink=0.85) > 0
    return np.median(img_bgr[inner].astype(np.float64), axis=0)


def has_colour_cast(img_bgr: Image) -> bool:
    """Return True if the background is strongly coloured instead of near grey."""
    bg = np.clip(background_colour(img_bgr), 0, 255).astype(np.uint8)
    sat = cv2.cvtColor(bg.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0, 1]
    return bool(sat > CAST_SATURATION)


def best_channel_image(img_bgr: Image) -> Image:
    """Return the most informative channel as a grey (3-channel) image.

    That is the channel whose background level is furthest from both empty
    (0) and clipped (255), so dark objects like the scale bar show best.
    """
    bg = background_colour(img_bgr)
    best = int(np.argmax(np.minimum(bg, 255 - bg)))
    channel = img_bgr[:, :, best]
    return cv2.merge([channel, channel, channel])


def _poly_terms(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the terms of a quadratic surface in x and y (scaled to about -1..1)."""
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)


def background_log_surface(img_bgr: Image, step: int = 8, n_iter: int = 4) -> NDArray[np.float32]:
    """Return log10 of each channel's background, as a smooth surface per pixel.

    A quadratic surface is fitted to log intensity inside the field of view on
    a coarse grid (every step pixels), then refitted without the pixels well
    below it (spores, debris, the scale bar) or well above it.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    center, radius = find_fov_circle(gray)
    inner = fov_mask(gray.shape, center, radius, shrink=0.95)[::step, ::step] > 0
    small = img_bgr[::step, ::step].astype(np.float64) + 1  # +1: no log of 0
    ys, xs = np.nonzero(inner)
    sx, sy = (w / step) / 2, (h / step) / 2
    terms = _poly_terms((xs - sx) / sx, (ys - sy) / sy)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    full_terms = _poly_terms((xx / step - sx) / sx, (yy / step - sy) / sy)
    surface = np.empty((h, w, 3), np.float32)
    for c in range(3):
        v = np.log10(small[:, :, c][inner])
        keep = np.ones(len(v), bool)
        for _ in range(n_iter):
            coef, *_ = np.linalg.lstsq(terms[keep], v[keep], rcond=None)
            res = v - terms @ coef
            mad = 1.4826 * np.median(np.abs(res[keep])) + 1e-4
            keep = (res > -2 * mad) & (res < 3 * mad)
        surface[:, :, c] = full_terms @ coef.astype(np.float32)
    return surface


def _scale_bar_samples(od: NDArray[np.float32], calib: Calibration) -> NDArray[np.float32]:
    """Return the OD of pixels along the scale bar's tick line (n x channels)."""
    h, w = od.shape[:2]
    cx, cy = calib["center"]
    a = np.radians(calib["angle_deg"])
    half_len = 50 * calib["px_per_tick"]  # the 100-tick scale, centred on its middle
    along = np.arange(-half_len, half_len)
    across = np.arange(-6, 7)
    t, n = np.meshgrid(along, across)
    x = np.clip(np.round(cx + t * np.cos(a) - n * np.sin(a)), 0, w - 1).astype(int)
    y = np.clip(np.round(cy + t * np.sin(a) + n * np.cos(a)), 0, h - 1).astype(int)
    return od[y.ravel(), x.ravel()]


def correct_colour_cast(img_bgr: Image, calib: Calibration) -> Image:
    """Return the photo corrected to a neutral grey background (see module docs)."""
    log_pixel = np.log10(img_bgr.astype(np.float32) + 1)
    od = background_log_surface(img_bgr) - log_pixel
    od_g, od_r = od[:, :, 1], od[:, :, 2]

    # red gain from the scale bar's darkest pixels, which are neutral
    samples = _scale_bar_samples(od, calib)
    g_line, r_line = samples[:, 1], samples[:, 2]
    dark = g_line >= np.percentile(g_line, 90)
    r_dark = float(np.median(r_line[dark]))
    gain = float(np.median(g_line[dark])) / r_dark if r_dark > 0 else RED_GAIN_LIMITS[1]
    od_r = od_r * float(np.clip(gain, *RED_GAIN_LIMITS))

    od_b = 2 * od_g - od_r
    out = CORRECTED_BACKGROUND * np.power(10.0, -np.stack([od_b, od_g, od_r], axis=2))
    corrected = np.clip(out, 0, 255).astype(np.uint8)
    # keep the dark surround outside the field of view dark
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    center, radius = find_fov_circle(gray)
    corrected[fov_mask(gray.shape, center, radius) == 0] = 0
    return corrected
