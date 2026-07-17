"""Image → lighting statistics. The numbers the analytic solver and the tonal critic run on.

Three engines, best available wins, all producing the IDENTICAL stats dict so the solver and
critic never care where numbers came from:
  1. Pillow (+numpy if present)  — dev venv / sidecar python / Max if user pip-installed it
  2. stdlib png_min              — always works on our own Max-rendered PNGs
  3. (bridge-side) Max transcode — JPEG/EXR refs are saved to small PNGs by the bridge first

Stats are deliberately about TONE and COLOR MOOD, not structure: the reference photo and the
render are different scenes, so SSIM-style spatial comparison is meaningless. What transfers
between different scenes is the tonal envelope (key, contrast, shadow/highlight placement)
and the chromatic mood (LAB means, hue distribution) — that is what we measure and match.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

LUM_BINS = 32
HUE_BINS = 12


# --------------------------------------------------------------------------- pixel loading
def _load_pixels(path: str, max_dim: int = 256):
    """→ (flat [(r,g,b)…], width, height) subsampled, or None. Pillow first, stdlib PNG
    floor second. Dimensions are kept so the key can be center-weighted."""
    try:
        from PIL import Image, ImageOps  # type: ignore

        with Image.open(path) as im:
            # phone references carry EXIF orientation — without honoring it a portrait
            # shot reads sideways and the DIRECTION grid (where the light lives) is
            # garbage for both the critic and the sun solve
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((max_dim, max_dim))
            w, h = im.size
            return list(im.getdata()), w, h
    except Exception:
        pass
    from . import png_min

    rows = png_min.read_png_rgb(path, max_dim=max_dim)
    if rows is None or not rows[0]:
        return None
    return [px for row in rows for px in row], len(rows[0]), len(rows)


# --------------------------------------------------------------------------- color math
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _f_lab(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)


def _rgb_to_lab(r8: int, g8: int, b8: int) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v / 255.0) for v in (r8, g8, b8))
    # sRGB D65
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _rgb_to_hue_chroma(r: int, g: int, b: int) -> Tuple[float, float]:
    mx, mn = max(r, g, b), min(r, g, b)
    c = (mx - mn) / 255.0
    if c <= 0:
        return 0.0, 0.0
    if mx == r:
        h = ((g - b) / (mx - mn)) % 6
    elif mx == g:
        h = (b - r) / (mx - mn) + 2
    else:
        h = (r - g) / (mx - mn) + 4
    return h * 60.0, c


# --------------------------------------------------------------------------- stats
def compute_stats(path: str, max_dim: int = 256) -> Optional[Dict]:
    loaded = _load_pixels(path, max_dim=max_dim)
    if not loaded:
        return None
    pixels, w, h = loaded
    n = len(pixels)
    # center-weighted key (photographic AE practice): the subject usually sits in the
    # middle half of the frame, so weighting it 60/40 over the full frame damps sky/floor
    # albedo contamination of the exposure solve
    cx0, cx1 = w // 4, (3 * w) // 4
    cy0, cy1 = h // 4, (3 * h) // 4
    lums: List[float] = []
    log_sum = 0.0
    log_sum_center, n_center = 0.0, 0
    lab_sum = [0.0, 0.0, 0.0]
    lab_sq = [0.0, 0.0, 0.0]
    hue_hist = [0.0] * HUE_BINS
    mean_rgb = [0.0, 0.0, 0.0]
    # 3×3 luminance grid — WHERE the light lives. Unlike content-bound structure metrics,
    # the bright-third pattern transfers across different scenes lit the same way (sun
    # camera-left brightens the left cells of ref AND render) — the critic's direction eye.
    grid_sum = [0.0] * 9
    grid_n = [0] * 9
    g5_sum = [0.0] * 25
    g5_n = [0] * 25
    for idx, (r, g, b) in enumerate(pixels):
        mean_rgb[0] += r
        mean_rgb[1] += g
        mean_rgb[2] += b
        lin_l = (0.2126 * _srgb_to_linear(r / 255.0)
                 + 0.7152 * _srgb_to_linear(g / 255.0)
                 + 0.0722 * _srgb_to_linear(b / 255.0))
        log_l = math.log(max(lin_l, 1e-5))
        log_sum += log_l
        x, y = idx % w, idx // w
        if cx0 <= x < cx1 and cy0 <= y < cy1:
            log_sum_center += log_l
            n_center += 1
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        lums.append(lum)
        cell = min(2, x * 3 // max(1, w)) + 3 * min(2, y * 3 // max(1, h))
        grid_sum[cell] += lum
        grid_n[cell] += 1
        c5 = min(4, x * 5 // max(1, w)) + 5 * min(4, y * 5 // max(1, h))
        g5_sum[c5] += lum
        g5_n[c5] += 1
        L, a, bb = _rgb_to_lab(r, g, b)
        for i, v in enumerate((L, a, bb)):
            lab_sum[i] += v
            lab_sq[i] += v * v
        hue, chroma = _rgb_to_hue_chroma(r, g, b)
        if chroma > 0.02:  # near-neutrals carry no hue information
            hue_hist[int(hue / 360.0 * HUE_BINS) % HUE_BINS] += chroma
    lums.sort()   # one sort serves both the percentiles and the highlight threshold
    # highlight chromaticity — the top luminance quartile carries the ILLUMINANT's color
    # (white-patch assumption), far less contaminated by scene albedo than the full mean
    hi_thresh = lums[max(0, int(0.75 * len(lums)) - 1)]
    hi_sum = [0.0, 0.0, 0.0]
    hi_n = 0
    for r, g, b in pixels:
        if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0 >= hi_thresh:
            L, a, bb = _rgb_to_lab(r, g, b)
            hi_sum[0] += L
            hi_sum[1] += a
            hi_sum[2] += bb
            hi_n += 1

    def pct(p: float) -> float:
        return lums[min(n - 1, int(p / 100.0 * n))]

    lum_hist = [0.0] * LUM_BINS
    for lum in lums:
        lum_hist[min(LUM_BINS - 1, int(lum * LUM_BINS))] += 1.0
    hist_total = sum(lum_hist) or 1.0
    hue_total = sum(hue_hist)
    lab_mean = [s / n for s in lab_sum]
    lab_std = [math.sqrt(max(0.0, lab_sq[i] / n - lab_mean[i] ** 2)) for i in range(3)]
    key_full = log_sum / n
    key_center = log_sum_center / n_center if n_center else key_full
    grid_mean = sum(grid_sum) / max(1, sum(grid_n)) or 1e-6
    grid = [(grid_sum[i] / grid_n[i] - grid_mean) if grid_n[i] else 0.0 for i in range(9)]
    g5_mean = sum(g5_sum) / max(1, sum(g5_n)) or 1e-6
    grid5 = [(g5_sum[i] / g5_n[i] - g5_mean) if g5_n[i] else 0.0 for i in range(25)]
    return {
        "count": n,
        "mean_rgb": [v / n / 255.0 for v in mean_rgb],
        "grid": [round(g, 5) for g in grid],   # mean-centered 3×3 luminance pattern
        "grid5": [round(g, 5) for g in grid5],  # 5×5 — finer directional acuity
        "lab_mean_hi": ([s / hi_n for s in hi_sum] if hi_n else lab_mean),
        # geometric mean of LINEAR luminance, 60% center-weighted (blend in log space)
        "log_key": math.exp(0.6 * key_center + 0.4 * key_full),
        "p": {"5": pct(5), "25": pct(25), "50": pct(50), "75": pct(75), "95": pct(95)},
        "contrast": pct(95) - pct(5),
        "lab_mean": lab_mean,
        "lab_std": lab_std,
        "lum_hist": [v / hist_total for v in lum_hist],
        "hue_hist": [v / hue_total for v in hue_hist] if hue_total > 0 else [0.0] * HUE_BINS,
        "saturation": hue_total / n,
    }


# --------------------------------------------------------------------------- comparisons
def hist_emd(a: Sequence[float], b: Sequence[float]) -> float:
    """1-D earth mover's distance between normalized histograms (cumsum difference)."""
    total, cum = 0.0, 0.0
    for x, y in zip(a, b):
        cum += x - y
        total += abs(cum)
    return total / max(1, len(a))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0 if na < 1e-9 and nb < 1e-9 else 0.0
    return dot / (na * nb)
