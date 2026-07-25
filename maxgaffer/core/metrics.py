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
            pixels = (im.get_flattened_data() if hasattr(im, "get_flattened_data")
                      else im.getdata())
            return list(pixels), w, h
    except Exception:
        pass
    from . import png_min

    try:
        rows = png_min.read_png_rgb(path, max_dim=max_dim)
    except Exception:                         # the stdlib floor degrades, never raises
        return None                           # (SPEC: loop stats can never fail)
    if rows is None or not rows[0]:
        return None
    return [px for row in rows for px in row], len(rows[0]), len(rows)


# --------------------------------------------------------------------------- color math
def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _f_lab(t: float) -> float:
    return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t + 16.0 / 116.0)


def _lab_from_linear(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """LAB from LINEAR-light rgb — lets the hot loop reuse its linear conversion
    instead of paying the sRGB decode twice per pixel."""
    # sRGB D65
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883
    fx, fy, fz = _f_lab(x), _f_lab(y), _f_lab(z)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def _rgb_to_lab(r8: int, g8: int, b8: int) -> Tuple[float, float, float]:
    r, g, b = (_srgb_to_linear(v / 255.0) for v in (r8, g8, b8))
    return _lab_from_linear(r, g, b)


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


def _unit3(vec: Sequence[float]) -> List[float]:
    """Unit-normalize a 3-vector. Degenerate input (near-zero norm — a flat or black
    frame) → [0.0, 0.0, 0.0], which every consumer reads as 'estimate unavailable'."""
    norm = math.sqrt(sum(c * c for c in vec))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    return [c / norm for c in vec]


# --------------------------------------------------------------------------- stats
def compute_stats(path: str, max_dim: int = 256) -> Optional[Dict]:
    loaded = _load_pixels(path, max_dim=max_dim)
    if not loaded:
        return None
    pixels, w, h = loaded
    n = len(pixels)
    if n == 0:      # degenerate decode — every other failure mode returns None too
        return None
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
    # SUN-PATCH map: how much of the frame is a bright directional highlight, and WHERE.
    # The luminance grids average their cells, which is exactly what erases a sun patch —
    # a patch is small and very bright, and a cell mean cannot tell it from the whole cell
    # being slightly brighter. Measured on-box 2026-07-25: a match with NO sun patches at
    # all scored 0.92 on the direction component against a reference covered in them.
    # Absolute threshold, not a percentile: a percentile would give every image the same
    # hot fraction by construction and measure nothing.
    hot_cells = [0] * 25
    hot_total = 0
    lum_flat: List[float] = []          # for the local-contrast pass below

    grid_sum = [0.0] * 9
    grid_n = [0] * 9
    g5_sum = [0.0] * 25
    g5_n = [0] * 25
    # illuminant-ESTIMATE accumulators (gray-world / gray-edge assumption — see illum_*
    # below): all read from the SAME linear channels the tonal loop already decodes, so both
    # engines agree exactly as the existing keys do
    sog_sum = [0.0, 0.0, 0.0]         # Σ linear_c**6 per channel → Shades-of-Gray L6
    grid_log_sum = [0.0] * 9          # per-pixel log-luminance, 3×3 spatial descriptor
    g5_log_sum = [0.0] * 25           # per-pixel log-luminance, 5×5 spatial descriptor
    lin_rgb: List[Tuple[float, float, float]] = []   # kept for the 1-pass gray-edge gradient
    for idx, (r, g, b) in enumerate(pixels):
        mean_rgb[0] += r
        mean_rgb[1] += g
        mean_rgb[2] += b
        # ONE sRGB decode per channel per pixel — reused for luminance AND LAB
        lin_r = _srgb_to_linear(r / 255.0)
        lin_g = _srgb_to_linear(g / 255.0)
        lin_b = _srgb_to_linear(b / 255.0)
        lin_rgb.append((lin_r, lin_g, lin_b))    # for the gray-edge gradient pass below
        sog_sum[0] += lin_r ** 6                  # Shades-of-Gray L6 (Minkowski p=6)
        sog_sum[1] += lin_g ** 6
        sog_sum[2] += lin_b ** 6
        lin_l = 0.2126 * lin_r + 0.7152 * lin_g + 0.0722 * lin_b
        log_l = math.log(max(lin_l, 1e-5))
        log_sum += log_l
        x, y = idx % w, idx // w
        if cx0 <= x < cx1 and cy0 <= y < cy1:
            log_sum_center += log_l
            n_center += 1
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        lums.append(lum)
        lum_flat.append(lum)
        cell = min(2, x * 3 // max(1, w)) + 3 * min(2, y * 3 // max(1, h))
        grid_sum[cell] += lum
        grid_n[cell] += 1
        grid_log_sum[cell] += log_l          # same cell, log-luminance (spatial descriptor)
        c5 = min(4, x * 5 // max(1, w)) + 5 * min(4, y * 5 // max(1, h))
        g5_sum[c5] += lum
        g5_n[c5] += 1
        g5_log_sum[c5] += log_l
        L, a, bb = _lab_from_linear(lin_r, lin_g, lin_b)
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
    hi_clipped = 0
    incl_sum = [0.0, 0.0, 0.0]   # INCLUSIVE mean (clipped pixels counted) — the symmetric
    incl_n = 0                   # same-scene signal; see lab_mean_hi_full below
    for r, g, b in pixels:
        if (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0 >= hi_thresh:
            # per-channel-saturated pixels read as NEUTRAL in LAB (a*=b*=0) — the
            # illuminant color they once carried is gone, and counting them drags the
            # highlight-quartile b* toward 0 (software-exposed 8-bit frames clip
            # exactly this way, over-driving the WB solve). Excluded unless the whole
            # quartile is clipped, in which case the old behavior is the only signal.
            L, a, bb = _rgb_to_lab(r, g, b)
            incl_sum[0] += L
            incl_sum[1] += a
            incl_sum[2] += bb
            incl_n += 1
            if max(r, g, b) >= 254:
                hi_clipped += 1
                continue
            hi_sum[0] += L
            hi_sum[1] += a
            hi_sum[2] += bb
            hi_n += 1
    hi_clip_frac = hi_clipped / max(1, incl_n)
    if hi_n == 0 and hi_clipped:      # fully blown quartile — fall back, don't zero out
        hi_sum, hi_n = incl_sum, incl_n

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
    # ---- sun-patch map: pixels bright relative to their own neighbourhood.
    # A summed-area table keeps this O(n) — compute_stats runs on every polish probe, and
    # a naive per-pixel window would add seconds to each of hundreds of renders.
    if w > 0 and h > 0 and len(lum_flat) >= w * h:
        sat = [0.0] * ((w + 1) * (h + 1))
        for yy in range(h):
            row_run = 0.0
            base, above, here = yy * w, yy * (w + 1), (yy + 1) * (w + 1)
            for xx in range(w):
                row_run += lum_flat[base + xx]
                sat[here + xx + 1] = sat[above + xx + 1] + row_run
        rad = HOT_LOCAL_RADIUS
        for yy in range(h):
            for xx in range(w):
                x0, x1 = max(0, xx - rad), min(w, xx + rad + 1)
                y0, y1 = max(0, yy - rad), min(h, yy + rad + 1)
                area = (x1 - x0) * (y1 - y0)
                if area <= 0:
                    continue
                total_win = (sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1]
                             - sat[y1 * (w + 1) + x0] + sat[y0 * (w + 1) + x0])
                lv = lum_flat[yy * w + xx]
                if lv > HOT_THRESHOLD and lv > total_win / area + HOT_LOCAL_LIFT:
                    hot_cells[min(4, yy * 5 // h) * 5 + min(4, xx * 5 // w)] += 1
                    hot_total += 1

    grid_mean = sum(grid_sum) / max(1, sum(grid_n)) or 1e-6
    grid = [(grid_sum[i] / grid_n[i] - grid_mean) if grid_n[i] else 0.0 for i in range(9)]
    g5_mean = sum(g5_sum) / max(1, sum(g5_n)) or 1e-6
    grid5 = [(g5_sum[i] / g5_n[i] - g5_mean) if g5_n[i] else 0.0 for i in range(25)]
    # illuminant ESTIMATES (gray-world / gray-edge) — valid only under a flat-albedo
    # assumption, NOT albedo-invariant: a strongly chromatic uniform albedo (a walnut
    # library) biases them toward the wall/wood, so consumers treat these as advisory WB
    # cues and NEVER feed them to the weighted score
    illum_sog = _unit3([(sog_sum[i] / n) ** (1.0 / 6.0) for i in range(3)])
    edge_sum = [0.0, 0.0, 0.0]                # 1st-order gray-edge: one |dx|+|dy| pass
    for i in range(n):
        pr, pg, pb = lin_rgb[i]
        if i % w + 1 < w:                     # right neighbor, same row
            nr, ng, nb = lin_rgb[i + 1]
            edge_sum[0] += abs(pr - nr)
            edge_sum[1] += abs(pg - ng)
            edge_sum[2] += abs(pb - nb)
        if i + w < n:                         # neighbor one row below
            nr, ng, nb = lin_rgb[i + w]
            edge_sum[0] += abs(pr - nr)
            edge_sum[1] += abs(pg - ng)
            edge_sum[2] += abs(pb - nb)
    illum_edge = _unit3(edge_sum)
    illum = _unit3([(illum_sog[i] + illum_edge[i]) / 2.0 for i in range(3)])
    # per-pixel log-luminance grids, mean-centered exactly like grid / grid5 above; spatial
    # descriptors of WHERE the light lives in log space, NOT albedo-invariant
    grid_log_mean = sum(grid_log_sum) / max(1, sum(grid_n)) or 1e-6
    grid_log = [(grid_log_sum[i] / grid_n[i] - grid_log_mean) if grid_n[i] else 0.0
                for i in range(9)]
    g5_log_mean = sum(g5_log_sum) / max(1, sum(g5_n)) or 1e-6
    grid5_log = [(g5_log_sum[i] / g5_n[i] - g5_log_mean) if g5_n[i] else 0.0
                 for i in range(25)]
    return {
        "count": n,
        "mean_rgb": [v / n / 255.0 for v in mean_rgb],
        "grid": [round(g, 5) for g in grid],   # mean-centered 3×3 luminance pattern
        "grid5": [round(g, 5) for g in grid5],  # 5×5 — finer directional acuity
        # sun-patch presence and placement (see HOT_THRESHOLD)
        "hot_frac": round(hot_total / n, 5),
        "hot_grid": [round(c / hot_total, 5) for c in hot_cells] if hot_total else [0.0] * 25,
        # illuminant ESTIMATES (unit RGB, linear) under the gray-world / gray-edge
        # assumption — advisory WB cues, deliberately OUTSIDE the weighted critic score;
        # [0,0,0] means "unavailable" (degenerate/flat frame)
        "illum_sog": illum_sog,
        "illum_edge": illum_edge,
        "illum": illum,
        # mean-centered per-pixel log-luminance grids (spatial descriptors, like grid/grid5)
        "grid_log": [round(g, 5) for g in grid_log],
        "grid5_log": [round(g, 5) for g in grid5_log],
        "lab_mean_hi": ([s / hi_n for s in hi_sum] if hi_n else lab_mean),
        # INCLUSIVE highlight mean (clipped pixels counted, neutral-drag and all) — the
        # solver switches to it when BOTH images clip heavily: same-scene matches need
        # symmetric highlight populations or the WB anchor drifts cool (v0.9.7 regression:
        # per-image exclusion mismatched the populations and stalled deep match at 97.8)
        "lab_mean_hi_full": ([s / incl_n for s in incl_sum] if incl_n else lab_mean),
        "hi_clip_frac": hi_clip_frac,
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


#: A sun patch is bright RELATIVE TO ITS SURROUNDINGS — that is what makes it a patch and
#: not just a bright room. An absolute threshold alone is satisfied by any globally
#: brightened frame, which is precisely the metamer this codebase keeps meeting: measured
#: on-box 2026-07-26, a match that cranked the dome to 2.29 (target 0.55) and white balance
#: to 10400 K (target 4800) carried MORE absolutely-bright pixels than the reference and
#: scored 0.891 on absolute-threshold agreement, with its sun 78 degrees out. Judged on
#: local contrast the same pair scores 0.514: the reference has 4.1% of frame in genuine
#: patches, the match 1.4%.
#:
#: So a pixel is a highlight when it clears BOTH: bright enough to read as light rather
#: than shade, and this far above its own neighbourhood's mean.
HOT_THRESHOLD = 0.35
HOT_LOCAL_LIFT = 0.18
#: Neighbourhood radius in pixels at the stats working resolution (max_dim 256).
HOT_LOCAL_RADIUS = 6


def highlight_similarity(ref: Dict, cur: Dict) -> Optional[float]:
    """Do both frames carry the same amount of bright directional light, in the same
    places? → 0..1, or None when neither side reports the map.

    This is the signal the weighted critic is missing. Measured on-box 2026-07-25 on a
    golden-hour interior: the reference carried sun patches in 15 of 25 cells at 17.3% of
    the frame; the match carried hot pixels in 3 cells at 5.8%, and all three were the
    blown window itself — no floor patch, no wall patch, no directional light anywhere.
    The two are unmistakable side by side. The critic's direction component scored them
    0.92.

    A rig that switches its sun off scores ZERO here, never "unmeasurable" — the candidate
    must not be able to delete a criterion by giving up on it."""
    if not isinstance(ref, dict) or not isinstance(cur, dict):
        return None
    if "hot_frac" not in ref or "hot_frac" not in cur:
        return None                       # stats predate the map — nothing to compare
    fr, fc = float(ref["hot_frac"]), float(cur["hot_frac"])
    if fr <= 0.0 and fc <= 0.0:
        return 1.0                        # both overcast: they agree there is no sun patch
    if fr <= 0.0 or fc <= 0.0:
        return 0.0                        # one has directional light and the other has none
    # presence, in octaves — 3x more or less sun patch is a different photograph
    octaves = abs(math.log2(fr / fc))
    presence = max(0.0, 1.0 - octaves / 2.0)
    # placement: where the patches landed
    placement = (cosine(ref.get("hot_grid") or [], cur.get("hot_grid") or []) + 1.0) / 2.0
    return round(0.5 * presence + 0.5 * placement, 4)


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


def _illum_vec(value) -> Optional[List[float]]:
    """Validated 3-vector for illuminant comparison: exactly-3, all finite, non-degenerate
    (norm above the noise floor). Anything else → None (read as 'estimate unavailable')."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        vec = [float(c) for c in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(c) for c in vec):
        return None
    if math.sqrt(sum(c * c for c in vec)) < 1e-9:
        return None
    return vec


def illuminant_similarity(ref: Dict, cur: Dict) -> Optional[float]:
    """Similarity of two illuminant ESTIMATES (gray-world / gray-edge). A diagnostic WB cue,
    NOT a scored metric and NOT albedo-invariant.

    Prefers illum_edge on both sides (gray-edge is the more albedo-robust estimator), then
    illum_sog, then the blended illum. GUARDS FIRST: any missing / wrong-length / degenerate
    ([0,0,0] / near-zero) vector → None BEFORE cosine() is ever called, so cosine's
    zero-vector convention (1.0 for two zeros, 0.0 for one) can never leak a fake score.

    Returns (cosine + 1) / 2. Physical illuminant vectors are non-negative unit vectors, so
    cosine ∈ [0, 1] and the EFFECTIVE range here is [0.5, 1] (not the full 0..1). Legacy
    stats dicts (no illum_* keys) → None, leaving every current consumer unaffected."""
    if not isinstance(ref, dict) or not isinstance(cur, dict):
        return None
    for key in ("illum_edge", "illum_sog", "illum"):
        v_ref = _illum_vec(ref.get(key))
        v_cur = _illum_vec(cur.get(key))
        if v_ref is not None and v_cur is not None:
            return (cosine(v_ref, v_cur) + 1.0) / 2.0
    return None
