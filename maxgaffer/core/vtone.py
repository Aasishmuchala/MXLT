"""Vantage-display → V-Ray-display tone transfer — the fit that makes a window grab a
measurement instead of a picture.

Why this exists: the artist's deliverable is rendered by Chaos Vantage, but every absolute
number in this plugin — ``metrics.HOT_THRESHOLD`` (0.35), the critic's histogram/key/
colour scoring, ``sunsolve``'s CROSS_DOMAIN_AGREEMENT / DECISIVE_MARGIN — was calibrated
on V-Ray display-encoded plates. A Vantage window grab is tonemapped by Vantage's own
pipeline, so today it is licensed for exactly one job: the sun solve's ORDINAL ranking,
where any monotone tone curve cancels out. This module holds the fitted transfer
``g: vantage display value → V-Ray display value`` that lets a grab land in the space all
of that machinery was calibrated for, without changing any of it.

Everything here was shaped by the 2026-07-31 adversarial stress-test of the design, whose
three decisive findings are load-bearing:

* THE FIT IS QUANTILE-MATCHED, NEVER PIXEL-REGISTERED. A V-Ray plate and a window grab of
  the same state do not share a pixel grid: the grab is a centre-crop of a client area
  whose framing model is unverified (vgrab.py's own SECOND CALIBRATION ITEM), the two
  images pass through different resampling PSFs (GDI HALFTONE box-average in display space
  vs V-Ray's AA filter in linear), and the V-Ray side is noisier per-pixel. A pixel-paired
  fit inherits every one of those as CURVE ERROR — measured consequence: a contrast-
  flattened g whose corrected grabs push fewer pixels over HOT_THRESHOLD, biasing
  ``hot_frac`` low, which is the exact metric this correction exists to serve. Matching
  sorted per-channel values (CDF matching) needs no registration at all, and for a
  monotone 1D transfer quantile matching IS the estimator. The pixel-registered path
  survives only as a DETECTOR (``pixel_pairs``): when its residual is far above the
  quantile fit's, something spatially varying (bloom, denoiser, vignette, DLSS) is in the
  window pixels, and no 1D curve should be trusted.

* IN-SAMPLE RESIDUAL IS NOT EVIDENCE. A single-axis calibration sweep (sun intensity
  alone) cannot falsify the static-curve premise, because a content-adaptive stage in
  Vantage (auto-exposure — real, documented, and not queryable from outside) responds
  monotonically to that one axis and is absorbed INTO the fitted curve: the in-sample
  residual comes back SMALL precisely when the transfer is broken. The honest numbers are
  (a) ``residual_on`` HELD-OUT states the fit never saw, and (b) ``curve_dispersion``
  across per-state fits — a static transfer produces the same curve from every state, an
  adaptive one produces a family of curves. Both gates live here; the calibration harness
  supplies the states.

* THE FIT MUST BE MONOTONIC, AND OBSERVED NON-MONOTONICITY IS EVIDENCE, NOT NOISE. The
  ordinal sun solve survives any monotone per-state transfer, so a fit that reordered
  brightness would break the one consumer that already works. Enforcement is
  pool-adjacent-violators over the binned medians; ``monotonic_observed`` records whether
  the raw relationship was monotone BEFORE enforcement, because a grossly non-monotone
  observation means the pairs never came from a 1D curve at all.

Pure python, stdlib only, no pymxs, no Qt (core/ house rule — enforced by test). Pixel
rows use the same structure as ``png_min``: lists of rows of (r, g, b) 0-255 tuples.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

#: Channel order everywhere in this module. Fitting is per-channel on purpose: a tint
#: difference between the two pipelines (one white-balance assumption vs another) shows up
#: as three DIFFERENT curves, and folding them into one luminance curve would launder a
#: colour cast into a brightness error.
CHANNELS = ("r", "g", "b")

#: Fewer DISTINCT populated input levels than this and ``fit`` refuses (returns None): a
#: curve interpolated from a handful of points is an assertion, not a measurement. A real
#: calibration sweep populates dozens of levels per channel.
MIN_LEVELS = 8

#: Held-out mean-absolute-error (0..255 display units) above which a fit must not be
#: trusted to feed absolute metrics. PROVISIONAL — set from first principles (≈2.5% of
#: full scale, comfortably under the ~0.05 display-unit spacing that separates
#: HOT_THRESHOLD from its neighbours after the sRGB decode) and MUST be re-set from real
#: paired plates once the calibration harness has run on the box. The arming path treats
#: this as a refusal boundary, so err small: a refused fit costs V-Ray render time, a
#: trusted bad fit costs wrong answers.
RESIDUAL_LIMIT = 6.0

#: Per-state curve spread (0..255 units, see ``curve_dispersion``) above which the
#: transfer is content-dependent — the auto-exposure signature — and no static curve may
#: arm regardless of how small its pooled residual is. PROVISIONAL, same re-measurement
#: obligation as RESIDUAL_LIMIT.
DISPERSION_LIMIT = 4.0

#: Sorted-value sample count for quantile pairing. 65 points spans 0..1 in 1/64 steps —
#: dense enough to shape a display curve, sparse enough that a 129600-pixel probe gives
#: every point ~2000 supporting pixels.
QUANTILE_POINTS = 65


# --------------------------------------------------------------------------- sampling
def _center_values(rows: Sequence[Sequence[Tuple[int, int, int]]],
                   crop: float = 0.6) -> Tuple[List[int], List[int], List[int]]:
    """Per-channel pixel values from the middle ``crop`` of the image, channels split.

    The same centre window vgrab's ``_all_black`` and ``_signature`` use, for the same
    reason: the edges of a window grab carry the window's own furniture, and the edges of
    a V-Ray plate carry vignetting and AA ramp — neither is evidence about the transfer.
    """
    if not rows or not rows[0]:
        return [], [], []
    h, w = len(rows), len(rows[0])
    margin = max(0.0, min(0.49, (1.0 - crop) / 2.0))
    y0, y1 = int(h * margin), max(int(h * margin) + 1, int(h * (1.0 - margin)))
    x0, x1 = int(w * margin), max(int(w * margin) + 1, int(w * (1.0 - margin)))
    rs: List[int] = []
    gs: List[int] = []
    bs: List[int] = []
    for row in rows[y0:y1]:
        for px in row[x0:x1]:
            rs.append(px[0])
            gs.append(px[1])
            bs.append(px[2])
    return rs, gs, bs


def _quantile(sorted_vals: Sequence[int], q: float) -> float:
    """Linear-interpolated quantile of an ALREADY SORTED sequence."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def quantile_pairs(vant_rows, vray_rows, points: int = QUANTILE_POINTS,
                   crop: float = 0.6) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """Registration-free fit samples: matched per-channel quantiles of the two plates.

    The two plates may have DIFFERENT pixel counts — sorting decouples them, and each
    side is sampled at the same fractional quantiles. → {"r": [(vantage_level,
    vray_level), ...], "g": [...], "b": [...]} or None when either plate is empty.
    """
    va = _center_values(vant_rows, crop)
    vr = _center_values(vray_rows, crop)
    if not va[0] or not vr[0]:
        return None
    out: Dict[str, List[Tuple[float, float]]] = {}
    n = max(2, int(points))
    for ci, ch in enumerate(CHANNELS):
        a = sorted(va[ci])
        b = sorted(vr[ci])
        out[ch] = [(_quantile(a, i / (n - 1)), _quantile(b, i / (n - 1)))
                   for i in range(n)]
    return out


def pixel_pairs(vant_rows, vray_rows,
                crop: float = 0.6) -> Optional[Dict[str, List[Tuple[float, float]]]]:
    """Registered per-pixel samples — the spatial-operator DETECTOR, never the estimator.

    Requires identical dimensions (returns None otherwise: a resample here would smuggle
    in exactly the PSF mismatch the quantile path exists to avoid). Interpretation rule:
    a fit evaluated on these pairs scoring far above the same fit on ``quantile_pairs``
    means the window pixels contain something spatially varying that no 1D curve models.
    """
    if not vant_rows or not vray_rows or not vant_rows[0] or not vray_rows[0]:
        return None
    if len(vant_rows) != len(vray_rows) or len(vant_rows[0]) != len(vray_rows[0]):
        return None
    h, w = len(vant_rows), len(vant_rows[0])
    margin = max(0.0, min(0.49, (1.0 - crop) / 2.0))
    y0, y1 = int(h * margin), max(int(h * margin) + 1, int(h * (1.0 - margin)))
    x0, x1 = int(w * margin), max(int(w * margin) + 1, int(w * (1.0 - margin)))
    out: Dict[str, List[Tuple[float, float]]] = {ch: [] for ch in CHANNELS}
    for y in range(y0, y1):
        va_row, vr_row = vant_rows[y], vray_rows[y]
        for x in range(x0, x1):
            va_px, vr_px = va_row[x], vr_row[x]
            out["r"].append((float(va_px[0]), float(vr_px[0])))
            out["g"].append((float(va_px[1]), float(vr_px[1])))
            out["b"].append((float(va_px[2]), float(vr_px[2])))
    return out


def merge_samples(sample_dicts: Sequence[Optional[Dict[str, List[Tuple[float, float]]]]]
                  ) -> Dict[str, List[Tuple[float, float]]]:
    """Concatenate per-channel sample dicts from several states into one pool. Nones —
    states whose pairing failed — are skipped, not fatal: the caller reports how many
    states actually contributed."""
    out: Dict[str, List[Tuple[float, float]]] = {ch: [] for ch in CHANNELS}
    for d in sample_dicts:
        if not d:
            continue
        for ch in CHANNELS:
            out[ch].extend(d.get(ch, ()))
    return out


# --------------------------------------------------------------------------- the fit
def _pav(values: List[float], weights: List[float]) -> List[float]:
    """Pool-adjacent-violators: the weighted least-squares NON-DECREASING sequence
    through ``values``. Standard isotonic regression on the knot list (≤256 entries) —
    each block carries (pooled value, pooled weight, source-knot count) so the pooled
    solution expands back over exactly the knots that violated."""
    blocks: List[List[float]] = []      # each: [value, weight, count]
    for v, wt in zip(values, weights):
        blocks.append([v, wt, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0] + 1e-12:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    out: List[float] = []
    for v, _wt, count in blocks:
        out.extend([v] * count)
    return out


def _iter_pairs(samples):
    """Yield well-formed (v, y) float pairs from ANYTHING — a None channel, a bare
    number, 1-tuples, 3-tuples, strings. Malformed entries are skipped, a malformed
    CONTAINER yields nothing. Found by the 2026-07-31 fuzz gauntlet: ``fit({"r": None})``
    raised TypeError out of ``list(None)`` before the refusal logic could run, and a
    module whose whole contract is 'None is a refusal, applying is safe' must extend
    that manner to its inputs."""
    try:
        items = list(samples or ())
    except TypeError:
        return
    for item in items:
        try:
            v, y = item
            yield float(v), float(y)
        except (TypeError, ValueError):
            continue


def _fit_channel(samples: Sequence[Tuple[float, float]]
                 ) -> Optional[Tuple[List[float], bool, int]]:
    """One channel's samples → (256-entry float LUT, monotonic_observed, levels_populated),
    or None when the channel is too sparse to trust (< MIN_LEVELS distinct input levels).
    """
    bins: Dict[int, List[float]] = {}
    for v, y in _iter_pairs(samples):
        try:
            iv = int(round(v))
        except (ValueError, OverflowError):        # nan/inf inputs round to nothing
            continue
        if 0 <= iv <= 255 and math.isfinite(y):
            bins.setdefault(iv, []).append(y)
    levels = sorted(bins)
    if len(levels) < MIN_LEVELS:
        return None
    # robust central estimate per populated level — median, so one noisy V-Ray probe
    # pixel population cannot drag a knot
    knots_v: List[int] = []
    knots_y: List[float] = []
    weights: List[float] = []
    for lv in levels:
        ys = sorted(bins[lv])
        mid = len(ys) // 2
        med = ys[mid] if len(ys) % 2 else (ys[mid - 1] + ys[mid]) / 2.0
        knots_v.append(lv)
        knots_y.append(med)
        weights.append(float(len(ys)))
    # observed monotonicity BEFORE enforcement — half a level of slack, because binned
    # medians of a genuinely monotone relationship jitter by quantisation
    monotone = all(b >= a - 0.5 for a, b in zip(knots_y, knots_y[1:]))
    iso = _pav(knots_y, weights)
    # interpolate the 256-entry LUT through the isotonic knots; flat extrapolation
    # outside the observed range — inventing slope where nothing was measured is how a
    # correction becomes an extrapolation error in exactly the shadows/highlights the
    # metrics care most about
    lut: List[float] = [0.0] * 256
    ki = 0
    for x in range(256):
        if x <= knots_v[0]:
            lut[x] = iso[0]
        elif x >= knots_v[-1]:
            lut[x] = iso[-1]
        else:
            while ki < len(knots_v) - 2 and knots_v[ki + 1] < x:
                ki += 1
            v0, v1 = knots_v[ki], knots_v[ki + 1]
            y0, y1 = iso[ki], iso[ki + 1]
            frac = (x - v0) / float(v1 - v0) if v1 > v0 else 0.0
            lut[x] = y0 + (y1 - y0) * frac
    return lut, monotone, len(levels)


class VTone:
    """A fitted per-channel Vantage→V-Ray display transfer, applied as three 256-entry
    LUTs. Immutable by convention: fit once, apply many."""

    def __init__(self, luts: Dict[str, List[float]],
                 provenance: Optional[Dict] = None,
                 residual: Optional[float] = None,
                 dispersion: Optional[float] = None,
                 monotonic_observed: bool = True,
                 levels: Optional[Dict[str, int]] = None):
        self.luts = {ch: list(luts[ch]) for ch in CHANNELS}
        self.provenance = dict(provenance or {})
        #: Held-out MAE in 0..255 units when the harness computed one; the ARMING gate.
        self.residual = residual
        #: Per-state curve spread when the harness computed one — the AE detector.
        self.dispersion = dispersion
        self.monotonic_observed = bool(monotonic_observed)
        self.levels = dict(levels or {})

    # -- fitting ------------------------------------------------------------------
    @classmethod
    def fit(cls, channel_samples: Optional[Dict[str, Sequence[Tuple[float, float]]]],
            provenance: Optional[Dict] = None) -> Optional["VTone"]:
        """Fit from per-channel (vantage_level, vray_level) samples — the dict shape
        ``quantile_pairs``/``merge_samples`` produce. → VTone, or None when any channel
        is too sparse (< MIN_LEVELS distinct input levels) or the input is empty. None is
        a REFUSAL, not an error: an untrustworthy fit must not exist at all, because a
        VTone object's whole contract is that applying it is safe."""
        if not isinstance(channel_samples, dict) or not channel_samples:
            return None
        luts: Dict[str, List[float]] = {}
        monotone_all = True
        levels: Dict[str, int] = {}
        for ch in CHANNELS:
            # no list() here — a None/non-iterable channel is _iter_pairs' job to refuse
            fitted = _fit_channel(channel_samples.get(ch))
            if fitted is None:
                return None
            lut, monotone, n_levels = fitted
            luts[ch] = lut
            monotone_all = monotone_all and monotone
            levels[ch] = n_levels
        return cls(luts, provenance=provenance, monotonic_observed=monotone_all,
                   levels=levels)

    # -- applying -----------------------------------------------------------------
    @property
    def monotonic(self) -> bool:
        """Always True for a constructed fit — PAV enforces it. The interesting flag is
        ``monotonic_observed``: False there means the RAW pairs were not a non-decreasing
        relationship, which is evidence about the transfer, not about the fit."""
        for ch in CHANNELS:
            lut = self.luts[ch]
            if any(b < a - 1e-9 for a, b in zip(lut, lut[1:])):
                return False
        return True

    def to_vray_space(self, rows: Sequence[Sequence[Tuple[int, int, int]]]
                      ) -> List[List[Tuple[int, int, int]]]:
        """Apply the transfer to a plate's pixel rows → new rows, 8-bit clamped. Three
        list-indexed LUTs per pixel — the cost that keeps a 129600-pixel probe correction
        in single-digit milliseconds, which matters because the entire point of the grab
        path is that it costs less than the render it replaces."""
        lr = [max(0, min(255, int(round(v)))) for v in self.luts["r"]]
        lg = [max(0, min(255, int(round(v)))) for v in self.luts["g"]]
        lb = [max(0, min(255, int(round(v)))) for v in self.luts["b"]]
        out: List[List[Tuple[int, int, int]]] = []
        for row in rows:
            out.append([(lr[px[0] & 0xFF], lg[px[1] & 0xFF], lb[px[2] & 0xFF])
                        for px in row])
        return out

    # -- honesty instruments ------------------------------------------------------
    def residual_on(self, channel_samples: Optional[
            Dict[str, Sequence[Tuple[float, float]]]]) -> Optional[float]:
        """Mean absolute |g(v) − y| in 0..255 units over the given samples — THE number
        that gates arming, and it only means something on samples the fit never saw
        (held-out states; see the module docstring for why in-sample is inverted
        evidence). → None on empty input."""
        if not isinstance(channel_samples, dict):
            return None
        total = 0.0
        count = 0
        for ch in CHANNELS:
            lut = self.luts[ch]
            for fv, fy in _iter_pairs(channel_samples.get(ch)):
                if not (math.isfinite(fv) and math.isfinite(fy)):
                    continue
                iv = max(0, min(255, int(round(fv))))
                total += abs(lut[iv] - fy)
                count += 1
        if not count:
            return None
        return total / count

    def delivery_gap(self) -> Dict[str, float]:
        """How far the V-Ray-space match will sit from the Vantage deliverable: the
        distribution of |g(x) − x| over the input range, per the luminance-weighted mean
        of the three channels. The artist-facing honesty line — 'your Vantage frame will
        differ from the matched VFB image by up to N levels, concentrated where the curve
        bends' — computed from the very fit the shim requires, so it is free."""
        diffs = sorted(
            abs((0.2126 * self.luts["r"][x] + 0.7152 * self.luts["g"][x]
                 + 0.0722 * self.luts["b"][x]) - x)
            for x in range(256))
        n = len(diffs)
        return {
            "p50": diffs[n // 2],
            "p90": diffs[int(n * 0.9)],
            "p99": diffs[int(n * 0.99)],
            "max": diffs[-1],
        }

    # -- persistence --------------------------------------------------------------
    def to_dict(self) -> Dict:
        """JSON-safe round trip, LUTs quantised to 3 decimals (0.001 of a display level —
        far below every noise floor in play — and it keeps the sidecar file small enough
        to read in a glance)."""
        return {
            "format": 1,
            "luts": {ch: [round(v, 3) for v in self.luts[ch]] for ch in CHANNELS},
            "provenance": dict(self.provenance),
            "residual": self.residual,
            "dispersion": self.dispersion,
            "monotonic_observed": self.monotonic_observed,
            "levels": dict(self.levels),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> Optional["VTone"]:
        """→ VTone, or None on anything malformed. None, not an exception: the loader's
        callers are arming paths whose correct response to a corrupt sidecar is 'do not
        arm, say why', never a traceback."""
        if not isinstance(d, dict):
            return None
        luts = d.get("luts")
        if not isinstance(luts, dict):
            return None
        clean: Dict[str, List[float]] = {}
        for ch in CHANNELS:
            arr = luts.get(ch)
            if not isinstance(arr, (list, tuple)) or len(arr) != 256:
                return None
            try:
                vals = [float(v) for v in arr]
            except (TypeError, ValueError):
                return None
            if not all(math.isfinite(v) for v in vals):
                return None
            clean[ch] = vals
        residual = d.get("residual")
        dispersion = d.get("dispersion")
        try:
            residual = None if residual is None else float(residual)
            dispersion = None if dispersion is None else float(dispersion)
        except (TypeError, ValueError):
            return None
        prov = d.get("provenance")
        levels = d.get("levels")
        return cls(clean,
                   provenance=prov if isinstance(prov, dict) else {},
                   residual=residual,
                   dispersion=dispersion,
                   monotonic_observed=bool(d.get("monotonic_observed", True)),
                   levels=levels if isinstance(levels, dict) else {})


def curve_dispersion(tones: Sequence[Optional["VTone"]]) -> Optional[float]:
    """Spread of a FAMILY of per-state fits — the auto-exposure detector.

    A static transfer produces the same curve from every calibration state; an adaptive
    one produces a curve per state. This is the mean over sampled input levels of the
    (max − min) across states of the luminance-weighted LUT value, in 0..255 units.
    Levels are sampled every 8 to keep it cheap; Nones (states whose fit refused) are
    skipped. → None with fewer than two usable curves, because dispersion of one curve
    is not a measurement."""
    usable = [t for t in tones if t is not None]
    if len(usable) < 2:
        return None
    total = 0.0
    count = 0
    for x in range(0, 256, 8):
        vals = [0.2126 * t.luts["r"][x] + 0.7152 * t.luts["g"][x]
                + 0.0722 * t.luts["b"][x] for t in usable]
        total += max(vals) - min(vals)
        count += 1
    return total / count if count else None
