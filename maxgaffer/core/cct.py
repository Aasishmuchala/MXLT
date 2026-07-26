"""What colour temperature is the light in this picture — measured, not guessed?

ANALYZE asks a vision model to eyeball a reference photo and report ``wb_kelvin_estimate``.
That guess is unreliable in the way that costs matches: measured 700 K from truth on one
run, 225 mireds out on a cross-domain match, 1754 K out on another. Meanwhile
``metrics.compute_stats`` has ALREADY measured the illuminant from the pixels themselves —
``illum_sog`` (Shades-of-Gray, Minkowski p=6), ``illum_edge`` (gray-edge) and their blend
``illum``, each a unit-normalised LINEAR RGB 3-vector. Turning an illuminant chromaticity
into a correlated colour temperature is textbook colour science, not a judgement call. This
module does that conversion so a measurement can replace the guess.

================================================================================
DIRECTION — read this before you use the number for anything
================================================================================
The kelvin returned here is the ILLUMINANT'S OWN colour temperature: the temperature of the
light falling on the scene, exactly as a gaffer means it saying "that's a 3200 K lamp". It
is NOT a camera setting, NOT a correction, and NOT an offset.

    returned 2856 K   the light is WARM  — orange, tungsten
    returned 6504 K   the light is NEUTRAL — noon daylight
    returned 9000 K   the light is COOL  — blue, open shade

To NEUTRALISE that light, a camera white balance takes THE SAME NUMBER, not its opposite:
a warm illuminant needs a LOW camera-WB kelvin. That is the standard photographic
convention and it is the one this project already uses — see ``colortemp.py``: a WB setting
of 3200 K makes the camera divide out orange, so the render comes back COOLER; raising the
spinner renders WARMER.

The bug this file is written to prevent is the intuition "the light is warm, so I must dial
the white balance UP to compensate". That is backwards, it is silent, and on a golden-hour
reference it lands the rig roughly 200 mireds from the target — the same magnitude as the
failures listed above. If you are about to write arithmetic that turns this kelvin into an
``exposure.wb_kelvin``, do not: call :func:`camera_wb_kelvin_for_illuminant`, which is the
identity and says so.

================================================================================
Honesty
================================================================================
Every entry point returns ``None`` rather than a number it cannot stand behind, and nothing
here raises. A chromaticity far from the Planckian locus has no correlated colour
temperature at all, so a deep-green or magenta illuminant is REFUSED — see
:func:`xy_to_cct`, where the measured consequence of not refusing is recorded.

Pure stdlib, sans-IO, no imports from the rest of the package: this is arithmetic on a dict
somebody else already produced.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- constants

#: Linear sRGB (D65 primaries) → CIE XYZ, IEC 61966-2-1. metrics.py carries the same matrix
#: rounded to 4 decimals, which is right for a LAB mood metric and wrong here: McCamy's
#: denominator is a small difference of two chromaticity numbers, so it amplifies input
#: rounding. Full precision costs nothing and keeps D65 landing on 6503.5 K instead of
#: drifting tens of kelvin.
SRGB_D65_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

#: Kim et al. (2002) cubic-spline approximation to the Planckian locus is defined over this
#: range; the Duv search below only ever asks inside it.
LOCUS_MIN_K = 1667.0
LOCUS_MAX_K = 25000.0

#: CIE 15 says the correlated-colour-temperature concept stops meaning anything beyond
#: 5×10⁻² from the locus in the 1960 UCS. That is a definition, not a tuning knob: past it
#: there is no "nearest" blackbody a human would accept as the same white.
CIE_MAX_DUV = 0.05

#: |Duv| at which the reading is worth half its confidence. NOT the ANSI C78.377 white band
#: (about ±0.006) — the daylight locus itself runs ≈ +0.0032 ABOVE the Planckian locus
#: (measured below: D50 +0.00317, D55 +0.00321, D65 +0.00318, D75 +0.00312), so halving
#: confidence at 0.006 would dock every correct daylight reading for being daylight. 0.010
#: is where the tint stops being physics and starts being a cast.
DUV_HALF_CONFIDENCE = 0.010

#: Disagreement between the two estimators worth half the confidence, in MIREDS —
#: transfer.py's scale, where equal steps look equal ("~15 mireds is a just-noticeable
#: shift; 120 is a different mood entirely"). Half confidence at twice the JND.
MIRED_HALF_CONFIDENCE = 30.0

#: Confidence ceiling when only ONE estimator was usable, so the answer could not be
#: corroborated. consensus.py exists because a single unreplicated read of this scene's
#: colour is exactly what has been wrong before; an uncheckable measurement does not get to
#: claim certainty just because nothing contradicted it.
UNCORROBORATED_CONFIDENCE = 0.6

#: Sanity band on the returned kelvin. This is a guard, not an accuracy claim: McCamy's
#: denominator (0.1858 − y) collapses near y ≈ 0.1858 and the polynomial then explodes —
#: measured 2026-07-26, a magenta illuminant at xy (0.3191, 0.1935) comes out of McCamy at
#: 28848 K, stated with perfect confidence. The Duv gate already refuses that point; this
#: band is the second lock on the same door.
PLAUSIBLE_MIN_K = 1000.0
PLAUSIBLE_MAX_K = 25000.0

#: Preference order for :func:`illuminant_kelvin`, as (metrics key, reported source).
#: ``illum`` leads because it is the mean of two estimators with different failure modes —
#: gray-edge reads chromatic gradients, Shades-of-Gray reads the p=6 channel norm — and an
#: average of two independent biases beats either.
_ORDER_CORROBORATED: Tuple[Tuple[str, str], ...] = (
    ("illum", "illum"),
    ("illum_edge", "illum_edge"),
    ("illum_sog", "illum_sog"),
)

#: The order used when the two constituents are NOT both usable. metrics builds ``illum``
#: from whatever it has — ``_unit3([(sog + edge) / 2 …])`` on a degenerate ``[0,0,0]``
#: partner returns the surviving estimator unchanged — so a non-degenerate ``illum`` is no
#: evidence that a blend happened. It therefore loses its place at the front and returns
#: under an honest name instead of claiming corroboration it never had.
_ORDER_PARTIAL: Tuple[Tuple[str, str], ...] = (
    ("illum_edge", "illum_edge"),
    ("illum_sog", "illum_sog"),
    ("illum", "illum_only"),
)


# --------------------------------------------------------------------------- guards
def _finite(value) -> Optional[float]:
    """A float, or None for anything that is not one — including NaN and ±inf, which are
    what a degenerate frame propagates into these vectors."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _illum_vec(value) -> Optional[List[float]]:
    """Validated illuminant 3-vector, or None for 'estimate unavailable'.

    Deliberately mirrors ``metrics._illum_vec`` instead of importing it: that name is
    private, metrics.py does file IO, and this module promises to be stdlib-only and
    sans-IO. Twelve duplicated lines are cheaper than that coupling.

    Rejects negatives as well as the degenerate cases. Both estimators are built from
    magnitudes — a p=6 channel sum and a sum of absolute gradients — so a negative component
    cannot come from a real frame; it means the dict was corrupted or hand-built, and a
    negative channel would sail straight through the chromaticity maths as a plausible
    colour."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    vec = [_finite(c) for c in value]
    if any(c is None or c < 0.0 for c in vec):
        return None
    if math.sqrt(sum(c * c for c in vec)) < 1e-9:    # [0,0,0] — metrics' 'unavailable'
        return None
    return vec  # type: ignore[return-value]


# --------------------------------------------------------------------------- chromaticity
def rgb_to_xy(linear_rgb: Sequence[float]) -> Optional[Tuple[float, float]]:
    """LINEAR sRGB (D65 primaries) → CIE 1931 xy chromaticity, or None.

    ``linear_rgb`` is light-linear, NOT display-encoded: it is what ``metrics.illum_*``
    already hold, having been decoded through ``_srgb_to_linear`` before accumulation.
    Passing 8-bit or gamma-encoded values here reads as a different colour and yields a
    wrong temperature without any error.

    Scale is irrelevant — chromaticity divides it out — so a unit-normalised illuminant
    vector and the same vector times 100 give identical xy.

    None on: wrong shape, non-finite or negative components, an all-zero vector (metrics'
    'unavailable' sentinel), or a non-positive XYZ sum."""
    vec = _illum_vec(linear_rgb)
    if vec is None:
        return None
    r, g, b = vec
    xyz = [row[0] * r + row[1] * g + row[2] * b for row in SRGB_D65_TO_XYZ]
    total = xyz[0] + xyz[1] + xyz[2]
    if not math.isfinite(total) or total <= 1e-12:
        return None
    x, y = xyz[0] / total, xyz[1] / total
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _planckian_xy(kelvin: float) -> Optional[Tuple[float, float]]:
    """Chromaticity of a blackbody at ``kelvin`` — Kim et al. (2002) cubic spline, valid
    1667..25000 K. Used only as the reference curve the Duv search measures against."""
    t = _finite(kelvin)
    if t is None or t < LOCUS_MIN_K or t > LOCUS_MAX_K:
        return None
    t2, t3 = t * t, t * t * t
    if t <= 4000.0:
        x = -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x ** 3 - 1.34811020 * x ** 2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x ** 3 - 1.37418593 * x ** 2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x ** 3 - 5.87338670 * x ** 2 + 3.75112997 * x - 0.37001483
    return x, y


def _xy_to_uv60(x: float, y: float) -> Optional[Tuple[float, float]]:
    """CIE 1931 xy → CIE 1960 UCS uv. Duv is DEFINED in the 1960 space, not 1976 u'v'
    (whose v' is 1.5× this v) — using the wrong one inflates every distance by half."""
    denom = -2.0 * x + 12.0 * y + 3.0
    if abs(denom) < 1e-9:
        return None
    return 4.0 * x / denom, 6.0 * y / denom


# --------------------------------------------------------------------------- duv
def duv(x: float, y: float) -> Optional[float]:
    """Signed distance from the Planckian locus in the CIE 1960 UCS, or None.

    Positive = ABOVE the locus (green cast), negative = BELOW (magenta/pink). This is the
    standard measure of how far off-locus a chromaticity is, and it is what licenses
    :func:`xy_to_cct` to refuse: a CCT names the nearest blackbody, and past a certain
    distance "nearest" stops being a useful description of a colour.

    Computed the honest way — find the closest point on the locus and measure — rather than
    via a fitted polynomial, so it can be checked against physics. It is: measured
    2026-07-26, chromaticities taken from the locus itself return 0.000000, and CIE
    Illuminant A (which IS a Planckian radiator at 2856 K) returns -0.00008. The CIE
    D-series, which by construction sits slightly above the locus, returns +0.0032 across
    D50/D55/D65/D75 — the textbook offset.

    The search runs in MIREDS because the locus is near-linear there, so uniform steps
    sample the warm and cool ends evenly; uniform kelvin steps would crowd the blue end and
    starve the warm one. Coarse scan, then ternary refine on the bracketing interval — the
    distance is unimodal near the locus, so this cannot land in a false minimum.

    The sign is taken by comparing v against the closest locus point, which is the standard
    practical rule and is exact enough in the near-locus band where Duv is meaningful.

    This function MEASURES and does not judge: any finite (x, y) gets a distance, however
    absurd — the origin returns -0.33. Deciding that a distance is too large to name a
    temperature is :func:`xy_to_cct`'s job, so there is exactly one gate rather than two
    that can drift apart."""
    fx, fy = _finite(x), _finite(y)
    if fx is None or fy is None:
        return None
    sample = _xy_to_uv60(fx, fy)
    if sample is None:
        return None
    su, sv = sample

    def sq_distance(mired: float) -> float:
        pt = _planckian_xy(1.0e6 / mired)
        if pt is None:
            return float("inf")
        uv = _xy_to_uv60(*pt)
        if uv is None:
            return float("inf")
        return (su - uv[0]) ** 2 + (sv - uv[1]) ** 2

    lo, hi = 1.0e6 / LOCUS_MAX_K, 1.0e6 / LOCUS_MIN_K       # 40 .. 600 mireds
    steps = 200
    best_m, best_d = lo, float("inf")
    for i in range(steps + 1):
        m = lo + (hi - lo) * i / steps
        d = sq_distance(m)
        if d < best_d:
            best_d, best_m = d, m
    if not math.isfinite(best_d):
        return None
    span = (hi - lo) / steps
    a, b = max(lo, best_m - span), min(hi, best_m + span)
    for _ in range(60):
        m1, m2 = a + (b - a) / 3.0, b - (b - a) / 3.0
        if sq_distance(m1) < sq_distance(m2):
            b = m2
        else:
            a = m1
    closest = _planckian_xy(1.0e6 / ((a + b) / 2.0))
    if closest is None:
        return None
    uv = _xy_to_uv60(*closest)
    if uv is None:
        return None
    distance = math.hypot(su - uv[0], sv - uv[1])
    return distance if sv >= uv[1] else -distance


# --------------------------------------------------------------------------- cct
def _cct_and_duv(x: float, y: float) -> Optional[Tuple[float, float]]:
    """(kelvin, duv) or None — the shared body, so callers wanting both do not pay for the
    locus search twice."""
    off_locus = duv(x, y)
    if off_locus is None or abs(off_locus) > CIE_MAX_DUV:
        return None                     # no meaningful nearest blackbody — refuse
    denom = 0.1858 - y
    if abs(denom) < 1e-6:               # McCamy's pole; unreachable past the Duv gate
        return None
    n = (x - 0.3320) / denom
    kelvin = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    if not math.isfinite(kelvin):
        return None
    if not (PLAUSIBLE_MIN_K <= kelvin <= PLAUSIBLE_MAX_K):
        return None
    return kelvin, off_locus


def xy_to_cct(x: float, y: float) -> Optional[float]:
    """CIE 1931 xy → correlated colour temperature in kelvin, or None.

    THE NUMBER IS THE LIGHT'S OWN TEMPERATURE, not a camera setting: low = warm/orange,
    high = cool/blue. See the module docstring before converting it into anything.

    McCamy's (1992) cubic approximation::

        n   = (x - 0.3320) / (0.1858 - y)
        CCT = 449 n³ + 3525 n² + 6823.3 n + 5520.33

    VALIDITY. Measured 2026-07-26 against the Kim et al. Planckian locus, feeding locus
    chromaticities back through the polynomial:

        1700 K  +24 K     6500 K   +1 K      12000 K   -362 K   (-3.0 %)
        2856 K  +10 K     7500 K  -11 K      15000 K  -1014 K   (-6.8 %)
        4000 K   +6 K     9000 K  -60 K      20000 K  -2869 K  (-14.3 %)

    So: within ±15 K from roughly 2000 K to 7500 K, which covers every practical scene
    illuminant from candle to overcast sky, and progressively UNDER-reading above that.
    The blue-end figures look disqualifying in kelvin and are not, because nothing in this
    plugin compares white balance in kelvin: transfer.py scores warmth in MIREDS, and the
    -2869 K miss at 20000 K is 8.4 mireds — inside the 15-mired just-noticeable step. That
    measurement is why there is no blue-end confidence penalty in
    :func:`illuminant_kelvin`; the error is real, and on the scale that decides matches it
    is invisible. It would matter to anyone using the raw kelvin for something
    kelvin-sensitive, which is what this paragraph is for.

    REFUSAL. None when |Duv| exceeds the CIE limit of 0.05, because past that a
    chromaticity has no correlated colour temperature to report. This is not defensive
    padding — McCamy answers anyway, and confidently wrong: measured 2026-07-26, a magenta
    illuminant reads 28848 K and a pure-blue one reads 1667 K. A blue light reported as
    candlelight is precisely the inversion this module exists to make impossible, and only
    the Duv gate catches it, because the polynomial itself has no idea it has left the
    locus."""
    fx, fy = _finite(x), _finite(y)
    if fx is None or fy is None:
        return None
    got = _cct_and_duv(fx, fy)
    return None if got is None else got[0]


def camera_wb_kelvin_for_illuminant(illuminant_kelvin_value: float) -> Optional[float]:
    """Camera white-balance kelvin that NEUTRALISES an illuminant of the given temperature.

    It is the identity, and that is the whole reason this function exists. A camera's WB
    control names the light it is being asked to cancel, so cancelling a 2856 K tungsten
    illuminant means setting the WB to 2856 K — the low number, not a high one. Anyone
    reaching for a conversion here is about to invent an inversion; this returns the number
    unchanged so they cannot.

    Matches ``colortemp.py``: a WB of 3200 K makes the camera divide out orange and the
    render comes back cooler, which is exactly what neutralising a 3200 K lamp means.

    Kept as a named function rather than a comment because a comment cannot be called."""
    return _finite(illuminant_kelvin_value)


# --------------------------------------------------------------------------- top level
def _mireds(kelvin: float) -> float:
    """Reciprocal megakelvin — the perceptually even scale, matching transfer.py."""
    return 1.0e6 / max(1.0, kelvin)


def illuminant_kelvin(stats: Optional[Dict]) -> Optional[Dict]:
    """A ``metrics.compute_stats()`` dict → the measured temperature OF THE SCENE LIGHT.

    Returns ``{"kelvin", "duv", "source", "confidence"}``, or None when nothing usable is
    present — legacy stats with no ``illum_*`` keys, a degenerate frame whose estimators
    all came back ``[0,0,0]``, or a chromaticity too far off the locus to name.

    ``kelvin``      the ILLUMINANT'S own colour temperature. Low = warm light, high = cool
                    light. NOT a camera setting; to neutralise it a camera white balance
                    takes the same number — :func:`camera_wb_kelvin_for_illuminant`.
    ``duv``         signed distance off the Planckian locus; + green, - magenta.
    ``source``      which estimate answered: ``illum`` (blend of both, both independently
                    verified), ``illum_edge``, ``illum_sog``, or ``illum_only`` (the blended
                    vector was all that was available, so what fed it is unknown).
    ``confidence``  0..1, the product of two honest doubts:
                    how far off-locus the chromaticity sits, and how far the two
                    independent estimators disagree — in mireds, since that is the scale on
                    which a warmth error is actually visible. When only one estimator is
                    usable the disagreement term cannot be measured and is capped at
                    UNCORROBORATED_CONFIDENCE rather than assumed perfect.

    Candidates are tried in order until one PRODUCES A NUMBER, not until one produces a
    vector: a gray-edge estimate driven into deep green by a chromatic scene is refused by
    the Duv gate and Shades-of-Gray still gets its turn. ``source`` names whichever actually
    answered.

    Caveats the confidence cannot see, and the reason this stays advisory:
      * these estimators assume flat average albedo, so a strongly chromatic scene — the
        walnut library metrics.py warns about — biases the reading toward the wall;
      * a reference JPEG that a camera already auto-white-balanced has had its cast
        REMOVED in-camera, so what is measured is the residue, biased toward neutral. A
        phone shot at golden hour will read far closer to 6500 K than the light really was.
    Neither shows up as low confidence, because both produce a clean, on-locus, perfectly
    self-consistent wrong answer."""
    if not isinstance(stats, dict):
        return None

    try:
        # Per-estimator temperatures, computed once. The two constituents are needed
        # independently anyway — for the agreement term, and to decide whether 'illum' has
        # earned the right to call itself a blend.
        solved: Dict[str, Tuple[float, float]] = {}
        for key in ("illum", "illum_edge", "illum_sog"):
            xy = rgb_to_xy(stats.get(key))
            if xy is None:
                continue
            got = _cct_and_duv(*xy)
            if got is not None:
                solved[key] = got

        both_present = (_illum_vec(stats.get("illum_sog")) is not None
                        and _illum_vec(stats.get("illum_edge")) is not None)

        chosen = None
        for key, source in (_ORDER_CORROBORATED if both_present else _ORDER_PARTIAL):
            if key in solved:
                chosen = (source, solved[key])
                break
        if chosen is None:
            return None
        source, (kelvin, off_locus) = chosen

        # Doubt 1 — distance off the locus. A Lorentzian, so it decays smoothly and never
        # reaches exactly zero: the reading gets quieter as the tint grows, and the hard
        # stop is the refusal in _cct_and_duv, not a cliff in the confidence.
        conf_duv = 1.0 / (1.0 + (abs(off_locus) / DUV_HALF_CONFIDENCE) ** 2)

        # Doubt 2 — do the two independent estimators tell the same story? Compared in
        # mireds: 5000 vs 5600 K is a 21-mired argument worth listening to, while 12000 vs
        # 14000 K is 12 mireds and barely a disagreement at all. Kelvin would have called
        # the second one four times worse than the first.
        if "illum_sog" in solved and "illum_edge" in solved:
            spread = abs(_mireds(solved["illum_sog"][0]) - _mireds(solved["illum_edge"][0]))
            conf_agree = 1.0 / (1.0 + (spread / MIRED_HALF_CONFIDENCE) ** 2)
        else:
            conf_agree = UNCORROBORATED_CONFIDENCE

        return {
            "kelvin": round(kelvin, 1),
            "duv": round(off_locus, 5),
            "source": source,
            "confidence": round(max(0.0, min(1.0, conf_duv * conf_agree)), 3),
        }
    except Exception:       # SPEC: this degrades, it never raises — a WB cue is not worth
        return None         # taking a match run down for
