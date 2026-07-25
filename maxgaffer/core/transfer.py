"""Lighting-TRANSFER score — did the rig end up where the reference's light actually is?

The critic scores PIXELS, which is the right question when the reference is reachable in
the scene and the wrong one when it is a photograph of a different building. Cross-domain
matches were being marked against an impossible target: MXLT scored a real dawn plate at
75 not because the lighting was wrong but because a CG room can never produce that
photograph's pixels. Measured 2026-07-25.

The answerable question is the one an artist actually asks: *is my sun where that photo's
sun is, at that colour temperature, with that hardness and haze?* This module scores the
recovered rig against the reference's own ANALYZE reading, in the reference's units:

  direction   sun bearing relative to the camera, the thing a DP notices first
  elevation   altitude against the analysed band, with band-width tolerance
  warmth      white balance against the analysed estimate, in perceptual mireds
  hardness    sun size against hard/soft/mixed
  atmosphere  haze density against none/light/heavy/fog
  presence    sun on/off agreement

Every sub-score is 0..1 and reports its own reasoning, so a low transfer score names what
is wrong rather than just being low. Pure stdlib; no renders; no pixels.
"""

from __future__ import annotations

from typing import Dict, Optional

from .genome import LightingState

#: Analysed band → the elevation it means, and how wide the band is (degrees). A band is a
#: RANGE, so "golden" recovered at 12° when the band centre is 10° is a hit, not a miss.
BAND_CENTRE_DEG: Dict[str, float] = {
    "just_set": -2.0, "golden": 8.0, "low": 18.0, "mid": 40.0,
    "high": 62.0, "overhead": 85.0,
}
BAND_HALF_WIDTH_DEG: Dict[str, float] = {
    "just_set": 5.0, "golden": 7.0, "low": 10.0, "mid": 14.0,
    "high": 14.0, "overhead": 8.0,
}

#: Analysed hardness → the sun size that expresses it (V-Ray size multiplier).
HARDNESS_SIZE = {"hard": 1.0, "mixed": 6.0, "soft": 14.0}

#: Analysed haze → visibility distance in metres (matches rules.ATMOSPHERE_DISTANCE_M).
ATMOSPHERE_DISTANCE_M = {"none": 4000.0, "light_haze": 250.0,
                         "heavy_haze": 70.0, "fog": 20.0}

#: Weights — direction dominates because it is what reads first and what a wrong match
#: gets wrong most visibly.
WEIGHTS = {"direction": 0.32, "elevation": 0.22, "warmth": 0.20,
           "hardness": 0.10, "atmosphere": 0.10, "presence": 0.06}


def _num(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if v == v and abs(v) != float("inf") else default


def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute separation between two compass angles, 0..180."""
    d = abs((float(a) - float(b)) % 360.0)
    return min(d, 360.0 - d)


def _mireds(kelvin: float) -> float:
    """Colour temperature in mireds — the perceptually even scale. 4000→5000 K is a big
    visible shift; 9000→10000 K is barely visible, and Kelvin arithmetic hides that."""
    k = max(1000.0, _num(kelvin, 6500.0))
    return 1.0e6 / k


def score(semantics: Optional[Dict], state: Optional[LightingState],
          camera_yaw_deg: float = 0.0) -> Dict:
    """→ {score 0..100, parts {name: 0..1}, notes [str], measurable [str]}.

    ``semantics`` is the reference's ANALYZE reading; ``state`` is the rig the match
    landed on. Sub-scores whose inputs are absent are SKIPPED and the weights renormalise
    over what was measurable — never silently scored as agreement."""
    parts: Dict[str, float] = {}
    notes = []
    if not isinstance(semantics, dict) or state is None:
        return {"score": 0.0, "parts": {}, "notes": ["no reference reading to compare"],
                "measurable": []}

    sun_active = bool(semantics.get("sun_active", True))
    has_sun = "sun.azimuth_deg" in state.values
    sun_on = state.get("sun.enabled", 1.0) >= 0.5 if "sun.enabled" in state.values else True

    # ---- presence: does the rig agree there IS (or is not) a sun
    if has_sun:
        parts["presence"] = 1.0 if sun_on == sun_active else 0.0
        if sun_on != sun_active:
            notes.append("reference sun is %s but the rig's sun is %s"
                         % ("ON" if sun_active else "OFF", "ON" if sun_on else "OFF"))

    # ---- direction: bearing relative to the camera — what a DP reads first
    if has_sun and sun_active and sun_on and "sun_bearing_deg" in semantics:
        want_az = (_num(camera_yaw_deg) + _num(semantics.get("sun_bearing_deg"))) % 360.0
        got_az = _num(state.get("sun.azimuth_deg")) % 360.0
        err = _angle_delta(want_az, got_az)
        # 15° is within a lighting artist's own margin; 90° is a different shot
        parts["direction"] = max(0.0, 1.0 - max(0.0, err - 15.0) / 75.0)
        notes.append("sun bearing off by %.0f°" % err)

    # ---- elevation: against the analysed BAND, not a point value
    band = str(semantics.get("sun_altitude_band", "na") or "na")
    if has_sun and sun_active and sun_on and band in BAND_CENTRE_DEG:
        got = _num(state.get("sun.altitude_deg"))
        slack = BAND_HALF_WIDTH_DEG.get(band, 10.0)
        err = max(0.0, abs(got - BAND_CENTRE_DEG[band]) - slack)
        parts["elevation"] = max(0.0, 1.0 - err / 35.0)
        if err:
            notes.append("sun elevation %.0f° is %.0f° outside the '%s' band"
                         % (got, err, band))

    # ---- warmth: in mireds, where equal steps look equal
    if "exposure.wb_kelvin" in state.values and semantics.get("wb_kelvin_estimate"):
        d_mired = abs(_mireds(state.get("exposure.wb_kelvin"))
                      - _mireds(semantics.get("wb_kelvin_estimate")))
        # ~15 mireds is a just-noticeable shift; 120 is a different mood entirely
        parts["warmth"] = max(0.0, 1.0 - max(0.0, d_mired - 15.0) / 105.0)
        if d_mired > 15.0:
            notes.append("white balance off by %.0f mireds" % d_mired)

    # ---- hardness: sun size against hard / soft / mixed
    quality = str(semantics.get("light_quality", "") or "")
    if has_sun and sun_on and quality in HARDNESS_SIZE and "sun.size" in state.values:
        want, got = HARDNESS_SIZE[quality], max(0.5, _num(state.get("sun.size"), 1.0))
        import math

        octaves = abs(math.log2(got / want))
        parts["hardness"] = max(0.0, 1.0 - octaves / 3.0)

    # ---- atmosphere: haze density
    atmo = str(semantics.get("atmosphere", "") or "")
    if atmo in ATMOSPHERE_DISTANCE_M and "atmosphere.distance_m" in state.values:
        import math

        want = ATMOSPHERE_DISTANCE_M[atmo]
        got = max(1.0, _num(state.get("atmosphere.distance_m"), want))
        octaves = abs(math.log2(got / want))
        parts["atmosphere"] = max(0.0, 1.0 - octaves / 4.0)

    if not parts:
        return {"score": 0.0, "parts": {},
                "notes": ["nothing comparable between the reading and the rig"],
                "measurable": []}
    total_w = sum(WEIGHTS[k] for k in parts) or 1.0
    total = sum(WEIGHTS[k] * v for k, v in parts.items()) / total_w
    return {"score": round(100.0 * total, 2),
            "parts": {k: round(v, 3) for k, v in parts.items()},
            "notes": notes, "measurable": sorted(parts)}
