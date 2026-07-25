"""ANALYZE self-consistency — the root-cause fix for single-sample semantic variance.

Live evidence (sim_match Phase B): the same reference read as "golden_hour @ 3800K" on one
fire and "midday @ 6500K" on another — and a wrong ANALYZE poisons the whole run (rules,
sweep and deltas all chase the wrong semantics). The cure is the oldest one: ask three
times, keep what the samples agree on. Enums/bools take the majority (ties broken by the
most-confident sample), numbers take the median, the sun bearing takes the CIRCULAR mean
(±180 wraps — an arithmetic median of [-170, 170, 0] would be nonsense).

Pure python; the controller fires the N samples, this consolidates.
"""

from __future__ import annotations

import math
from typing import Dict, List

NUMERIC_KEYS = ("wb_kelvin_estimate", "confidence")
CIRCULAR_KEYS = ("sun_bearing_deg",)
BOOL_KEYS = ("sun_active", "practicals_on")


def _median(values: List[float]) -> float:
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def circular_spread_deg(values: List[float]) -> float:
    """How far the samples scatter, in degrees: 0 = unanimous, 90 = no agreement at all.

    The circular standard deviation, from the resultant length R. Unlike a min/max range
    this is not hostage to one outlier deciding the whole number, and it is the quantity
    that says how much the reading should be TRUSTED downstream."""
    n = len(values)
    if n < 2:
        return 0.0
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    r = math.hypot(sin_sum, cos_sum) / n          # 1 = identical, 0 = evenly scattered
    r = max(1e-9, min(1.0, r))
    return round(math.degrees(math.sqrt(-2.0 * math.log(r))), 1)


def _circular_median_deg(values: List[float]) -> float:
    """The sample minimising total angular distance to the others — a true median, so a
    single wild read cannot drag it.

    A circular MEAN was used here, and the mean has no outlier rejection: measured
    2026-07-25, samples [60, 70, -50] average to 35 degrees when two of the three agree
    within ten. That 30-degree injected error is the size of the sun-placement misses seen
    on-box, and it matters far more now that the bearing carries a quarter of the match
    objective. With N=3 and one bad read — the exact observed failure — the median keeps
    the two that agree. Even N falls back to the mean of the two central candidates, which
    is still computed circularly so the -170/+170 wrap the old docstring worried about
    stays correct."""
    n = len(values)
    if n == 1:
        return values[0]

    def total_distance(c: float) -> float:
        return sum(min(d, 360.0 - d) for d in
                   (abs((c - v) % 360.0) for v in values))

    ranked = sorted(values, key=total_distance)
    if n % 2:
        best = ranked[0]
    else:                       # even: average the two most central, circularly
        best = _circular_mean_deg(ranked[:2])
    return (best + 180.0) % 360.0 - 180.0


def _circular_mean_deg(values: List[float]) -> float:
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return values[0]
    deg = math.degrees(math.atan2(sin_sum, cos_sum))
    return (deg + 180.0) % 360.0 - 180.0     # keep the -180..180 bearing convention


def consolidate_analyses(samples: List[Dict]) -> Dict:
    """N validated ANALYZE dicts → one consensus dict (same shape). N=1 passes through."""
    if not samples:
        raise ValueError("no analysis samples to consolidate")
    if len(samples) == 1:
        return dict(samples[0])
    best = max(samples, key=lambda s: float(s.get("confidence", 0.0)))
    out: Dict = {}
    for key in samples[0]:
        values = [s.get(key) for s in samples if key in s]
        if key in CIRCULAR_KEYS:
            out[key] = round(_circular_median_deg([float(v) for v in values]), 1)
        elif key in NUMERIC_KEYS:
            out[key] = round(_median([float(v) for v in values]), 2)
        elif key in BOOL_KEYS:
            true_votes = sum(bool(v) for v in values)
            if true_votes * 2 == len(values):
                out[key] = bool(best.get(key))   # tie → the most-confident sample,
            else:                                # exactly like the enum path below
                out[key] = true_votes * 2 > len(values)
        elif key == "key_notes":
            out[key] = best.get(key, "")
        else:                                   # enums: majority, confidence-broken ties
            counts: Dict[str, int] = {}
            for v in values:
                counts[str(v)] = counts.get(str(v), 0) + 1
            top = max(counts.values())
            winners = [v for v, c in counts.items() if c == top]
            out[key] = (best.get(key) if str(best.get(key)) in winners else winners[0])
    # How much the samples agreed. This was measured on time_of_day ALONE, which let the
    # sun bearing scatter 130 degrees across reads of one image while all three samples
    # said "morning" — agreement 1.0, no warning raised, and nothing downstream knew the
    # direction was a coin flip (measured on-box 2026-07-25 across four runs of the same
    # golden-hour reference: 45.0, -52.5, 77.6, 64.9). The bearing now drives a quarter of
    # the match objective, so its scatter is reported too, and the headline agreement is
    # the WORSE of the two — an unreliable direction is not hidden by a confident clock.
    times = [s.get("time_of_day") for s in samples]
    enum_agreement = times.count(out.get("time_of_day")) / len(times)
    bearings = [float(s["sun_bearing_deg"]) for s in samples if "sun_bearing_deg" in s]
    spread = circular_spread_deg(bearings) if len(bearings) > 1 else 0.0
    out["sun_bearing_spread_deg"] = spread
    # 0 deg of scatter is full confidence, 60+ is none; linear between
    out["sun_bearing_agreement"] = round(max(0.0, 1.0 - spread / 60.0), 2)
    out["consensus_agreement"] = round(min(enum_agreement,
                                           out["sun_bearing_agreement"]), 2)
    return out
