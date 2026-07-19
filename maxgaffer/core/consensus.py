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
            out[key] = round(_circular_mean_deg([float(v) for v in values]), 1)
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
    # disagreement level: how far the samples scattered on the load-bearing enum
    times = [s.get("time_of_day") for s in samples]
    out["consensus_agreement"] = round(times.count(out.get("time_of_day")) / len(times), 2)
    return out
