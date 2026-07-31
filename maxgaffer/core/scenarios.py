"""Scenario board — diverse candidate rigs, rendered and MEASURED, pick one, refine.

Chaos Light Gen's UX insight (generate lighting scenarios, browse, click) grafted onto
MaxGaffer's split of powers: every candidate is built by the SAME deterministic craft
tables as the first guess (rules.initial_state fed variant semantics), probe-rendered by
the bridge, and — when a reference is bound — scored by the tonal critic, so the board
shows numbers, not vibes. Adopting a candidate is just applying its state; MATCH/DEEP
continue from it like any other start.

Variants are written in ANALYZE-semantics vocabulary (time_of_day, sky, bearing…) rather
than raw genome values, deliberately: they flow through the identical semantics→state
mapping as a real reference analysis, so a craft fix there (e.g. the on-box WB direction
check) corrects the board and the first guess together.

Pure python, zero pymxs. The bridge renders; this builds.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

from . import rules
from .genome import LightingState

# a neutral base for boards run with NO reference bound (Light Gen's original mode) —
# every variant overrides what makes it itself
DEFAULT_SEMANTICS: Dict = {
    "scene_type": "interior_with_view",
    "time_of_day": "afternoon",
    "sky": "clear",
    "sun_active": True,
    "sun_bearing_deg": -60.0,
    "sun_altitude_band": "mid",
    "light_quality": "mixed",
    "wb_kelvin_estimate": 5500.0,
    "practicals_on": False,
    "atmosphere": "none",
    "contrast_character": "balanced",
    # complete ANALYZE shape — this dict also serves as the gateway-down fallback
    # semantics, so every key run_match logs or prompts with must exist
    "key_notes": "neutral base — no reference analysis available",
    "confidence": 0.0,
    # ABSENT EVIDENCE, stated. This dict is also the gateway-down fallback, and the
    # -60° bearing above is an INVENTION, not a reading. Without these keys the
    # controller's `.get("sun_bearing_agreement", 1.0)` read the invention at FULL trust
    # and gave it a quarter of the match objective, then printed "sun mid @ bearing -60°"
    # in the same sentence shape it uses for a real measurement. (2026-07-31)
    "sun_bearing_agreement": 0.0,
    "sun_bearing_spread_deg": None,
}

# (key, label, why, semantics overrides) — bearing convention matches ANALYZE:
# 0 = sun straight ahead of camera (backlight), ±90 = side, ±180 = front-lit
VARIANTS = (
    ("as_analyzed", "As analyzed",
     "the reference's own read — the first-guess rig",
     {}),
    ("golden_low", "Golden low sun",
     "late warm key raking the frame, hazy falloff",
     {"time_of_day": "golden_hour", "sky": "clear", "sun_active": True,
      "sun_altitude_band": "golden", "light_quality": "hard",
      "atmosphere": "light_haze", "wb_kelvin_estimate": 3800.0,
      "contrast_character": "moody"}),
    ("overcast_soft", "Overcast soft",
     "sky as the key — shadowless product-light",
     {"time_of_day": "overcast_day", "sky": "overcast", "sun_active": False,
      "sun_altitude_band": "na", "light_quality": "soft",
      "atmosphere": "light_haze", "wb_kelvin_estimate": 6800.0,
      "practicals_on": False, "contrast_character": "airy"}),
    ("backlit_rim", "Backlit rim",
     "low sun straight into camera — silhouettes and halos",
     {"time_of_day": "afternoon", "sky": "clear", "sun_active": True,
      "sun_bearing_deg": 0.0, "sun_altitude_band": "low",
      "light_quality": "hard", "atmosphere": "none",
      "wb_kelvin_estimate": 5200.0, "contrast_character": "moody"}),
    ("cool_north", "Cool north light",
     "sunless clear-sky fill — the atelier look",
     {"time_of_day": "morning", "sky": "clear", "sun_active": False,
      "sun_altitude_band": "na", "light_quality": "soft",
      "atmosphere": "none", "wb_kelvin_estimate": 7500.0,
      "practicals_on": False, "contrast_character": "airy"}),
    ("practicals_dusk", "Practicals at dusk",
     "blue hour outside, the room's own lamps carrying",
     {"time_of_day": "blue_hour", "sky": "clear", "sun_active": False,
      "sun_altitude_band": "just_set", "light_quality": "soft",
      "wb_kelvin_estimate": 8000.0, "practicals_on": True,
      "contrast_character": "moody"}),
)


# numeric ANALYZE fields with their accepted ranges (mirror parse.validate_analysis): a
# hand-edited/corrupted sidecar cache reaches here RAW — json.load accepts NaN/Infinity
# literals — and rules.initial_state would float() them straight into genome.clamp, whose
# wrap path (fmod(nan, 360) = nan) lets a NaN azimuth ride all the way into a scene write
_NUMERIC_BOUNDS = {
    "sun_bearing_deg": (-180.0, 180.0),
    "wb_kelvin_estimate": (2000.0, 15000.0),
    "confidence": (0.0, 1.0),
}


def sanitize_semantics(semantics: Optional[Dict]) -> Dict:
    """Cached-semantics gate: drop anything rules.initial_state can't safely consume —
    NaN/±inf, non-numeric numerics, non-scalar junk. Out-of-range numbers are clamped
    (the same move parse._num makes on the live-LLM path); rejected keys fall back to
    the DEFAULT_SEMANTICS base instead of corrupting the board."""
    out: Dict = {}
    for k, v in (semantics or {}).items():
        if k in _NUMERIC_BOUNDS:
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(f):
                continue
            lo, hi = _NUMERIC_BOUNDS[k]
            out[k] = min(hi, max(lo, f))
        elif isinstance(v, bool) or isinstance(v, str) or v is None:
            out[k] = v
        elif isinstance(v, (int, float)) and math.isfinite(v):
            out[k] = v
        # anything else (dict/list/non-finite float) isn't ANALYZE vocabulary — drop it
    return out


def build_scenarios(
    semantics: Optional[Dict],
    current: LightingState,
    camera_yaw_deg: float,
    locks: Optional[Set[str]] = None,
    overcast_sun_mode: str = "dim",
    max_count: int = 6,
) -> List[Dict]:
    """→ [{key, label, why, state}] — distinct, rig-capable, lock-respecting candidates.

    ``semantics`` may be None (no reference bound): the board still builds from the
    neutral base, minus the "as analyzed" slot which would be meaningless. Variants whose
    state collapses onto an earlier one (a rig with no sun makes golden == backlit) are
    dropped — the board shows CHOICES, not duplicates."""
    if not current.values and not current.groups:
        return []          # no writable rig → every "candidate" would be a no-op card
    max_count = max(0, int(max_count))
    base = dict(DEFAULT_SEMANTICS)
    have_ref = bool(semantics)
    if have_ref:
        base.update({k: v for k, v in sanitize_semantics(semantics).items()
                     if v is not None})
    out: List[Dict] = []
    for key, label, why, overrides in VARIANTS:
        if len(out) >= max_count:
            break
        if key == "as_analyzed" and not have_ref:
            continue
        if key == "practicals_dusk" and not current.groups:
            continue                       # no dimmer boards — the lamps aren't there
        sem = dict(base)
        sem.update(overrides)
        state, _rationale = rules.initial_state(
            sem, current, camera_yaw_deg, locks,
            overcast_sun_mode=overcast_sun_mode)
        if any(not state.diff(prev["state"]) for prev in out):
            continue
        out.append({"key": key, "label": label, "why": why, "state": state,
                    "semantics": sem})
    return out
