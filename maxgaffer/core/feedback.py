"""Director's notes — the conversation layer. "Exposure is too much" must DO something.

Two mechanisms, in the house pattern (math first, model second):
  * a CRAFT TABLE parses common critiques deterministically into immediate bounded nudges
    ("too bright" → +0.7 EV, "sun more left" → −20° azimuth relative to the camera) — the
    note takes effect in the very next render, no model in the loop;
  * the raw note text is pinned into every subsequent LLM prompt as the DIRECTOR'S NOTE
    (highest priority), and the ensemble lenses attack it from three angles.

Notes are matched on word stems, case-insensitive; "very/way/much too X" doubles the
nudge. Unknown phrasing simply parses to no nudges — the LLM still gets the text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# (regex, key, delta, is_relative_to_camera) — deltas are ADDED (log2 for log-scale keys
# is not needed here; intensity critiques go through EV, the measured channel)
_RULES: Tuple[Tuple[str, str, float], ...] = (
    (r"\b(too bright|overexposed|over-exposed|blown|exposure is too (much|high))", "exposure.ev", +0.7),
    (r"\b(too dark|underexposed|under-exposed|exposure is too low)", "exposure.ev", -0.7),
    (r"\b(too warm|too orange|too yellow)", "exposure.wb_kelvin", -800.0),
    (r"\b(too cool|too blue|too cold)", "exposure.wb_kelvin", +800.0),
    (r"\b(shadows? (are )?too soft|softer than|less crisp)", "sun.size", -0.5),   # log2 handled below
    (r"\b(shadows? (are )?too (hard|sharp|crisp)|soften the (sun|shadows?))", "sun.size", +0.5),
    (r"\b(sun (more |to the )?left|light (from |more )?(the )?left)", "sun.azimuth_deg", -20.0),
    (r"\b(sun (more |to the )?right|light (from |more )?(the )?right)", "sun.azimuth_deg", +20.0),
    (r"\b(sun (too )?(high|steep)|lower the sun|sun down)", "sun.altitude_deg", -10.0),
    (r"\b(sun (too )?low|raise the sun|sun up|higher sun)", "sun.altitude_deg", +10.0),
    (r"\b(too hazy|less haze|too much haze|too milky)", "sun.turbidity", -1.5),
    (r"\b(more haze|hazier|too crisp|too clear)", "sun.turbidity", +1.5),
    (r"\b(practicals? (too )?(bright|strong|hot))", "group.*", -0.5),
    (r"\b(practicals? (too )?(dim|weak)|more practicals?)", "group.*", +0.5),
)
_INTENSIFIERS = re.compile(r"\b(way|much|far|very|really) too\b")
_LOG_KEYS = ("sun.size",)          # these deltas are applied in log2 space
_GROUP_WILDCARD = "group.*"


def nudges_from_note(note: str, current_keys: List[str],
                     group_names: List[str]) -> Dict[str, float]:
    """Note text → {param: ABSOLUTE-DELTA dict entries} for params the rig has. The caller
    turns deltas into values (so bounds/steps apply through the normal genome gate)."""
    low = " " + (note or "").lower() + " "
    deltas: Dict[str, float] = {}
    for pattern, key, delta in _RULES:
        m = re.search(pattern, low)
        if m:
            # intensifier is LOCAL to its own clause — "way too dark, shadows too hard"
            # doubles only the darkness nudge; the scan never crosses , ; . and/but
            lead = low[max(0, m.start() - 24):m.start()]
            for sep in (",", ";", ".", " and ", " but "):
                if sep in lead:
                    lead = lead.rsplit(sep, 1)[1]
            scale = 2.0 if _INTENSIFIERS.search(lead + m.group(0)) else 1.0
            if key == _GROUP_WILDCARD:
                for g in group_names:
                    if g.startswith("MG_"):
                        continue   # MG_* are our own plan-created match instruments,
                                   # not scene practicals (same exemption as rules.py)
                    deltas[f"group.{g}"] = deltas.get(f"group.{g}", 0.0) + delta * scale
            elif key in current_keys:
                deltas[key] = deltas.get(key, 0.0) + delta * scale
    return deltas


def apply_note_deltas(state, deltas: Dict[str, float], locks=None):
    """Deltas → target values on a copy of ``state`` (log-space where the key demands),
    then through the genome gate so bounds still rule. → (new_state, changes_dict).
    A note is a request, a lock is a veto — locks are honored here exactly like in the
    LLM and polish paths ("never touched — not by the solver, not by the model")."""
    from .genome import apply_changes

    changes: Dict[str, float] = {}
    for key, delta in deltas.items():
        cur = state.get(key)
        if key in _LOG_KEYS:
            changes[key] = cur * (2.0 ** delta)
        elif key.startswith("group."):
            changes[key] = max(0.0, cur * (2.0 ** delta))
        else:
            changes[key] = cur + delta
    new, accepted, _rej = apply_changes(state, changes, locks=locks, limit=False)
    return new, accepted


# ----------------------------------------------------------------- ensemble lenses
LENSES: Tuple[Tuple[str, str], ...] = (
    ("exposure-first", "Your PRIORITY this round: tonal accuracy — key, contrast envelope, "
                       "highlight/shadow placement. Geometry only if tone demands it."),
    ("geometry-first", "Your PRIORITY this round: light DIRECTION — sun azimuth/altitude, "
                       "dome rotation, where shadows fall. Tone only if direction demands it."),
    ("mood-first", "Your PRIORITY this round: color mood and quality — warmth, haze, "
                   "shadow softness, group balance. Structure only if mood demands it."),
)


def lens_system(base_system: str, lens_line: str) -> str:
    return base_system + "\n\nLENS: " + lens_line
