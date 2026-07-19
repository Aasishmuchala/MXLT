"""LLM reply validation — every gateway reply passes through here before touching anything.

The genome enforces bounds; this enforces SHAPE. Missing fields get safe defaults, junk gets
dropped, and a reply with no parseable JSON raises ParseError so the loop can retry once and
then degrade gracefully instead of applying garbage to a client scene.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence

from .omega import parse_json_from_text

ALTITUDE_BANDS = ("just_set", "golden", "low", "mid", "high", "overhead", "na")
TIMES = ("night", "blue_hour", "golden_hour", "morning", "midday", "afternoon", "overcast_day")
SKIES = ("clear", "hazy", "overcast", "dramatic", "night")
ATMOSPHERES = ("none", "light_haze", "heavy_haze", "fog")


class ParseError(ValueError):
    pass


def _num(d: Dict, key: str, lo: float, hi: float, default: float) -> float:
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return default
    # JSON accepts bare NaN/Infinity tokens — they must land on the DEFAULT, not on a
    # bound (min/max ordering silently coerces NaN to lo)
    return min(hi, max(lo, v)) if math.isfinite(v) else default


def _enum(d: Dict, key: str, allowed: Sequence[str], default: str) -> str:
    v = d.get(key)
    return v if isinstance(v, str) and v in allowed else default


def _flag(d: Dict, key: str, default: bool) -> bool:
    v = d.get(key)
    return v if isinstance(v, bool) else default


def validate_analysis(reply_text: str) -> Dict:
    obj = parse_json_from_text(reply_text)
    if obj is None:
        raise ParseError("analysis reply contained no JSON object")
    return {
        "scene_type": _enum(obj, "scene_type",
                            ("exterior", "interior", "interior_with_view"), "exterior"),
        "time_of_day": _enum(obj, "time_of_day", TIMES, "afternoon"),
        "sky": _enum(obj, "sky", SKIES, "clear"),
        "sun_active": _flag(obj, "sun_active", True),
        "sun_bearing_deg": _num(obj, "sun_bearing_deg", -180.0, 180.0, 0.0),
        "sun_altitude_band": _enum(obj, "sun_altitude_band", ALTITUDE_BANDS, "mid"),
        "light_quality": _enum(obj, "light_quality", ("hard", "soft", "mixed"), "mixed"),
        "wb_kelvin_estimate": _num(obj, "wb_kelvin_estimate", 2000.0, 15000.0, 6500.0),
        "practicals_on": _flag(obj, "practicals_on", False),
        "atmosphere": _enum(obj, "atmosphere", ATMOSPHERES, "none"),
        "contrast_character": _enum(obj, "contrast_character",
                                    ("airy", "balanced", "moody"), "balanced"),
        "key_notes": str(obj.get("key_notes") or "")[:400],
        "confidence": _num(obj, "confidence", 0.0, 1.0, 0.5),
    }


def validate_deltas(reply_text: str, max_changes: int = 4) -> Dict:
    """→ {"assessment": str, "changes": {param: value}, "reasons": {param: why}, "stop": bool}
    Only shape-checks here — genome.apply_changes does bounds/locks/step limiting."""
    obj = parse_json_from_text(reply_text)
    if obj is None:
        raise ParseError("deltas reply contained no JSON object")
    changes: Dict[str, float] = {}
    reasons: Dict[str, str] = {}
    raw = obj.get("changes")
    if isinstance(raw, dict):
        # a common deviation from the prompted schema — accept it rather than drop silently
        raw = [{"param": k, "value": v} for k, v in raw.items()]
    if isinstance(raw, list):
        for item in raw[:max_changes]:
            if not isinstance(item, dict):
                continue
            param = item.get("param")
            try:
                value = float(item.get("value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue          # NaN/∞ proposals are refused, never step-coerced
            if isinstance(param, str) and param:
                changes[param] = value
                reasons[param] = str(item.get("why") or "")[:120]
    return {
        "assessment": str(obj.get("assessment") or "")[:500],
        "changes": changes,
        "reasons": reasons,
        "stop": _flag(obj, "stop", False),
    }


def validate_sweep(reply_text: str, n_candidates: int) -> Dict:
    obj = parse_json_from_text(reply_text)
    if obj is None:
        raise ParseError("sweep reply contained no JSON object")
    idx = int(_num(obj, "best_index", 0, max(0, n_candidates - 1), 0))
    return {
        "best_index": idx,
        "altitude_hint": _enum(obj, "altitude_hint", ALTITUDE_BANDS, "na"),
        "why": str(obj.get("why") or "")[:200],
    }
