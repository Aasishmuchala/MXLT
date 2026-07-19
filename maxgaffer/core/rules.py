"""Reference semantics → first lighting guess. Deterministic gaffer craft, no LLM freestyle.

The ANALYZE call returns *semantics* (golden hour, backlit-left, hazy…) precisely because
mapping semantics to V-Ray numbers is a solved craft problem — tables beat vibes. The LLM
then only has to REFINE from a sane starting point, which is a bounded task it is good at.

Camera yaw convention: world compass bearing of the camera's LOOK direction in degrees,
0 = +Y (north), clockwise positive (matches sun.azimuth_deg in the genome). The bridge
computes it from the camera transform; here it's just arithmetic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from .genome import GROUP_PREFIX, LightingState

ALTITUDE_DEG = {
    "just_set": -2.0,
    "golden": 6.0,
    "low": 18.0,
    "mid": 38.0,
    "high": 62.0,
    "overhead": 80.0,
    "na": 35.0,
}

TIME_FALLBACK_ALTITUDE = {
    "night": -4.0,
    "blue_hour": -2.0,
    "golden_hour": 6.0,
    "morning": 25.0,
    "midday": 65.0,
    "afternoon": 40.0,
    "overcast_day": 45.0,
}

TURBIDITY = {"none": 2.5, "light_haze": 4.5, "heavy_haze": 7.0, "fog": 8.5}

# sun size multiplier: hard light = small apparent sun, soft = bigger (haze/diffusion)
SUN_SIZE = {"hard": 1.0, "mixed": 3.0, "soft": 8.0}


def _fnum(value, default: float) -> float:
    """float() + finiteness — cached semantics may come from a hand-edited sidecar,
    so a junk value falls back to the default instead of crashing the match at start."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def initial_state(
    semantics: Dict,
    current: LightingState,
    camera_yaw_deg: float,
    locks: Optional[Set[str]] = None,
    overcast_sun_mode: str = "disable",
) -> Tuple[LightingState, List[str]]:
    """First guess written over a copy of ``current`` (only unlocked params, only params the
    rig actually has — i.e. keys already present in ``current``). Returns (state, rationale).

    ``overcast_sun_mode``: "disable" turns the VRaySun off for overcast references;
    "dim" keeps it on at minimum intensity with a huge apparent size instead — the escape
    hatch for rigs whose VRaySky brightness is coupled to the sun node (checklist #13)."""
    locks = locks or set()
    st = current.copy()
    why: List[str] = []
    camera_yaw_deg = _fnum(camera_yaw_deg, 0.0)   # a degenerate camera transform ≠ crash

    def put(key: str, value: float, note: str) -> None:
        has = (key in st.values) or (
            key.startswith(GROUP_PREFIX) and key[len(GROUP_PREFIX):] in st.groups)
        if not has or key in locks:
            return
        st.set(key, value)
        why.append(f"{key} → {st.get(key):.2f}  ({note})")

    time_of_day = semantics.get("time_of_day", "afternoon")
    sky = semantics.get("sky", "clear")
    sun_active = bool(semantics.get("sun_active", True)) and sky != "overcast" \
        and time_of_day not in ("night",)

    # ---- sun geometry
    if sun_active:
        put("sun.enabled", 1, f"{time_of_day}, {sky} sky — direct sun shapes the frame")
        band = semantics.get("sun_altitude_band", "na")
        alt = ALTITUDE_DEG.get(band, 35.0)
        if band == "na":
            alt = TIME_FALLBACK_ALTITUDE.get(time_of_day, 35.0)
        put("sun.altitude_deg", alt, f"altitude band '{band}' / {time_of_day}")
        bearing = _fnum(semantics.get("sun_bearing_deg"), 0.0)
        put("sun.azimuth_deg", camera_yaw_deg + bearing,
            f"camera yaw {camera_yaw_deg:.0f}° + bearing {bearing:+.0f}° from reference shadows")
        put("sun.size", SUN_SIZE.get(semantics.get("light_quality", "mixed"), 3.0),
            f"{semantics.get('light_quality', 'mixed')} shadow edges")
        if time_of_day in ("golden_hour", "blue_hour"):
            put("sun.intensity", 1.0, "low sun — physical sky handles the falloff")
    elif sky == "overcast" and overcast_sun_mode == "dim" and time_of_day != "night":
        put("sun.enabled", 1, "overcast (dim mode) — sun kept on so VRaySky stays alive")
        put("sun.intensity", 0.05, "overcast dim mode — sun contributes ~nothing")
        put("sun.size", 12.0, "overcast — any residual shadow is soft mush")
    else:
        put("sun.enabled", 0, f"no direct sun ({'overcast' if sky == 'overcast' else time_of_day})")

    put("sun.turbidity", TURBIDITY.get(semantics.get("atmosphere", "none"), 2.5),
        f"atmosphere '{semantics.get('atmosphere', 'none')}'")

    # ---- dome / sky fill
    if "dome.enabled" in st.values:
        put("dome.enabled", 1, "environment fill always contributes")
        if sky == "overcast" or not sun_active:
            put("dome.intensity", max(1.0, st.get("dome.intensity", 1.0)),
                "sky is the key light — dome carries the frame")

    # ---- white balance first guess (analytic solver refines every iteration)
    put("exposure.wb_kelvin", _fnum(semantics.get("wb_kelvin_estimate"), 6500.0),
        "reference white-balance feel")

    # ---- practicals: on → they contribute (and at night they carry the frame);
    # off in a daytime reference → kill them; off at night → leave authored (moonlight looks).
    # MG_ groups are exempt from the kill — those are MaxGaffer's OWN plan-created match
    # instruments (fills, rims), not scene practicals; zeroing them would undo the plan.
    practicals = bool(semantics.get("practicals_on", False))
    for group in st.groups:
        key = GROUP_PREFIX + group
        if practicals:
            put(key, max(1.0, st.get(key, 1.0)), "reference shows practicals contributing")
        elif group.startswith("MG_"):
            continue
        elif time_of_day not in ("night", "blue_hour"):
            put(key, 0.0, "daylight reference with no practical contribution")

    return st, why


def sweep_azimuths(n: int = 8) -> List[float]:
    """Evenly spaced compass azimuths for the sun-direction grid solve."""
    n = max(2, int(n))
    return [i * 360.0 / n for i in range(n)]
