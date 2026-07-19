"""The lighting genome — every parameter MaxGaffer is allowed to touch, with hard bounds,
per-iteration step limits, and user locks.

Design contract (mirrors MaxDirector's "bounded LLM" principle):
  * The LLM never invents parameter names — proposals are validated against this table and
    anything unknown is DROPPED, anything out-of-bounds is CLAMPED, anything locked is REFUSED.
  * Exposure + white balance are marked ``analytic`` — the deterministic solver owns them and
    the LLM is told hands-off (an LLM eyeballing EV is exactly the hallucination MaxDirector's
    geometric critic was built to kill).
  * Angles are canonical degrees; intensities move in LOG space (a 2x light change is one
    perceptual step regardless of absolute value), which is what ``step`` means for them.

Pure python, zero pymxs — the bridge translates a LightingState into scene writes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

GROUP_PREFIX = "group."          # dynamic artificial-light groups: "group.<name>"
GROUP_BOUNDS = (0.0, 10.0)       # multiplier factor bounds for a light group
GROUP_STEP = 1.0                 # max log2 change per iteration for a group


@dataclass(frozen=True)
class ParamSpec:
    key: str
    lo: float
    hi: float
    step: float            # max |change| per LLM iteration (log2 units when log_scale)
    wrap: bool = False     # circular parameter (azimuth, dome rotation)
    log_scale: bool = False
    analytic: bool = False  # owned by the deterministic solver, not the LLM
    doc: str = ""


PARAMS: Tuple[ParamSpec, ...] = (
    ParamSpec("sun.enabled", 0, 1, 1, doc="1 = VRaySun on, 0 = off (overcast/dusk looks)"),
    ParamSpec("sun.azimuth_deg", 0.0, 360.0, 60.0, wrap=True,
              doc="world compass azimuth of the sun, degrees, 0=+Y north, clockwise"),
    ParamSpec("sun.altitude_deg", -4.0, 88.0, 25.0,
              doc="sun elevation above horizon; 2-10 golden hour, <0 just-set glow"),
    ParamSpec("sun.intensity", 0.05, 20.0, 1.0, log_scale=True,
              doc="VRaySun intensity multiplier"),
    ParamSpec("sun.size", 0.5, 30.0, 1.0, log_scale=True,
              doc="VRaySun size multiplier — bigger = softer shadow edges"),
    ParamSpec("sun.turbidity", 1.8, 10.0, 1.5,
              doc="atmosphere haze; 2-3 crisp blue, 5-7 warm hazy, 8+ smog"),
    ParamSpec("dome.enabled", 0, 1, 1, doc="1 = dome/HDRI light on"),
    ParamSpec("dome.rotation_deg", 0.0, 360.0, 90.0, wrap=True,
              doc="HDRI horizontal rotation (dome node Z spin), degrees"),
    ParamSpec("dome.intensity", 0.0, 20.0, 1.0, log_scale=True,
              doc="dome light multiplier"),
    ParamSpec("exposure.ev", -4.0, 20.0, 2.5, analytic=True,
              doc="camera/exposure-control EV — SOLVED analytically from histograms"),
    ParamSpec("exposure.wb_kelvin", 2000.0, 15000.0, 1500.0, analytic=True,
              doc="white balance kelvin — SOLVED analytically; higher K renders warmer"),
)

SPEC_BY_KEY: Dict[str, ParamSpec] = {p.key: p for p in PARAMS}


def spec_for(key: str) -> Optional[ParamSpec]:
    """Spec for a fixed param or a dynamic ``group.<name>`` multiplier."""
    if key in SPEC_BY_KEY:
        return SPEC_BY_KEY[key]
    if key.startswith(GROUP_PREFIX) and len(key) > len(GROUP_PREFIX):
        return ParamSpec(key, GROUP_BOUNDS[0], GROUP_BOUNDS[1], GROUP_STEP, log_scale=True,
                         doc="artificial light-group intensity multiplier factor")
    return None


def _wrap_deg(v: float) -> float:
    if not math.isfinite(v):
        return 0.0          # inf/NaN can never be a heading — deterministic fallback
    v = math.fmod(v, 360.0)
    return v + 360.0 if v < 0 else v


def clamp(key: str, value: float) -> float:
    spec = spec_for(key)
    if spec is None:
        raise KeyError(f"unknown lighting parameter: {key}")
    v = float(value)
    if spec.wrap:
        return _wrap_deg(v)
    if math.isnan(v):
        return spec.lo
    return min(spec.hi, max(spec.lo, v))


def limit_step(key: str, current: float, proposed: float, scale: float = 1.0) -> float:
    """Clamp ``proposed`` so one iteration never moves further than the spec's step.
    Log-scale params limit the log2 ratio; wrapped params take the short way around.

    ``scale`` anneals the step budget: 1.0 while exploring, shrinking as the match closes
    in — a 60° azimuth swing is how you FIND the sun, not how you land the last two points.
    """
    spec = spec_for(key)
    if spec is None:
        raise KeyError(f"unknown lighting parameter: {key}")
    step = spec.step * max(0.05, float(scale))
    cur, prop = float(current), float(proposed)
    if not math.isfinite(prop):
        # a NaN/∞ proposal must never become a move — min/max ordering would otherwise
        # silently coerce it into a FULL step in one direction (found by stress fuzzing)
        return clamp(key, cur)
    if not math.isfinite(cur):
        cur = spec.lo
    if spec.wrap:
        delta = (prop - cur + 180.0) % 360.0 - 180.0
        if delta == -180.0:   # antipode is ambiguous — deterministically go clockwise
            delta = 180.0
        delta = max(-step, min(step, delta))
        return _wrap_deg(cur + delta)
    if spec.log_scale and cur > 1e-6 and prop > 1e-6:
        ratio = math.log2(prop / cur)
        ratio = max(-step, min(step, ratio))
        return clamp(key, cur * (2.0 ** ratio))
    delta = max(-step, min(step, prop - cur))
    return clamp(key, cur + delta)


@dataclass
class LightingState:
    """One complete lighting rig setting. ``values`` holds fixed params; ``groups`` holds
    ``{group_name: multiplier_factor}`` for artificial lights (factor 1.0 = as-authored)."""

    values: Dict[str, float] = field(default_factory=dict)
    groups: Dict[str, float] = field(default_factory=dict)

    # -------------------------------------------------------------- access
    def get(self, key: str, default: float = 0.0) -> float:
        if key.startswith(GROUP_PREFIX):
            return self.groups.get(key[len(GROUP_PREFIX):], default)
        return self.values.get(key, default)

    def set(self, key: str, value: float) -> None:
        v = clamp(key, value)
        if key.startswith(GROUP_PREFIX):
            self.groups[key[len(GROUP_PREFIX):]] = v
        else:
            self.values[key] = v

    def keys(self) -> List[str]:
        return list(self.values.keys()) + [GROUP_PREFIX + g for g in self.groups]

    def copy(self) -> "LightingState":
        return LightingState(values=dict(self.values), groups=dict(self.groups))

    # -------------------------------------------------------------- json
    def to_dict(self) -> Dict[str, Any]:
        return {"values": dict(self.values), "groups": dict(self.groups)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LightingState":
        """Sidecar/preset load — hand-edited files are INVITED (the sidecar is documented
        as human-readable), so junk values are skipped, never raised."""
        st = cls()
        if not isinstance(d, dict):
            return st
        values = d.get("values")
        groups = d.get("groups")
        for k, v in (values.items() if isinstance(values, dict) else ()):
            if spec_for(k) is not None:
                try:
                    st.values[k] = clamp(k, v)
                except (TypeError, ValueError):
                    continue
        for g, v in (groups.items() if isinstance(groups, dict) else ()):
            try:
                st.groups[str(g)] = clamp(GROUP_PREFIX + str(g), v)
            except (TypeError, ValueError):
                continue
        return st

    # -------------------------------------------------------------- diff
    def diff(self, other: "LightingState") -> Dict[str, Tuple[float, float]]:
        """{key: (mine, theirs)} for every key whose value meaningfully differs."""
        out: Dict[str, Tuple[float, float]] = {}
        for key in sorted(set(self.keys()) | set(other.keys())):
            a, b = self.get(key), other.get(key)
            if abs(a - b) > 1e-4:
                out[key] = (a, b)
        return out


def rig_keys(state: LightingState) -> set:
    """The capability set of the rig ``state`` was read from — the ONLY keys a
    proposal may touch. ``group.*`` names come from the rig's real layer groups."""
    return set(state.values) | {GROUP_PREFIX + g for g in state.groups}


def apply_changes(
    state: LightingState,
    changes: Dict[str, float],
    locks: Optional[set] = None,
    limit: bool = True,
    step_scale: float = 1.0,
    known: Optional[set] = None,
) -> Tuple[LightingState, Dict[str, float], List[str]]:
    """Validated apply: returns (new_state, accepted{key: value}, rejected[reason...]).
    Unknown keys are dropped, locked keys refused, bounds clamped, steps limited
    (``step_scale`` anneals the per-iteration budget). ``known`` (usually
    ``rig_keys(state)``) rejects genome-valid keys the RIG doesn't have — without it
    a hallucinated ``dome.*``/``group.*`` on a rig that lacks them would be fabricated
    at the spec's lower bound and persist into the session as a phantom parameter."""
    locks = locks or set()
    new = state.copy()
    accepted: Dict[str, float] = {}
    rejected: List[str] = []
    for key, raw in changes.items():
        spec = spec_for(key)
        if spec is None:
            rejected.append(f"{key}: unknown parameter")
            continue
        if known is not None and key not in known:
            rejected.append(f"{key}: rig has no such parameter")
            continue
        if key in locks:
            rejected.append(f"{key}: locked by user")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            rejected.append(f"{key}: non-numeric value {raw!r}")
            continue
        if not math.isfinite(value):
            rejected.append(f"{key}: non-finite value {raw!r}")
            continue
        if limit:
            value = limit_step(key, new.get(key, spec.lo), value, step_scale)
        else:
            value = clamp(key, value)
        new.set(key, value)
        accepted[key] = value
    return new, accepted, rejected


def state_table(state: LightingState, locks: Optional[set] = None) -> str:
    """Human/LLM-readable table of the current state with bounds + lock + analytic flags —
    embedded verbatim in the delta prompt so proposals stay inside the rails."""
    locks = locks or set()
    lines = ["param | current | min..max | max_step | flags | meaning"]
    for key in sorted(state.keys()):
        spec = spec_for(key)
        if spec is None:
            continue
        flags = []
        if key in locks:
            flags.append("LOCKED")
        if spec.analytic:
            flags.append("ANALYTIC(hands-off)")
        if spec.log_scale:
            flags.append("log2-step")
        if spec.wrap:
            flags.append("wraps")
        lines.append(
            f"{key} | {state.get(key):.3f} | {spec.lo:g}..{spec.hi:g} | "
            f"{spec.step:g} | {','.join(flags) or '-'} | {spec.doc}"
        )
    return "\n".join(lines)
