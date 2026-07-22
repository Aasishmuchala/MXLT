"""LightingState → scene, and scene → LightingState. One undo record per apply.

Group multipliers are FACTORS over the lights' AUTHORED values. Baselines are keyed by
light NAME and live in the Session (adopt-once, never overwrite) — names survive Max
restarts, and adopt-once is what makes re-scanning a rig safe after MaxGaffer itself has
dimmed a group (re-capturing would read 0 and poison the group forever). Everything is
candidates-based and per-parameter fault-isolated — one missing property must not stop the
sun from moving.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from ..core.genome import LightingState
from . import scene as sc
from .exposure import ExposureHost

BASELINE_EPSILON = 1e-4  # at/below this a captured value is "dimmed", not "authored"


def _rt():
    import pymxs

    return pymxs.runtime


def _light_name(node) -> str:
    try:
        return str(node.name)
    except Exception:
        return ""


def _baseline_of(baselines: Dict[str, float], name: str) -> float:
    """A light's authored multiplier — never ~0 (a 0 baseline is dimmer-poison, not an
    authored value; capture and adoption both refuse it). Read and apply paths share this
    helper so an adopted baseline behaves identically in both directions."""
    try:
        base = float(baselines.get(name, 1.0))
    except (TypeError, ValueError):
        return 1.0
    return base if abs(base) > BASELINE_EPSILON else 1.0


def _authored_off(baselines: Dict[str, float], name: str, current_mult) -> bool:
    """True for a light whose 0 baseline is REAL: the artist authored it off (baseline 0
    AND the light sits at ~0 right now). Its dimmer carries no signal — reads report 0,
    writes keep it pinned at 0. A 0 baseline on a light that is visibly lit is legacy
    POISON instead (stale sidecar / hand edit) and is treated as authored 1.0 by
    ``_baseline_of`` — same discriminator on the read and apply paths."""
    try:
        if not (name in baselines and float(baselines[name]) == 0.0):
            return False
    except (TypeError, ValueError):
        return False
    try:
        return current_mult is None or abs(float(current_mult)) <= BASELINE_EPSILON
    except (TypeError, ValueError):
        return True


def _state_float(state: LightingState, key: str, warnings: List[str]):
    """float() of a state value, fault-isolated — states are clamped floats via set()/
    from_dict, but the dicts are public, so a bad raw value downgrades to a warning
    instead of detonating inside the undo record. None = skip this param."""
    try:
        value = float(state.get(key))
        if not math.isfinite(value):
            raise ValueError("non-finite")
        return value
    except (TypeError, ValueError):
        warnings.append(f"{key}: non-numeric value {state.get(key)!r} — skipped")
        return None


class CaptureResult(dict):
    """Plain {light_name: multiplier} dict PLUS ``.notes`` — the skip reasons the caller
    should surface in the log. Subclasses dict so existing consumers
    (``Session.adopt_baselines``, on-box scripts) take it unchanged."""
    def __init__(self, *args, notes: List[str] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.notes: List[str] = list(notes)


def capture_baselines(rig: Dict[str, Any]) -> CaptureResult:
    """{light_name: current multiplier} for every group light — a CANDIDATE set. Feed it
    through ``Session.adopt_baselines`` (adopt-only-new); never use it to overwrite.
    A light currently at ~0 is almost certainly DIMMED, not authored that way — adopting
    0 would poison the group (0 × factor is 0 forever), so it is skipped with a note in
    ``.notes`` (the artist can re-author explicitly via ``forget_baseline``)."""
    out: Dict[str, float] = {}
    notes: List[str] = []
    for lights in (rig.get("groups") or {}).values():
        for lt in lights:
            name = _light_name(lt)
            if not name:
                continue
            v = sc.get_prop(lt, sc.LIGHT_MULT, 1.0)
            try:
                mult = float(v)
            except (TypeError, ValueError):
                mult = 1.0
            if abs(mult) <= BASELINE_EPSILON:
                notes.append(f"baseline: light '{name}' reads {mult:g} — declining to "
                             "capture a ~0 baseline (likely dimmed, not authored); use "
                             "forget_baseline to force a re-author")
                continue
            out[name] = mult
    return CaptureResult(out, notes=notes)


def _role_nodes(rig: Dict[str, Any], plural: str, singular: str) -> List[Any]:
    nodes = list(rig.get(plural) or [])
    primary = rig.get(singular)
    if not nodes and primary is not None:
        nodes = [primary]
    elif primary is not None and primary not in nodes:
        nodes.insert(0, primary)
    return nodes[:4]


def _role_prefix(base: str, index: int) -> str:
    return base if index == 0 else f"{base}{index + 1}"


def read_state(rig: Dict[str, Any], baselines: Dict[str, float],
               camera=None) -> LightingState:
    """Current scene → genome (only the params this rig actually supports)."""
    st = LightingState()
    for index, sun in enumerate(_role_nodes(rig, "suns", "sun")):
        prefix = _role_prefix("sun", index)
        az, alt, _ = sc.read_sun_angles(sun)
        st.set(f"{prefix}.azimuth_deg", az)
        st.set(f"{prefix}.altitude_deg", alt)
        on = sc.get_prop(sun, sc.LIGHT_ON, True)
        st.set(f"{prefix}.enabled", 1.0 if on else 0.0)
        for suffix, props in (("intensity", sc.SUN_INTENSITY),
                              ("size", sc.SUN_SIZE),
                              ("turbidity", sc.SUN_TURBIDITY)):
            v = sc.get_prop(sun, props)
            if v is not None:
                try:
                    st.set(f"{prefix}.{suffix}", float(v))
                except (TypeError, ValueError):
                    pass
    for index, dome in enumerate(_role_nodes(rig, "domes", "dome")):
        prefix = _role_prefix("dome", index)
        st.set(f"{prefix}.enabled",
               1.0 if sc.get_prop(dome, sc.LIGHT_ON, True) else 0.0)
        st.set(f"{prefix}.rotation_deg", sc.read_dome_rotation(dome))
        v = sc.get_prop(dome, sc.LIGHT_MULT)
        if v is not None:
            try:
                st.set(f"{prefix}.intensity", float(v))
            except (TypeError, ValueError):
                pass
    atmosphere = rig.get("atmosphere")
    if atmosphere is not None:
        st.set("atmosphere.enabled",
               1.0 if sc.get_prop(atmosphere, sc.FOG_ON, True) else 0.0)
        distance = sc.get_prop(atmosphere, sc.FOG_DISTANCE)
        height = sc.get_prop(atmosphere, sc.FOG_HEIGHT)
        try:
            if distance is not None:
                st.set("atmosphere.distance_m", sc.metres_from_world_units(distance))
        except (TypeError, ValueError):
            pass
        try:
            if height is not None:
                st.set("atmosphere.height_m", sc.metres_from_world_units(height))
        except (TypeError, ValueError):
            pass
    for group, lights in (rig.get("groups") or {}).items():
        factors: List[float] = []
        for lt in lights:
            name = _light_name(lt)
            base = _baseline_of(baselines, name)
            v = sc.get_prop(lt, sc.LIGHT_MULT, base)
            if _authored_off(baselines, name, v):
                # authored-off light (baseline 0 AND sitting at 0): writes pin it at
                # 0 × factor = 0, so it carries no dimmer signal — read it as 0, not
                # against a phantom 1.0 baseline. A 0 baseline on a VISIBLY LIT light
                # is legacy poison instead: _baseline_of reads it as authored 1.0.
                factors.append(0.0)
                continue
            try:
                factors.append(float(v) / base)   # base is never 0 (_baseline_of floor)
            except (TypeError, ValueError):
                factors.append(1.0)
        if factors:
            st.groups[group] = sum(factors) / len(factors)
    host = ExposureHost(camera)
    ev = host.read_ev()
    if ev is not None:
        st.set("exposure.ev", ev)
    wb = host.read_wb_kelvin()
    if wb is not None:
        st.set("exposure.wb_kelvin", wb)
    elif host.kind != "none":
        st.set("exposure.wb_kelvin", 6500.0)   # color-swatch host: track our own kelvin
    return st


def apply_state(rig: Dict[str, Any], baselines: Dict[str, float], state: LightingState,
                camera=None, undo: bool = True) -> List[str]:
    """Write the state to the scene — one undo record by default. ``undo=False`` is for
    loop/probe applies (130+ per deep match would flood the artist's undo stack); their
    official revert is the pre-match snapshot. Returns warnings (params the rig couldn't
    take)."""
    import pymxs

    warnings: List[str] = []
    ctx = pymxs.undo(True, "MaxGaffer lighting") if undo else pymxs.undo(False)
    with ctx:
        _apply_inner(rig, baselines, state, camera, warnings)
    if undo:
        # probe/loop applies (undo=False) fire 130+ times per deep match — redrawing the
        # viewport for each is a redraw storm; manual applies stay live
        try:
            _rt().redrawViews()
        except Exception:
            pass
    return warnings


def bake_animation(rig: Dict[str, Any], baselines: Dict[str, float],
                   frames, camera=None) -> List[str]:
    """Write sampled lighting states as real 3ds Max animation keys in one undo record."""
    import pymxs

    warnings: List[str] = []
    with pymxs.undo(True, "MaxGaffer lighting animation"):
        with pymxs.animate(True):
            for frame, state in frames:
                with pymxs.attime(int(frame)):
                    _apply_inner(rig, baselines, state, camera, warnings)
    try:
        _rt().redrawViews()
    except Exception:
        pass
    return warnings


def _apply_inner(rig, baselines, state: LightingState, camera, warnings: List[str]) -> None:
    sun_nodes = _role_nodes(rig, "suns", "sun")
    for index, sun in enumerate(sun_nodes):
        prefix = _role_prefix("sun", index)
        enabled_key = f"{prefix}.enabled"
        az_key = f"{prefix}.azimuth_deg"
        alt_key = f"{prefix}.altitude_deg"
        if enabled_key in state.values:
            enabled = _state_float(state, enabled_key, warnings)
            if enabled is not None:
                if sc.set_prop(sun, sc.LIGHT_ON, bool(enabled >= 0.5)) is None:
                    warnings.append(f"{prefix} on/off property not found")
        if az_key in state.values or alt_key in state.values:
            current_az, current_alt, _ = sc.read_sun_angles(sun)
            az = (_state_float(state, az_key, warnings)
                  if az_key in state.values else current_az)
            alt = (_state_float(state, alt_key, warnings)
                   if alt_key in state.values else current_alt)
            if az is not None and alt is not None and not sc.write_sun_angles(sun, az, alt):
                warnings.append(f"could not move {prefix} (controller-locked transform?)")
        for suffix, props in (("intensity", sc.SUN_INTENSITY),
                              ("size", sc.SUN_SIZE),
                              ("turbidity", sc.SUN_TURBIDITY)):
            key = f"{prefix}.{suffix}"
            if key in state.values:
                value = _state_float(state, key, warnings)
                if value is not None:
                    if sc.set_prop(sun, props, value) is None:
                        warnings.append(f"{key}: no matching property on role VRaySun")
    if not sun_nodes and any(k.startswith("sun.") or k.startswith(("sun2.", "sun3.", "sun4."))
                             for k in state.values):
        warnings.append("state has sun.* but the rig has no VRaySun")

    dome_nodes = _role_nodes(rig, "domes", "dome")
    for index, dome in enumerate(dome_nodes):
        prefix = _role_prefix("dome", index)
        enabled_key = f"{prefix}.enabled"
        intensity_key = f"{prefix}.intensity"
        rotation_key = f"{prefix}.rotation_deg"
        if enabled_key in state.values:
            enabled = _state_float(state, enabled_key, warnings)
            if enabled is not None:
                sc.set_prop(dome, sc.LIGHT_ON, bool(enabled >= 0.5))
        if intensity_key in state.values:
            intensity = _state_float(state, intensity_key, warnings)
            if intensity is not None:
                if sc.set_prop(dome, sc.LIGHT_MULT, intensity) is None:
                    warnings.append(f"{intensity_key}: no multiplier property")
        if rotation_key in state.values:
            rotation = _state_float(state, rotation_key, warnings)
            if rotation is not None:
                how = sc.write_dome_rotation(dome, rotation)
                if how == "failed":
                    warnings.append(f"{rotation_key}: could not rotate texmap or node")
    if not dome_nodes and any(k.startswith("dome.") or
                              k.startswith(("dome2.", "dome3.", "dome4."))
                              for k in state.values):
        # same contract as the sun branch — an orphaned saved state (dome deleted
        # or replaced since the match) must never be dropped silently
        warnings.append("state has dome.* but the rig has no dome light")

    atmosphere = rig.get("atmosphere")
    atmosphere_keys = {k for k in state.values if k.startswith("atmosphere.")}
    if atmosphere is not None:
        if "atmosphere.enabled" in atmosphere_keys:
            enabled = _state_float(state, "atmosphere.enabled", warnings)
            if enabled is not None and sc.set_prop(
                    atmosphere, sc.FOG_ON, bool(enabled >= 0.5)) is None:
                warnings.append("atmosphere.enabled: no writable on/off property")
        for key, props in (("atmosphere.distance_m", sc.FOG_DISTANCE),
                           ("atmosphere.height_m", sc.FOG_HEIGHT)):
            if key in atmosphere_keys:
                metres = _state_float(state, key, warnings)
                if metres is not None and sc.set_prop(
                        atmosphere, props, sc.world_units(metres)) is None:
                    warnings.append(f"{key}: no matching V-Ray atmosphere property")
    elif atmosphere_keys:
        warnings.append("state has atmosphere.* but the rig has no supported V-Ray fog/aerial effect")

    rig_groups = rig.get("groups") or {}
    for group, factor in state.groups.items():
        lights = rig_groups.get(group, [])
        if not lights:
            warnings.append(f"group.{group}: no such light group in this rig "
                            "(layer renamed or lights removed?) — value dropped")
            continue
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            warnings.append(f"group.{group}: non-numeric factor {factor!r} — skipped")
            continue
        lit = 0
        for lt in lights:
            if bool(sc.get_prop(lt, sc.LIGHT_ON)):
                lit += 1
            name = _light_name(lt)
            if _authored_off(baselines, name, sc.get_prop(lt, sc.LIGHT_MULT, None)):
                # authored-off light: 0 × factor = 0 — the dimmer must not switch it on
                sc.set_prop(lt, sc.LIGHT_MULT, 0.0)
                continue
            base = _baseline_of(baselines, name)
            if sc.set_prop(lt, sc.LIGHT_MULT, base * factor) is None:
                warnings.append(f"group.{group}: light '{_light_name(lt) or '?'}' "
                                "has no multiplier")
        if lit == 0 and float(factor) > 1e-6:
            warnings.append(f"group.{group}: all {len(lights)} light(s) are DISABLED — "
                            "the dimmer moves nothing until they're switched on")

    host = ExposureHost(camera)
    if "exposure.ev" in state.values:
        ev = _state_float(state, "exposure.ev", warnings)
        if ev is not None and not host.write_ev(ev):
            warnings.append(f"exposure.ev: no writable exposure host (kind={host.kind})")
    if "exposure.wb_kelvin" in state.values:
        kelvin = _state_float(state, "exposure.wb_kelvin", warnings)
        if kelvin is not None and not host.write_wb_kelvin(kelvin):
            warnings.append("exposure.wb_kelvin: no writable WB property")
