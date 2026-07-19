"""LightingState → scene, and scene → LightingState. One undo record per apply.

Group multipliers are FACTORS over the lights' AUTHORED values. Baselines are keyed by
light NAME and live in the Session (adopt-once, never overwrite) — names survive Max
restarts, and adopt-once is what makes re-scanning a rig safe after MaxGaffer itself has
dimmed a group (re-capturing would read 0 and poison the group forever). Everything is
candidates-based and per-parameter fault-isolated — one missing property must not stop the
sun from moving.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..core.genome import LightingState
from . import scene as sc
from .exposure import ExposureHost


def _rt():
    import pymxs

    return pymxs.runtime


def _light_name(node) -> str:
    try:
        return str(node.name)
    except Exception:
        return ""


def capture_baselines(rig: Dict[str, Any]) -> Dict[str, float]:
    """{light_name: current multiplier} for every group light — a CANDIDATE set. Feed it
    through ``Session.adopt_baselines`` (adopt-only-new); never use it to overwrite."""
    out: Dict[str, float] = {}
    for lights in (rig.get("groups") or {}).values():
        for lt in lights:
            name = _light_name(lt)
            if not name:
                continue
            v = sc.get_prop(lt, sc.LIGHT_MULT, 1.0)
            try:
                out[name] = float(v)
            except (TypeError, ValueError):
                out[name] = 1.0
    return out


def read_state(rig: Dict[str, Any], baselines: Dict[str, float],
               camera=None) -> LightingState:
    """Current scene → genome (only the params this rig actually supports)."""
    st = LightingState()
    sun = rig.get("sun")
    if sun is not None:
        az, alt, _ = sc.read_sun_angles(sun)
        st.set("sun.azimuth_deg", az)
        st.set("sun.altitude_deg", alt)
        on = sc.get_prop(sun, sc.LIGHT_ON, True)
        st.set("sun.enabled", 1.0 if on else 0.0)
        for key, props in (("sun.intensity", sc.SUN_INTENSITY),
                           ("sun.size", sc.SUN_SIZE),
                           ("sun.turbidity", sc.SUN_TURBIDITY)):
            v = sc.get_prop(sun, props)
            if v is not None:
                try:
                    st.set(key, float(v))
                except (TypeError, ValueError):
                    pass
    dome = rig.get("dome")
    if dome is not None:
        st.set("dome.enabled", 1.0 if sc.get_prop(dome, sc.LIGHT_ON, True) else 0.0)
        st.set("dome.rotation_deg", sc.read_dome_rotation(dome))
        v = sc.get_prop(dome, sc.LIGHT_MULT)
        if v is not None:
            try:
                st.set("dome.intensity", float(v))
            except (TypeError, ValueError):
                pass
    for group, lights in (rig.get("groups") or {}).items():
        factors: List[float] = []
        for lt in lights:
            name = _light_name(lt)
            try:
                base = float(baselines[name]) if name in baselines else 1.0
            except (TypeError, ValueError):
                base = 1.0
            if name in baselines and base == 0:
                # authored-off light: writes pin it at 0 × factor = 0, so it carries no
                # dimmer signal — read it as 0, not against a phantom 1.0 baseline
                factors.append(0.0)
                continue
            v = sc.get_prop(lt, sc.LIGHT_MULT, base)
            try:
                factors.append(float(v) / base)   # base is never 0 here (handled above)
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


def _apply_inner(rig, baselines, state: LightingState, camera, warnings: List[str]) -> None:
    sun = rig.get("sun")
    if sun is not None:
        if "sun.enabled" in state.values:
            if sc.set_prop(sun, sc.LIGHT_ON, bool(state.get("sun.enabled") >= 0.5)) is None:
                warnings.append("sun on/off property not found")
        if "sun.azimuth_deg" in state.values or "sun.altitude_deg" in state.values:
            az = state.get("sun.azimuth_deg", sc.read_sun_angles(sun)[0])
            alt = state.get("sun.altitude_deg", sc.read_sun_angles(sun)[1])
            if not sc.write_sun_angles(sun, az, alt):
                warnings.append("could not move the sun (controller-locked transform?)")
        for key, props in (("sun.intensity", sc.SUN_INTENSITY),
                           ("sun.size", sc.SUN_SIZE),
                           ("sun.turbidity", sc.SUN_TURBIDITY)):
            if key in state.values:
                if sc.set_prop(sun, props, float(state.get(key))) is None:
                    warnings.append(f"{key}: no matching property on VRaySun")
    elif any(k.startswith("sun.") for k in state.values):
        warnings.append("state has sun.* but the rig has no VRaySun")

    dome = rig.get("dome")
    if dome is not None:
        if "dome.enabled" in state.values:
            sc.set_prop(dome, sc.LIGHT_ON, bool(state.get("dome.enabled") >= 0.5))
        if "dome.intensity" in state.values:
            if sc.set_prop(dome, sc.LIGHT_MULT, float(state.get("dome.intensity"))) is None:
                warnings.append("dome.intensity: no multiplier property")
        if "dome.rotation_deg" in state.values:
            how = sc.write_dome_rotation(dome, state.get("dome.rotation_deg"))
            if how == "failed":
                warnings.append("dome.rotation_deg: could not rotate texmap or node")
    elif any(k.startswith("dome.") for k in state.values):
        # same contract as the sun branch — an orphaned saved state (dome deleted
        # or replaced since the match) must never be dropped silently
        warnings.append("state has dome.* but the rig has no dome light")

    rig_groups = rig.get("groups") or {}
    for group, factor in state.groups.items():
        lights = rig_groups.get(group, [])
        if not lights:
            warnings.append(f"group.{group}: no such light group in this rig "
                            "(layer renamed or lights removed?) — value dropped")
            continue
        lit = 0
        for lt in lights:
            if bool(sc.get_prop(lt, sc.LIGHT_ON)):
                lit += 1
            base = baselines.get(_light_name(lt), 1.0)
            if sc.set_prop(lt, sc.LIGHT_MULT, float(base) * float(factor)) is None:
                warnings.append(f"group.{group}: light '{getattr(lt, 'name', '?')}' "
                                "has no multiplier")
        if lit == 0 and float(factor) > 1e-6:
            warnings.append(f"group.{group}: all {len(lights)} light(s) are DISABLED — "
                            "the dimmer moves nothing until they're switched on")

    host = ExposureHost(camera)
    if "exposure.ev" in state.values:
        if not host.write_ev(state.get("exposure.ev")):
            warnings.append(f"exposure.ev: no writable exposure host (kind={host.kind})")
    if "exposure.wb_kelvin" in state.values:
        if not host.write_wb_kelvin(state.get("exposure.wb_kelvin")):
            warnings.append("exposure.wb_kelvin: no writable WB property")
