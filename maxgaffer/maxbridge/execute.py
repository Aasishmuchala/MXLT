"""Plan execution — validated ops → scene, with BEFORE/AFTER capture for the report popup.

One undo record for the whole plan. Every op is fault-isolated: a failed set records a
warning and the plan continues. Created lights are always MG_-prefixed and placed on the
"MG_lights" layer, so a whole session's additions can be selected, dimmed, or deleted as
one board. Placement is resolved HERE from real camera geometry — the model only ever
supplied bearing plus physical distance/height (or legacy raw-unit values), or a node name.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from . import scene as sc
from .digest import camera_basis


def _rt():
    import pymxs

    return pymxs.runtime


def _resolve_target(target: str):
    rt = _rt()
    try:
        if target == "renderer":
            return rt.renderers.current
        if target == "environment":
            return rt.environmentMap
        if target == "exposure":
            return rt.SceneExposureControl.exposureControl
        if target.startswith("node:"):
            return rt.getNodeByName(target[len("node:"):], exact=True)
    except Exception:
        return None
    return None


def _coerce(current, value):
    """Match the incoming JSON value to the property's current MAXScript type. Rejects
    non-finite floats (a NaN/±inf written into a live light silently poisons renders) and
    maps bool-ish strings explicitly — ``bool("false") is True`` must never happen."""
    rt = _rt()
    if isinstance(value, (list, tuple)) and len(value) == 3:
        comps = [float(v) for v in value]
        if not all(math.isfinite(c) for c in comps):
            raise ValueError(f"non-finite component in {value!r}")
        try:
            if current is not None and rt.classOf(current) == rt.Point3:
                return rt.Point3(comps[0], comps[1], comps[2])
        except Exception:
            pass
        return rt.color(comps[0], comps[1], comps[2])
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value {value!r}")
    if isinstance(current, bool) or isinstance(value, bool):
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("true", "1", "yes", "on"):
                return True
            if low in ("false", "0", "no", "off"):
                return False
            raise ValueError(f"cannot coerce string {value!r} to bool")
        return bool(value)
    if isinstance(current, int) and isinstance(value, (int, float)):
        return int(value)
    return value


def _read(obj, prop: str):
    try:
        v = getattr(obj, prop)
        try:
            rt = _rt()
            if rt.classOf(v) == rt.Color:
                return [round(float(v.r), 1), round(float(v.g), 1), round(float(v.b), 1)]
        except Exception:
            pass
        if isinstance(v, (bool, int, float, str)):
            return v
        return str(v)[:60]
    except Exception:
        return None


LIGHT_MAKERS = {
    "VRayLight_plane": ("VRayLight", {"type": 0}),
    "VRayLight_dome": ("VRayLight", {"type": 1}),
    "VRayLight_sphere": ("VRayLight", {"type": 2}),
    "VRayLight_disc": ("VRayLight", {"type": 4}),
    "VRaySun": ("VRaySun", {}),
    "VRayIES": ("VRayIES", {}),
}


def _world_units(metres: float) -> float:
    """Convert physical metres through Max's current system-unit settings."""
    return sc.world_units(metres)


def _place_from(basis: Dict[str, Any], placement: Dict[str, Any]):
    rt = _rt()
    if "at_node" in placement:
        node = rt.getNodeByName(placement["at_node"], exact=True)
        if node is not None:
            p = node.pos
            return rt.Point3(float(p.x), float(p.y), float(p.z) + _world_units(0.5))
        return rt.Point3(0.0, 0.0, _world_units(1.0))
    yaw = math.radians(basis["yaw_deg"] + float(placement.get("bearing_deg", 0.0)))
    if "distance_m" in placement:
        dist = _world_units(float(placement.get("distance_m", 2.0)))
    elif "distance" in placement:
        dist = float(placement["distance"])
    else:
        dist = _world_units(2.0)
    if "height_m" in placement:
        height = _world_units(float(placement.get("height_m", 0.0)))
    else:
        height = float(placement.get("height", 0.0))
    cx, cy, cz = basis["pos"]
    return rt.Point3(cx + math.sin(yaw) * dist, cy + math.cos(yaw) * dist,
                     cz + height)


def _ensure_mg_layer():
    rt = _rt()
    try:
        layer = rt.LayerManager.getLayerFromName("MG_lights")
        if layer is None:
            layer = rt.LayerManager.newLayerFromName("MG_lights")
        return layer
    except Exception:
        return None


def execute_plan(ops: List[Dict], camera=None) -> Dict[str, Any]:
    """→ report {"changes": [{target,prop,before,after,why}], "created": [{name,type,at}],
    "warnings": [str]} — the popup renders exactly this."""
    import pymxs

    report: Dict[str, Any] = {"changes": [], "created": [], "warnings": []}
    basis = camera_basis(camera) if camera is not None else None
    rt = _rt()
    with pymxs.undo(True, "MaxGaffer plan"):
        for op in ops:
            try:
                if op["op"] == "set":
                    obj = _resolve_target(op["target"])
                    if obj is None:
                        report["warnings"].append(f"{op['target']}: target vanished")
                        continue
                    before = _read(obj, op["prop"])
                    try:
                        setattr(obj, op["prop"], _coerce(getattr(obj, op["prop"], None),
                                                         op["value"]))
                    except Exception as e:  # noqa: BLE001
                        report["warnings"].append(
                            f"{op['target']}.{op['prop']}: set failed ({e})")
                        continue
                    report["changes"].append({
                        "target": op["target"], "prop": op["prop"],
                        "before": before, "after": _read(obj, op["prop"]),
                        "why": op.get("why", "")})
                elif op["op"] == "create_light":
                    maker, presets = LIGHT_MAKERS[op["light_type"]]
                    # validate BEFORE creating anything — a malformed op must not leak
                    # an untracked orphan light into the scene
                    name = op["name"]
                    placement = op["placement"]
                    try:
                        node = getattr(rt, maker)()
                    except Exception as e:  # noqa: BLE001
                        report["warnings"].append(
                            f"create {op['light_type']}: class unavailable ({e})")
                        continue
                    # the ctor already put a node in the scene — any failure between
                    # here and the layer-add must delete it, or it leaks off MG_lights
                    tgt = None
                    try:
                        node.name = name
                        for k, v in presets.items():
                            sc.set_prop(node, (k,), v)
                        if basis is not None or "at_node" in placement:
                            try:
                                node.pos = _place_from(basis or {"pos": [0, 0, 0],
                                                                 "yaw_deg": 0.0,
                                                                 "look": [0, 200, 0]},
                                                       placement)
                            except Exception:
                                report["warnings"].append(f"{name}: placement failed")
                        if op["light_type"] == "VRaySun" and basis is not None:
                            # a targetless scripted VRaySun aims straight down — give it
                            # a target at the camera's subject so its direction is real
                            try:
                                tgt = rt.Targetobject()
                                lx, ly, lz = basis["look"]
                                tgt.pos = rt.Point3(lx, ly, lz)
                                tgt.name = name + "_target"
                                node.target = tgt
                            except Exception:
                                if tgt is not None:     # don't leak the named helper
                                    try:
                                        rt.delete(tgt)
                                    except Exception:
                                        pass
                                    tgt = None
                                report["warnings"].append(
                                    f"{name}: could not create a sun target")
                        if op.get("aim_at_camera_target") and basis is not None:
                            try:
                                lx, ly, lz = basis["look"]
                                p = node.pos
                                d = rt.Point3(lx - float(p.x), ly - float(p.y),
                                              lz - float(p.z))
                                node.dir = d
                            except Exception:
                                report["warnings"].append(f"{name}: aim failed")
                        for prop, value in (op.get("props") or {}).items():
                            try:
                                setattr(node, prop,
                                        _coerce(getattr(node, prop, None), value))
                            except Exception:
                                report["warnings"].append(
                                    f"{name}.{prop}: not settable on "
                                    f"{op['light_type']}")
                        layer = _ensure_mg_layer()
                        if layer is not None:
                            for n in (node, tgt):       # sun target lives on-layer too
                                if n is None:
                                    continue
                                try:
                                    layer.addNode(n)
                                except Exception:
                                    pass
                        where = placement.get("at_node")
                        if not where:
                            if "distance_m" in placement:
                                where = (f"{placement.get('bearing_deg', 0):+.0f}° / "
                                         f"{placement.get('distance_m', 0):.2f}m / "
                                         f"h{placement.get('height_m', 0):+.2f}m")
                            else:
                                where = (f"{placement.get('bearing_deg', 0):+.0f}° / "
                                         f"{placement.get('distance', 0):.0f}u / "
                                         f"h{placement.get('height', 0):+.0f}u")
                        report["created"].append({"name": name,
                                                  "type": op["light_type"], "at": where,
                                                  "why": op.get("why", "")})
                    except Exception as e:  # noqa: BLE001 — roll back the orphan node
                        for n in (node, tgt):
                            try:
                                if n is not None:
                                    rt.delete(n)
                            except Exception:
                                pass
                        report["warnings"].append(
                            f"create {op.get('name', '?')}: failed — node removed ({e})")
            except Exception as e:  # noqa: BLE001 one op must never kill the plan
                report["warnings"].append(f"op failed: {e}")
    try:
        rt.redrawViews()
    except Exception:
        pass
    return report
