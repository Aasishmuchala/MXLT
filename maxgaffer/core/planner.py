"""The PLAN stage — scene-wide change plans, grounded in the digest catalog.

Flow: digest (current scene, ALL settings) + reference image + semantics → the model writes
an explicit ChangePlan → validated here → previewed to the human (or auto-approved) →
executed by the bridge with before/after capture → change-report popup.

"No restrictions" is implemented as grounded generality, not blind trust:
  * a `set` op may touch ANY property on ANY target — but only targets and property names
    that actually exist in the digest catalog (hallucinated names are dropped with a note);
  * `create_light` places NEW lights — but placement is camera-relative (bearing/distance/
    height) or at a named node, never raw world coordinates from the model's imagination
    (MaxDirector's "camera in the void" lesson);
  * everything executes inside one undo record, after the pre-match snapshot.

Pure python; unit-tested off-Max.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Sequence, Set, Tuple

from .omega import parse_json_from_text
from .parse import ParseError

MAX_OPS = 12
CREATABLE_LIGHTS = ("VRayLight_plane", "VRayLight_sphere", "VRayLight_disc",
                    "VRayLight_dome", "VRaySun", "VRayIES")
# New plans use physical metres. 3ds Max system units may be millimetres, centimetres,
# inches, feet, or a custom scale, so a bare numeric distance is not portable. Legacy
# ``distance``/``height`` plans remain valid and retain their raw-scene-unit meaning.
PLACEMENT_LIMITS = {"bearing_deg": (-180.0, 180.0), "distance_m": (0.1, 1000.0),
                    "height_m": (-100.0, 1000.0)}
LEGACY_PLACEMENT_LIMITS = {"bearing_deg": (-180.0, 180.0),
                           "distance": (10.0, 100000.0),
                           "height": (-10000.0, 100000.0)}
# "anything related to LIGHT" is the mandate — output paths, DR/net-render and system
# plumbing on the renderer are never light, and a stray write there hurts a client job
RENDERER_DENY_SUBSTR = ("savefilename", "rawfilename", "distributed", "netrender",
                        "autosave", "region", "system_")

PLAN_SYSTEM = """You are a master gaffer with FULL access to a 3ds Max + V-Ray 7 scene. You
will receive the scene digest (current renderer settings, environment, exposure control,
every light with its properties, cameras) and a lighting REFERENCE image. Plan the changes
that make the SCENE'S LIGHT match the REFERENCE.

You may:
- change ANY property listed in the digest on targets: "renderer", "environment",
  "exposure", or "node:<exact name>" — use the exact property spellings from the digest;
- create NEW lights (types: VRayLight_plane, VRayLight_sphere, VRayLight_disc,
  VRayLight_dome, VRaySun, VRayIES) placed RELATIVE TO THE CAMERA — "placement" MUST
  contain ALL THREE numbers: bearing_deg (0 = in front of camera, +right/-left),
  distance_m (physical metres), height_m (physical metres above camera) — OR use
  {"at_node": "<existing name>"}.
  A create_light op missing any placement number is dropped. Set key properties via
  the "props" field.

Hard rules:
- AT MOST 12 operations, highest impact first. Prefer adjusting existing lights over
  creating new ones; create only what the reference clearly needs and the scene lacks.
- VRayIES lights are useless without an .ies profile — only create one if you set its
  file via props, otherwise use a VRayLight_sphere/disc instead.
- Only property names that appear in the digest. Never invent names.
- Values: numbers, true/false, [r,g,b] 0-255 colors, or strings (file paths).
- Do not change resolution, output paths, or anything unrelated to LIGHT.
- Every op carries a short "why" tied to the reference.

Reply with ONLY a JSON object:
{
  "read": "<= 60 words: what the current scene's light is doing vs the reference",
  "ops": [
    {"op": "set", "target": "node:VRaySun001", "prop": "turbidity", "value": 5.0,
     "why": "hazier warm sky per reference"},
    {"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_window_fill",
     "placement": {"bearing_deg": -70, "distance_m": 2.5, "height_m": 1.2},
     "aim_at_camera_target": true,
     "props": {"multiplier": 8.0, "color": [255, 230, 200]},
     "why": "soft warm fill from camera-left as in reference"}
  ],
  "expects": "<= 30 words: what the render should look like after"
}"""


def plan_user_text(digest_text: str, semantics: Dict, camera_name: str) -> str:
    return (f"Active camera: {camera_name}\n\nReference lighting analysis:\n"
            f"{json.dumps(semantics, indent=1)}\n\nCURRENT SCENE DIGEST:\n{digest_text}\n\n"
            "The image attached is the REFERENCE. Write the change plan. "
            "Reply with only the JSON object.")


def _valid_value(v) -> bool:
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float)):
        return math.isfinite(v)             # NaN/±inf must never reach a scene setattr
    if isinstance(v, str):
        return len(v) < 500
    if (isinstance(v, (list, tuple)) and len(v) == 3
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)):
        return True
    return False


def validate_plan(reply_text: str, cat: Dict[str, Set[str]],
                  max_ops: int = MAX_OPS) -> Tuple[List[Dict], List[str], Dict[str, str]]:
    """→ (ops, rejected_notes, meta{read, expects}). Grounding enforced here: unknown
    targets/props/types dropped with a reason, placements clamped, count capped."""
    obj = parse_json_from_text(reply_text)
    if obj is None:
        raise ParseError("plan reply contained no JSON object")
    ops: List[Dict] = []
    rejected: List[str] = []
    raw_ops = obj.get("ops")
    existing_names = {t[len("node:"):] for t in cat if t.startswith("node:")}
    created_names: Set[str] = set()   # lights this plan creates — settable later in-plan
    for item in (raw_ops if isinstance(raw_ops, list) else [])[:max_ops * 2]:
        if len(ops) >= max_ops:
            rejected.append("plan truncated at 12 operations")
            break
        if not isinstance(item, dict):
            continue
        kind = item.get("op")
        why = str(item.get("why") or "")[:160]
        if kind == "set":
            target = item.get("target")
            prop = item.get("prop")
            value = item.get("value")
            on_created = (isinstance(target, str) and target.startswith("node:")
                          and target[len("node:"):] in created_names)
            if not (isinstance(target, str) and (target in cat or on_created)):
                rejected.append(f"set: unknown target {target!r}")
                continue
            if not on_created and not (isinstance(prop, str) and prop in cat[target]):
                rejected.append(f"set {target}: property {prop!r} not in the scene digest")
                continue
            if on_created and not isinstance(prop, str):
                continue
            if target == "renderer" and any(d in prop.lower()
                                            for d in RENDERER_DENY_SUBSTR):
                rejected.append(f"set renderer.{prop}: output/system plumbing is off-limits "
                                "(not light)")
                continue
            if not _valid_value(value):
                rejected.append(f"set {target}.{prop}: unsupported value {value!r}")
                continue
            ops.append({"op": "set", "target": target, "prop": prop,
                        "value": list(value) if isinstance(value, tuple) else value,
                        "why": why})
        elif kind == "create_light":
            ltype = item.get("light_type")
            if ltype not in CREATABLE_LIGHTS:
                rejected.append(f"create_light: type {ltype!r} not allowed")
                continue
            name = str(item.get("name") or f"MG_light_{len(ops)}")[:60]
            if not name.startswith("MG_"):
                name = "MG_" + name
            if name in existing_names:
                rejected.append(f"create_light: name {name!r} already exists")
                continue
            placement = item.get("placement")
            at_node = item.get("at_node")
            if isinstance(at_node, str) and at_node in existing_names:
                place = {"at_node": at_node}
            elif isinstance(placement, dict):
                place = {}
                ok = True
                # Do not accept a mixture of metres and raw scene units: physical
                # fields select the canonical schema, while old plans remain readable.
                physical = any(k in placement for k in ("distance_m", "height_m"))
                limits = PLACEMENT_LIMITS if physical else LEGACY_PLACEMENT_LIMITS
                for k, (lo, hi) in limits.items():
                    try:
                        pv = float(placement.get(k))
                        if not math.isfinite(pv):
                            raise ValueError
                        place[k] = min(hi, max(lo, pv))
                    except (TypeError, ValueError):
                        rejected.append(f"create_light {name}: bad placement.{k}")
                        ok = False
                        break
                if not ok:
                    continue
            else:
                rejected.append(f"create_light {name}: needs placement{{bearing_deg,"
                                "distance_m,height_m}} or at_node")
                continue
            props = item.get("props") if isinstance(item.get("props"), dict) else {}
            clean_props = {str(k): (list(v) if isinstance(v, tuple) else v)
                           for k, v in props.items()
                           if isinstance(k, str) and _valid_value(v)}
            ops.append({"op": "create_light", "light_type": ltype, "name": name,
                        "placement": place,
                        "aim_at_camera_target": bool(item.get("aim_at_camera_target")),
                        "props": clean_props, "why": why})
            existing_names.add(name)
            created_names.add(name)
        else:
            rejected.append(f"unknown op {kind!r}")
    meta = {"read": str(obj.get("read") or "")[:600],
            "expects": str(obj.get("expects") or "")[:300]}
    return ops, rejected, meta


def describe_plan(ops: Sequence[Dict]) -> List[str]:
    """One human line per op — the preview dialog and the log both use this."""
    out: List[str] = []
    for o in ops:
        if o["op"] == "set":
            out.append(f"set  {o['target']} · {o['prop']} → {o['value']}   ({o['why']})")
        else:
            if "at_node" in o["placement"]:
                where = f"at node {o['placement'].get('at_node')}"
            elif "distance_m" in o["placement"]:
                where = (f"bearing {o['placement'].get('bearing_deg', 0):+.0f}° · "
                         f"dist {o['placement'].get('distance_m', 0):.2f}m · "
                         f"h {o['placement'].get('height_m', 0):+.2f}m")
            else:
                where = (f"bearing {o['placement'].get('bearing_deg', 0):+.0f}° · "
                         f"dist {o['placement'].get('distance', 0):.0f}u · "
                         f"h {o['placement'].get('height', 0):+.0f}u")
            out.append(f"new  {o['light_type']} '{o['name']}' {where}   ({o['why']})")
    return out
