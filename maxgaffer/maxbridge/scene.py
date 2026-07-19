"""Scene introspection — cameras and the lighting rig, read defensively.

Property names across V-Ray builds drift, so every read/write goes through CANDIDATES lists
(first property that exists wins) and the classifier records what it could not find in
``rig["notes"]`` instead of raising. The README's on-box checklist walks these exact names.

Conventions shared with core:
  * camera yaw / sun azimuth: world compass bearing, 0 = +Y, clockwise, degrees;
  * sun altitude: degrees above horizon;
  * Max cameras look down their local -Z axis.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def _rt():
    import pymxs

    return pymxs.runtime


def _warn(msg: str) -> None:
    """Bridge defects must be LOUD (Max listener) but never fatal."""
    try:
        print("[MaxGaffer] scene: " + msg)
    except Exception:
        pass


# ------------------------------------------------------------------ candidates plumbing
def get_prop(obj, names: Tuple[str, ...], default=None):
    rt = _rt()
    for n in names:
        try:
            if rt.isProperty(obj, rt.Name(n)):
                return getattr(obj, n)
        except Exception:
            continue
    return default


def set_prop(obj, names: Tuple[str, ...], value) -> Optional[str]:
    """Set the first existing property; returns the name used (None = none existed)."""
    rt = _rt()
    for n in names:
        try:
            if rt.isProperty(obj, rt.Name(n)):
                setattr(obj, n, value)
                return n
        except Exception:
            continue
    return None


SUN_INTENSITY = ("intensity_multiplier", "intensityMultiplier", "intensity")
SUN_SIZE = ("size_multiplier", "sizeMultiplier")
SUN_TURBIDITY = ("turbidity",)
LIGHT_ON = ("enabled", "on")
LIGHT_MULT = ("multiplier", "intensity")
DOME_TEX_ROT = ("horizontalRotation", "horizontal_rotation", "hRot", "tex_hrotation")
DOME_TEX_FILE = ("HDRIMapName", "fileName", "filename", "bitmap_filename")
DOME_TEX_ON = ("texmap_on", "useTexmap", "use_texture")


# ------------------------------------------------------------------ cameras
def _class_name(obj) -> str:
    try:
        return str(_rt().classOf(obj))
    except Exception:
        return ""


def _node_name(obj) -> str:
    try:
        return str(obj.name)
    except Exception:
        return "<unreadable>"


def camera_yaw_deg(cam) -> float:
    """World compass bearing of the camera's look direction (0 = +Y, clockwise)."""
    try:
        row3 = cam.transform.row3            # local Z axis in world space
        dx, dy = -float(row3.x), -float(row3.y)   # look = -Z
    except (AttributeError, RuntimeError) as e:
        # stale node handle / deleted camera — NOT the same as a zero-length axis
        _warn(f"camera_yaw_deg: transform unreadable ({e}) — yaw guesses will use 0°")
        return 0.0
    except (TypeError, ValueError) as e:
        # transform read fine but the math rejected it (corrupt TM values)
        _warn(f"camera_yaw_deg: bad transform values ({e}) — yaw guesses will use 0°")
        return 0.0
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(dx, dy)) % 360.0


_DUPE_WARNED: set = set()          # duplicate names already shouted about this session


def list_cameras() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        rt = _rt()   # inside the guard — query functions must degrade, never raise
        for o in rt.cameras:                  # cameras collection excludes targets? — filter anyway
            cname = _class_name(o)
            if "target" in cname.lower() and "camera" not in cname.lower():
                continue                       # Targetobject helpers
            try:
                out.append({"name": str(o.name), "class": cname,
                            "yaw_deg": camera_yaw_deg(o)})
            except Exception:
                continue
    except Exception:
        pass
    # Max permits duplicate node names (merged scenes); get_camera resolves the FIRST in
    # collection order, so flag every entry whose name is not unique.
    counts: Dict[str, int] = {}
    for c in out:
        counts[c["name"]] = counts.get(c["name"], 0) + 1
    for c in out:
        if counts[c["name"]] > 1:
            c["duplicate"] = True
            if c["name"] not in _DUPE_WARNED:
                _DUPE_WARNED.add(c["name"])
                _warn(f"{counts[c['name']]} cameras named '{c['name']}' — "
                      "MaxGaffer always uses the first in scene order")
    return out


def get_camera(name: str):
    """Resolve a camera by name. With duplicate names, getNodeByName's pick is
    unspecified — walk the same collection list_cameras() shows and take the FIRST match,
    so the node acted on is always the first one the board lists."""
    try:
        rt = _rt()
    except Exception:
        return None
    try:
        for o in rt.cameras:
            cname = _class_name(o)
            if "target" in cname.lower() and "camera" not in cname.lower():
                continue
            try:
                if str(o.name) == name:
                    return o
            except Exception:
                continue
    except Exception:
        pass
    try:
        return rt.getNodeByName(name, exact=True)   # legacy fallback (non-camera nodes)
    except Exception:
        return None


def set_active_camera(name: str) -> bool:
    cam = get_camera(name)
    if cam is None:
        return False
    try:
        rt = _rt()
        rt.viewport.setCamera(cam)
        rt.redrawViews()
        return True
    except Exception:
        return False


def scene_path() -> str:
    try:
        rt = _rt()
        p = str(rt.maxFilePath or "")
        n = str(rt.maxFileName or "")
        return (p + n) if (p and n) else ""
    except Exception:
        return ""


# every unsaved scene stringifies to "" — a path-keyed cache can't tell scene A
# (unsaved) from scene B (a different, also-unsaved scene) across File > New/Reset,
# so cached sessions could silently apply one scene's lighting in another. The
# generation counter, bumped by Max's file callbacks, is the missing cache key.
_generation = 0


def scene_generation() -> int:
    return _generation


def _bump_generation() -> None:
    global _generation
    _generation += 1


def register_scene_callbacks() -> None:
    """Idempotent: file new/reset/open bump the generation counter (no-op off-Max)."""
    try:
        rt = _rt()
        rt.callbacks.removeScripts(id=rt.Name("MaxGafferSceneGen"))
        code = ('python.Execute "import maxgaffer.maxbridge.scene as _mgs; '
                '_mgs._bump_generation()"')
        for ev in ("systemPostNew", "systemPostReset", "filePostOpen"):
            rt.callbacks.addScript(rt.Name(ev), code, id=rt.Name("MaxGafferSceneGen"))
    except Exception:
        pass


# ------------------------------------------------------------------ rig classification
def _is_class(obj, class_names: Tuple[str, ...]) -> bool:
    return _class_name(obj).lower() in tuple(c.lower() for c in class_names)


def _dome_type_value(light) -> Optional[int]:
    v = get_prop(light, ("type",))
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def classify_rig() -> Dict[str, Any]:
    """→ {"sun": node|None, "dome": node|None, "sky_env": bool,
          "groups": {name: [nodes]}, "notes": [str]}

    Groups = every non-dome VRayLight (and VRayIES), keyed by the light's LAYER name —
    archviz scenes organize practicals by layer, which makes layers the natural dimmer
    boards. Lights on the default layer land in group "practicals".
    """
    rt = _rt()
    rig: Dict[str, Any] = {"sun": None, "dome": None, "sky_env": False,
                           "groups": {}, "notes": []}
    try:
        lights = list(rt.lights)
    except Exception:
        lights = []
    for lt in lights:
        cname = _class_name(lt).lower()
        if cname == "targetobject":
            continue    # the lights collection yields light TARGETS too — same gotcha as cameras
        if cname == "vraysun":
            if rig["sun"] is None:
                rig["sun"] = lt
                try:
                    tgt = getattr(lt, "target", None)
                    # a deleted target reads back as a dead-node wrapper, not None
                    if tgt is None or not rt.isValidNode(tgt):
                        rig["notes"].append(
                            f"sun '{lt.name}' has no target — an untargeted VRaySun aims "
                            "by node rotation, so azimuth/altitude writes will not re-aim it")
                except Exception:
                    pass
                try:
                    ctrl = str(rt.classOf(lt.transform.controller)).lower()
                    if "position" not in ctrl and "prs" not in ctrl:
                        rig["notes"].append(
                            f"sun '{lt.name}' transform controller is {ctrl} — a Daylight "
                            "assembly may fight MaxGaffer's sun moves")
                except Exception:
                    pass
            else:
                rig["notes"].append(f"extra VRaySun '{_node_name(lt)}' ignored (first one wins)")
        elif cname == "vraylight":
            # dome detection: V-Ray light .type — dome is expected to be 1 (VERIFY ON BOX)
            if _dome_type_value(lt) == 1 and rig["dome"] is None:
                rig["dome"] = lt
            elif _dome_type_value(lt) == 1:
                rig["notes"].append(f"extra dome '{_node_name(lt)}' ignored")
            else:
                _add_group_light(rig, lt)
        elif cname in ("vrayies", "vrayambientlight"):
            _add_group_light(rig, lt)
        elif cname in ("free_light", "target_light", "photometriclight",
                       "omnilight", "targetspot", "freespot",
                       "directionallight", "targetdirectionallight"):
            # photometric + standard lights join the dimmer boards too — LIGHT_MULT
            # candidates cover both conventions (.multiplier / .intensity)
            _add_group_light(rig, lt)
    try:
        env = rt.environmentMap
        rig["sky_env"] = env is not None and "vraysky" in _class_name(env).lower()
    except Exception:
        pass
    if rig["sun"] is None:
        rig["notes"].append("no VRaySun found — sun.* parameters disabled")
    if rig["dome"] is None:
        rig["notes"].append("no VRayLight dome found — dome.* parameters disabled")
    return rig


def _add_group_light(rig: Dict[str, Any], lt) -> None:
    try:
        layer = str(lt.layer.name)
    except Exception:
        layer = "0"
    group = "practicals" if layer in ("0", "") else layer
    rig["groups"].setdefault(group, []).append(lt)


# ------------------------------------------------------------------ sun geometry
def _sun_pivot(sun):
    rt = _rt()
    try:
        tgt = getattr(sun, "target", None)
        if tgt is not None:
            return tgt.pos
    except Exception:
        pass
    return rt.Point3(0.0, 0.0, 0.0)


def sun_readable(sun) -> bool:
    """False when the sun handle is stale (deleted mid-session while the rig is cached).
    Callers should skip sun math rather than act on placeholder angles."""
    try:
        sun.pos
        return True
    except Exception:
        return False


def read_sun_angles(sun) -> Tuple[float, float, float]:
    """→ (azimuth_deg, altitude_deg, distance). Direction FROM pivot TO sun position."""
    pivot = _sun_pivot(sun)
    try:
        d = sun.pos - pivot
        dx, dy, dz = float(d.x), float(d.y), float(d.z)
    except Exception as e:
        # LOUD placeholder: callers keep their tuple contract, but this must never be
        # mistaken for real geometry (write_sun_angles refuses to act on it)
        _warn(f"read_sun_angles: sun position unreadable ({e}) — returning placeholder "
              "angles; do not trust them")
        return 0.0, 45.0, 10000.0
    dist = math.sqrt(dx * dx + dy * dy + dz * dz) or 10000.0
    horiz = math.sqrt(dx * dx + dy * dy)
    altitude = math.degrees(math.atan2(dz, horiz))
    azimuth = math.degrees(math.atan2(dx, dy)) % 360.0 if horiz > 1e-6 else 0.0
    return azimuth, altitude, dist


def write_sun_angles(sun, azimuth_deg: float, altitude_deg: float) -> bool:
    rt = _rt()
    if not sun_readable(sun):
        # a stale handle would silently orbit the WORLD ORIGIN at 10000 units off the
        # read_sun_angles placeholder — refuse to mutate instead
        _warn("write_sun_angles: sun unreadable (stale handle?) — sun NOT moved")
        return False
    pivot = _sun_pivot(sun)
    _, _, dist = read_sun_angles(sun)
    az, alt = math.radians(azimuth_deg), math.radians(altitude_deg)
    try:
        sun.pos = rt.Point3(
            float(pivot.x) + dist * math.sin(az) * math.cos(alt),
            float(pivot.y) + dist * math.cos(az) * math.cos(alt),
            float(pivot.z) + dist * math.sin(alt),
        )
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ dome rotation
def read_dome_rotation(dome) -> float:
    tex = get_prop(dome, ("texmap",))
    if tex is not None:
        v = get_prop(tex, DOME_TEX_ROT)
        if v is not None:
            try:
                return float(v) % 360.0
            except (TypeError, ValueError):
                pass
    try:  # fall back to the node's world-Z euler
        rt = _rt()
        eul = rt.quatToEuler(dome.transform.rotationpart)
        return float(eul.z) % 360.0
    except Exception:
        return 0.0


def write_dome_rotation(dome, degrees_: float) -> str:
    """Prefer the HDRI texmap's horizontal-rotation spinner; fall back to spinning the node
    about WORLD Z at its own pivot. Returns which path was used (log/verify checklist).

    The fallback composes W' = W · T(−p) · Rz(Δ) · T(p) explicitly instead of rt.rotate(),
    whose working coordsys is context-dependent — a dome that a previous artist tilted
    would otherwise spin around the wrong axis."""
    tex = get_prop(dome, ("texmap",))
    if tex is not None:
        used = set_prop(tex, DOME_TEX_ROT, float(degrees_ % 360.0))
        if used:
            return f"texmap.{used}"
    try:
        rt = _rt()
        current = read_dome_rotation(dome)
        delta = (degrees_ - current + 180.0) % 360.0 - 180.0
        tm = dome.transform
        p = tm.translationpart
        neg_p = rt.Point3(-float(p.x), -float(p.y), -float(p.z))
        dome.transform = (tm * rt.transMatrix(neg_p)
                          * rt.rotateZMatrix(float(delta)) * rt.transMatrix(p))
        return "node_z_rotation"
    except Exception:
        return "failed"


def get_dome_texture(dome) -> str:
    """Current HDRI file path on the dome's texmap ("" = no texmap / no file)."""
    tex = get_prop(dome, ("texmap",))
    if tex is None:
        return ""
    v = get_prop(tex, DOME_TEX_FILE)
    return str(v) if v else ""


def set_dome_texture(dome, hdri_path: str) -> str:
    """Point the dome light at an HDRI file, creating a VRayBitmap texmap if the dome has
    none. Returns how it was done ('failed' = nothing writable found — and NOTHING was
    changed: a fresh texmap gets its file set FIRST and is only bound on success, so a
    failure never leaves an empty texmap blacking out the dome)."""
    rt = _rt()
    tex = get_prop(dome, ("texmap",))
    created = False
    if tex is None:
        for maker in ("VRayBitmap", "VRayHDRI"):
            try:
                tex = getattr(rt, maker)()
                break
            except Exception:
                tex = None
        if tex is None:
            return "failed"
        created = True
    used = set_prop(tex, DOME_TEX_FILE, hdri_path)
    if used is None:
        return "failed"                     # created texmap was never bound — clean
    if created:
        try:
            dome.texmap = tex
        except Exception:
            return "failed"                 # dome.texmap untouched — still clean
    set_prop(dome, DOME_TEX_ON, True)   # best-effort; missing prop is fine
    return f"texmap.{used}"
