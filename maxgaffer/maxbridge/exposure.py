"""Exposure host abstraction — one EV + one WB-kelvin, whatever the scene actually uses.

Property names verified against docs 2026-07-16 (Autodesk MAXScript help "Physical :
Camera"; Chaos VRayExposureControl page + forums):

  Max Physical Camera (native):
    exposure_gain_type   0 = Manual(ISO) · 1 = Target EV (the DEFAULT)
    exposure_value       direct target EV — the clean write path
    iso · f_number · shutter_length_seconds (SECONDS; shutter_unit_type selects units)
    white_balance_type   0 = Illuminant · 1 = Temperature · 2 = Custom
    white_balance_kelvin · white_balance_custom

  V-Ray exposure control (scene-level):
    created via the documented global  vrayCreateVRayExposureControl()  and assigned to
    SceneExposureControl.exposureControl; its own property spellings remain
    candidates-based (checklist #4). Requires "Use 3ds Max photometric scale" in V-Ray
    global switches.

  Legacy VRayPhysicalCamera: ISO / f_number / shutter_speed (a SPEED, 1/s — note the unit
  difference vs the native camera's shutter_length_seconds; handled per-property below).

Host priority: scene V-Ray exposure control → active Physical/VRayPhysical camera → none
(params absent from genome → UI auto-locks). EV convention: HIGHER = DARKER.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..core.colortemp import wb_color_for_kelvin
from .scene import get_prop, set_prop

EC_EV = ("ev", "EV", "exposure_value")
EC_MODE = ("mode", "exposure_mode")
EC_WB_MODE = ("whitebalance_mode", "wb_mode", "white_balance_preset")
EC_WB_KELVIN = ("temperature", "whitebalance_temperature", "wb_temperature")
EC_WB_COLOR = ("whitebalance", "white_balance", "wb_color")

CAM_EV = ("exposure_value",)                    # native Physical, gain type 1 (verified)
CAM_EV_TYPE = ("exposure_gain_type",)           # 0 Manual · 1 Target (verified)
CAM_ISO = ("iso", "ISO", "film_speed")
CAM_FNUM = ("f_number", "fnumber", "f_stop")    # f_number verified on native Physical
CAM_SHUTTER_SECONDS = ("shutter_length_seconds",)   # native Physical (verified, seconds)
CAM_SHUTTER_SPEED = ("shutter_speed",)              # legacy VRayPhysical (1/s)
CAM_WB_KELVIN = ("white_balance_kelvin", "temperature", "whiteBalance_temperature")
CAM_WB_COLOR = ("white_balance_custom", "whiteBalance", "wb_color")
CAM_WB_TYPE = ("white_balance_type", "whiteBalance_mode", "wb_mode")
WB_TYPE_TEMPERATURE = 1                          # verified enum on native Physical
WB_TYPE_CUSTOM = 2


def _rt():
    import pymxs

    return pymxs.runtime


def _scene_exposure_control():
    """Whatever is assigned to the scene exposure slot, ANY class (native
    Photographic/Logarithmic included), None when the slot is truly empty."""
    try:                # _rt() inside the try — ExposureHost must be constructible off-Max
        return _rt().SceneExposureControl.exposureControl
    except Exception:
        return None


def _find_exposure_control():
    """The scene's V-RAY exposure control, or None — MaxGaffer only drives V-Ray's.
    Class names carry underscores on real boxes (the renderer is
    V_Ray_GPU_7__update_2_hotfix_2) — normalize before matching."""
    ec = _scene_exposure_control()
    if ec is not None:
        try:
            if "vray" in str(_rt().classOf(ec)).lower().replace("_", ""):
                return ec
        except Exception:
            # Deleted/hostile controls can remain referenced by the scene slot.  Treat
            # them as unusable instead of breaking every state read/apply.
            return None
    return None


def ensure_exposure_control(camera=None) -> Optional[str]:
    """Create + assign an exposure host when the scene slot is EMPTY.
    An existing NON-V-Ray control is the artist's own exposure setup — it is never
    clobbered; we say so and let the physical-camera host (or an auto-lock) take over.
    The creation is wrapped in its own undo record so it is reversible (apply_state's
    record doesn't cover this call). Returns a log line, or None if a V-Ray EC already
    exists or creation isn't available on this build.

    Native Physical camera → Max's own Physical_Camera_Exposure_Control: measured
    on-box 2026-07-16, V-Ray honors the camera's EV AND WB only through it (the VRay
    EC never applies its WB, and its EV only registers in mode 106 on an offset
    scale). The host resolver then routes writes to the camera — the doc-verified
    ``exposure_value`` / ``white_balance_kelvin`` path."""
    if _find_exposure_control() is not None:
        return None
    existing = _scene_exposure_control()
    if existing is not None:
        try:
            cls = str(_rt().classOf(existing))
        except Exception:
            cls = "unknown class"
        return ("⚠ scene already has a non-V-Ray exposure control "
                f"({cls}) — leaving it untouched; exposure falls through to the "
                "physical-camera host (or stays locked)")
    import pymxs

    rt = _rt()
    try:
        if (camera is not None
                and "physical" in str(rt.classOf(camera)).lower()
                and get_prop(camera, CAM_EV) is not None):
            with pymxs.undo(True, "MaxGaffer exposure control"):
                rt.SceneExposureControl.exposureControl = rt.Physical_Camera_Exposure_Control()
            return ("assigned the Physical Camera Exposure Control — EV/WB now drive "
                    "through the active Physical camera (scene had no exposure host)")
    except Exception:
        pass
    for fn in ("vrayCreateVRayExposureControl",):
        try:
            with pymxs.undo(True, "MaxGaffer exposure control"):
                ec = getattr(rt, fn)()
                rt.SceneExposureControl.exposureControl = ec
                # mode 106 = "from EV" — measured on-box 2026-07-16: .ev is silently
                # IGNORED in the default mode (107, from-camera with no camera set)
                set_prop(ec, EC_MODE, 106)
            return ("created a V-Ray exposure control (scene exposure slot was empty; "
                    "undo-safe) — requires 'Use 3ds Max photometric scale' in V-Ray "
                    "global switches")
        except Exception:
            continue
    return None


def shutter_seconds(prop_name: str, value: float) -> float:
    """Normalize a shutter property to SECONDS — the native camera stores a duration,
    the legacy VRayPhysicalCamera stores a speed (1/s). Pure; unit-tested."""
    v = max(1e-6, float(value))
    if "speed" in prop_name.lower():
        return 1.0 / v
    return v


class ExposureHost:
    """Resolved once per apply/read; ``kind`` ∈ exposure_control | physical_cam | none."""

    def __init__(self, camera=None):
        self.ec = _find_exposure_control()
        self.cam = None
        self.kind = "none"
        if camera is None:
            try:
                # camera-less callers (UI sliders, probes): the active viewport
                # camera is the artist's context — required since the native
                # Physical-camera host carries EV/WB on the camera itself
                camera = _rt().viewport.getCamera()
            except Exception:
                camera = None
        if self.ec is not None and get_prop(self.ec, EC_EV) is not None:
            self.kind = "exposure_control"
        elif camera is not None:
            cname = ""
            try:
                cname = str(_rt().classOf(camera)).lower()
            except Exception:
                pass
            if "physical" in cname and (get_prop(camera, CAM_EV) is not None
                                        or get_prop(camera, CAM_ISO) is not None):
                self.cam = camera
                self.kind = "physical_cam"

    # ------------------------------------------------------------------ EV
    def read_ev(self) -> Optional[float]:
        if self.kind == "exposure_control":
            try:
                return float(get_prop(self.ec, EC_EV))
            except (TypeError, ValueError):
                return None
        if self.kind == "physical_cam":
            # native Physical in Target mode: exposure_value IS the EV (verified)
            gain_type = get_prop(self.cam, CAM_EV_TYPE)
            ev_direct = get_prop(self.cam, CAM_EV)
            try:
                target_mode = gain_type is None or int(gain_type) == 1
            except (TypeError, ValueError):
                # a non-int enum (broken plugin prop) must not kill read_state —
                # treat as not-Target and fall through to the exposure triangle
                target_mode = False
            if ev_direct is not None and target_mode:
                try:
                    return float(ev_direct)
                except (TypeError, ValueError):
                    pass
            try:  # manual mode / legacy camera: EV100 from the exposure triangle
                iso = float(get_prop(self.cam, CAM_ISO, 100.0))
                n = float(get_prop(self.cam, CAM_FNUM, 8.0))
                t = None
                for props in (CAM_SHUTTER_SECONDS, CAM_SHUTTER_SPEED):
                    for name in props:
                        v = get_prop(self.cam, (name,))
                        if v is not None:
                            t = shutter_seconds(name, v)
                            break
                    if t is not None:
                        break
                if t is None:
                    t = 1.0 / 200.0
                return math.log2((n * n) / t) - math.log2(max(1e-6, iso) / 100.0)
            except Exception:
                return None
        return None

    def write_ev(self, ev: float) -> bool:
        try:
            ev = float(ev)
            if not math.isfinite(ev):
                return False
        except (TypeError, ValueError):
            return False
        if self.kind == "exposure_control":
            # .ev only registers in "from EV" mode (106) — enforce like the camera
            # path enforces gain type below (measured on-box 2026-07-16: default
            # mode 107 silently ignores .ev writes)
            if get_prop(self.ec, EC_MODE) is not None:
                set_prop(self.ec, EC_MODE, 106)
            return set_prop(self.ec, EC_EV, ev) is not None
        if self.kind == "physical_cam":
            # preferred: native Target-EV mode — exact, no side effects on DOF/motion
            if get_prop(self.cam, CAM_EV) is not None:
                if get_prop(self.cam, CAM_EV_TYPE) is not None:
                    set_prop(self.cam, CAM_EV_TYPE, 1)   # 1 = Target (verified enum)
                return set_prop(self.cam, CAM_EV, ev) is not None
            current = self.read_ev()   # legacy fallback: move ISO only
            if current is None:
                return False
            try:
                iso = float(get_prop(self.cam, CAM_ISO, 100.0))
                new_iso = min(51200.0, max(6.0, iso * (2.0 ** (current - ev))))
                return set_prop(self.cam, CAM_ISO, new_iso) is not None
            except Exception:
                return False
        return False

    # ------------------------------------------------------------------ WB
    def read_wb_kelvin(self) -> Optional[float]:
        host = self.ec if self.kind == "exposure_control" else self.cam
        if host is None:
            return None
        v = get_prop(host, EC_WB_KELVIN if self.kind == "exposure_control" else CAM_WB_KELVIN)
        try:
            k = float(v)
            if 1000.0 <= k <= 40000.0:
                return k
        except (TypeError, ValueError):
            pass
        return None

    def write_wb_kelvin(self, kelvin: float) -> bool:
        try:
            kelvin = float(kelvin)
            if not math.isfinite(kelvin):
                return False
        except (TypeError, ValueError):
            return False
        host = self.ec if self.kind == "exposure_control" else self.cam
        if host is None:
            return False
        kelvin_props = EC_WB_KELVIN if self.kind == "exposure_control" else CAM_WB_KELVIN
        mode_props = EC_WB_MODE if self.kind == "exposure_control" else CAM_WB_TYPE
        if get_prop(host, kelvin_props) is not None:
            ok = set_prop(host, kelvin_props, kelvin) is not None
            if ok:
                _set_wb_mode(host, mode_props, WB_TYPE_TEMPERATURE)
            return ok
        # color-swatch-only host: write the illuminant color (same spinner convention)
        color_props = EC_WB_COLOR if self.kind == "exposure_control" else CAM_WB_COLOR
        r, g, b = wb_color_for_kelvin(kelvin)
        try:
            rt = _rt()
            ok = set_prop(host, color_props,
                          rt.color(r * 255.0, g * 255.0, b * 255.0)) is not None
            if ok:
                _set_wb_mode(host, mode_props, WB_TYPE_CUSTOM)
            return ok
        except Exception:
            return False

    def describe(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ev": self.read_ev(), "wb_kelvin": self.read_wb_kelvin()}


def _set_wb_mode(host, mode_props, value: int) -> None:
    """Native Physical enum verified (0 Illuminant · 1 Temperature · 2 Custom); V-Ray EC's
    enum ints remain a checklist item — failure is silent and harmless."""
    try:
        set_prop(host, mode_props, value)
    except Exception:
        pass
