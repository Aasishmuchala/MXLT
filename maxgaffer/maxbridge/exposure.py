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


def _find_exposure_control():
    try:
        rt = _rt()      # inside the try — ExposureHost must be constructible off-Max
        ec = rt.SceneExposureControl.exposureControl
        # class names carry underscores on real boxes (the renderer is
        # V_Ray_GPU_7__update_2_hotfix_2) — normalize before matching
        if ec is not None and "vray" in str(rt.classOf(ec)).lower().replace("_", ""):
            return ec
    except Exception:
        pass
    return None


def ensure_exposure_control() -> Optional[str]:
    """Create + assign a V-Ray exposure control when the scene has none (documented call:
    ``SceneExposureControl.exposureControl = vrayCreateVRayExposureControl()``).
    Returns a log line, or None if creation isn't available on this build."""
    if _find_exposure_control() is not None:
        return None
    rt = _rt()
    for fn in ("vrayCreateVRayExposureControl",):
        try:
            ec = getattr(rt, fn)()
            rt.SceneExposureControl.exposureControl = ec
            return ("created a V-Ray exposure control (scene had no exposure host) — "
                    "requires 'Use 3ds Max photometric scale' in V-Ray global switches")
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
            if ev_direct is not None and (gain_type is None or int(gain_type) == 1):
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
        if self.kind == "exposure_control":
            return set_prop(self.ec, EC_EV, float(ev)) is not None
        if self.kind == "physical_cam":
            # preferred: native Target-EV mode — exact, no side effects on DOF/motion
            if get_prop(self.cam, CAM_EV) is not None:
                if get_prop(self.cam, CAM_EV_TYPE) is not None:
                    set_prop(self.cam, CAM_EV_TYPE, 1)   # 1 = Target (verified enum)
                return set_prop(self.cam, CAM_EV, float(ev)) is not None
            current = self.read_ev()   # legacy fallback: move ISO only
            if current is None:
                return False
            try:
                iso = float(get_prop(self.cam, CAM_ISO, 100.0))
                new_iso = min(51200.0, max(6.0, iso * (2.0 ** (current - float(ev)))))
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
        host = self.ec if self.kind == "exposure_control" else self.cam
        if host is None:
            return False
        kelvin_props = EC_WB_KELVIN if self.kind == "exposure_control" else CAM_WB_KELVIN
        mode_props = EC_WB_MODE if self.kind == "exposure_control" else CAM_WB_TYPE
        if get_prop(host, kelvin_props) is not None:
            ok = set_prop(host, kelvin_props, float(kelvin)) is not None
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
