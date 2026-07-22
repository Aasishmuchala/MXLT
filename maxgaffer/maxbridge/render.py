"""Rendering + image plumbing on the Max side.

Loop renders are the match loop's heartbeat: small (config.loop_width), VFB off, current
V-Ray settings otherwise — resolution is the honest speed lever that never changes the
lighting character (dialing samplers would). Render size is saved/restored around every
call so the user's render setup is untouched.

Also owns reference transcoding: Max's own bitmap I/O converts a JPEG/EXR/TIFF reference to
a small PNG that the stdlib stats reader can always ingest — the zero-dependency fallback
path when neither Pillow nor a sidecar python exists.
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def _rt():
    import pymxs

    return pymxs.runtime


def render_frame(camera, out_path: str, width: int, height: int) -> Optional[str]:
    """One still through the CURRENT renderer at the given size. Returns path or None.

    A pre-existing target is deleted before rendering, so success can never be a stale
    file from an earlier run; the bitmap is closed in a finally, so a failed ``save``
    leaks no framebuffer."""
    rt = _rt()
    try:
        if os.path.exists(out_path):   # success can never be a stale file from an earlier run
            os.remove(out_path)
    except OSError:
        pass
    old_w, old_h = None, None
    try:
        # inside the try: a path-length/permission failure here must degrade to the
        # same None every other failure mode returns, not raise through the loop
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            rt.renderSceneDialog.close()   # open dialog blocks programmatic size changes
        except Exception:
            pass
        old_w, old_h = int(rt.renderWidth), int(rt.renderHeight)
        try:
            # sky/background pixels carry alpha 0 — saved as-is the LLM sees a
            # transparent (often white-composited) sky. Flatten at the PNG writer.
            rt.pngio.setAlpha(False)
        except Exception:
            pass
        # render()'s size-arg spelling varies across Max releases — a single spelling here
        # would be a single point of failure for EVERY loop render. Layered candidates:
        # outputwidth/height kwargs → outputSize:Point2 → the globals (restored in finally).
        bm = None
        try:
            bm = rt.render(camera=camera, outputwidth=int(width),
                           outputheight=int(height), vfb=False, quiet=True)
        except Exception:
            bm = None
        if bm is None:
            try:
                bm = rt.render(camera=camera,
                               outputSize=rt.Point2(int(width), int(height)),
                               vfb=False, quiet=True)
            except Exception:
                bm = None
        if bm is None:
            rt.renderWidth, rt.renderHeight = int(width), int(height)
            bm = rt.render(camera=camera, vfb=False, quiet=True)
        try:
            if bm is None:
                return None
            bm.filename = out_path
            rt.save(bm)
            return out_path if os.path.exists(out_path) else None
        finally:                       # a throwing save must not leak the framebuffer
            if bm is not None:
                try:
                    rt.close(bm)
                except Exception:
                    pass
    except Exception:
        return None
    finally:
        try:
            if old_w:
                rt.renderWidth, rt.renderHeight = old_w, old_h
        except Exception:
            pass


def transcode_to_png(src_path: str, dst_png: str, max_dim: int = 1024) -> Optional[str]:
    """Any Max-readable image → small PNG via Max bitmap I/O (maxscript ``copy`` rescales
    between differently-sized bitmaps). The universal reference-ingest fallback."""
    rt = _rt()
    src = None
    dst = None
    try:
        # A failed/silent Max save must never make an older file look like a successful
        # decode of the newly selected reference.
        if os.path.exists(dst_png):
            os.remove(dst_png)
        src = rt.openBitMap(src_path)
        if src is None:
            return None
        w, h = int(src.width), int(src.height)
        scale = min(1.0, float(max_dim) / max(w, h, 1))
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        os.makedirs(os.path.dirname(dst_png) or ".", exist_ok=True)
        dst = rt.bitmap(nw, nh, filename=dst_png)
        rt.copy(src, dst)
        rt.save(dst)
        return dst_png if os.path.exists(dst_png) else None
    except Exception:
        return None
    finally:
        for bm in (src, dst):
            try:
                if bm is not None:
                    rt.close(bm)
            except Exception:
                pass


# ColorPipelineMgr (Max 2024+) accessor candidates. Only ``.mode`` is load-bearing — it
# drives the gamma-vs-OCIO classification and its name is doc-stable; the view-transform and
# display-gamma spellings below DRIFT across Max builds and were unverifiable off-box, so
# every read is a best-effort candidate probe that degrades to None (ON-BOX CALIBRATION item).
CS_VIEW_TRANSFORM = ("OCIOViewTransform", "displayViewTransform", "viewTransform",
                     "OCIODisplayView", "displayView", "view_transform")
CS_DISPLAY_GAMMA = ("displayGamma", "gamma", "outputGamma", "dispGamma")
# The active renderer's VFB display-correction intent (V-Ray). Spelling is a V-Ray-build
# calibration item; unknown → None.
VRAY_DISPLAY_CORRECTION = ("output_srgb", "output_gamma", "display_srgb", "srgb_display",
                           "vfb_display_correction", "display_correction")

#: The honest caveat every colorspace report carries: the loop's saved plate is a RAW
#: framebuffer. render_frame's ``rt.save(bm)`` writes the framebuffer as rendered, so a
#: color-managed VFB view (OCIO view transform / GPU sRGB display correction) is a VIEW-only
#: correction that is NOT baked into the saved PNG — the critic scores the raw plate, which
#: can differ from what the artist sees in the VFB whenever a display transform is active.
_COLORSPACE_RISK = (
    "GPU/VFB display transforms (OCIO view, sRGB display correction) are VIEW-only and may "
    "NOT bake into saved PNGs — render_frame's rt.save(bm) writes the raw framebuffer, so a "
    "color-managed VFB view is not reflected in the saved loop image the critic scores; when "
    "color_management is 'ocio'/'aces' the saved plate can differ from the VFB view."
)


def _first_attr(obj, names, coerce=None):
    """First readable attribute in ``names`` (each getattr guarded), optionally passed
    through ``coerce`` (a coercion failure skips to the next candidate). → the value or None.
    Interface spellings drift across Max builds, so this never raises — it degrades to None."""
    if obj is None:
        return None
    for n in names:
        try:
            val = getattr(obj, n)
        except Exception:
            continue
        if val is None:
            continue
        if coerce is None:
            return val
        try:
            return coerce(val)
        except (TypeError, ValueError):
            continue
    return None


def _probe_vray_display_correction(rt) -> Optional[str]:
    """Best-effort read of the ACTIVE renderer's VFB display-correction intent (sRGB / OCIO /
    none). The property spelling is a V-Ray-build calibration item; unknown → None. This is a
    VIEW-only correction — see ``_COLORSPACE_RISK``."""
    try:
        renderer = rt.renderers.current
    except Exception:
        return None
    val = _first_attr(renderer, VRAY_DISPLAY_CORRECTION)
    if val is None:
        return None
    try:
        return str(val)
    except Exception:
        return None


def probe_colorspace() -> Dict:
    """NON-destructive report of the scene's colour management / display-correction mode
    (sRGB/gamma vs OCIO/ACES). Pure read via ColorPipelineMgr (Max 2024+); fires no render
    and changes no settings. Degrades to ``legacy``/``unknown`` when the interface is absent
    (pre-2024 Max) or off-Max.

    Why this matters: the loop's scores are only as trustworthy as the saved plate. A
    color-managed VFB view is a display-side transform that render_frame's raw-framebuffer
    ``rt.save`` does NOT bake in (see ``_COLORSPACE_RISK`` / render_frame line ~76), so an
    active OCIO/ACES display can silently desync the saved PNG from what the artist sees. The
    on-box smoke run flags that mismatch; this probe surfaces the mode so it can.

    Classification is by NORMALIZED substring (lower/strip, drop a leading ``#``) rather than
    an exact ``#gamma`` literal, because pymxs Name-value stringification varies across
    builds. The exact OCIO enum literals, the view-transform accessor, and the display-gamma
    /V-Ray display-correction spellings are ON-BOX CALIBRATION ITEMS — each is probed
    best-effort and None-guarded so an unverified name stays safe.

    Keys: available, color_management ("gamma"|"ocio"|"legacy"|"unknown"), mode (raw str),
    ocio_active, view_transform, display_gamma, vray_display_correction, risk."""
    result: Dict = {
        "available": False,
        "color_management": "unknown",
        "mode": "",
        "ocio_active": False,
        "view_transform": None,
        "display_gamma": None,
        "vray_display_correction": None,
        "risk": _COLORSPACE_RISK,
    }
    try:
        rt = _rt()
    except Exception:                       # off-Max: mode is undetectable, saved plate is raw
        return result
    try:
        mgr = rt.ColorPipelineMgr
    except Exception:
        mgr = None
    if mgr is None:                         # pre-2024 Max → the classic gamma/sRGB pipeline
        result["color_management"] = "legacy"
        return result
    raw = _first_attr(mgr, ("mode",))
    if raw is None:                         # interface present but no mode → treat as legacy
        result["color_management"] = "legacy"
        return result
    result["available"] = True
    result["mode"] = str(raw)
    norm = str(raw).lower().strip().lstrip("#")
    if "ocio" in norm or "aces" in norm:
        result["color_management"] = "ocio"
        result["ocio_active"] = True
    elif "gamma" in norm or "srgb" in norm:
        result["color_management"] = "gamma"
    else:                                   # an unrecognized mode string — do not guess
        result["color_management"] = "unknown"
    view = _first_attr(mgr, CS_VIEW_TRANSFORM, coerce=str)
    result["view_transform"] = view if view else None
    result["display_gamma"] = _first_attr(mgr, CS_DISPLAY_GAMMA, coerce=float)
    result["vray_display_correction"] = _probe_vray_display_correction(rt)
    return result
