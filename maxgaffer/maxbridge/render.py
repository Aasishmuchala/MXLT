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
from typing import Optional


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
