"""Chaos Vantage live-link WINDOW GRAB — a direction probe in ~50 ms instead of ~60 s.

Why this exists: a match's cost IS its render cost, and on TULA (2026-07-30 — 18M
triangles, 460 lights) one V-Ray probe took ~60 seconds while a standard profile fires up
to 133 of them. The global sun solve is 44 of those, and it does not need an accurate
picture: it needs a RANKING of directions, it scores them on the hot-patch map, and its
answer is an ANGLE. Vantage is already streaming that scene in real time on the artist's
second monitor. Reading its window is the cheapest honest answer to "which way is the
light". Nothing else in the plugin is allowed near this path — the sweep, the basin
picker, every tone stage and the delivered render are all V-Ray (controller.py:1197).

Stdlib ``ctypes`` + ``core.png_min`` only — NO Pillow. Max 2026's embedded Python ships
neither numpy nor Pillow (that is why png_min exists at all), so a capture path that needs
either is a capture path that does not run on the box. The Win32 DLLs are loaded through
``_dlls()`` into handles PRIVATE to this module — never ``ctypes.windll``, which returns one
shared object per process and caches argtypes/restype on it, so configuring it would break
every other plugin in Max (see the G-11 note below). Loading is lazy, so this module still
imports cleanly off-Max and off-Windows.

Nothing here raises, ever. Every failure degrades to ``None`` plus a ``last_error()``
string the caller logs, because the caller's fallback (render it in V-Ray) is correct —
just slow — and a wrongly-attributed frame is worse than a slow one: the critic will
happily rank it.

Four refusals are measured, not theoretical, from the 2026-07-30 spike:
  * OCCLUSION — the first grab came back with 3ds Max's own viewport showing through the
    target's rect, with every API reporting success. Silent wrong pixels are the one
    failure this module must not have, so a grab over ``OCCLUSION_LIMIT`` is refused.
  * BLACK — a fully black capture scored "29% non-black" on a whole-frame test, because
    the window's own drop shadow is not black. The black test is centre-crop only.
  * MINIMISED / CLOAKED — a minimised or virtual-desktop-cloaked window still has a rect
    and still returns pixels; they are stale or empty.
  * NOT YET DELIVERED — the fourth, and the only one that is silent by nature: a window
    that is on screen, unobscured and lit, still showing the PREVIOUS probe. See
    ``SETTLE_LIMIT_S``.

ON-BOX CALIBRATION ITEM: a Vantage plate is TONEMAPPED. ``metrics.HOT_THRESHOLD`` is an
ABSOLUTE 0.35 and ``sunsolve`` scores on ``highlight_similarity`` alone, so a different
tone curve changes which pixels clear that threshold and biases the presence half of the
metric; ``CROSS_DOMAIN_AGREEMENT`` / ``DECISIVE_MARGIN`` were calibrated on V-Ray plates.
The remedy is the fitted transfer in ``core.vtone`` (armed per run via ``arm_tone`` once
``scripts/vantage_calibrate.py`` has measured this box) — NOT tone-aligning on Vantage.
An earlier revision of this docstring claimed Vantage's exposure "cannot be written from
Max"; the vendor support table (Chaos docs page 124621427) says otherwise — Physical
Camera ISO / f-number / shutter / EV / white balance all stream over the live link. The
REAL hazards, both documented, are (a) Vantage's auto-exposure toggle, which "ignores set
camera exposure" when on (page 125272932) and is not queryable from outside, and (b)
Vantage 3.3's preserved-local-overrides behaviour (changelog 908558346): a property the
artist once edited INSIDE Vantage stops following live-link updates, silently. Both are
measured — never assumed absent — by the calibration harness before any fit may arm.
``config.probe_backend`` defaults to "vray" so this ships dark.

SECOND CALIBRATION ITEM: the grab is the client area centre-cropped to the probe's aspect
(``_client_rect`` / ``_aspect_crop``), which is the closest this side can get to the
camera's framing without asking Vantage what it is showing. If Vantage draws docked panels
INSIDE its client area they are still in the plate, and they are still bright UI over
``HOT_THRESHOLD``. Eyeball one ``sunsolve_a000.png`` from a real run before trusting an
angle this backend produced.
"""

from __future__ import annotations

import ctypes          # stdlib everywhere; the Win32 DLL loads stay lazy (see _dlls)
import os
import time
from typing import List, Optional, Sequence, Tuple

from ..core import png_min

VANTAGE_TITLE = "Vantage"
VANTAGE_IMAGE = "vantage.exe"

#: A window smaller than this on either side is a tooltip, a splash or a collapsed shell —
#: never the live-link viewport, and downscaling it to a probe would be noise.
MIN_WINDOW_PX = 32

#: Fraction of the window that may be covered before the grab is refused. The screen DC
#: reads what is ON SCREEN, so anything on top of Vantage lands in the probe. Two percent
#: is a stray tooltip edge; more than that is another window.
OCCLUSION_LIMIT = 0.02

#: Sampling grid for the occlusion test. 144 points costs microseconds and catches a
#: window covering any meaningful part of the viewport.
_OCCLUSION_GRID = 12

#: FRESHNESS. Vantage is a SEPARATE PROCESS: the sun rotation has to cross the live-link
#: socket and be re-rendered there before it is on screen, and the probe apply path
#: deliberately suppresses Max's own viewport redraw (apply.py:218-222, undo=False), so
#: nothing even nudges the notification. Both probe producers apply and render in the same
#: breath (sunsolve.py:118-121, director.py:1284-1286) — with V-Ray that is safe because
#: the render IS the 60 s of new work, but a grab lands milliseconds after the pymxs write
#: and returns whatever was already composited. Every sample would be shifted one slot and
#: NOTHING would refuse it: the plate is not black, not occluded, not minimised. That is
#: worse than the 3-hour run it replaces, because it is fast and confidently wrong.
#:
#: So the grab POLLS until the picture differs from the previous probe's, bounded. 0.6 s
#: worst case is still 100x cheaper than one V-Ray probe, and it never blocks Max's main
#: thread for an unbounded time.
SETTLE_LIMIT_S = 0.6
SETTLE_STEP_S = 0.05

#: MEASUREMENT-GRADE convergence, on top of the ordinal freshness poll above. The settle
#: test proves "the picture CHANGED" and deliberately accepts the FIRST frame that moved —
#: which is the least-converged, most-denoiser-biased frame in Vantage's accumulation
#: sequence. That is fine for a RANKING (every probe is early by the same rule) and wrong
#: for a MEASUREMENT: a tone fit calibrated on converged frames does not apply to
#: first-motion frames, and vice versa. ``converged=True`` grabs therefore keep polling
#: after first motion until two consecutive signatures agree within CONVERGE_EPSILON —
#: stationarity, the cheapest observable proxy for "accumulation has stopped moving the
#: picture". Both numbers are CALIBRATION ITEMS: the harness's dwell experiment measures
#: the real knee per box, and 8 s is a ceiling chosen so a heavy scene converges and a
#: hung link still refuses inside one V-Ray probe's budget.
CONVERGE_LIMIT_S = 8.0
CONVERGE_EPSILON = 0.75

#: How far a 4x4 block mean (0..255 per channel) must move before the live link is believed
#: to have delivered. NOT "any difference at all": Vantage keeps accumulating while it sits
#: idle, so bit-equality is satisfied by path-tracer noise, which would wave the stale frame
#: through. At block-mean scale that noise is a fraction of a level while a coarse-pass
#: azimuth step (30 deg) moves whole blocks by tens. Calibration item — measure it on the
#: box before widening it.
SETTLE_DELTA = 1.5

_LAST_ERROR = ""
#: The previous probe's picture, as a signature. The freshness test is inductive — each
#: grab proves it is newer than the one before it — so the chain needs a base, and
#: ``reset_settle()`` (called once per run when the backend is armed) provides it by
#: making the first probe wait out the whole budget instead of comparing.
_LAST_SIGNATURE: Optional[List[float]] = None

#: The armed ``core.vtone.VTone`` transfer, or None. Armed per run by the controller
#: AFTER the fit's provenance and held-out residual pass their gates — this module never
#: decides whether a fit is trustworthy, it only applies or refuses. None means grabs
#: are UNCORRECTED, which is exactly today's shipped behaviour: legal for the sun
#: solve's ordinal ranking, illegal for anything scored on absolute thresholds.
_TONE = None


def arm_tone(tone) -> None:
    """Arm a fitted tone transfer for this run's grabs. The caller (the controller's
    arming path) owns validation; passing None is an explicit disarm."""
    global _TONE
    _TONE = tone


def disarm_tone() -> None:
    """Back to uncorrected (ordinal-only) grabs — the state every run must START from,
    so a previous run's fit can never leak into one whose gates were not checked."""
    global _TONE
    _TONE = None


def armed_tone():
    """The currently armed VTone, or None. Read by callers that need to LABEL a plate's
    provenance (corrected vs ordinal-only) — a plate whose correction state cannot be
    named is the 'silent wrong pixels' failure this module exists to prevent."""
    return _TONE


def last_error() -> str:
    """Why the last grab was refused — the string the caller puts in the transcript. A
    backend that fails silently is a backend nobody can debug at 2am."""
    return _LAST_ERROR


def _fail(msg: str) -> None:
    global _LAST_ERROR
    _LAST_ERROR = msg


# --------------------------------------------------------------------------- win32 handles
#
# G-11, fixed 2026-07-31. ``ctypes.windll.user32`` returns ONE shared WinDLL object for the
# whole Max process, and ctypes caches ``restype``/``argtypes`` ON THAT OBJECT. This module
# was setting ``restype = c_void_p`` on GetDC/CreateCompatibleDC/CreateDIBSection/
# SelectObject and — worse — installing a freshly-created private ``_Point`` class as
# ``WindowFromPoint.argtypes`` on EVERY call. Any other plugin or startup script that later
# called ``user32.WindowFromPoint(pt)`` then got
#
#     ctypes.ArgumentError: expected _Point instance instead of POINT
#
# because argtypes demanded THIS module's private class, freshly constructed, so even a
# structurally identical POINT failed the identity check. Permanent until Max restarts, and
# the blast radius is OTHER PEOPLE'S CODE.
#
# ``WinDLL(...)`` constructs a fresh object where ``windll.x`` returns the shared one, so
# these three handles are private to this module and nothing configured on them escapes.
# ``_Point`` is hoisted to module scope for the same reason: one class, defined once.
_U32 = None
_G32 = None
_DWM = None
_K32 = None


class _Point(ctypes.Structure):
    """ONE class, defined once. It used to be redefined per call and installed as
    ``WindowFromPoint.argtypes`` on the process-shared user32 handle, so every later
    caller in the whole Max process had to pass an instance of a class that no longer
    existed anywhere they could reach."""
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _dlls():
    """→ (user32, gdi32, dwmapi, kernel32), PRIVATE to this module, or Nones off-Windows."""
    global _U32, _G32, _DWM, _K32
    if _U32 is None:
        try:
            _U32 = ctypes.WinDLL("user32", use_last_error=True)
            _G32 = ctypes.WinDLL("gdi32", use_last_error=True)
            _DWM = ctypes.WinDLL("dwmapi")
            _K32 = ctypes.WinDLL("kernel32", use_last_error=True)
        except Exception:  # noqa: BLE001 — not Windows, or a locked-down box
            return None, None, None, None
    return _U32, _G32, _DWM, _K32


def _rows_from_bgrx(raw: bytes, width: int, height: int
                    ) -> List[List[Tuple[int, int, int]]]:
    """A GDI 32-bit top-down DIB buffer → rows of (r, g, b).

    Extracted 2026-07-31 so the BGRX→RGB swizzle can be tested off-box. It is the single
    most classic place in Win32 imaging to get red and blue backwards, it was untested by
    all thirteen tests in test_vgrab.py, and an R/B swap materially changes the answer:
    luminance is 0.2126·R + 0.0722·B, so swapping shifts which pixels clear
    ``metrics.HOT_THRESHOLD`` and therefore the entire ranking the sun solve depends on.
    """
    rows: List[List[Tuple[int, int, int]]] = []
    stride = width * 4
    for y in range(height):
        base = y * stride
        line = raw[base:base + stride]
        # BGRX → RGB, dropping the unused alpha byte GDI leaves at 0
        rows.append([(line[x + 2], line[x + 1], line[x]) for x in range(0, stride, 4)])
    return rows


def reset_settle() -> None:
    """Forget the last run's final frame. A run that starts by comparing against a picture
    from an hour ago would pass its freshness test on the strength of the artist having
    moved the camera in between, which is not evidence about THIS probe."""
    global _LAST_SIGNATURE
    _LAST_SIGNATURE = None


def last_signature() -> Optional[List[float]]:
    """The coarse signature of the last ACCEPTED grab, or None after ``reset_settle``.

    Public so preflight's stability gate can compare two grabs without re-deriving the
    crop. Added 2026-07-31: ``check_vantage_armed``'s docstring promised a two-grab
    comparison and its body took ONE grab and checked only that it returned a path, which
    is the one condition a scene mid-ingest also satisfies.
    """
    return list(_LAST_SIGNATURE) if _LAST_SIGNATURE is not None else None


# --------------------------------------------------------------------------- process ids
def _pids_for_image(name: str = VANTAGE_IMAGE) -> Tuple[int, ...]:
    """PIDs of every running process whose image name matches (case-insensitive).

    PID-FIRST identification is the whole point: a browser tab open on the Vantage docs,
    an Explorer window in a folder called Vantage and the installer's own progress dialog
    all match the TITLE. Only the process that is actually vantage.exe can be streaming
    the live link.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001 — off-Windows / stripped python: no processes to find
        return ()

    class _ProcessEntry32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]

    want = str(name).lower()
    pids: List[int] = []
    snapshot = None
    try:
        _u, _g, _d, k32 = _dlls()
        if k32 is None:
            return ()
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        snapshot = k32.CreateToolhelp32Snapshot(0x00000002, 0)   # TH32CS_SNAPPROCESS
        if not snapshot or snapshot == wintypes.HANDLE(-1).value:
            return ()
        entry = _ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(_ProcessEntry32W)
        ok = k32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            if str(entry.szExeFile).lower() == want:
                pids.append(int(entry.th32ProcessID))
            ok = k32.Process32NextW(snapshot, ctypes.byref(entry))
    except Exception:  # noqa: BLE001 — an enumeration hiccup is "not running", not a crash
        return tuple(pids)
    finally:
        try:
            if snapshot:
                _dlls()[3].CloseHandle(snapshot)
        except Exception:  # noqa: BLE001
            pass
    return tuple(pids)


# --------------------------------------------------------------------------- window census
def _window_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """(left, top, right, bottom) in SCREEN pixels, from DWM's extended frame bounds.

    The window's OUTLINE — used to size and rank candidates. The pixels are taken from
    ``_client_rect`` instead; see there for why.

    ``GetWindowRect`` over-reports on composited Windows: it includes the invisible
    resize border, so a rect framed on it carries a strip of whatever is behind the
    window down two edges — desktop, or 3ds Max. DWM's bounds are what the user sees.
    Falls back to GetWindowRect when dwmapi is unavailable (pre-Vista / server core).

    Sizes are always READ BACK, never assumed or scaled by a DPI factor:
    ``SetProcessDpiAwarenessContext`` is process-wide and one-shot, and inside Max the
    host already made that call — so the only honest number is the one the OS reports.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None
    try:
        rect = wintypes.RECT()
        try:
            # DWMWA_EXTENDED_FRAME_BOUNDS = 9
            hr = _dlls()[2].DwmGetWindowAttribute(
                wintypes.HWND(hwnd), ctypes.c_uint(9), ctypes.byref(rect),
                ctypes.sizeof(rect))
        except Exception:  # noqa: BLE001
            hr = -1
        if hr != 0:
            if not _dlls()[0].GetWindowRect(wintypes.HWND(hwnd),
                                                      ctypes.byref(rect)):
                return None
        return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    except Exception:  # noqa: BLE001
        return None


def _client_rect(hwnd: int) -> Optional[Tuple[int, int, int, int]]:
    """The CLIENT area in SCREEN pixels — the window without its caption or its borders.

    This, not ``_window_rect``, is what gets grabbed, and the difference is not cosmetic.
    ``metrics`` bins hot pixels into a 5x5 grid over the WHOLE image (metrics.py:253) and
    ``hot_frac`` is a fraction of the WHOLE image, so anything that is in the plate but not
    in the picture does not average out: a ~32 px caption and a border shift every sun
    patch into a neighbouring cell of ``hot_grid``, and light-on-dark UI text clears the
    ABSOLUTE ``HOT_THRESHOLD`` and reads as a permanent sun patch in fixed border cells of
    every probe. ``highlight_similarity`` is half placement and half presence, and both
    halves are measured against the artist's REFERENCE — so chrome biases the comparison in
    a direction that does not cancel between probes.

    Refused rather than approximated when it cannot be read: the V-Ray fallback is always
    correct, and the whole contract of this module is that it never hands back a picture
    whose framing it cannot name.
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None
    try:
        user32 = _dlls()[0]
        if user32 is None:
            return None
        rect = wintypes.RECT()
        if not user32.GetClientRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        origin = wintypes.POINT(0, 0)      # GetClientRect is client-relative: (0,0,w,h)
        if not user32.ClientToScreen(wintypes.HWND(hwnd), ctypes.byref(origin)):
            return None
        left, top = int(origin.x), int(origin.y)
        return left, top, left + int(rect.right), top + int(rect.bottom)
    except Exception:  # noqa: BLE001
        return None


def _aspect_crop(rect: Tuple[int, int, int, int], width: int,
                 height: int) -> Tuple[int, int, int, int]:
    """Centre-crop ``rect`` to the probe's aspect — never squash it into one.

    ``_grab_rows`` StretchBlts whatever it is given into ``width`` x ``height``, and the
    probe box carries the RENDER's aspect (profiles.py:122). A 21:9 Vantage window on a
    3440x1440 monitor squeezed into a 16:9 probe is horizontally compressed by a third, and
    both metrics these probes are scored on are POSITIONAL — the 5x5 ``hot_grid`` and the
    3x3 ``grid`` the sweep cosines against the reference. A sun patch that belongs in cell
    (1,3) lands in (1,2), the argmax over azimuth moves with it, and the basin and polish
    stages inherit the answer. Cropping loses field of view at the edges; squashing moves
    everything, which is the error the grids actually measure.
    """
    try:
        left, top, right, bottom = (int(v) for v in rect)
        want = float(width) / float(height)
    except (TypeError, ValueError, ZeroDivisionError):
        return rect
    src_w, src_h = right - left, bottom - top
    if src_w <= 0 or src_h <= 0 or want <= 0:
        return rect
    if src_w > src_h * want:                       # too wide — trim the sides
        keep = min(src_w, max(1, int(round(src_h * want))))
        cut = (src_w - keep) // 2
        return left + cut, top, left + cut + keep, bottom
    keep = min(src_h, max(1, int(round(src_w / want))))   # too tall — trim top and bottom
    cut = (src_h - keep) // 2
    return left, top + cut, right, top + cut + keep


def _enum_candidates(pids: Sequence[int]) -> List[Tuple]:
    """Every top-level window owned by ``pids`` as plain tuples:
    ``(hwnd, pid, title, rect, minimized, cloaked)``.

    Minimised and cloaked state is RECORDED rather than filtered here so the ranking —
    the part worth unit-testing — stays a pure function over data (see ``_best_window``).
    A cloaked window is one on another virtual desktop or a suspended UWP host: it has a
    perfectly good rect and returns perfectly stale pixels.
    """
    wanted = set(int(p) for p in pids or ())
    if not wanted:
        return []
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return []
    found: List[Tuple] = []
    try:
        user32 = _dlls()[0]
        if user32 is None:
            return []
        proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def _visit(hwnd, _lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True
                pid = wintypes.DWORD(0)
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                if int(pid.value) not in wanted:
                    return True
                length = int(user32.GetWindowTextLengthW(hwnd))
                buf = ctypes.create_unicode_buffer(length + 2)
                user32.GetWindowTextW(hwnd, buf, length + 2)
                rect = _window_rect(int(hwnd))
                if rect is None:
                    return True
                cloaked = ctypes.c_int(0)
                try:   # DWMWA_CLOAKED = 14
                    _dlls()[2].DwmGetWindowAttribute(
                        hwnd, ctypes.c_uint(14), ctypes.byref(cloaked),
                        ctypes.sizeof(cloaked))
                except Exception:  # noqa: BLE001 — no dwmapi: nothing is cloaked
                    cloaked = ctypes.c_int(0)
                found.append((int(hwnd), int(pid.value), str(buf.value), rect,
                              bool(user32.IsIconic(hwnd)), bool(cloaked.value)))
            except Exception:  # noqa: BLE001 — one bad window must not end the census
                pass
            return True

        user32.EnumWindows(proc_type(_visit), 0)
    except Exception:  # noqa: BLE001
        return found
    return found


def _best_window(candidates: Sequence[Tuple], title_substr: str = VANTAGE_TITLE):
    """Rank ``(hwnd, pid, title, rect, minimized, cloaked)`` tuples → the winner or None.

    PURE FUNCTION, no ctypes on purpose: this is the part with actual judgement in it, so
    it is the part that takes plain data and can be unit-tested without faking the Win32
    API. Minimised, cloaked and sub-``MIN_WINDOW_PX`` windows are dropped; a title match
    beats a bigger untitled sibling (Vantage owns several helper windows); among equals,
    the largest area wins — the viewport is the big one.
    """
    want = str(title_substr or "").lower()
    scored = []
    for cand in candidates or ():
        try:
            _hwnd, _pid, title, rect, minimized, cloaked = cand
            left, top, right, bottom = rect
        except (TypeError, ValueError):
            continue                     # not a candidate tuple — not evidence
        if minimized or cloaked:
            continue
        w, h = int(right) - int(left), int(bottom) - int(top)
        if w < MIN_WINDOW_PX or h < MIN_WINDOW_PX:
            continue
        titled = 1 if (want and want in str(title).lower()) else 0
        scored.append((titled, w * h, cand))
    if not scored:
        return None
    return max(scored, key=lambda s: (s[0], s[1]))[2]


def find_window(title_substr: str = VANTAGE_TITLE,
                image_name: str = VANTAGE_IMAGE) -> Optional[int]:
    """The HWND of Vantage's viewport window, or None (with ``last_error()`` set)."""
    pids = _pids_for_image(image_name)
    if not pids:
        _fail(f"{image_name} is not running")
        return None
    best = _best_window(_enum_candidates(pids), title_substr)
    if best is None:
        _fail(f"{image_name} is running but its window was not found "
              "(minimised, on another virtual desktop, or too small)")
        return None
    return int(best[0])


# --------------------------------------------------------------------------- occlusion
def _occluded_fraction(hwnd: int, rect: Tuple[int, int, int, int]) -> Optional[float]:
    """What fraction of a 12×12 grid over ``rect`` belongs to some OTHER top-level window.

    The grab reads the composited desktop, so anything sitting on top of Vantage ends up
    in the probe — and the 2026-07-30 spike's first capture was exactly that: 3ds Max's
    viewport, delivered through the target's rect with every API returning success. A
    probe whose provenance cannot be named is worse than no probe.

    → 0..1, or None when the test itself could not run (which the caller also refuses:
    unverifiable is not the same as verified, and the V-Ray fallback is always correct).
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None
    try:
        left, top, right, bottom = (int(v) for v in rect)
    except (TypeError, ValueError):
        return None
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None

    try:
        user32 = _dlls()[0]
        if user32 is None:
            return None
        user32.WindowFromPoint.restype = wintypes.HWND
        user32.WindowFromPoint.argtypes = [_Point]
        user32.GetAncestor.restype = wintypes.HWND
        n = _OCCLUSION_GRID
        sampled = covered = 0
        for gy in range(n):
            for gx in range(n):
                px = left + (2 * gx + 1) * w // (2 * n)
                py = top + (2 * gy + 1) * h // (2 * n)
                hit = user32.WindowFromPoint(_Point(px, py))
                if not hit:
                    continue
                root = user32.GetAncestor(wintypes.HWND(hit), 2)   # GA_ROOT
                sampled += 1
                if int(root or hit) != int(hwnd):
                    covered += 1
        if not sampled:
            return None
        return covered / float(sampled)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- the grab
def _grab_rows(rect: Tuple[int, int, int, int], width: int,
               height: int) -> Optional[List[List[Tuple[int, int, int]]]]:
    """The window's pixels, already downscaled to ``width`` × ``height``, as png_min rows.

    Through the SCREEN DC (``GetDC(None)``) — the DWM-composed desktop — which is why
    Vantage's D3D12 content arrives at all. ``GetWindowDC`` + ``PrintWindow`` returns a
    fully black client area on this window class (all four flag combinations, measured
    2026-07-30) and is a synchronous SendMessage into the target's UI thread with no
    timeout, so a busy Vantage would hang Max. It is not used here and should not be.

    ``StretchBlt`` with HALFTONE does the downscale in GDI, in C: a 4K window is 8.3M
    pixels and pure python must never walk them — the whole point of this path is that it
    costs milliseconds. ``CAPTUREBLT`` is deliberately unset (it forces layered windows
    into the copy and flickers the screen).
    """
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:  # noqa: BLE001
        return None
    try:
        left, top, right, bottom = (int(v) for v in rect)
        w_out, h_out = max(1, int(width)), max(1, int(height))
    except (TypeError, ValueError):
        return None
    src_w, src_h = right - left, bottom - top
    if src_w <= 0 or src_h <= 0:
        return None

    class _BitmapInfoHeader(ctypes.Structure):
        _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long),
                    ("biHeight", ctypes.c_long), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                    ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD)]

    class _BitmapInfo(ctypes.Structure):
        _fields_ = [("bmiHeader", _BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

    screen_dc = mem_dc = dib = old = None
    try:
        user32, gdi32, _dwm, _k32 = _dlls()
        if user32 is None:
            return None
        for fn in (user32.GetDC, gdi32.CreateCompatibleDC, gdi32.CreateDIBSection,
                   gdi32.SelectObject):
            fn.restype = ctypes.c_void_p          # 64-bit handles must not truncate
        screen_dc = user32.GetDC(None)
        if not screen_dc:
            return None
        mem_dc = gdi32.CreateCompatibleDC(ctypes.c_void_p(screen_dc))
        if not mem_dc:
            return None
        info = _BitmapInfo()
        info.bmiHeader.biSize = ctypes.sizeof(_BitmapInfoHeader)
        info.bmiHeader.biWidth = w_out
        info.bmiHeader.biHeight = -h_out          # negative = TOP-DOWN rows, as png wants
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = 0          # BI_RGB
        bits = ctypes.c_void_p()
        dib = gdi32.CreateDIBSection(ctypes.c_void_p(mem_dc), ctypes.byref(info), 0,
                                     ctypes.byref(bits), None, 0)
        if not dib or not bits:
            return None
        old = gdi32.SelectObject(ctypes.c_void_p(mem_dc), ctypes.c_void_p(dib))
        gdi32.SetStretchBltMode(ctypes.c_void_p(mem_dc), 4)        # HALFTONE
        try:
            gdi32.SetBrushOrgEx(ctypes.c_void_p(mem_dc), 0, 0, None)   # HALFTONE's pair
        except Exception:  # noqa: BLE001
            pass
        ok = gdi32.StretchBlt(ctypes.c_void_p(mem_dc), 0, 0, w_out, h_out,
                              ctypes.c_void_p(screen_dc), left, top, src_w, src_h,
                              0x00CC0020)        # SRCCOPY, no CAPTUREBLT
        if not ok:
            return None
        raw = ctypes.string_at(bits, w_out * h_out * 4)
    except Exception:  # noqa: BLE001 — a GDI failure is a fallback, never an exception
        return None
    finally:
        try:
            _u32, _g32, _dwm2, _k322 = _dlls()
            if old:
                _g32.SelectObject(ctypes.c_void_p(mem_dc), ctypes.c_void_p(old))
            if dib:
                _g32.DeleteObject(ctypes.c_void_p(dib))
            if mem_dc:
                _g32.DeleteDC(ctypes.c_void_p(mem_dc))
            if screen_dc:
                _u32.ReleaseDC(None, ctypes.c_void_p(screen_dc))
        except Exception:  # noqa: BLE001
            pass
    return _rows_from_bgrx(raw, w_out, h_out)


def _all_black(rows: Sequence[Sequence[Tuple[int, int, int]]]) -> bool:
    """Is the CENTRE of this grab entirely black? Same rule as ``metrics.is_black`` — one
    definition of black in this codebase — but restricted to the middle 60%, because a
    whole-frame test passes on the window's own drop shadow: the spike's fully black
    capture measured "29% non-black" and was accepted."""
    if not rows or not rows[0]:
        return True
    h, w = len(rows), len(rows[0])
    y0, y1 = int(h * 0.2), max(int(h * 0.2) + 1, int(h * 0.8))
    x0, x1 = int(w * 0.2), max(int(w * 0.2) + 1, int(w * 0.8))
    for row in rows[y0:y1]:
        for px in row[x0:x1]:
            if px[0] or px[1] or px[2]:
                return False
    return True


def _signature(rows: Sequence[Sequence[Tuple[int, int, int]]]) -> List[float]:
    """A 4x4 grid of block means (R,G,B) over the middle 60% → 48 numbers.

    Deliberately COARSE and deliberately centre-only: it answers "is this a different
    picture", never "is this a good picture", and it must survive the accumulation noise a
    real-time path tracer produces while it sits still. The centre crop is the same window
    ``_all_black`` uses, for the same reason — the edges carry the window's own furniture.
    """
    if not rows or not rows[0]:
        return []
    h, w = len(rows), len(rows[0])
    y0, y1 = int(h * 0.2), max(int(h * 0.2) + 1, int(h * 0.8))
    x0, x1 = int(w * 0.2), max(int(w * 0.2) + 1, int(w * 0.8))
    ch, cw = y1 - y0, x1 - x0
    sums = [0.0] * 48
    counts = [0] * 16
    for yy in range(y0, y1):
        by = min(3, (yy - y0) * 4 // ch)
        row = rows[yy]
        for xx in range(x0, x1):
            cell = by * 4 + min(3, (xx - x0) * 4 // cw)
            px = row[xx]
            sums[cell * 3] += px[0]
            sums[cell * 3 + 1] += px[1]
            sums[cell * 3 + 2] += px[2]
            counts[cell] += 1
    return [sums[i] / counts[i // 3] if counts[i // 3] else 0.0 for i in range(48)]


def _moved(sig: Sequence[float], prev: Sequence[float]) -> bool:
    """Has the picture changed by more than accumulation noise? See ``SETTLE_DELTA``."""
    if not sig or not prev or len(sig) != len(prev):
        return True                       # nothing comparable — not evidence of staleness
    return max(abs(a - b) for a, b in zip(sig, prev)) >= SETTLE_DELTA


def _settled_rows(rect: Tuple[int, int, int, int], width: int, height: int, rows,
                  should_cancel=None):
    """Poll until the window shows a picture the PREVIOUS probe did not → (rows, sig).

    → (None, []) when the budget runs out without the frame moving, which the caller turns
    into a refusal. Three consecutive refusals disarm the backend for the run and the rest
    finishes in V-Ray, slowly and correctly. That is the right trade — a sun angle solved
    off frames that all predate their own apply is not a cheaper answer, it is a different
    question answered confidently.

    ``should_cancel`` is consulted every step (2026-07-31). This loop sleeps up to 0.6 s
    per probe and 44 probes is up to 26 seconds of unresponsive dock, which is main-thread
    blocking added by a commit whose stated motivation was an artist unable to cancel.
    """
    prev = _LAST_SIGNATURE
    sig = _signature(rows)
    deadline = time.monotonic() + max(0.0, float(SETTLE_LIMIT_S))
    # prev is None on the FIRST probe of a run: there is nothing to be newer than, so wait
    # out the budget instead of comparing and take what is on screen at the end of it. One
    # 0.6 s pause per run is the induction base for every probe after it.
    while (prev is None or not _moved(sig, prev)) and time.monotonic() < deadline:
        if should_cancel is not None:
            try:
                if should_cancel():
                    _fail("cancelled while waiting for the live link to deliver")
                    return None, []
            except Exception:  # noqa: BLE001 — a broken predicate must not wedge the poll
                pass
        time.sleep(max(0.0, float(SETTLE_STEP_S)))
        fresh = _grab_rows(rect, width, height)
        if not fresh:
            break                          # the window went away — the black/empty tests
        rows, sig = fresh, _signature(fresh)          # downstream name it properly
    if prev is not None and not _moved(sig, prev):
        _fail(f"the Vantage window has not changed in {SETTLE_LIMIT_S:g}s — the live link "
              "did not deliver this probe's sun, so the grab is the PREVIOUS probe")
        return None, []
    return rows, sig


def _stationary(sig: Sequence[float], prev: Sequence[float]) -> bool:
    """Have two CONSECUTIVE grabs agreed within accumulation noise? The convergence
    counterpart of ``_moved`` — same signature space, tighter epsilon."""
    if not sig or not prev or len(sig) != len(prev):
        return False
    return max(abs(a - b) for a, b in zip(sig, prev)) < CONVERGE_EPSILON


def _converged_rows(rect: Tuple[int, int, int, int], width: int, height: int, rows,
                    sig: List[float], should_cancel=None):
    """Poll AFTER first motion until the picture stops moving → (rows, sig) or (None, []).

    Runs only for ``converged=True`` grabs (the calibration harness and any future
    measurement-grade consumer). The refusal on timeout is the point: a frame still
    accumulating visibly after CONVERGE_LIMIT_S is a frame whose value depends on WHEN it
    was taken, and a measurement with an uncontrolled timestamp in it is not a
    measurement. Ordinal probes never pay this cost.
    """
    deadline = time.monotonic() + max(0.0, float(CONVERGE_LIMIT_S))
    prev_sig = sig
    while time.monotonic() < deadline:
        if should_cancel is not None:
            try:
                if should_cancel():
                    _fail("cancelled while waiting for Vantage to converge")
                    return None, []
            except Exception:  # noqa: BLE001 — a broken predicate must not wedge the poll
                pass
        time.sleep(max(0.01, float(SETTLE_STEP_S)))
        fresh = _grab_rows(rect, width, height)
        if not fresh:
            _fail("the Vantage window went away while converging")
            return None, []
        fresh_sig = _signature(fresh)
        if _stationary(fresh_sig, prev_sig):
            return fresh, fresh_sig
        rows, prev_sig = fresh, fresh_sig
    _fail(f"the Vantage frame was still accumulating after {CONVERGE_LIMIT_S:g}s — a "
          "measurement-grade grab needs a stationary picture, and this one's value "
          "would depend on when it was taken")
    return None, []


def capture_window_png(title_substr: str, out_path: str,
                       width: int, height: int, should_cancel=None,
                       converged: bool = False) -> Optional[str]:
    """Grab the Vantage live-link window into an 8-bit RGB PNG at probe size → path/None.

    find → client rect → aspect crop → occlusion → grab → settle → [converge] →
    re-occlusion → black test → [tone-correct] → write.
    NOTHING is written on any refusal, and a pre-existing file at ``out_path`` is removed
    first, so a stale frame from an earlier probe can never be mistaken for this one's
    render (``render_frame`` holds the same contract for exactly the same reason).

    ``converged=True`` additionally waits for stationarity after first motion — the
    measurement-grade mode (see ``CONVERGE_LIMIT_S``). When a tone transfer is armed
    (``arm_tone``) the corrected rows are what lands on disk; a correction that FAILS
    refuses the grab entirely rather than silently writing uncorrected pixels, because a
    consumer that armed the correction is by definition one that scores absolute values,
    and an uncorrected plate it believes corrected is the exact silent-wrong-pixels
    failure the four refusals above exist to prevent. Ordinal callers simply never arm.
    """
    global _LAST_SIGNATURE

    hwnd = find_window(title_substr)
    if not hwnd:
        return None                       # find_window already named the reason
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except OSError:
        pass
    client = _client_rect(int(hwnd))
    if client is None:
        _fail("the Vantage window's client area could not be read")
        return None
    rect = _aspect_crop(client, width, height)
    # the occlusion test asks about the pixels actually taken, not the whole window: a
    # tooltip over Vantage's title bar is not in the plate, and a dialog over the middle
    # of the viewport is 100% of the problem even if it is 8% of the window
    covered = _occluded_fraction(int(hwnd), rect)
    if covered is None:
        _fail("could not verify the Vantage window is unobscured — refusing to guess")
        return None
    if covered > OCCLUSION_LIMIT:
        _fail(f"the Vantage window is {covered:.0%} occluded — the grab would carry "
              "another window's pixels")
        return None
    rows = _grab_rows(rect, width, height)
    if not rows:
        _fail("the screen capture returned no pixels")
        return None
    rows, sig = _settled_rows(rect, width, height, rows, should_cancel)
    if not rows:
        return None                       # _settled_rows already named the reason
    if converged:
        rows, sig = _converged_rows(rect, width, height, rows, sig, should_cancel)
        if not rows:
            return None                   # _converged_rows already named the reason
    # G-5, 2026-07-31: RE-VERIFY OCCLUSION ON THE FRAME THAT IS ACTUALLY RETURNED.
    # The check above ran on a frame _settled_rows then threw away — it re-grabs up to
    # twelve more times, unchecked — so a toast or a tooltip appearing during the 0.6 s
    # poll was not merely missed, it was CERTIFIED: the popup moves the picture by far
    # more than SETTLE_DELTA, so _moved returns True immediately and the freshness guard
    # waves the popup's pixels straight through as "the live link delivered". 144
    # WindowFromPoint calls, microseconds.
    covered_now = _occluded_fraction(int(hwnd), rect)
    if covered_now is None or covered_now > OCCLUSION_LIMIT:
        _fail("something appeared over the Vantage window while the live link was "
              "delivering — the frame that was going to be returned is not verifiably "
              "Vantage's")
        return None
    if _all_black(rows):
        _fail("the Vantage grab came back black (window covered, minimising, or the "
              "viewport is not drawing)")
        return None
    if _TONE is not None:
        # REFUSE over lying, decided deliberately: the alternative — fall through to the
        # uncorrected rows — hands a consumer that armed a correction a plate in the
        # WRONG SPACE with nothing refusing it, and every absolute threshold downstream
        # (HOT_THRESHOLD first) silently measures the wrong thing. A refusal costs one
        # V-Ray render; the V-Ray fallback is always correct.
        try:
            rows = _TONE.to_vray_space(rows)
        except Exception as e:  # noqa: BLE001 — a broken fit must refuse, never leak raw pixels
            _fail(f"the armed tone correction failed ({type(e).__name__}: {e}) — "
                  "refusing to hand back an uncorrected plate while a corrected "
                  "surface is armed")
            return None
    written = png_min.write_png_rgb(out_path, rows)
    if not written:
        _fail(f"the probe PNG could not be written to {out_path}")
        return None
    # only an ACCEPTED plate becomes the next probe's baseline: a refused frame is one the
    # caller never scored, so proving the frame after it differs from THAT proves nothing
    _LAST_SIGNATURE = sig
    return written
