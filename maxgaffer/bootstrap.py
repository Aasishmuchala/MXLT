"""Launch entry point — the startup macro calls ``maxgaffer.bootstrap.launch()``.

MaxGaffer's hard floor is stdlib-only (the Omega client is urllib, loop-render stats decode
via the stdlib PNG reader), so unlike MaxDirector there are NO required pip packages.
Pillow in Max's user-site is optional and self-detected: it upgrades reference ingestion
(JPEG stats without transcode) and slims LLM payloads.
"""

from __future__ import annotations

import os
import sys

OPTIONAL = ("PIL",)


def _ensure_usersite_on_path() -> None:
    candidates = []
    try:
        import site

        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    for env_var in ("APPDATA", "LOCALAPPDATA"):
        base = os.environ.get(env_var)
        if base:
            candidates.append(os.path.join(base, "Python", "Python311", "site-packages"))
    for path in candidates:
        for cand in (path, os.path.realpath(path) if path else None):
            if cand and cand not in sys.path and os.path.isdir(cand):
                # APPEND, never insert(0): user-site must not shadow Max's own bundled
                # modules — a stray PySide6/Qt wheel there is the classic embedded-Max
                # hard crash (binary Qt conflict at widget creation)
                sys.path.append(cand)


def _pyside_conflict() -> str:
    """Resolve where `import PySide6` would land WITHOUT importing it. Max 2026 bundles
    its own PySide6 against its own Qt DLLs; a second PySide6 earlier on sys.path
    (typically %APPDATA%\\Python\\Python311\\site-packages, shared by every Max python
    tool) wins the import and hard-crashes the process at widget creation — a Python
    try/except cannot catch a binary Qt conflict, so check BEFORE the import happens.
    Returns the offending path, or "" when it's Max's own copy (or absent)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("PySide6")
    except Exception:
        return ""
    if spec is None or not spec.origin:
        return ""
    origin = os.path.realpath(spec.origin)
    exe_dir = os.path.realpath(os.path.dirname(sys.executable or " ") or " ")
    if origin.startswith(exe_dir):
        return ""                     # Max's bundled copy — the safe one
    return origin


def launch():
    _ensure_usersite_on_path()
    conflict = _pyside_conflict()
    if conflict:
        msg = ("MaxGaffer did NOT open — a second PySide6 shadows 3ds Max's own:\n\n"
               f"{conflict}\n\nImporting it would crash Max (binary Qt conflict). "
               "Remove/rename the PySide6 (and PySide6-Essentials) folders from that "
               "site-packages, then restart Max. MaxGaffer needs only Max's bundled Qt.")
        try:
            from pymxs import runtime as rt  # type: ignore

            rt.messageBox(msg, title="MaxGaffer")
        except Exception:
            print("[MaxGaffer] " + msg)
        return None
    try:
        from .ui.dock import show_dock

        return show_dock()
    except Exception as e:  # noqa: BLE001 surface a readable message inside Max
        msg = f"MaxGaffer failed to launch:\n{type(e).__name__}: {e}"
        try:
            from pymxs import runtime as rt  # type: ignore

            rt.messageBox(msg, title="MaxGaffer")
        except Exception:
            print("[MaxGaffer] " + msg)
        return None
