"""Hot-reload MaxGaffer inside a running 3ds Max, then reopen the dock.

    python.ExecuteFile @"C:\\Users\\aasis\\MXLT\\scripts\\reload_dock.py"

Why this is needed at all: Python caches imported modules for the life of the process, and
`show_dock` deliberately REUSES an existing panel rather than rebuilding it. So editing the
plugin and calling launch() again gets you the old code in the old widget — nothing you
changed appears, which reads exactly like the change not working.

It also clears a dock stuck busy. `_busy` is reset in a finally, so a stuck MATCH button
means the handler is still executing — usually blocked in a gateway call, which sits at
near-zero CPU and can look identical to a hang. Rebuilding the dock abandons that widget
and starts clean.

Safe to run any time. It does not touch the scene, the session sidecar, or the config.
"""

import sys


def _close_existing() -> str:
    dockmod = sys.modules.get("maxgaffer.ui.dock")
    if dockmod is None:
        return "no dock was open"
    note = "closed the old dock"
    for attr in ("_dock_wrapper", "_dock_instance"):
        widget = getattr(dockmod, attr, None)
        if widget is not None:
            try:
                widget.close()
                widget.deleteLater()
            except Exception as err:                       # noqa: BLE001
                note = f"old dock would not close cleanly ({err}) — rebuilding anyway"
            try:
                setattr(dockmod, attr, None)
            except Exception:                              # noqa: BLE001
                pass
    return note


def main() -> None:
    print("[reload] " + _close_existing())
    # Drop every maxgaffer module so the next import reads the files from disk. Reversed
    # so submodules go before the packages that hold references to them.
    stale = sorted((m for m in sys.modules if m == "maxgaffer"
                    or m.startswith("maxgaffer.")), reverse=True)
    for name in stale:
        sys.modules.pop(name, None)
    print("[reload] dropped %d cached module(s)" % len(stale))

    import maxgaffer.bootstrap as bootstrap                # noqa: E402 — after the purge

    if bootstrap.launch() is None:
        print("[reload] launch() returned nothing — check the Max listener for the reason")
    else:
        print("[reload] dock reopened on current code")


main()
