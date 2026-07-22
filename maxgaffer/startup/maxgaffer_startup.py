"""Copied into Max's scripts/startup by install.bat — registers the MaxGaffer macro.

Reads the clone path from %LOCALAPPDATA%/MaxGaffer/config.json (written by the installer),
puts it on sys.path, and defines a macroscript so the action can live on any toolbar:
Customize → category "MaxGaffer".

A startup script must NEVER break Max's launch: every failure mode (missing/corrupt/
non-dict config, moved clone, pymxs hiccup) degrades to a loud listener print.
"""

import json
import os
import sys


def _repo_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    try:
        with open(os.path.join(base, "MaxGaffer", "config.json"), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):           # a hand-edited/non-dict config must not
            return str(data.get("repo_path") or "")  # raise AttributeError out of startup
    except (OSError, ValueError, RecursionError):
        pass
    return os.environ.get("MAXGAFFER", "")


def _register():
    try:
        repo = _repo_path()
        if repo and os.path.isdir(repo):
            if repo not in sys.path:
                sys.path.insert(0, repo)
        else:
            # registering the macro anyway would only defer the failure to a raw
            # MAXScript runtime-error dialog on click — say so NOW, in the listener
            print("[MaxGaffer] repo path missing or moved (no valid repo_path in "
                  "config.json and MAXGAFFER unset) — macro NOT registered; reinstall "
                  "or set MAXGAFFER to the clone path")
            return
        from pymxs import runtime as rt

        rt.execute(r'''
macroScript MaxGaffer category:"MaxGaffer" tooltip:"MaxGaffer — reference lighting match" (
    on execute do (
        python.Execute "import maxgaffer.bootstrap; maxgaffer.bootstrap.launch()"
    )
)
''')
    except Exception as e:  # noqa: BLE001
        print("[MaxGaffer] startup registration failed:", e)


_register()
