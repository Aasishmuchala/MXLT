"""``scripts/onbox_effects_smoke.py`` only ever EXECUTES on the box — it must still be
VERIFIABLE off it, the same guarantee ``test_scripts_static.py`` gives the other spikes.

Three floors, none of which need pymxs / a real Max:

1. it COMPILES and every ``from maxgaffer... import X`` names a real X (a symbol *or* a
   submodule) — the round-3 lesson (a stale import swallowed by the inside-Max guard would
   print "must run INSIDE 3ds Max" ON the box and kill the one-command bring-up with a lie);
2. the pymxs ``ImportError`` guard wraps ONLY ``import pymxs`` — ``main()`` runs from the
   guard's ``else`` branch so a stale maxgaffer import tracebacks loudly instead of
   masquerading as "you're not inside Max";
3. the no-V-Ray degrade path actually SURVIVES off-Max: driven against the hostile fake
   runtime with every V-Ray class removed, ``main()`` returns without raising, each of the
   six isolated sections still records a result, and the three sections the script itself
   gates on a constructable-node check self-report ``status: degraded`` (never a crash).
"""

import ast
import importlib
import importlib.util
import json
import os
import sys

from tests.mock_pymxs import CHAOS_SEED, FakeMaxRuntime, install

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = "scripts/onbox_effects_smoke.py"

#: V-Ray class/function globals whose ABSENCE is what "V-Ray not installed" looks like to
#: the script's candidate-name maker helper (``_make`` -> None for every one of them).
_VRAY_MAKERS = ("VRaySun", "VRayLight", "VRayIES", "VRayBitmap", "VRayHDRI",
                "vrayCreateVRayExposureControl")


def _read(rel):
    with open(os.path.join(REPO, rel), encoding="utf-8") as f:
        return f.read()


def _load_module(rel):
    """Import a ``scripts/*.py`` file as a throwaway module (``scripts/`` is deliberately
    not a package). NOTE: the file's bottom-of-module pymxs guard runs DURING exec — control
    ``sys.modules['pymxs']`` before calling this to decide whether ``main()`` auto-runs."""
    path = os.path.join(REPO, rel)
    spec = importlib.util.spec_from_file_location("mg_onbox_effects_static", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _install_vray_less_runtime(monkeypatch):
    """A seeded hostile ``pymxs.runtime`` with every V-Ray global removed — the process
    stays clean (monkeypatch reverts the ``sys.modules`` entry at teardown)."""
    rt = FakeMaxRuntime(seed=CHAOS_SEED)
    for cls in _VRAY_MAKERS:
        rt.remove_maker(cls)
    monkeypatch.setitem(sys.modules, "pymxs", install(rt))
    return rt


# ---------------------------------------------------------------- 1 compile + imports
def test_script_compiles_and_maxgaffer_imports_resolve():
    src = _read(SCRIPT)
    compile(src, SCRIPT, "exec")                        # syntax floor
    tree = ast.parse(src)
    checked = 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module
                and node.module.startswith("maxgaffer") and not node.level):
            continue
        if ".ui" in node.module:                        # dock needs PySide6 — this never imports it
            continue
        mod = importlib.import_module(node.module)
        for alias in node.names:
            name = alias.name
            resolved = hasattr(mod, name)
            if not resolved:
                # a `from pkg import submodule` (e.g. `from maxgaffer.core import metrics`)
                # binds the child lazily — importing it proves it exists just as well as an
                # attribute would, so both import styles are honoured.
                try:
                    importlib.import_module(node.module + "." + name)
                    resolved = True
                except ImportError:
                    resolved = False
            assert resolved, (
                f"{SCRIPT}: `from {node.module} import {name}` — '{name}' does not resolve "
                f"(stale symbol/submodule after a refactor?)")
            checked += 1
    assert checked > 0, f"{SCRIPT}: no maxgaffer imports found — did main() move?"


# ---------------------------------------------------------------- 2 the pymxs guard
def test_pymxs_guard_wraps_only_the_import_not_main():
    """The inside-Max message must be reachable ONLY from ``import pymxs`` failing — any
    other ImportError (a refactor that split a maxgaffer symbol out from under ``main()``)
    has to traceback. So ``main()`` lives in the guard's ``else``, never in its ``try``."""
    tree = ast.parse(_read(SCRIPT))
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Try)
              and any(isinstance(h.type, ast.Name) and h.type.id == "ImportError"
                      for h in n.handlers if h.type is not None)]
    assert guards, "the pymxs ImportError guard is gone"
    saw_main_in_else = False
    for g in guards:
        body_calls = {n.id for stmt in g.body for n in ast.walk(stmt)
                      if isinstance(n, ast.Name)}
        assert "main" not in body_calls, (
            "main() is inside the ImportError-guarded try again — a stale maxgaffer import "
            "would masquerade as 'not inside Max' (the round-3 lesson)")
        else_calls = {n.id for stmt in g.orelse for n in ast.walk(stmt)
                      if isinstance(n, ast.Name)}
        saw_main_in_else = saw_main_in_else or ("main" in else_calls)
    assert saw_main_in_else, "main() is no longer invoked from the pymxs guard's else branch"


# ---------------------------------------------------------------- 3 off-box import
def test_off_box_import_prints_notice_and_does_not_run_main(monkeypatch, capsys):
    """Off-box (no pymxs) the file is import/compile-only: it prints the inside-Max notice
    and never touches ``main()`` — so importing it here can never render, mutate, or hang."""
    monkeypatch.delitem(sys.modules, "pymxs", raising=False)
    mod = _load_module(SCRIPT)
    out = capsys.readouterr().out
    assert "must run INSIDE 3ds Max" in out
    assert mod.results == [], "main() ran during import even though pymxs was absent"


# ---------------------------------------------------------------- 4 no-V-Ray degrade path
def test_no_vray_degrade_path_survives_and_self_reports_degraded(tmp_path, monkeypatch, capsys):
    """With a hostile runtime that has NO V-Ray, ``main()`` must run every section to
    completion and never raise (surviving + logging is the whole point of the harness), and
    the sections the script itself gates on a constructable node must degrade, not crash."""
    # load WITHOUT pymxs so the bottom guard does NOT auto-run main() during exec — we drive
    # it ourselves with the report redirected off the repo and the temp dir inside tmp_path.
    monkeypatch.delitem(sys.modules, "pymxs", raising=False)
    mod = _load_module(SCRIPT)
    assert mod.results == []
    capsys.readouterr()                                 # drop the off-box notice

    _install_vray_less_runtime(monkeypatch)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))   # keep config.load hermetic
    report = tmp_path / "onbox_effects_report.json"
    monkeypatch.setattr(mod, "REPORT", str(report))

    workdir = tmp_path / "maxwork"                       # keep the harness's mkdtemp off %TEMP%
    workdir.mkdir()
    monkeypatch.setattr(mod.tempfile, "mkdtemp", lambda *a, **k: str(workdir))

    assert mod.main() is None, "main() must never propagate — a step crash is logged, not raised"

    # every one of the six isolated sections records exactly one result
    assert len(mod.results) == 6, [r["step"] for r in mod.results]
    by_name = {r["step"].split(":")[0]: r for r in mod.results}
    # the sections the SCRIPT gates on _make()->None (independent of any sibling probe
    # landing) must self-report degraded rather than blow up when V-Ray is absent
    for key in ("fog aliases", "keyframe baking", "multi-light"):
        r = by_name[key]
        assert r["ok"] is True, r
        assert isinstance(r["detail"], dict) and r["detail"].get("status") == "degraded", r

    # the report was written OFF the repo, carrying the honest V-Ray-absent header
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["vray_active"] is False
    assert len(data["results"]) == 6
