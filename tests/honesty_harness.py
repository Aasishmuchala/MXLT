"""A real Controller against a fake scene — the seam the honesty-layer tests drive.

Shared by test_cost.py, test_cancel_bound.py, test_gates.py and test_result_honesty.py.
Deliberately NOT a conftest fixture: these four files each want to bend a different part
of it, and a helper you can read top to bottom beats a fixture you have to go and find.

Two rules it exists to enforce:

  * patch ``_render_raw``, never ``_render_exposed``. Plate validation, the cancel latch
    and the timing all live IN ``_render_exposed`` (2026-07-31), so a harness that
    replaces that function replaces the thing under test;
  * the reference is a REAL file on disk, because preflight blocks a run whose reference
    is not — matching against a file that is not there produces numbers, not measurements.
"""

from __future__ import annotations

import types

from maxgaffer.core.director import MatchResult
from maxgaffer.core.genome import LightingState
from maxgaffer.core.scenarios import DEFAULT_SEMANTICS
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import controller as ctl

SEMANTICS = dict(DEFAULT_SEMANTICS)
SEMANTICS["key_notes"] = "test reference"
SEMANTICS["sun_bearing_agreement"] = 1.0
SEMANTICS["sun_bearing_spread_deg"] = 0.0

#: a plate that is lit, structured and not the same picture twice
LIT = {"mean_rgb": [0.35, 0.34, 0.33], "p": {"5": 0.1, "50": 0.4, "95": 0.85},
       "contrast": 0.75, "hot_frac": 0.04, "log_key": 0.22,
       "grid_log": [0.1, 0.2, 0.3, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0],
       "hot_grid": [0.04] * 25, "count": 65536}
BLACK = {"mean_rgb": [0.0, 0.0, 0.0], "p": {"5": 0.0, "95": 0.0}, "contrast": 0.0,
         "hot_frac": 0.0, "log_key": 1e-6, "grid_log": [0.0] * 9, "count": 65536}


def make_state(ev=12.0):
    st = LightingState()
    st.set("sun.enabled", 1)
    st.set("sun.azimuth_deg", 180.0)
    st.set("sun.altitude_deg", 30.0)
    st.set("sun.intensity", 1.0)
    st.set("exposure.ev", ev)
    st.set("exposure.wb_kelvin", 6500.0)
    st.groups["practicals"] = 1.0
    return st


def build(tmp_path, monkeypatch, **cfg_over):
    """→ a Controller wired to a fake scene, with .h holding the harness knobs."""
    cfg_over.setdefault("auto_exposure_control", False)
    cfg_over.setdefault("analyze_samples", 1)
    cfg_over.setdefault("preflight_level", "block")
    cfg_over.setdefault("uncertainty_policy", "assume")
    real_host = cfg_over.pop("real_exposure_host", False)
    cfg = cfgmod.Config(**cfg_over)
    c = ctl.Controller(cfg)

    scene = {"state": make_state()}
    applies = []

    def _apply(rig, b, st, cam=None, undo=True):
        applies.append(st.copy())
        scene["state"] = st.copy()
        return []

    sun = types.SimpleNamespace(name="SUN A", target=None)
    monkeypatch.setattr(ctl.sc, "scene_path", lambda: str(tmp_path / "t.max"))
    monkeypatch.setattr(ctl.sc, "classify_rig", lambda: {
        "sun": sun, "suns": [sun], "dome": None, "domes": [],
        "groups": {"practicals": [object()]}, "notes": [], "sky_env": False})
    monkeypatch.setattr(ctl.sc, "get_camera", lambda name, camera_id="": object())
    monkeypatch.setattr(ctl.sc, "set_active_camera", lambda name, camera_id="": True)
    monkeypatch.setattr(ctl.sc, "camera_yaw_deg", lambda cam: 0.0)
    monkeypatch.setattr(ctl.sc, "list_cameras",
                        lambda: [{"name": "CamA", "id": "1", "duplicate": False}])
    monkeypatch.setattr(ctl.sc, "active_camera_identity", lambda: "1")
    monkeypatch.setattr(ctl.sc, "active_camera_name", lambda: "CamA")
    monkeypatch.setattr(ctl.sc, "get_prop", lambda obj, names, default=None: default)
    monkeypatch.setattr(ctl.sc, "matched_prop", lambda obj, names: None)
    monkeypatch.setattr(ctl.sc, "_rt",
                        lambda: (_ for _ in ()).throw(ImportError("no pymxs")))
    monkeypatch.setattr(ctl.ap, "capture_baselines", lambda rig: {})
    monkeypatch.setattr(ctl.ap, "read_state",
                        lambda rig, b, cam=None: scene["state"].copy())
    monkeypatch.setattr(ctl.ap, "apply_state", _apply)
    monkeypatch.setattr(cfgmod, "sessions_dir", lambda: str(tmp_path / "sessions"))
    monkeypatch.setattr(c, "analyze_reference", lambda name: dict(SEMANTICS))
    monkeypatch.setattr(c, "_image_block", lambda path: {"type": "image"})
    if not real_host:
        monkeypatch.setattr(c, "_verify_exposure_host",
                            lambda *a, **k: None)  # its own tests own the canary
    monkeypatch.setattr(ctl.vt, "link_running", lambda *a, **k: None)

    ref = tmp_path / "ref.jpg"
    ref.write_bytes(b"\xff\xd8\xff\xe0a-real-file-on-disk")
    c.session.entry("CamA").reference = str(ref)
    monkeypatch.setattr(c, "ref_stats", lambda p: dict(LIT))

    c.h = types.SimpleNamespace(applies=applies, scene=scene, ref=str(ref),
                                tmp=tmp_path, renders=[], asked=[])
    return c


def stub_renders(ctrl, monkeypatch, stats=None, seconds=0.0, clock=None):
    """Route every render through a fake _render_raw and a fake stats reader.

    ``stats`` may be a dict (every plate the same) or a callable taking the render index.
    ``seconds`` advances the injected clock per render — tests must never sleep.
    """
    if stats is None:
        # VARY the plate by default. A harness that returns one constant dict makes every
        # render pixel-identical, which the frozen detector correctly reports — so a
        # constant fixture would make every unrelated test fight gate 4. Real renders of
        # different states differ; the fixture should too.
        def stats(i):
            out = dict(LIT)
            out["log_key"] = 0.20 + 0.001 * i
            out["grid_log"] = [0.1 + 0.001 * i] + LIT["grid_log"][1:]
            return out
    now = {"t": 1000.0}
    if clock is None:
        monkeypatch.setattr(ctl.time, "time", lambda: now["t"])

    def _raw(cam, out, w, h, **kw):
        ctrl.h.renders.append((out, w, h))
        now["t"] += float(seconds)
        ctrl._last_render_backend = "vray"
        return out

    def _stats(path):
        i = max(0, len(ctrl.h.renders) - 1)
        return dict(stats(i)) if callable(stats) else dict(stats)

    monkeypatch.setattr(ctrl, "_render_raw", _raw)
    monkeypatch.setattr(ctrl, "stats_for", _stats)
    return now


def stub_director(monkeypatch, fn):
    """Replace core.director.run_match as the controller sees it."""
    monkeypatch.setattr(ctl, "run_match", fn)


def done(score=88.0, state=None, **kw):
    return MatchResult(best_state=state or make_state(), best_score=score,
                       best_render=None, stop_reason="target_reached", **kw)


def quiet_sun(monkeypatch):
    """Silence the real sun solve.

    A test that wants to drive the frozen counter (or anything else) through the DIRECTOR
    must not also be running 44 real grid probes through the same counter — the stage that
    fires first would own the finding and the test would be measuring the wrong thing."""
    monkeypatch.setattr(ctl.sunsolve, "solve_sun_angles", lambda *a, **k: None)
