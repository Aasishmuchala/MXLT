"""Controller-layer bugfix regressions — each test encodes a reproduced defect.

The fake-scene fixture mirrors test_no_renders.py: bridge functions are monkeypatched
(applies persist so read-back agrees), no pymxs anywhere, LLM calls faked at the seam.
"""

import json
import os
import types

import pytest

from maxgaffer.core.director import MatchResult
from maxgaffer.core.genome import LightingState
from maxgaffer.core.scenarios import DEFAULT_SEMANTICS
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import controller as ctl

SEMANTICS = dict(DEFAULT_SEMANTICS)
SEMANTICS.setdefault("key_notes", "test reference")


def make_state():
    st = LightingState()
    st.set("sun.enabled", 1)
    st.set("sun.azimuth_deg", 180.0)
    st.set("sun.altitude_deg", 30.0)
    st.set("sun.intensity", 1.0)
    st.set("exposure.ev", 12.0)
    st.set("exposure.wb_kelvin", 6500.0)
    st.groups["practicals"] = 1.0
    return st


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    cfg = cfgmod.Config(no_renders=True, auto_exposure_control=False,
                        analyze_samples=1)
    c = ctl.Controller(cfg)
    applies = []
    scene = {"state": make_state()}

    def _apply(rig, b, st, cam=None, undo=True):
        applies.append((st.copy(), undo))
        scene["state"] = st.copy()
        return []

    monkeypatch.setattr(ctl.sc, "scene_path", lambda: str(tmp_path / "t.max"))
    monkeypatch.setattr(ctl.sc, "classify_rig", lambda: {
        "sun": object(), "dome": None, "groups": {"practicals": [object()]},
        "notes": [], "sky_env": False})
    monkeypatch.setattr(ctl.sc, "get_camera", lambda name: object())
    monkeypatch.setattr(ctl.sc, "set_active_camera", lambda name: None)
    monkeypatch.setattr(ctl.sc, "camera_yaw_deg", lambda cam: 0.0)
    monkeypatch.setattr(ctl.ap, "capture_baselines", lambda rig: {})
    monkeypatch.setattr(ctl.ap, "read_state",
                        lambda rig, b, cam=None: scene["state"].copy())
    monkeypatch.setattr(ctl.ap, "apply_state", _apply)
    monkeypatch.setattr(cfgmod, "sessions_dir", lambda: str(tmp_path / "sessions"))
    monkeypatch.setattr(c, "analyze_reference", lambda name: dict(SEMANTICS))
    monkeypatch.setattr(c, "_image_block", lambda path: {"type": "image"})
    monkeypatch.setattr(c, "ref_stats", lambda path: {"log_key": 0.2})
    c._test_applies = applies
    c._test_scene = scene
    return c


def bound(ctrl, name="CamA"):
    ctrl.session.entry(name).reference = "ref.jpg"
    return ctrl.session.cameras[name]


# ------------------------------------------- (a) cancel/fail must not wipe accepted state
@pytest.mark.parametrize("reason", ["cancelled", "render_failed"])
def test_unmeasured_run_keeps_previous_state(ctrl, monkeypatch, reason):
    """run_match returning best_score=None with stop_reason cancelled/render_failed used
    to record_match(best_state, None) — overwriting the accepted state+score with an
    unmeasured first guess."""
    ctrl.cfg.no_renders = False
    bound(ctrl)
    accepted = make_state()
    accepted.set("exposure.ev", 9.0)
    ctrl.session.record_match("CamA", accepted, 88.0)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=None, best_render=None, stop_reason=reason))
    logs = []
    ctrl.run_match("CamA", logs.append)
    e = ctrl.session.cameras["CamA"]
    assert e.state.get("exposure.ev") == 9.0               # accepted state intact
    assert e.score == 88.0                                 # accepted score intact
    assert any("kept previous lighting" in ln for ln in logs)
    # and the scene was put back to the accepted light
    assert ctrl._test_applies[-1][0].get("exposure.ev") == 9.0


def test_unmeasured_run_falls_back_to_pre_match(ctrl, monkeypatch):
    """No accepted state yet → re-apply the pre-match snapshot, leave state/score empty."""
    ctrl.cfg.no_renders = False
    bound(ctrl)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=None, best_render=None,
        stop_reason="render_failed"))
    logs = []
    ctrl.run_match("CamA", logs.append)
    e = ctrl.session.cameras["CamA"]
    assert e.state is None and e.score is None             # nothing recorded
    # pre_match (EV 12.0 from the fake scene) was re-applied after the fake loop's mess
    assert ctrl._test_applies[-1][0].get("exposure.ev") == 12.0


def test_measured_run_still_records(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    bound(ctrl)
    best = make_state()
    best.set("exposure.ev", 10.0)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=best, best_score=91.0, best_render=None,
        stop_reason="target_reached"))
    ctrl.run_match("CamA", lambda ln: None)
    e = ctrl.session.cameras["CamA"]
    assert e.score == 91.0 and e.state.get("exposure.ev") == 10.0


# ------------------------------------------- (b) draft restore covers verify + sweep
def test_draft_restored_when_exposure_check_raises(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    ctrl.cfg.draft_sampler = True
    bound(ctrl)
    restored = []
    monkeypatch.setattr(ctl.df, "apply_draft", lambda: ["draft: on"])
    monkeypatch.setattr(ctl.df, "pending_snapshot", lambda: True)
    monkeypatch.setattr(ctl.df, "restore_draft",
                        lambda: restored.append(True) or ["draft: off"])

    def boom(*a, **k):
        raise RuntimeError("exposure probe exploded")

    monkeypatch.setattr(ctrl, "_verify_exposure_host", boom)
    with pytest.raises(RuntimeError):
        ctrl.run_match("CamA", lambda ln: None)
    assert restored == [True]


def test_draft_restored_when_sweep_raises(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    ctrl.cfg.draft_sampler = True
    bound(ctrl)
    restored = []
    monkeypatch.setattr(ctl.df, "apply_draft", lambda: ["draft: on"])
    monkeypatch.setattr(ctl.df, "pending_snapshot", lambda: True)
    monkeypatch.setattr(ctl.df, "restore_draft",
                        lambda: restored.append(True) or ["draft: off"])

    def boom(*a, **k):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(ctl, "run_sun_sweep", boom)
    with pytest.raises(RuntimeError):
        ctrl.run_match("CamA", lambda ln: None, do_sweep=True)
    assert restored == [True]


# ------------------------------------------- (c) board restores the scene on a raise
def test_board_reapplies_pre_board_state_when_apply_raises(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    bound(ctrl)
    monkeypatch.setattr(ctl.rd, "render_frame", lambda *a, **k: None)
    calls = []

    def flaky_apply(rig, b, st, cam=None, undo=True):
        calls.append(st.copy())
        if len(calls) == 2:                # the FIRST candidate's apply explodes
            raise RuntimeError("apply exploded")
        return []

    monkeypatch.setattr(ctl.ap, "apply_state", flaky_apply)
    with pytest.raises(RuntimeError):
        ctrl.run_scenarios("CamA", lambda ln: None)
    assert calls, "at least one candidate was applied"
    # the finally re-applied the pre-board light — the scene is not stranded mid-board
    assert calls[-1].diff(make_state()) == {}


# ------------------------------------------- (d) double-junk PLAN degrades, never aborts
def test_make_plan_double_junk_returns_none(ctrl, monkeypatch):
    bound(ctrl)
    monkeypatch.setattr(ctl.dg, "build_digest", lambda: {"lights": [], "cameras": []})
    monkeypatch.setattr(ctl.scenedigest, "catalog", lambda raw: {})
    monkeypatch.setattr(ctl.scenedigest, "to_text", lambda raw: "digest")
    monkeypatch.setattr(ctl.planner, "plan_user_text", lambda *a: "plan please")
    monkeypatch.setattr(ctl.omega, "call", lambda *a, **k: "not json at all")

    def junk(*a, **k):
        raise ctl.ParseError("junk")

    monkeypatch.setattr(ctl.planner, "validate_plan", junk)
    logs = []
    assert ctrl.make_plan("CamA", logs.append) is None
    assert any("plan skipped" in ln for ln in logs)


# ------------------------------------------- (e) refine honors the passed locks
def test_refine_note_deltas_respect_locks(ctrl):
    """no-render path, behavior-level: a locked param is not nudged by the note."""
    bound(ctrl)
    ctrl.refine("CamA", "exposure is too much", lambda ln: None,
                locks={"exposure.ev"})
    applied = ctrl._test_applies[-1][0]
    assert applied.get("exposure.ev") == 12.0              # lock vetoed the darker nudge


def test_refine_defaults_to_persisted_locks(ctrl):
    bound(ctrl).locks = {"exposure.ev"}
    ctrl.refine("CamA", "exposure is too much", lambda ln: None)   # locks=None → e.locks
    assert ctrl._test_applies[-1][0].get("exposure.ev") == 12.0


def _fake_inner_match(captured):
    def fake(camera_name, log, should_cancel=lambda: False, locks=None, **kw):
        captured["locks"] = locks
        return MatchResult(best_state=make_state(), best_score=90.0,
                           best_render=None, stop_reason="target_reached")
    return fake


def test_refine_passes_locks_to_inner_match(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    bound(ctrl)
    captured = {}
    monkeypatch.setattr(ctrl, "run_match", _fake_inner_match(captured))
    monkeypatch.setattr(ctl.rd, "render_frame", lambda *a, **k: None)
    monkeypatch.setattr(ctl.omega, "call", lambda *a, **k: "junk")   # lenses die quietly
    ctrl.refine("CamA", "too warm", lambda ln: None, locks={"exposure.ev"})
    assert captured["locks"] == {"exposure.ev"}


def test_refine_ensemble_rejects_locked_params(ctrl, monkeypatch):
    """A lens proposing a locked param must get zero accepted changes."""
    ctrl.cfg.no_renders = False
    bound(ctrl)
    monkeypatch.setattr(ctrl, "run_match", _fake_inner_match({}))
    monkeypatch.setattr(ctl.rd, "render_frame", lambda *a, **k: None)
    monkeypatch.setattr(ctl.omega, "call", lambda *a, **k: json.dumps({
        "assessment": "", "stop": False,
        "changes": [{"param": "exposure.ev", "value": 9.0, "why": "darker"}]}))
    logs = []
    ctrl.refine("CamA", "a note", logs.append, locks={"exposure.ev"})
    assert any("no valid changes" in ln for ln in logs)


# ------------------------------------------- (f) pre_seed restores without pre_match
def test_restore_pre_seed_without_pre_match(ctrl, monkeypatch):
    dome = object()
    monkeypatch.setattr(ctl.sc, "classify_rig", lambda: {
        "sun": None, "dome": dome, "groups": {}, "notes": [], "sky_env": False})
    tex_calls = []
    monkeypatch.setattr(ctl.sc, "set_prop",
                        lambda *a: tex_calls.append(a) or "texmap_on")
    monkeypatch.setattr(ctl.sc, "write_dome_rotation", lambda d, r: "node_z_rotation")
    e = bound(ctrl)
    e.pre_match = None                                     # fresh camera: seed only
    e.pre_seed = {"file": "", "rotation": 45.0}
    e.seed_hdri = "/runs/seed_CamA_x.hdr"
    assert ctrl.restore_pre_match("CamA") is True          # was unreachable (returned False)
    assert e.pre_seed == {} and e.seed_hdri == ""
    assert any(a[0] is dome for a in tex_calls)            # dome texture switch touched


def test_restore_pre_match_still_applies_snapshot(ctrl):
    e = bound(ctrl)
    e.pre_match = make_state()
    e.pre_seed = {}
    assert ctrl.restore_pre_match("CamA") is True
    assert ctrl._test_applies[-1][0].get("exposure.ev") == 12.0


def test_restore_nothing_returns_false(ctrl):
    bound(ctrl)
    assert ctrl.restore_pre_match("CamA") is False


# ------------------------------------------- (g) _safe / _stamp collisions
def test_safe_disambiguates_sanitized_names():
    assert ctl._safe("CamA") == "CamA"                     # stable names byte-identical
    assert ctl._safe("Cam/A") != ctl._safe("Cam\\A")       # was the same "Cam_A"
    assert ctl._safe("Cam/A").startswith("Cam_A_")
    assert ctl._safe("Cam/A") == ctl._safe("Cam/A")        # stable hash


def test_stamp_has_subsecond_resolution(monkeypatch):
    import re

    now = [1752864500.001]
    monkeypatch.setattr(ctl.time, "time", lambda: now[0])
    s1 = ctl._stamp()
    now[0] = 1752864500.002                                # same wall-clock second
    s2 = ctl._stamp()
    assert s1 != s2                                        # run dirs no longer collide
    assert re.fullmatch(r"\d{8}-\d{6}-\d{3}", s1)          # still chronologically sortable


# ------------------------------------------- (h) contested-consensus flag lifecycle
def _fresh_analyzer(ctrl, monkeypatch, samples):
    """Un-fake the fixture's instance-level analyze_reference; drive the real one."""
    del ctrl.analyze_reference
    monkeypatch.setattr(ctl.omega, "call", lambda *a, **k: "{}")
    replies = iter(samples)
    monkeypatch.setattr(ctl, "validate_analysis", lambda reply: dict(next(replies)))


def test_fresh_unanimous_analysis_resets_stale_agreement_flag(ctrl, monkeypatch):
    ctrl.cfg.analyze_samples = 2
    bound(ctrl)
    ctrl._last_analyze_agreement = 0.5                     # stale from a previous run
    _fresh_analyzer(ctrl, monkeypatch, [SEMANTICS, SEMANTICS])
    ctrl.analyze_reference("CamA")
    assert getattr(ctrl, "_last_analyze_agreement", None) is None


def test_contested_analysis_still_sets_the_flag(ctrl, monkeypatch):
    ctrl.cfg.analyze_samples = 2
    bound(ctrl)
    other = dict(SEMANTICS)
    other["time_of_day"] = "night"                         # samples disagree
    _fresh_analyzer(ctrl, monkeypatch, [SEMANTICS, other])
    ctrl.analyze_reference("CamA")
    assert ctrl._last_analyze_agreement is not None
    assert ctrl._last_analyze_agreement < 1.0


# ------------------------------------------- (i) sidecar I/O runs on the io runner
def test_sidecar_stats_runs_on_io_runner(ctrl, monkeypatch, tmp_path):
    py = tmp_path / "python.exe"
    py.write_bytes(b"x")
    ctrl.cfg.system_python = str(py)
    used = []

    def io_spy(fn):
        used.append(fn)
        return fn()

    ctrl.io = io_spy
    monkeypatch.setattr(ctl.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout="null"))
    assert ctrl.stats_for("/nonexistent/ref.jpg") is None  # sidecar ran, found nothing
    assert used, "the sidecar subprocess did not go through the io runner"


def test_image_block_sidecar_runs_on_io_runner(ctrl, monkeypatch, tmp_path):
    del ctrl._image_block                                  # undo the fixture's fake
    py = tmp_path / "python.exe"
    py.write_bytes(b"x")
    ctrl.cfg.system_python = str(py)
    used = []
    ctrl.io = lambda fn: (used.append(fn), fn())[1]
    payload = json.dumps([{"b64": "aGVsbG8=", "media_type": "image/jpeg"}])
    monkeypatch.setattr(ctl.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=payload))
    img = tmp_path / "ref.jpg"
    img.write_bytes(b"not actually a jpeg")                # Pillow path fails → sidecar
    block = ctrl._image_block(str(img))
    assert used, "the sidecar subprocess did not go through the io runner"
    assert block is not None and "image" in str(block)


# ------------------------------------------- (j) finals cancellation + fault isolation
def _three_matched(ctrl):
    for name in ("A", "B", "C"):
        ctrl.session.record_match(name, make_state(), 80.0)


def test_finals_cancel_between_cameras(ctrl, monkeypatch, tmp_path):
    ctrl.cfg.no_renders = False
    _three_matched(ctrl)
    rendered = []
    monkeypatch.setattr(ctrl, "_render_exposed",
                        lambda cam, out, w, h, **kw: rendered.append(out) or out)
    out = ctrl.render_finals_vray(["A", "B", "C"], str(tmp_path / "out"),
                                  lambda n, s: None,
                                  should_cancel=lambda: len(rendered) >= 1)
    assert out == {"A": "ok", "B": "cancelled", "C": "cancelled"}
    assert len(rendered) == 1


def test_finals_one_camera_failure_does_not_abort_batch(ctrl, monkeypatch, tmp_path):
    ctrl.cfg.no_renders = False
    _three_matched(ctrl)

    def flaky_render(cam, out, w, h, **kw):
        if out.endswith("A.png"):
            raise RuntimeError("boom")
        return out

    monkeypatch.setattr(ctrl, "_render_exposed", flaky_render)
    out = ctrl.render_finals_vray(["A", "B", "C"], str(tmp_path / "out"),
                                  lambda n, s: None)
    assert out["A"].startswith("error:")
    assert out["B"] == "ok" and out["C"] == "ok"


def test_vantage_jobs_thread_should_cancel(ctrl, monkeypatch):
    ctrl.cfg.no_renders = False
    captured = {}

    def fake_stills(jobs, exe, w, h, on_progress, should_cancel=None):
        captured["should_cancel"] = should_cancel
        return {j["camera"]: "ok" for j in jobs}

    monkeypatch.setattr(ctl.vt, "render_stills", fake_stills)
    jobs = [{"camera": "A", "scene_file": "a.vrscene", "output": "a.png"}]
    out = ctrl.run_vantage_jobs(jobs, lambda n, s: None, should_cancel=lambda: True)
    assert captured["should_cancel"] is not None
    assert captured["should_cancel"]() is True
    assert out == {"A": "ok"}


# ------------------------------------------- (k) refs transcode cache is pruned
def test_refs_dir_pruned_under_keep_runs(ctrl):
    ctrl.cfg.keep_runs = 3
    refs = ctrl._ensure_run_dir("refs")
    for i in range(6):
        p = os.path.join(refs, f"ref_{i}.png")
        with open(p, "wb") as f:
            f.write(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    ctrl._ensure_run_dir("refs")                           # access prunes oldest > keep
    assert sorted(os.listdir(refs)) == ["ref_3.png", "ref_4.png", "ref_5.png"]


def test_refs_pruning_disabled_at_zero(ctrl):
    ctrl.cfg.keep_runs = 0
    refs = ctrl._ensure_run_dir("refs")
    for i in range(4):
        with open(os.path.join(refs, f"llm_{i}.png"), "wb") as f:
            f.write(b"x")
    ctrl._ensure_run_dir("refs")
    assert len(os.listdir(refs)) == 4
