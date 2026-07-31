"""Controller-layer bugfix regressions — each test encodes a reproduced defect.

The fake-scene fixture mirrors test_no_renders.py: bridge functions are monkeypatched
(applies persist so read-back agrees), no pymxs anywhere, LLM calls faked at the seam.
"""

import json
import os
import types

import pytest

from maxgaffer.core.director import MatchResult
from maxgaffer.core.errors import PreflightBlocked
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
    c._test_tmp = tmp_path
    return c


def bound(ctrl, name="CamA"):
    """Bind a REAL file on disk as the reference.

    It used to be the string "ref.jpg" and nothing checked. Preflight's REFERENCE_PRESENT
    now blocks a run whose reference is not on disk — the 2026-07-30 lesson is that a
    match against a file that is not there produces numbers, not measurements — so the
    fixture has to be as honest as the product is."""
    path = ctrl._test_tmp / "ref.jpg"
    if not path.exists():
        path.write_bytes(b"\xff\xd8\xff\xe0not-really-a-jpeg-but-it-is-on-disk")
    ctrl.session.entry(name).reference = str(path)
    return ctrl.session.cameras[name]


def test_select_deleted_camera_fails_before_applying_its_saved_light(ctrl, monkeypatch):
    ctrl.session.record_match("Gone", make_state(), 80.0)
    monkeypatch.setattr(ctl.sc, "set_active_camera", lambda name: False)

    with pytest.raises(RuntimeError, match="no longer available"):
        ctrl.select_camera("Gone")

    assert ctrl._test_applies == []


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
    monkeypatch.setattr(ctl.df, "apply_draft", lambda seconds=0.0: ["draft: on"])
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
    monkeypatch.setattr(ctl.df, "apply_draft", lambda seconds=0.0: ["draft: on"])
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


# ------------------------------------------- (l) the draft gate says which way it is set
def test_draft_off_is_stated_in_the_transcript(ctrl, monkeypatch):
    """The 2026-07-30 bug was UNFALSIFIABLE from the log: draft_sampler:true and a 20 s
    probe cap sat in config.json from 15:27, the Options menu wrote false over them at
    match time, and not one line said the branch was False. apply_draft always emits at
    least one line on every path, so silence was the one state nobody could read."""
    ctrl.cfg.no_renders = False
    ctrl.cfg.draft_sampler = False
    ctrl.cfg.probe_max_seconds = 20.0
    bound(ctrl)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=90.0, best_render=None,
        stop_reason="target_reached"))
    logs = []
    ctrl.run_match("CamA", logs.append)
    off = [ln for ln in logs if ln.startswith("draft: OFF")]
    assert off, "a closed gate must say so"
    assert "20" in off[0] and "NOT in effect" in off[0]


def test_draft_on_hands_apply_draft_the_configured_cap(ctrl, monkeypatch):
    """One source of truth: the CAP travels with the gate, from the same cfg object."""
    ctrl.cfg.no_renders = False
    ctrl.cfg.draft_sampler = True
    ctrl.cfg.probe_max_seconds = 20.0
    bound(ctrl)
    seen = []
    monkeypatch.setattr(ctl.df, "apply_draft",
                        lambda seconds=None: seen.append(seconds) or ["draft: on"])
    monkeypatch.setattr(ctl.df, "pending_snapshot", lambda: False)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=90.0, best_render=None,
        stop_reason="target_reached"))
    ctrl.run_match("CamA", lambda ln: None)
    assert seen == [20.0]


# ------------------------------------------- (m) the Vantage probe backend
def test_vantage_plate_is_never_software_exposed(ctrl, monkeypatch):
    """A Vantage plate ALREADY carries its exposure and its own tonemap. Software
    exposure assumes a PRE-exposure buffer, so applying it here would multiply the same
    EV delta in twice, the solver would measure the residual and ask for another step,
    and the loop would overshoot ~2x -- the failure _plate_linear documents."""
    ctrl.cfg.software_exposure = True
    ctrl._plate_linear = True
    ctrl._probe_backend = "vantage"
    touched = []
    monkeypatch.setattr(ctl.expose, "display_encode_png",
                        lambda *a, **k: touched.append("encode") or "p")
    monkeypatch.setattr(ctl.expose, "expose_image_file",
                        lambda *a, **k: touched.append("expose") or "p")
    monkeypatch.setattr(ctl.rd, "render_probe", lambda *a, **k: ("p", "vantage"))
    st = make_state()

    assert ctrl._render_exposed(object(), "p.png", 8, 8, state=st, probe=True) == "p"
    assert touched == []

    # ...and the same call on a V-Ray plate is exposed exactly as before
    ctrl._probe_backend = "vray"
    monkeypatch.setattr(ctl.rd, "render_frame", lambda cam, out, w, h: "p")
    ctrl._render_exposed(object(), "p.png", 8, 8, state=st, probe=True)
    assert touched == ["encode", "expose"]


def test_a_refused_vantage_grab_is_skipped_not_replaced_by_a_vray_plate(ctrl,
                                                                        monkeypatch):
    """G-2, 2026-07-31. render_probe used to fall through to render_frame on a refusal,
    so probe a210 came back a V-Ray plate appended to the SAME sunsolve table and the same
    argmax as the tonemapped Vantage grabs before it — and highlight_similarity's presence
    half is gated on an ABSOLUTE luminance threshold, so that is a systematic offset
    between the two halves of one comparison. A refused grab is now a SKIP.

    And one refusal no longer disarms the run: _settled_rows reads a legitimately static
    dusk viewport as "stale", so a single settle timeout used to disarm the fast path on
    exactly the scene class where 60 s → 50 ms matters most. Three in a row does."""
    ctrl._probe_backend = "vantage"
    monkeypatch.setattr(ctl.rd, "render_probe", lambda *a, **k: (None, "vantage"))
    logs = []
    for _ in range(2):
        assert ctrl._render_exposed(object(), "p.png", 8, 8, probe=True,
                                    log=logs.append) is None
        assert ctrl._probe_backend == "vantage"      # one, then two, is not a verdict
    assert ctrl._render_exposed(object(), "p.png", 8, 8, probe=True,
                                log=logs.append) is None
    assert ctrl._probe_backend == "vray"             # three consecutive is
    assert any("demoted to V-Ray" in ln for ln in logs)


def test_a_refused_grab_never_reaches_render_frame_at_all(ctrl, monkeypatch):
    """The half of the name above that nothing tested. The stub there was
    ``lambda *a, **k: (None, "vantage")`` — a ``**k`` that swallows the very kwarg the
    behaviour lives in, so reverting ``fallback=False`` to ``fallback=True`` passed it
    unchanged. This drives the REAL render_probe and fails the test if V-Ray is asked to
    substitute a plate into the sunsolve table. (2026-07-31)"""
    ctrl._probe_backend = "vantage"
    monkeypatch.setattr(ctl.rd.vgrab, "capture_window_png",
                        lambda *a, **k: None)
    monkeypatch.setattr(ctl.rd.vgrab, "last_error", lambda: "no live link is streaming")
    monkeypatch.setattr(ctl.rd, "render_frame",
                        lambda *a, **k: pytest.fail(
                            "a refused Vantage grab must be SKIPPED, never replaced by a "
                            "V-Ray plate in the same ranked table"))
    logs = []
    assert ctrl._render_exposed(object(), "p.png", 8, 8, probe=True,
                                log=logs.append) is None
    assert any("SKIPPED rather than mixed into the grid" in ln for ln in logs)


def test_an_empty_preflight_report_does_not_arm_the_grab(ctrl, monkeypatch):
    """2026-07-31. run_match always passes a report object and never None, so
    preflight_level='off' (or a preflight that raised and was degraded) handed
    _arm_probe_backend an EMPTY report — and reading only ``demotions`` armed the window
    grab having checked nothing at all: not the live link, not the window, not the
    checklist #14 GPU conflict, not the camera."""
    from maxgaffer.maxbridge import preflight as pf
    from maxgaffer.maxbridge import vgrab

    ctrl.cfg.probe_backend = "vantage"
    monkeypatch.setattr(ctl.vt, "link_running", lambda *a, **k: None)
    monkeypatch.setattr(vgrab, "find_window",
                        lambda *a, **k: pytest.fail("unreachable if it armed blindly"))
    logs = []
    ctrl._arm_probe_backend(logs.append, report=pf.PreflightReport())
    assert ctrl._probe_backend == "vray"
    assert any("rendering every probe in V-Ray" in ln for ln in logs)


def test_the_report_decision_is_what_arms_it_in_either_direction(ctrl, monkeypatch):
    from maxgaffer.maxbridge import preflight as pf
    from maxgaffer.maxbridge import vgrab

    ctrl.cfg.probe_backend = "vantage"
    monkeypatch.setattr(vgrab, "find_window",
                        lambda *a, **k: pytest.fail("the decision is preflight's"))
    yes = pf.PreflightReport(demotions={"probe_backend": "vantage"})
    ctrl._arm_probe_backend(lambda _ln: None, report=yes)
    assert ctrl._probe_backend == "vantage"

    no = pf.PreflightReport(demotions={"probe_backend": "vray"})
    logs = []
    ctrl._arm_probe_backend(logs.append, report=no)
    assert ctrl._probe_backend == "vray"
    assert any("see the preflight line above" in ln for ln in logs)


def test_an_unpreflighted_arming_says_that_it_was_not_preflighted(ctrl, monkeypatch):
    """I6. The direct-probe fallback is still allowed — it is the documented behaviour for
    automation with preflight off — but it must not read like a verified arming."""
    from maxgaffer.maxbridge import vgrab

    ctrl.cfg.probe_backend = "vantage"
    monkeypatch.setattr(ctl.vt, "link_running", lambda *a, **k: 20701)
    monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: 4242)
    logs = []
    ctrl._arm_probe_backend(logs.append, report=None)
    assert ctrl._probe_backend == "vantage"
    assert any("NOT preflighted" in ln for ln in logs)


def test_only_the_sun_solve_uses_the_probe_backend(ctrl, monkeypatch):
    """Named by PREFIX, not by exception. The global sun solve is the one stage whose
    probes exist purely to be RANKED against each other and whose output is an angle, and
    it is 44 of the 133 probes in a standard profile. Its neighbours are excluded because
    each of them compares ACROSS sources: sunsolve_tonealign sets the artist's EXPOSURE
    via solve_ev; sweep_basin_* scores on the FULL weighted critic (min over every
    measured component) and caps the whole run; and sweep### is cosined against the
    reference PHOTO's 3x3 grid and uploaded to the model by llm_pick."""
    ctrl.cfg.no_renders = False
    bound(ctrl)
    monkeypatch.setattr(ctrl, "ref_stats", lambda p: None)   # no tone-align / sun solve
    seen = []
    monkeypatch.setattr(ctrl, "_render_exposed",
                        lambda cam, out, w, h, **kw: seen.append(
                            (os.path.basename(out)[:-4], bool(kw.get("probe")))) or out)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for tag in ("sunsolve_a007", "sunsolve_fine_a012", "sunsolve_tonealign",
                    "sweep090", "sweep_basin_dusk", "iter00", "polish_a"):
            hooks.render(tag)
        return MatchResult(best_state=make_state(), best_score=90.0,
                           best_render=None, stop_reason="target_reached")

    monkeypatch.setattr(ctl, "run_match", fake)
    ctrl.run_match("CamA", lambda ln: None, multi_start=False)
    assert dict(seen) == {"sunsolve_a007": True, "sunsolve_fine_a012": True,
                          "sunsolve_tonealign": False, "sweep090": False,
                          "sweep_basin_dusk": False,
                          "iter00": False, "polish_a": False}


def test_arming_needs_a_live_link_and_a_window_and_clears_the_settle_baseline(
        ctrl, monkeypatch):
    """An unarmed backend costs nothing; an armed-but-broken one costs a log line per
    probe. And the freshness chain is inductive, so a new run must not inherit the last
    run's final frame as the thing its first probe has to differ from."""
    from maxgaffer.maxbridge import vgrab

    ctrl.cfg.probe_backend = "vantage"
    vgrab._LAST_SIGNATURE = [1.0] * 48
    monkeypatch.setattr(ctl.vt, "link_running", lambda *a, **k: 20701)
    monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: 4242)
    logs = []
    ctrl._arm_probe_backend(logs.append)
    assert ctrl._probe_backend == "vantage"
    assert vgrab._LAST_SIGNATURE is None
    assert any("VANTAGE" in ln for ln in logs)

    # ...and every way of not being ready degrades to V-Ray, out loud
    for port, hwnd in ((None, 4242), (20701, None)):
        monkeypatch.setattr(ctl.vt, "link_running", lambda *a, **k: port)
        monkeypatch.setattr(vgrab, "find_window", lambda *a, **k: hwnd)
        logs = []
        ctrl._arm_probe_backend(logs.append)
        assert ctrl._probe_backend == "vray"
        assert any("rendering every probe in V-Ray" in ln for ln in logs)

    ctrl.cfg.probe_backend = "vray"
    monkeypatch.setattr(vgrab, "find_window",
                        lambda *a, **k: pytest.fail("nothing may be probed for"))
    ctrl._arm_probe_backend(lambda ln: pytest.fail("and nothing may be said"))
    assert ctrl._probe_backend == "vray"


def test_plan_probe_and_final_stay_on_vray(ctrl, monkeypatch, tmp_path):
    """The hard requirement: the plan-effect measurement and the DELIVERED render never
    move off V-Ray, whatever the probe backend is armed to."""
    ctrl.cfg.no_renders = False
    ctrl._probe_backend = "vantage"
    bound(ctrl)
    used = []
    monkeypatch.setattr(ctl.rd, "render_frame",
                        lambda cam, out, w, h: used.append(out) or out)
    monkeypatch.setattr(ctl.rd, "render_probe",
                        lambda *a, **k: pytest.fail("this render must never reach Vantage"))
    monkeypatch.setattr(ctrl, "stats_for", lambda p: {"log_key": 0.2})
    monkeypatch.setattr(ctl.critic, "score",
                        lambda ref, cur, w: types.SimpleNamespace(score=77.0))
    assert ctrl.probe_score("CamA", "preplan") == 77.0

    ctrl.session.record_match("CamA", make_state(), 90.0)
    assert ctrl.render_finals_vray(["CamA"], str(tmp_path / "out"),
                                   lambda n, s: None) == {"CamA": "ok"}
    assert len(used) == 2


# ------------------------------------------- (n) the black-frame guard
_BLACK = {"mean_rgb": [0.0, 0.0, 0.0], "p": {"95": 0.0}, "log_key": 1e-6}


def _black_probe_harness(ctrl, monkeypatch, stats):
    """Patch _render_RAW, not _render_exposed.

    2026-07-31: plate validation moved INTO _render_exposed, so a harness that replaces
    that function replaces the thing under test. _render_raw is the seam below it — the
    part that actually talks to V-Ray or vgrab — so patching there keeps the validation,
    the timing and the cancel latch in the picture, which is the point."""
    ctrl.cfg.no_renders = False
    ctrl.cfg.draft_sampler = True
    bound(ctrl)
    monkeypatch.setattr(ctrl, "ref_stats", lambda p: None)
    rendered = []
    monkeypatch.setattr(ctrl, "_render_raw",
                        lambda cam, out, w, h, **kw: rendered.append(out) or out)
    monkeypatch.setattr(ctrl, "stats_for", lambda p: dict(stats))
    restored = []
    monkeypatch.setattr(ctl.df, "apply_draft", lambda seconds=0.0: ["draft: on"])
    monkeypatch.setattr(ctl.df, "pending_snapshot", lambda: True)
    monkeypatch.setattr(ctl.df, "restore_draft",
                        lambda: restored.append(True) or ["draft: off"])
    return rendered, restored


# --------------------------------------------------------- PREFLIGHT B: the canary
def _plate(tmp_path, name, w, h, value=90):
    from maxgaffer.core import png_min

    rows = [[(value, value, value)] * w for _ in range(h)]
    return png_min.write_png_rgb(str(tmp_path / name), rows)


_LIT_STATS = {"mean_rgb": [0.3, 0.3, 0.3], "p": {"5": 0.1, "95": 0.9}, "contrast": 0.8,
              "hot_frac": 0.02, "log_key": 0.2}
_BLACK_STATS = {"mean_rgb": [0.0, 0.0, 0.0], "p": {"5": 0.0, "95": 0.0}, "contrast": 0.0,
                "hot_frac": 0.0, "log_key": 1e-6}


def test_run_match_points_the_viewport_at_the_camera_even_with_preflight_off(
        ctrl, monkeypatch):
    """The call was DELETED from run_match and left only inside check_camera_active, and
    pf.run returns before building a Context when preflight_level='off'. V-Ray is
    unaffected — it renders the camera node it is handed — but the Vantage live link
    mirrors the ACTIVE VIEWPORT camera, so every direction probe was a grab of whatever
    shot the artist last looked at, ranked against the reference. (2026-07-31)"""
    bound(ctrl)
    ctrl.cfg.preflight_level = "off"
    pointed = []
    monkeypatch.setattr(ctl.sc, "set_active_camera",
                        lambda name, camera_id="": pointed.append(name) or True)
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=90.0, best_render=None,
        stop_reason="target_reached"))
    ctrl.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)
    assert pointed == ["CamA"]


def test_the_canary_blocks_when_the_renderer_saved_no_file(ctrl):
    """Three BLOCK conditions and the only coverage any of them had was
    `any("canary" in ln for ln in log.lines)` — a log-string presence check that passes
    against a canary which never blocks at all. (2026-07-31)"""
    with pytest.raises(PreflightBlocked, match="no file at all"):
        ctrl._canary(None, None, 1.0, lambda _m: None)


def test_the_canary_blocks_on_the_wrong_size(ctrl):
    """render_frame has three layered size spellings, the third of which mutates
    rt.renderWidth/Height globally. A build where all three fall through saves a valid
    frame at SCENE resolution, compute_stats downsamples it to 256 px, and every number
    looks entirely normal while every "probe-resolution" frame costs full price."""
    path = _plate(ctrl._test_tmp, "big.png", 320, 180)
    with pytest.raises(PreflightBlocked, match="came back 320×180"):
        ctrl._canary(path, dict(_LIT_STATS), 1.0, lambda _m: None)


def test_the_canary_accepts_the_size_it_asked_for(ctrl):
    path = _plate(ctrl._test_tmp, "ok.png", 160, 90)
    logs = []
    ctrl._canary(path, dict(_LIT_STATS), 12.0, logs.append)
    assert any("160×90 frame in 12.0 s" in ln for ln in logs)


def test_a_black_canary_with_a_named_cause_blocks_and_without_one_only_reports(ctrl):
    """DR-on-with-a-dead-port PLUS a black plate is not an unlit scene, it is a dead
    renderer. Without that pairing a black canary is legitimate — nothing has been applied
    yet — and blocking would lock an artist out of the scene they came to light."""
    path = _plate(ctrl._test_tmp, "dark.png", 160, 90, value=0)
    logs = []
    ctrl._preflight_black_cause = ""
    ctrl._canary(path, dict(_BLACK_STATS), 1.0, logs.append)
    assert any("100% BLACK" in ln and ln.startswith("⚠") for ln in logs)

    ctrl._preflight_black_cause = "distributed rendering is ON and NOTHING is listening"
    with pytest.raises(PreflightBlocked, match="preflight already named the cause"):
        ctrl._canary(path, dict(_BLACK_STATS), 1.0, logs.append)


def test_preflight_warn_does_not_block_the_run_through_the_canary(ctrl, monkeypatch):
    """preflight_level='warn' promises "report everything, stop for nothing". A Finding
    keeps its authored severity in the report even when pf.run downgraded it, so reading
    .severity alone made 'warn' STRICTER than 'off' — the canary raised PreflightBlocked
    on a legitimately unlit scene. (2026-07-31)"""
    from maxgaffer.maxbridge import preflight as pf

    bound(ctrl)
    ctrl.cfg.no_renders = False
    report = pf.PreflightReport(findings=[
        pf.Finding("DR_DEAD_PORT", "block", "nothing is listening on 192.168.1.50")])
    monkeypatch.setattr(pf, "run", lambda *a, **k: report)
    monkeypatch.setattr(ctl.rd, "render_frame",
                        lambda cam, out, w, h: _plate(ctrl._test_tmp, "c.png", 160, 90))
    monkeypatch.setattr(ctl.metrics, "compute_stats", lambda p, **k: dict(_LIT_STATS))
    monkeypatch.setattr(ctl, "run_match", lambda *a, **k: MatchResult(
        best_state=make_state(), best_score=90.0, best_render=None,
        stop_reason="target_reached"))

    for level, expected in (("block", "nothing is listening on 192.168.1.50"),
                            ("warn", ""), ("off", "")):
        ctrl.cfg.preflight_level = level
        ctrl._exposure_host_checked = False
        ctrl.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)
        assert ctrl._preflight_black_cause == expected, level


def test_black_first_probe_aborts_the_match(ctrl, monkeypatch):
    """The abort must be STICKY, not merely an exception: sunsolve.probe swallows every
    exception per probe by design, so a bare raise would be caught 44 times and each
    catch would fire another 60-second render. The second hooks.render below mirrors that
    swallow -- and must cost nothing."""
    rendered, restored = _black_probe_harness(ctrl, monkeypatch, _BLACK)
    caught = []

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for tag in ("iter00", "iter01"):
            try:
                hooks.render(tag)
            except RuntimeError as err:          # exactly what sunsolve.probe does
                caught.append(err)
        raise caught[-1]

    monkeypatch.setattr(ctl, "run_match", fake)
    logs = []
    with pytest.raises(RuntimeError, match="BLACK"):
        ctrl.run_match("CamA", logs.append)
    assert len(rendered) == 1, "the sticky flag must short-circuit before rendering"
    assert len(caught) == 2                      # both probes refused, one render fired
    assert any("distributed rendering" in ln for ln in logs)
    assert restored == [True]                    # the artist's sampler still came back


def test_a_plate_that_goes_black_at_probe_fifty_still_aborts(ctrl, monkeypatch):
    """THE regression the 29bbae6 one-shot guard permitted, and its own comment named.

    That guard checked the FIRST probe and latched. If the distributed-rendering link dies
    at probe 50 — which is exactly what a Vantage live link closing mid-run does — the
    remaining 83 probes AND the delivered final are ranked unvalidated. Validation now
    lives in _render_exposed and runs on every plate, so the fiftieth black frame stops
    the run just as the first would."""
    rendered, _restored = _black_probe_harness(ctrl, monkeypatch, _BLACK)
    lit = {"mean_rgb": [0.4, 0.4, 0.4], "p": {"95": 0.8, "5": 0.1},
           "contrast": 0.7, "log_key": 0.2, "hot_frac": 0.02}
    seq = [dict(lit)] * 49 + [dict(_BLACK)]
    stats = iter(seq)
    monkeypatch.setattr(ctrl, "stats_for", lambda p: next(stats))
    caught = []

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for i in range(60):
            try:
                hooks.render(f"iter{i:02d}")
            except RuntimeError as err:
                caught.append(err)
                break
        raise caught[-1]

    monkeypatch.setattr(ctl, "run_match", fake)
    with pytest.raises(RuntimeError, match="BLACK"):
        ctrl.run_match("CamA", lambda ln: None)
    assert len(rendered) == 50        # 49 good plates, then the black one stops it
    assert len(caught) == 1


def test_an_unmeasurable_plate_is_not_evidence_and_the_next_one_is_still_checked(
        ctrl, monkeypatch):
    """Absence of evidence is not evidence: a frame whose stats would not compute has not
    been checked, so the next one still has to be."""
    rendered, _restored = _black_probe_harness(ctrl, monkeypatch, _BLACK)
    stats = iter((None, dict(_BLACK)))
    monkeypatch.setattr(ctrl, "stats_for", lambda p: next(stats))
    caught = []

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for tag in ("iter00", "iter01"):
            try:
                hooks.render(tag)
            except RuntimeError as err:
                caught.append(err)
        raise caught[-1]

    monkeypatch.setattr(ctl, "run_match", fake)
    with pytest.raises(RuntimeError, match="BLACK"):
        ctrl.run_match("CamA", lambda ln: None)
    assert len(rendered) == 2


def test_black_guard_ignores_a_dark_but_lit_frame(ctrl, monkeypatch):
    """A moonlit interior is not a dead render -- one lit pixel in 576 lifts the mean."""
    dark = {"mean_rgb": [0.002, 0.002, 0.002], "p": {"95": 0.01}, "log_key": 0.001}
    rendered, _restored = _black_probe_harness(ctrl, monkeypatch, dark)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        hooks.render("iter00")
        hooks.render("iter01")
        return MatchResult(best_state=make_state(), best_score=90.0,
                           best_render=None, stop_reason="target_reached")

    monkeypatch.setattr(ctl, "run_match", fake)
    result = ctrl.run_match("CamA", lambda ln: None)
    assert result.stop_reason == "target_reached"
    assert len(rendered) == 2


def test_exposure_host_check_ignores_black_probes_and_says_so(ctrl, monkeypatch,
                                                              tmp_path):
    """Two black plates give moved == 0.0, i.e. "this host is display-stage only", which
    switches software_exposure ON for the session off a dead render. And these are the
    EARLIEST two V-Ray plates of the run, so a gate that closes in silence here is the
    same unfalsifiable-transcript failure the draft-OFF line was added to fix. It reports
    rather than aborts: nothing has been applied yet, so a scene the artist left unlit is
    legitimately black at this point and the post-apply probe guard is the one that can
    tell those apart."""
    ctrl.cfg.no_renders = False
    ctrl.cfg.software_exposure = False
    monkeypatch.setattr(ctl.rd, "render_frame", lambda cam, out, w, h: out)
    monkeypatch.setattr(ctl.metrics, "compute_stats", lambda p, **kw: dict(_BLACK))

    class _Host:
        def __init__(self, cam):
            pass

        def read_ev(self):
            return 10.0

        def write_ev(self, value):
            pass

    from maxgaffer.maxbridge import exposure as exmod

    monkeypatch.setattr(exmod, "ExposureHost", _Host)
    logs = []
    ctrl._verify_exposure_host(object(), str(tmp_path), logs.append)
    assert ctrl.cfg.software_exposure is False
    assert any("BLACK" in ln and "distributed rendering" in ln for ln in logs)
    assert not getattr(ctrl, "_probe_abort", "")     # reports, never aborts, from here
