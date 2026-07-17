"""no_renders (apply-only) mode — the plugin must NEVER render: MATCH applies the
first guess once (undoable) and verifies by read-back; REFINE applies the note's
deterministic nudges; the board lists candidates unrendered; probe_score and V-Ray
finals refuse. Every test arms a render_frame trip-wire that fails the run."""
import pytest

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
    scene = {"state": make_state()}       # applies persist, so read-back agrees

    def _apply(rig, b, st, cam=None, undo=True):
        applies.append((st, undo))
        scene["state"] = st.copy()
        return []

    monkeypatch.setattr(ctl.sc, "scene_path", lambda: str(tmp_path / "t.max"))
    monkeypatch.setattr(ctl.sc, "classify_rig", lambda: {
        "sun": object(), "dome": None, "groups": {"practicals": []},
        "notes": [], "sky_env": False})
    monkeypatch.setattr(ctl.sc, "get_camera", lambda name: object())
    monkeypatch.setattr(ctl.sc, "set_active_camera", lambda name: None)
    monkeypatch.setattr(ctl.sc, "camera_yaw_deg", lambda cam: 0.0)
    monkeypatch.setattr(ctl.ap, "capture_baselines", lambda rig: {})
    monkeypatch.setattr(ctl.ap, "read_state",
                        lambda rig, b, cam=None: scene["state"].copy())
    monkeypatch.setattr(ctl.ap, "apply_state", _apply)
    monkeypatch.setattr(
        ctl.rd, "render_frame",
        lambda *a, **k: pytest.fail("render_frame fired in no-render mode"))
    monkeypatch.setattr(cfgmod, "sessions_dir", lambda: str(tmp_path / "sessions"))
    monkeypatch.setattr(c, "analyze_reference", lambda name: dict(SEMANTICS))
    monkeypatch.setattr(c, "_image_block", lambda path: None)
    monkeypatch.setattr(c, "ref_stats", lambda path: None)
    c._test_applies = applies
    return c


def test_config_field_default_and_roundtrip(tmp_path, monkeypatch):
    assert cfgmod.Config().no_renders is False
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(tmp_path / "config.json"))
    cfgmod.Config(no_renders=True).save()
    assert cfgmod.load().no_renders is True


def test_match_applies_without_rendering(ctrl):
    ctrl.session.entry("CamA").reference = "ref.jpg"
    logs = []
    result = ctrl.run_match("CamA", logs.append)
    assert result.stop_reason == "applied (no-render mode)"
    assert result.best_score is None and result.best_render is None
    # exactly one scene apply, and it was the undoable kind
    assert len(ctrl._test_applies) == 1 and ctrl._test_applies[0][1] is True
    assert ctrl.session.cameras["CamA"].state is not None
    assert any("verified by read-back" in ln for ln in logs)


def test_probe_score_refuses(ctrl):
    assert ctrl.probe_score("CamA", "preplan") is None


def test_refine_applies_note_nudges_only(ctrl):
    ctrl.session.entry("CamA").reference = "ref.jpg"
    logs = []
    result = ctrl.refine("CamA", "exposure is too much", logs.append)
    assert result.stop_reason == "applied (no-render mode)"
    assert len(ctrl._test_applies) == 1
    applied_state = ctrl._test_applies[0][0]
    # "too much exposure" must darken: EV rises above the 12.0 start
    assert applied_state.get("exposure.ev") > 12.0


def test_board_lists_candidates_unrendered(ctrl):
    logs = []
    board = ctrl.run_scenarios("CamA", logs.append)
    assert board and all(c["render"] is None and c["score"] is None for c in board)
    assert not ctrl._test_applies       # candidates listed, scene untouched
    assert any("no-render mode" in ln for ln in logs)


def test_finals_refuse(ctrl, tmp_path):
    notes = []
    out = ctrl.render_finals_vray(["CamA", "CamB"], str(tmp_path / "out"),
                                  lambda n, s: notes.append((n, s)))
    assert out == {"CamA": "skipped (no-render mode)",
                   "CamB": "skipped (no-render mode)"}
    assert not (tmp_path / "out").exists()


def test_vantage_cli_finals_refuse(ctrl, monkeypatch):
    monkeypatch.setattr(
        ctl.vt, "render_stills",
        lambda *a, **k: pytest.fail("vantage CLI render fired in no-render mode"))
    jobs = [{"camera": "CamA", "scene_file": "a.vrscene", "output": "a.png"}]
    assert ctrl.run_vantage_jobs(jobs, lambda n, s: None) == {
        "CamA": "skipped (no-render mode)"}


def test_refine_after_match_reports_only_the_note(ctrl):
    """A 2nd exploration must resnapshot pre_match: the change report shows ONLY this
    refine's nudge, and Restore returns to the state right before it — not to the
    original pre-match light with the whole MATCH lumped in."""
    ctrl.session.entry("CamA").reference = "ref.jpg"
    ctrl.run_match("CamA", lambda ln: None)          # exploration 1: first guess
    m1 = ctrl.session.cameras["CamA"].state.copy()
    logs = []
    ctrl.refine("CamA", "exposure is too much", logs.append)
    e = ctrl.session.cameras["CamA"]
    assert not e.pre_match.diff(m1)                  # snapshot = state before THIS refine
    applied = [ln for ln in logs if ln.startswith("applied:")]
    assert len(applied) == 1 and "exposure.ev" in applied[0]
