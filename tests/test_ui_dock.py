"""The redesigned dock, driven for real — every user journey through the dropdown UI,
offscreen, against a faithful fake controller. Skips cleanly where PySide6 is absent
(the box's Max python runs this suite too; CI without Qt just skips).

What this proves that import-smoke cannot: mode dropdown → engine kwargs, Locks ▾ and
Options ▾ actually reaching run_match, plan preview accept/decline branches, the CHANGES
panel contents, refine note delivery, busy-lifecycle button states, and error recovery.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from maxgaffer.core.director import MatchResult  # noqa: E402
from maxgaffer.core.genome import LightingState  # noqa: E402
from maxgaffer.core.session import Session  # noqa: E402
from maxgaffer.maxbridge.config import Config  # noqa: E402


def demo_state():
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 208.0, "sun.altitude_deg": 8.0,
                 "sun.intensity": 1.0, "exposure.ev": 11.5,
                 "exposure.wb_kelvin": 5200.0}.items():
        st.set(k, v)
    return st


class FakeController:
    """Faithful stand-in: real Session, real result types, records every engine call."""

    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.session = Session()
        self.calls = []
        self._run_dir = "/tmp"
        self.io = lambda fn: fn()
        self.raise_on_match = None
        self.best_render = None

    # scene/session -------------------------------------------------
    def cameras(self):
        return [{"name": n, "class": "Physical", "yaw_deg": 0.0,
                 "reference": e.reference, "score": e.score, "has_state": bool(e.state)}
                for n, e in self.session.cameras.items()]

    def read_state(self, camera_name=""):
        return demo_state()

    def apply_state(self, state, camera_name=""):
        self.calls.append(("apply_state", camera_name))
        return []

    def select_camera(self, name, apply_saved=True):
        self.calls.append(("select_camera", name))
        return []

    def save_session(self):
        return True

    def state_change_rows(self, camera_name):
        return [{"target": camera_name, "prop": "sun.altitude_deg",
                 "before": 55.0, "after": 8.0, "why": ""}]

    # engine --------------------------------------------------------
    def make_plan(self, camera_name, log):
        self.calls.append(("make_plan", camera_name))
        log("plan built")
        ops = [{"op": "set", "target": "exposure", "prop": "ev", "value": 11.5,
                "why": "match"}]
        return ops, ["set  exposure · ev → 11.5"], {"read": "scene read",
                                                    "expects": "dusk"}, {}

    def execute_plan(self, ops, camera_name, log, measure=True):
        self.calls.append(("execute_plan", camera_name, len(ops)))
        log("executed")
        return {"changes": [{"target": "exposure", "prop": "ev", "before": 13.2,
                             "after": 11.5, "why": ""}],
                "created": [], "warnings": [], "effect": {"before": 40.0, "after": 80.0}}

    def run_match(self, camera_name, log, should_cancel=lambda: False, locks=None,
                  do_sweep=False, deep=False, start_override=None, director_note=""):
        self.calls.append(("run_match", camera_name, frozenset(locks or set()),
                           do_sweep, deep))
        if self.raise_on_match:
            raise self.raise_on_match
        log("matched")
        return MatchResult(best_state=demo_state(), best_score=98.2,
                           best_render=self.best_render, stop_reason="target_reached")

    def refine(self, camera_name, note, log, should_cancel=lambda: False, locks=None):
        self.calls.append(("refine", camera_name, note))
        self.refine_locks = locks                  # the Locks menu's live selection
        log("refined")
        return MatchResult(best_state=demo_state(), best_score=98.9,
                           best_render=self.best_render, stop_reason="target_reached")

    def match_all(self, log, should_cancel=lambda: False, do_sweep=True):
        self.calls.append(("match_all", do_sweep))
        return {"CamA": "98.2"}

    def seed_dome(self, camera_name, log):
        self.calls.append(("seed_dome", camera_name))
        log("dome seed: seed_cam.hdr (reference, texmap.HDRIMapName, rotation zeroed)")
        return {"path": "/tmp/seed.hdr", "source": "reference",
                "sun": {"azimuth_deg": 120.0, "altitude_deg": 30.0, "kelvin": 4300.0}}

    def run_scenarios(self, camera_name, log, should_cancel=lambda: False):
        self.calls.append(("run_scenarios", camera_name))
        log("board rendered")
        return [{"key": "golden_low", "label": "Golden low sun", "why": "warm rake",
                 "state": demo_state(), "render": self.best_render, "score": 77.7},
                {"key": "overcast_soft", "label": "Overcast soft", "why": "sky key",
                 "state": demo_state(), "render": self.best_render, "score": 61.0}]

    def adopt_scenario(self, camera_name, state, score=None):
        self.calls.append(("adopt", camera_name, score))
        return []

    def set_dome_hdri(self, path):
        self.calls.append(("set_dome_hdri", path))
        return "texmap.HDRIMapName"

    # misc ----------------------------------------------------------
    def restore_pre_match(self, camera_name):
        self.calls.append(("restore", camera_name))
        return True

    def start_live_link(self):
        return True, "actionMan"

    def prepare_vantage_jobs(self, cams, out_dir, on_progress, use_saved_states=True):
        return [{"camera": c, "scene_file": "x.vrscene", "output": "o.png"} for c in cams]

    def run_vantage_jobs(self, jobs, on_progress, should_cancel=None):
        self.vantage_cancel = should_cancel
        return {j["camera"]: "ok" for j in jobs}

    def render_finals_vray(self, cams, out_dir, on_progress, should_cancel=None):
        self.calls.append(("finals", tuple(cams)))
        self.finals_cancel = should_cancel
        return {c: "ok" for c in cams}

    def export_and_open_vantage(self, cams, on_progress):
        return [], False, "/tmp"


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def dock(app, monkeypatch, tmp_path):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    monkeypatch.setattr(dockmod.PlanPreviewDialog, "exec", lambda self: True)
    monkeypatch.setattr(dockmod.ChangeReportDialog, "exec", lambda self: True)
    d = dockmod.MaxGafferDock()
    ref = tmp_path / "ref.png"
    try:
        from PIL import Image

        Image.new("RGB", (32, 18), (220, 170, 110)).save(str(ref))
    except Exception:
        ref.write_bytes(b"")
    for cam in ("CamA", "CamB"):
        d.ctrl.session.set_reference(cam, str(ref))
    d.ctrl.best_render = str(ref)
    d.act_popup.setChecked(False)
    d.refresh_cameras()
    return d


def names(dock):
    return [c[0] for c in dock.ctrl.calls]


# --------------------------------------------------------------------- journeys
def test_camera_dropdown_switch(dock):
    dock.cam_combo.setCurrentIndex(1)
    assert ("select_camera", "CamB") in dock.ctrl.calls
    assert dock._current_camera() == "CamB"


def test_standard_match_runs_plan_then_loop_and_fills_changes(dock):
    dock.cmb_mode.setCurrentIndex(0)
    dock._start_match()
    assert names(dock).count("make_plan") == 1
    assert names(dock).count("execute_plan") == 1
    assert names(dock).count("run_match") == 1
    top = dock.changes_tree.topLevelItem(0)
    assert top is not None and "98.2" in top.text(0)
    rows = [top.child(i).text(0) for i in range(top.childCount())]
    assert any("exposure · ev" in r for r in rows)          # plan change
    assert any("sun.altitude_deg" in r for r in rows)       # loop diff
    assert any("plan effect" in r for r in rows)            # measured effect
    assert dock.match_thumb.pixmap() is not None            # latest-match thumb set
    assert dock.btn_match.isEnabled()                       # lifecycle restored


def test_loop_only_mode_skips_plan(dock):
    dock.cmb_mode.setCurrentIndex(2)
    dock._start_match()
    assert "make_plan" not in names(dock)
    assert "run_match" in names(dock)


def test_deep_mode_reaches_engine(dock):
    dock.cmb_mode.setCurrentIndex(1)
    dock._start_match()
    call = next(c for c in dock.ctrl.calls if c[0] == "run_match")
    assert call[4] is True                                   # deep=True


def test_locks_and_options_reach_engine(dock):
    for a in dock.lock_menu.actions():
        if a.text() == "exposure.ev":
            a.setChecked(True)
    dock.act_sweep.setChecked(False)
    dock.cmb_mode.setCurrentIndex(2)
    dock._start_match()
    call = next(c for c in dock.ctrl.calls if c[0] == "run_match")
    assert "exposure.ev" in call[2]                          # lock delivered
    assert call[3] is False                                  # sweep off delivered


def test_plan_declined_still_matches(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod.PlanPreviewDialog, "exec", lambda self: False)
    dock.cmb_mode.setCurrentIndex(0)
    dock._start_match()
    assert "make_plan" in names(dock)
    assert "execute_plan" not in names(dock)                 # declined
    assert "run_match" in names(dock)                        # loop still ran


def test_refine_delivers_and_clears_note(dock):
    dock.cmb_note.setCurrentText("too warm, softer shadows")
    dock._start_refine()
    assert ("refine", "CamA", "too warm, softer shadows") in dock.ctrl.calls
    assert dock.cmb_note.currentText() == ""
    assert dock.changes_tree.topLevelItemCount() == 1


def test_match_all_confirm_and_dispatch(dock, monkeypatch):
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes))
    dock._start_match_all()
    assert ("match_all", True) in dock.ctrl.calls


def test_error_path_restores_buttons_and_logs(dock):
    from maxgaffer.core.omega import OmegaError

    dock.ctrl.raise_on_match = OmegaError("gateway down", "network")
    dock.cmb_mode.setCurrentIndex(2)
    dock._start_match()
    assert "gateway down" in dock.log.toPlainText()
    assert dock.btn_match.isEnabled() and not dock._busy


def test_transcript_toggle_and_thumb_line(dock, tmp_path):
    assert not dock.log.isVisible()
    dock._toggle_transcript()
    assert dock.btn_transcript.text().endswith("▴")
    dock._log("THUMB::/tmp/nonexistent.png")                 # must not crash
    dock._log("plain <line> & escaped")
    assert "plain <line> & escaped" in dock.log.toPlainText()


def test_finals_button_uses_vray_backend(dock, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: "/tmp"))
    dock.ctrl.session.record_match("CamA", demo_state(), 98.0)
    dock._render_finals(selected_only=True)
    assert ("finals", ("CamA",)) in dock.ctrl.calls


def test_seed_dome_action_reaches_controller(dock):
    dock._seed_dome()
    assert ("seed_dome", "CamA") in dock.ctrl.calls
    assert "dome seeded" in dock.log.toPlainText()
    assert not dock._busy


def test_seed_dome_needs_reference(dock):
    dock.ctrl.session.cameras["CamA"].reference = ""
    dock._seed_dome()
    assert "seed_dome" not in names(dock)
    assert "bind a reference" in dock.log.toPlainText()


def test_scenario_board_adopts_measured_winner(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    # dialog auto-accepts with its preselection — the best-scoring card
    monkeypatch.setattr(dockmod.ScenarioBoardDialog, "exec", lambda self: True)
    dock._open_scenarios()
    assert ("run_scenarios", "CamA") in dock.ctrl.calls
    adopt = next(c for c in dock.ctrl.calls if c[0] == "adopt")
    assert adopt[1] == "CamA" and adopt[2] == 77.7   # 77.7 beats 61.0 — winner preselected
    assert dock.btn_match.isEnabled() and not dock._busy


def test_scenario_board_keep_current_adopts_nothing(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod.ScenarioBoardDialog, "exec", lambda self: False)
    dock._open_scenarios()
    assert ("run_scenarios", "CamA") in dock.ctrl.calls
    assert not any(c[0] == "adopt" for c in dock.ctrl.calls)
    assert dock.btn_board.isEnabled()


def test_scenario_dialog_preselects_best_and_reports_choice(dock, app):
    from maxgaffer.ui.dock import ScenarioBoardDialog

    cands = [{"key": "a", "label": "A", "why": "", "state": demo_state(),
              "render": None, "score": 50.0},
             {"key": "b", "label": "B", "why": "", "state": demo_state(),
              "render": None, "score": 88.0},
             {"key": "c", "label": "C", "why": "", "state": demo_state(),
              "render": None, "score": None}]
    dlg = ScenarioBoardDialog(cands)
    assert dlg.chosen == 1                      # measured winner preselected
    dlg._cards[0].click()                       # user changes their mind
    assert dlg.chosen == 0


def test_manual_hdri_pick_releases_the_seed(dock, monkeypatch, tmp_path):
    """Round-2 regression: _rebind_seed re-binds on every camera switch, so a manual
    HDRI pick must clear the camera's seed or the artist's choice silently reverts."""
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(tmp_path / "sky.hdr"), "")))
    dock.ctrl.session.cameras["CamA"].seed_hdri = "/runs/seed_CamA_11112222.hdr"
    dock._pick_hdri()
    assert any(c[0] == "set_dome_hdri" for c in dock.ctrl.calls)
    assert dock.ctrl.session.cameras["CamA"].seed_hdri == ""
    assert "seed released" in dock.log.toPlainText()


def test_settings_tuning_persists(dock, app):
    from maxgaffer.ui.dock import SettingsDialog

    s = SettingsDialog(dock.cfg)
    s.sp_iters.setValue(8)
    s.sp_target.setValue(90.0)
    s._save()
    assert dock.cfg.max_iterations == 8
    assert dock.cfg.target_score == 90.0
