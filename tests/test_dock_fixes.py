"""Dock bugfix regressions — busy-guard holes, refine lock wiring, finals Cancel.

Skips cleanly where PySide6 is absent (same contract as test_ui_dock.py); run with the
review Qt libs via PYTHONPATH=/tmp/mxlt_review/pylibs for the real drive.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from maxgaffer.maxbridge.config import Config  # noqa: E402
from test_ui_dock import FakeController  # noqa: E402


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
    ref.write_bytes(b"")
    for cam in ("CamA", "CamB"):
        d.ctrl.session.set_reference(cam, str(ref))
    d.act_popup.setChecked(False)
    d.refresh_cameras()
    return d


# ------------------------------------------------------------------ busy guards
def test_pick_reference_blocked_while_busy(dock, monkeypatch):
    """A reference swap under a running match used to clear score/semantics mid-run."""
    called = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: called.append(a) or ("x.jpg", "")))
    before = dock.ctrl.session.cameras["CamA"].reference
    dock._busy = True
    dock._pick_reference()
    assert not called                                       # dialog never opened
    assert dock.ctrl.session.cameras["CamA"].reference == before
    assert "busy" in dock.log.toPlainText()


def test_reference_picker_exposes_max_native_formats_and_rejects_missing(dock,
                                                                          monkeypatch):
    seen = {}

    def choose(*args, **kwargs):
        seen["filter"] = args[3]
        return ("Z:/missing/reference.exr", "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", staticmethod(choose))
    before = dock.ctrl.session.cameras["CamA"].reference
    dock._pick_reference()
    assert "*.exr" in seen["filter"] and "*.tiff" in seen["filter"]
    assert dock.ctrl.session.cameras["CamA"].reference == before
    assert "not found" in dock.log.toPlainText()


def test_missing_bound_reference_is_reported_as_missing_not_unbound(dock, tmp_path):
    missing = tmp_path / "moved.jpg"
    dock.ctrl.session.cameras["CamA"].reference = str(missing)
    dock._show_reference("CamA")
    assert dock.ref_thumb.text() == "reference missing"
    assert "moved or was deleted" in dock.lbl_ref_info.text()


def test_open_settings_blocked_while_busy(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    built = []
    monkeypatch.setattr(dockmod, "SettingsDialog",
                        lambda *a, **k: built.append(a) or object())
    dock._busy = True
    dock._open_settings()
    assert not built
    assert "busy" in dock.log.toPlainText()


def test_live_link_blocked_while_busy(dock, monkeypatch):
    called = []
    monkeypatch.setattr(dock.ctrl, "start_live_link", lambda: called.append(1) or (True, "x"))
    dock._busy = True
    dock._start_live_link()
    assert not called
    assert "busy" in dock.log.toPlainText()


def test_save_preset_blocked_while_busy(dock, monkeypatch):
    called = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: called.append(a) or ("x.json", "")))
    dock._busy = True
    dock._save_preset()
    assert not called
    assert "busy" in dock.log.toPlainText()


def test_open_run_dir_blocked_while_busy(dock, monkeypatch):
    from PySide6 import QtGui

    opened = []
    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl",
                        staticmethod(lambda url: opened.append(url)))
    dock._busy = True
    dock._open_run_dir()
    assert not opened
    assert "busy" in dock.log.toPlainText()


def test_guards_release_after_run(dock, monkeypatch):
    """The same slots must work again once the run is over."""
    called = []
    monkeypatch.setattr(dock.ctrl, "start_live_link", lambda: called.append(1) or (True, "x"))
    dock._busy = True
    dock._start_live_link()
    dock._busy = False
    dock._start_live_link()
    assert called == [1]


# ------------------------------------------------------------------ refine locks
def test_refine_passes_the_locks_menus_current_selection(dock):
    """run_match synced menu locks into the run; refine used the stale persisted set."""
    for a in dock.lock_menu.actions():
        if a.text() == "exposure.ev":
            a.setChecked(True)
    dock.cmb_note.setCurrentText("too warm")
    dock._start_refine()
    assert dock.ctrl.refine_locks == {"exposure.ev"}


def test_refine_without_checked_locks_passes_empty(dock):
    dock.cmb_note.setCurrentText("softer shadows")
    dock._start_refine()
    assert dock.ctrl.refine_locks == set()


# ------------------------------------------------------------------ finals cancel
def test_finals_enable_cancel_and_thread_the_flag(dock, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: "/tmp"))
    seen = {}

    def fake_finals(cams, out_dir, on_progress, should_cancel=None):
        seen["cancel_enabled_mid_run"] = dock.btn_cancel.isEnabled()
        dock._cancel_match()                       # the artist hits ✕ between cameras
        seen["flag_fires"] = bool(should_cancel and should_cancel())
        return {c: "cancelled" for c in cams}

    monkeypatch.setattr(dock.ctrl, "render_finals_vray", fake_finals)
    dock._render_finals(selected_only=True)
    assert seen == {"cancel_enabled_mid_run": True, "flag_fires": True}
    assert not dock.btn_cancel.isEnabled()         # restored afterwards
    assert not dock._busy


def test_vantage_cli_finals_thread_the_flag(dock, monkeypatch):
    from test_ui_dock import demo_state

    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a, **k: "/tmp"))
    dock.ctrl.session.record_match("CamA", demo_state(), 90.0)   # → finals target
    dock.cfg.final_render_backend = "vantage_cli"
    seen = {}
    monkeypatch.setattr(dock.ctrl, "run_vantage_jobs",
                        lambda jobs, on_progress, should_cancel=None:
                        seen.setdefault("should_cancel", should_cancel) or
                        {j["camera"]: "ok" for j in jobs})
    dock._render_finals(selected_only=False)
    assert callable(seen.get("should_cancel"))
    dock._cancel = True
    assert seen["should_cancel"]() is True


# ------------------------------------------------------------------ plan-less match
def test_double_junk_plan_proceeds_planless(dock, monkeypatch):
    """make_plan → None (junk reply twice) must skip the plan, not abort the match."""
    monkeypatch.setattr(dock.ctrl, "make_plan", lambda cam, log: None)
    dock.cmb_mode.setCurrentIndex(0)
    dock._start_match()
    names = [c[0] for c in dock.ctrl.calls]
    assert "execute_plan" not in names
    assert "run_match" in names                     # the match still ran
    assert "no operations proposed" in dock.log.toPlainText()


# ------------------------------------------------------------------ options ▾ vs config
def _dock_on(cfg, monkeypatch, tmp_path):
    """A dock built on a SPECIFIC config — the 2026-07-30 one, on purpose."""
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: cfg)
    monkeypatch.setattr(dockmod.PlanPreviewDialog, "exec", lambda self: True)
    monkeypatch.setattr(dockmod.ChangeReportDialog, "exec", lambda self: True)
    d = dockmod.MaxGafferDock()
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"")
    d.ctrl.session.set_reference("CamA", str(ref))
    d.refresh_cameras()
    return d


def test_options_menu_no_longer_overwrites_the_config_file(app, monkeypatch, tmp_path):
    """THE 2026-07-30 BUG, exactly. config.json carried draft_sampler:true and a 20 s
    probe cap, written at 15:27; the dock was launched fresh at ~16:30 and a match ran
    with neither in effect, because _start_match copied the Options ▾ item's state over
    cfg before every run — and that item drew no check mark, so nobody could see it was
    off. The menu is a MIRROR of the config now, not a second copy of it."""
    cfg = Config(api_key="oc_test", draft_sampler=True, probe_max_seconds=20.0)
    d = _dock_on(cfg, monkeypatch, tmp_path)
    assert d.act_draft.isChecked() is True         # the file seeded the menu
    d._start_match()
    assert d.cfg.draft_sampler is True             # ...and the match did not undo it
    assert d.cfg.probe_max_seconds == 20.0
    assert d.act_draft.isChecked() is True


def test_toggling_the_menu_item_is_the_only_thing_that_moves_the_flag(dock):
    """The binding REPLACES the clobber: the value moves when the artist toggles it."""
    dock.act_draft.setChecked(False)
    assert dock.cfg.draft_sampler is False
    dock.act_draft.setChecked(True)
    assert dock.cfg.draft_sampler is True
    dock.act_autoexec.setChecked(True)
    assert dock.cfg.auto_execute_plan is True


def test_checkable_menu_items_render_their_state():
    """The affordance bug underneath the config bug: styling QMenu::item without an
    indicator rule silences Qt's own check mark, so a checkable item becomes a switch
    with no position — the artist clicked "Draft sampler" to turn it ON. Only assertable
    statically headless; worth two minutes of on-box eyeballing after this lands."""
    from maxgaffer.ui import dock as dockmod

    assert "QMenu::indicator" in dockmod.STYLE
    assert "QMenu::indicator:checked" in dockmod.STYLE


# ------------------------------------------------------------------ settings ↔ menu
def test_settings_round_trips_the_probe_budget(app):
    """The two knobs that do not scale with the scene had no UI at all — they were
    hand-edited into config.json, which is how they came to be silently discarded."""
    from maxgaffer.ui import dock as dockmod

    cfg = Config()
    dlg = dockmod.SettingsDialog(cfg)
    dlg.cb_draft.setChecked(True)
    dlg.sp_probe_secs.setValue(20.0)
    dlg.cmb_probe_backend.setCurrentText("vantage")
    dlg._save()
    assert cfg.draft_sampler is True
    assert cfg.probe_max_seconds == 20.0
    assert cfg.probe_backend == "vantage"

    again = dockmod.SettingsDialog(cfg)
    assert again.cb_draft.isChecked() is True
    assert again.sp_probe_secs.value() == 20.0
    assert again.cmb_probe_backend.currentText() == "vantage"


def test_settings_save_resyncs_the_options_menu(dock, monkeypatch):
    """Two controls for one value is only safe while they agree; disagreeing silently is
    the bug that cost 2026-07-30."""
    from maxgaffer.ui import dock as dockmod

    dock.act_draft.setChecked(False)
    assert dock.cfg.draft_sampler is False
    saves = []
    monkeypatch.setattr(type(dock.cfg), "save", lambda self: saves.append(1))

    def fake_exec(self):
        self.cfg.draft_sampler = True              # what SettingsDialog._save does
        return True

    monkeypatch.setattr(dockmod.SettingsDialog, "exec", fake_exec)
    dock._open_settings()
    assert dock.act_draft.isChecked() is True      # the menu followed the file
    assert saves == [1]
