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
