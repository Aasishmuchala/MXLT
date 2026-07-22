"""Qt regression coverage for the dock's P0-class behaviors (wave-Q).

  * dock close → reopen re-SHOWS the same live instance (a dead macro button is
    how this regressed once — the wrapper hid and nothing ever re-showed it);
  * a camera-combo poke while a match runs reverts to the running camera instead
    of silently switching rigs mid-loop;
  * the Settings dialog's "Test gateway" button routes the blocking ping through
    the dock's io worker — never the main thread (SPEC §2 threading is law: a
    synchronous gateway wait on Max's main thread is a UI freeze).

Every test skips cleanly where PySide6 is absent (managed CI python). The Qt
platform is chosen by tests/conftest.py — never hard-code QT_QPA_PLATFORM here:
Max's Qt ships qwindows.dll only, and requesting a missing platform hangs
QApplication() in native code.
"""

import threading

import pytest

try:
    from PySide6 import QtWidgets
    HAS_QT = True
except ImportError:                                  # off-Max CI: everything below skips
    QtWidgets = None
    HAS_QT = False

requires_qt = pytest.mark.skipif(not HAS_QT, reason="PySide6 unavailable off-Max")

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
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.session = Session()
        self.calls = []
        self._run_dir = "/tmp"
        self.io = lambda fn: fn()

    def cameras(self):
        return [{"name": n, "class": "Physical", "yaw_deg": 0.0,
                 "reference": e.reference, "score": e.score, "has_state": bool(e.state)}
                for n, e in self.session.cameras.items()]

    def read_state(self, camera_name=""):
        return demo_state()

    def select_camera(self, name, apply_saved=True, camera_id=""):
        self.calls.append(("select_camera", name))
        return []

    def save_session(self):
        return True


@pytest.fixture(scope="module")
def app():
    if not HAS_QT:
        pytest.skip("PySide6 unavailable off-Max")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def dock(app, monkeypatch, tmp_path):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    d = dockmod.MaxGafferDock()
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"")                             # existence is all _show_reference needs
    for cam in ("CamA", "CamB"):
        d.ctrl.session.set_reference(cam, str(ref))
    d.refresh_cameras()
    yield d
    d._drain_workers()
    d.deleteLater()


# ------------------------------------------------------------- lifecycle
@requires_qt
def test_dock_close_then_reopen_reshows(app, monkeypatch):
    """User closes the panel, clicks the macro again: the SAME dock must come back
    on screen — not a rebuilt widget, not a hidden window."""
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    dockmod._dock_instance = None
    dockmod._dock_wrapper = None
    try:
        w1 = dockmod.show_dock()
        assert w1.isVisible()
        w1.close()                                   # the X button / close macro path
        assert not w1.isVisible()
        w2 = dockmod.show_dock()
        assert w2 is w1                              # same live instance, not a zombie
        assert w1.isVisible()                        # …and actually back on screen
    finally:
        inst = dockmod._dock_instance
        dockmod._dock_instance = None
        dockmod._dock_wrapper = None
        if inst is not None:
            inst._drain_workers()
            inst.deleteLater()
        app.processEvents()


# ------------------------------------------------------------- busy guard
@requires_qt
def test_busy_camera_switch_reverts_combo(dock):
    """Mid-match the artist pokes the camera dropdown: the combo must snap back to
    the camera that is actually being matched, and no select may reach the scene."""
    dock._busy = True
    try:
        dock.cam_combo.setCurrentIndex(1)            # → CamB attempted mid-run
        assert dock._current_camera() == "CamA"      # combo reverted to the running cam
        assert ("select_camera", "CamB") not in dock.ctrl.calls
    finally:
        dock._busy = False


# ------------------------------------------------------------- gateway threading (P0)
@requires_qt
def test_test_gateway_routes_through_io_worker(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "ping", lambda key, model: "gateway reachable: 'OK'")
    used = []
    real_io = dock._run_blocking_io
    monkeypatch.setattr(dock, "_run_blocking_io",
                        lambda fn: used.append(fn) or real_io(fn))
    dlg = dockmod.SettingsDialog(dock.cfg, dock)
    dlg._test()
    assert len(used) == 1                            # the ping went through the io worker
    assert dlg.lbl_status.text() == "gateway reachable: 'OK'"
    assert dlg.btn_test.isEnabled()                  # button restored after the reply


@requires_qt
def test_test_gateway_ping_runs_off_the_main_thread(dock, monkeypatch):
    """The blocking ping body must execute on the worker QThread — if this ever
    regresses to the main thread, Max's UI freezes for the whole gateway wait."""
    from maxgaffer.ui import dock as dockmod

    main_thread = threading.current_thread()
    seen = []

    def _ping(key, model):
        seen.append(threading.current_thread())
        return "gateway reachable: 'OK'"

    monkeypatch.setattr(dockmod, "ping", _ping)
    dlg = dockmod.SettingsDialog(dock.cfg, dock)
    dlg._test()
    assert len(seen) == 1
    assert seen[0] is not main_thread                # SPEC §2: never block Max's UI
    assert dlg.lbl_status.text() == "gateway reachable: 'OK'"
