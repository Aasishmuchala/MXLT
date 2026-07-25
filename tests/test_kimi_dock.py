"""Cluster K regressions — dock threading/lifecycle + startup-script robustness.

Qt tests follow tests/test_ui_dock.py (offscreen, faithful fake controller) and skip
cleanly where PySide6 is absent; the startup-script tests are pure python and run
everywhere (pymxs stubbed in sys.modules, the script exec'd from its real path).
"""

import importlib.util
import os
import sys
import time
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6 import QtWidgets
    HAS_QT = True
except ImportError:                                  # off-Max CI: Qt tests skip, the
    QtWidgets = None                                 # startup-script tests still run
    HAS_QT = False

requires_qt = pytest.mark.skipif(not HAS_QT, reason="PySide6 unavailable off-Max")

from maxgaffer.core.genome import LightingState  # noqa: E402
from maxgaffer.core.session import Session  # noqa: E402
from maxgaffer.maxbridge.config import Config  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTUP = os.path.join(REPO, "maxgaffer", "startup", "maxgaffer_startup.py")


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
        self.hdri_error = None
        self.restore_error = None

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

    def set_dome_hdri(self, path):
        self.calls.append(("set_dome_hdri", path))
        if self.hdri_error:
            raise self.hdri_error
        return "texmap.HDRIMapName"

    def restore_pre_match(self, camera_name):
        self.calls.append(("restore", camera_name))
        if self.restore_error:
            raise self.restore_error
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
    try:
        from PIL import Image

        Image.new("RGB", (32, 18), (220, 170, 110)).save(str(ref))
    except Exception:
        ref.write_bytes(b"")
    for cam in ("CamA", "CamB"):
        d.ctrl.session.set_reference(cam, str(ref))
    d.refresh_cameras()
    yield d
    d._drain_workers()
    d.deleteLater()


# ------------------------------------------------------------- io worker pump
@requires_qt
def test_run_blocking_io_returns_result(dock):
    assert dock._run_blocking_io(lambda: 41 + 1) == 42
    assert dock._workers == []                       # reaped after a clean run


@requires_qt
def test_run_blocking_io_reraises_worker_failure(dock):
    def _boom():
        raise ValueError("net down")

    with pytest.raises(RuntimeError, match="net down"):
        dock._run_blocking_io(_boom)
    assert dock._workers == []


@requires_qt
def test_run_blocking_io_survives_systemexit(dock):
    """A BaseException inside the io fn must still emit failed — otherwise the nested
    wait never ends and Max's main thread wedges forever."""
    def _bye():
        raise SystemExit("bye")

    with pytest.raises(RuntimeError, match="bye"):
        dock._run_blocking_io(_bye)
    assert dock._workers == []


@requires_qt
def test_run_blocking_io_cancel_escape(dock):
    """Cancel must break the wait-poll instead of blocking on a job that never ends."""
    dock._cancel = True
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="cancelled"):
        dock._run_blocking_io(lambda: time.sleep(3))
    assert time.monotonic() - t0 < 2.5               # escaped, not wedged on the 3s job
    for w in list(dock._workers):                    # cancelled worker finishes off-thread
        w.wait(5000)
    dock._cancel = False


@requires_qt
def test_drain_workers_quits_and_waits(dock):
    class FakeWorker:
        def __init__(self):
            self.quit_called = False
            self.waited = None

        def quit(self):
            self.quit_called = True

        def wait(self, ms):
            self.waited = ms
            return True

    w = FakeWorker()
    dock._workers.append(w)
    dock._drain_workers()
    assert w.quit_called and w.waited == 3000
    assert dock._cancel is True
    dock._workers.remove(w)
    dock._cancel = False


# ------------------------------------------------------------- busy guards
@requires_qt
def test_busy_camera_switch_reverts_combo(dock):
    dock._busy = True
    dock.cam_combo.setCurrentIndex(1)                # user pokes the combo mid-run
    assert dock._current_camera() == "CamA"          # combo reverted to the running cam
    assert ("select_camera", "CamB") not in dock.ctrl.calls
    dock._busy = False


@requires_qt
def test_pick_reference_ignored_while_busy(dock, monkeypatch):
    called = []
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: called.append(a) or ("x.png", "")))
    dock._busy = True
    dock._pick_reference()
    assert called == []                              # dialog never opened mid-run
    dock._busy = False


@requires_qt
def test_camera_switch_works_after_busy_clears(dock):
    dock._busy = True
    dock.cam_combo.setCurrentIndex(1)
    dock._busy = False
    dock.cam_combo.setCurrentIndex(1)
    assert dock._current_camera() == "CamB"
    assert ("select_camera", "CamB") in dock.ctrl.calls


# ------------------------------------------------------------- slot hardening
@requires_qt
def test_pick_hdri_error_logged_and_rig_recovered(dock, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: ("sky.hdr", "")))
    dock.ctrl.hdri_error = RuntimeError("dome deleted")
    dock._pick_hdri()                                # must not raise out of the slot
    assert "✗ HDRI: dome deleted" in dock.log.toPlainText()


@requires_qt
def test_restore_pre_match_error_logged_not_raised(dock):
    dock.ctrl.restore_error = RuntimeError("light gone")
    dock._restore_pre_match()                        # must not raise out of the slot
    assert "✗ restore: light gone" in dock.log.toPlainText()


@requires_qt
def test_show_reference_tolerates_nonnumeric_semantics(dock):
    e = dock.ctrl.session.cameras["CamA"]
    e.semantics = {"time_of_day": "dusk", "sky": "clear", "wb_kelvin_estimate": "warm"}
    dock._show_reference("CamA")                     # hand-edited sidecar must not crash
    txt = dock.lbl_ref_info.text()
    assert "dusk" in txt and "wb ~" not in txt


@requires_qt
def test_settings_save_failure_logged(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod.SettingsDialog, "exec", lambda self: True)

    def _fail():
        raise OSError("access denied")

    monkeypatch.setattr(dock.cfg, "save", _fail)
    dock._open_settings()                            # must not raise out of the slot
    assert "settings not persisted" in dock.log.toPlainText()


# ------------------------------------------------------------- P0: test gateway
@requires_qt
def test_test_gateway_routes_through_io_worker(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "ping", lambda key, model, **kw: "gateway reachable: 'OK'")
    used = []
    real_io = dock._run_blocking_io
    monkeypatch.setattr(dock, "_run_blocking_io",
                        lambda fn: used.append(fn) or real_io(fn))
    dlg = dockmod.SettingsDialog(dock.cfg, dock)
    dlg._test()
    assert len(used) == 1                            # SPEC §2: gateway on the io worker
    assert dlg.lbl_status.text() == "gateway reachable: 'OK'"
    assert dlg.btn_test.isEnabled()                  # re-enabled after the ping lands


@requires_qt
def test_test_gateway_failure_lands_in_label(dock, monkeypatch):
    from maxgaffer.ui import dock as dockmod
    from maxgaffer.core.omega import OmegaError

    def _down(key, model, **kw):
        raise OmegaError("gateway down", "network")

    monkeypatch.setattr(dockmod, "ping", _down)
    dlg = dockmod.SettingsDialog(dock.cfg, dock)
    dlg._test()                                      # real io worker re-raises as RuntimeError
    assert "gateway down" in dlg.lbl_status.text()
    assert dlg.btn_test.isEnabled()


# ------------------------------------------------------------- show_dock lifecycle
@requires_qt
def test_show_dock_reshows_hidden_window(app, monkeypatch):
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    dockmod._dock_instance = None
    dockmod._dock_wrapper = None
    try:
        w1 = dockmod.show_dock()
        assert dockmod._dock_wrapper is w1           # dev fallback: window IS the wrapper
        w1.hide()                                    # user closed the panel (hide-on-close)
        w2 = dockmod.show_dock()
        assert w2 is w1
        assert w1.isVisible()                        # re-shown, not a dead macro
    finally:
        inst = dockmod._dock_instance
        dockmod._dock_instance = None
        dockmod._dock_wrapper = None
        if inst is not None:
            inst._drain_workers()
            inst.deleteLater()
        app.processEvents()


@requires_qt
def test_show_dock_recreates_after_cpp_delete(app, monkeypatch):
    shiboken = pytest.importorskip("shiboken6")
    from maxgaffer.ui import dock as dockmod

    monkeypatch.setattr(dockmod, "Controller", FakeController)
    monkeypatch.setattr(dockmod.cfgmod, "load", lambda: Config(api_key="oc_test"))
    dockmod._dock_instance = None
    dockmod._dock_wrapper = None
    try:
        w1 = dockmod.show_dock()
        w1._drain_workers()
        shiboken.delete(dockmod._dock_wrapper)       # wrapper finalized behind our back
        w2 = dockmod.show_dock()                     # RuntimeError → clean rebuild
        assert w2 is not w1 and w2.isVisible()
    finally:
        inst = dockmod._dock_instance
        dockmod._dock_instance = None
        dockmod._dock_wrapper = None
        if inst is not None:
            try:
                inst._drain_workers()
                inst.deleteLater()
            except RuntimeError:
                pass
        app.processEvents()


# ------------------------------------------------------------- startup script (pure python)
def _load_startup(monkeypatch, tmp_path, config_text=None, env_repo=""):
    """Exec the real startup script off-Max with pymxs stubbed; returns (module, executed)."""
    executed = []
    fake_pymxs = types.ModuleType("pymxs")
    fake_pymxs.runtime = types.SimpleNamespace(execute=lambda s: executed.append(s))
    monkeypatch.setitem(sys.modules, "pymxs", fake_pymxs)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    if env_repo:
        monkeypatch.setenv("MAXGAFFER", env_repo)
    else:
        monkeypatch.delenv("MAXGAFFER", raising=False)
    if config_text is not None:
        cfgdir = tmp_path / "MaxGaffer"
        cfgdir.mkdir(parents=True, exist_ok=True)
        (cfgdir / "config.json").write_text(config_text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mg_startup_under_test", STARTUP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # runs _register() — must never raise
    return mod, executed


def test_startup_non_dict_config_does_not_break_launch(monkeypatch, tmp_path, capsys):
    mod, executed = _load_startup(monkeypatch, tmp_path, config_text='["not", "a", "dict"]')
    assert executed == []                            # no macro registered without a repo
    assert "macro NOT registered" in capsys.readouterr().out
    assert mod._repo_path() == ""                    # AttributeError contained


def test_startup_non_dict_config_falls_back_to_env(monkeypatch, tmp_path):
    mod, executed = _load_startup(monkeypatch, tmp_path,
                                  config_text='"just a string"', env_repo=str(tmp_path))
    assert mod._repo_path() == str(tmp_path)         # env fallback survives the bad json
    assert executed                                  # repo valid → macro registered


def test_startup_registers_macro_when_repo_valid(monkeypatch, tmp_path):
    import json as _json

    _, executed = _load_startup(monkeypatch, tmp_path,
                                config_text=_json.dumps({"repo_path": str(tmp_path)}))
    assert executed and "macroScript MaxGaffer" in executed[0]
    assert str(tmp_path) in sys.path


def test_startup_missing_repo_skips_macro_loudly(monkeypatch, tmp_path, capsys):
    _, executed = _load_startup(monkeypatch, tmp_path)   # no config, no env
    assert executed == []
    out = capsys.readouterr().out
    assert "macro NOT registered" in out and "MaxGaffer" in out


def test_startup_register_survives_repo_path_explosion(monkeypatch, tmp_path, capsys):
    mod, _ = _load_startup(monkeypatch, tmp_path)
    monkeypatch.setattr(mod, "_repo_path",
                        lambda: (_ for _ in ()).throw(RuntimeError("disk on fire")))
    mod._register()                                  # a startup script must never raise
    assert "startup registration failed" in capsys.readouterr().out
