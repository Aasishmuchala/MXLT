"""Crash-guard regressions (2026-07-18) — the three protections against on-box Max crashes:

1. bootstrap must never let user-site shadow Max's bundled binaries, and must REFUSE to
   build the dock when a conflicting PySide6 would be imported (binary Qt conflicts are
   hard crashes a try/except cannot catch — the guard runs before the import);
2. vantage.link_running must detect a live link non-destructively (the V-Ray action is a
   toggle — probing by executing it would flip the artist's link off);
3. run_match must warn BEFORE rendering when the link is up and the renderer is V-Ray
   GPU (checklist #14 — one card doing both is the documented VRAM crash).
"""

from __future__ import annotations

import socket
import sys

import pytest

from maxgaffer import bootstrap
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import controller as ctl
from maxgaffer.maxbridge import vantage
from maxgaffer.core.genome import LightingState
from maxgaffer.core.scenarios import DEFAULT_SEMANTICS


# --------------------------------------------------------------------------- bootstrap
class TestPySideConflict:
    def _fake_spec(self, monkeypatch, origin):
        import importlib.util as ilu

        class _Spec:
            def __init__(self, o):
                self.origin = o

        monkeypatch.setattr(ilu, "find_spec", lambda name: _Spec(origin))
        monkeypatch.setattr(sys, "executable", "/Autodesk/3ds Max 2026/3dsmax.exe")

    def test_bundled_pyside_is_no_conflict(self, monkeypatch):
        self._fake_spec(monkeypatch,
                        "/Autodesk/3ds Max 2026/Python/Lib/site-packages/PySide6/__init__.py")
        assert bootstrap._pyside_conflict() == ""

    def test_user_site_pyside_is_flagged(self, monkeypatch):
        self._fake_spec(monkeypatch,
                        "/Users/x/AppData/Roaming/Python/Python311/site-packages/"
                        "PySide6/__init__.py")
        out = bootstrap._pyside_conflict()
        assert out and "PySide6" in out

    def test_missing_pyside_is_no_conflict(self, monkeypatch):
        import importlib.util as ilu

        monkeypatch.setattr(ilu, "find_spec", lambda name: None)
        assert bootstrap._pyside_conflict() == ""

    def test_usersite_is_appended_never_prepended(self, tmp_path, monkeypatch):
        import site as site_mod

        d = str(tmp_path)
        monkeypatch.setattr(site_mod, "getusersitepackages", lambda: d)
        monkeypatch.delitem(__import__("os").environ, "APPDATA", raising=False)
        monkeypatch.delitem(__import__("os").environ, "LOCALAPPDATA", raising=False)
        before = list(sys.path)
        try:
            bootstrap._ensure_usersite_on_path()
            assert sys.path[-1] == d and sys.path[0] == (before[0] if before else d)
        finally:
            if d in sys.path:
                sys.path.remove(d)

    def test_launch_refuses_on_conflict(self, monkeypatch):
        monkeypatch.setattr(bootstrap, "_pyside_conflict", lambda: "/bad/PySide6")
        monkeypatch.setattr(bootstrap, "_ensure_usersite_on_path", lambda: None)
        assert bootstrap.launch() is None   # messageBox path, never the dock import


# --------------------------------------------------------------------------- vantage
class TestLinkRunning:
    def test_detects_listener(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            assert vantage.link_running((port,)) == port
        finally:
            srv.close()

    def test_no_listener_returns_none(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
        s.close()
        assert vantage.link_running((free,)) is None


# --------------------------------------------------------------------------- run_match
SEMANTICS = dict(DEFAULT_SEMANTICS)
SEMANTICS.setdefault("key_notes", "test reference")


def _state():
    st = LightingState()
    st.set("sun.enabled", 1)
    st.set("exposure.ev", 12.0)
    st.set("exposure.wb_kelvin", 6500.0)
    return st


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    cfg = cfgmod.Config(no_renders=True, auto_exposure_control=False, analyze_samples=1)
    c = ctl.Controller(cfg)
    scene = {"state": _state()}
    monkeypatch.setattr(ctl.sc, "scene_path", lambda: str(tmp_path / "t.max"))
    monkeypatch.setattr(ctl.sc, "classify_rig", lambda: {
        "sun": object(), "dome": None, "groups": {}, "notes": [], "sky_env": False})
    monkeypatch.setattr(ctl.sc, "get_camera", lambda name: object())
    monkeypatch.setattr(ctl.sc, "set_active_camera", lambda name: None)
    monkeypatch.setattr(ctl.sc, "camera_yaw_deg", lambda cam: 0.0)
    monkeypatch.setattr(ctl.ap, "capture_baselines", lambda rig: {})
    monkeypatch.setattr(ctl.ap, "read_state",
                        lambda rig, b, cam=None: scene["state"].copy())
    monkeypatch.setattr(ctl.ap, "apply_state", lambda rig, b, st, cam=None, undo=True: [])
    monkeypatch.setattr(cfgmod, "sessions_dir", lambda: str(tmp_path / "sessions"))
    monkeypatch.setattr(c, "analyze_reference", lambda name: dict(SEMANTICS))
    monkeypatch.setattr(c, "_image_block", lambda path: {"type": "image"})
    monkeypatch.setattr(c, "ref_stats", lambda path: {"log_key": 0.2})
    c.session.entry("CamA").reference = "ref.jpg"
    return c


class TestGpuLinkWarning:
    def _fake_rt(self, cls_name):
        class _RT:
            class renderers:
                current = object()

            @staticmethod
            def classOf(obj):
                return cls_name

        return _RT

    def test_warns_when_link_up_and_gpu(self, ctrl, monkeypatch):
        monkeypatch.setattr(vantage, "link_running", lambda ports=None: 20701)
        monkeypatch.setattr(ctl.sc, "_rt", lambda: self._fake_rt("V_Ray_GPU_7__update_2"))
        lines = []
        ctrl.run_match("CamA", log=lines.append)
        assert any("V-Ray GPU" in m and "20701" in m for m in lines)

    def test_no_warning_when_link_down(self, ctrl, monkeypatch):
        monkeypatch.setattr(vantage, "link_running", lambda ports=None: None)
        monkeypatch.setattr(ctl.sc, "_rt", lambda: self._fake_rt("V_Ray_GPU_7__update_2"))
        lines = []
        ctrl.run_match("CamA", log=lines.append)
        assert not any("V-Ray GPU" in m and "20701" in m for m in lines)

    def test_no_warning_on_cpu_renderer(self, ctrl, monkeypatch):
        monkeypatch.setattr(vantage, "link_running", lambda ports=None: 20701)
        monkeypatch.setattr(ctl.sc, "_rt", lambda: self._fake_rt("V_Ray_7__update_2"))
        lines = []
        ctrl.run_match("CamA", log=lines.append)
        assert not any("V-Ray GPU" in m and "20701" in m for m in lines)

    def test_off_max_silently_skips(self, ctrl, monkeypatch):
        # no pymxs anywhere and link down — the guard must degrade, never break a match
        monkeypatch.setattr(vantage, "link_running", lambda ports=None: None)
        monkeypatch.setattr(ctl.sc, "_rt",
                            lambda: (_ for _ in ()).throw(ImportError("no pymxs")))
        lines = []
        ctrl.run_match("CamA", log=lines.append)
        assert any("match" in m.lower() or "applied" in m.lower() for m in lines)


# ---------------------------------------------------------------- dock image safety
class TestBoundedPixmap:
    def test_decode_is_bounded_not_full_res(self, tmp_path):
        pytest.importorskip("PySide6.QtWidgets")
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtGui, QtWidgets
        from maxgaffer.ui import dock as d

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        big = tmp_path / "big.jpg"
        QtGui.QImage(4000, 3000, QtGui.QImage.Format_RGB32).save(str(big))
        pix = d._bounded_pixmap(str(big), __import__("PySide6.QtCore", fromlist=["QSize"])
                                .QSize(240, 135))
        assert not pix.isNull()
        assert pix.width() <= 240 and pix.height() <= 135
        _ = app

    def test_missing_file_returns_null(self, tmp_path):
        pytest.importorskip("PySide6.QtWidgets")
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6 import QtCore, QtWidgets  # noqa: F401
        from maxgaffer.ui import dock as d

        QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        assert d._bounded_pixmap(str(tmp_path / "nope.jpg"),
                                 QtCore.QSize(240, 135)).isNull()


class TestLogMirror:
    def test_mirror_writes_and_reset_truncates(self, tmp_path, monkeypatch):
        pytest.importorskip("PySide6.QtWidgets")   # importing the dock needs Qt present
        from maxgaffer.ui import dock as d

        mirror = tmp_path / "last_session.log"
        monkeypatch.setattr(d, "_LOG_MIRROR", str(mirror))
        d._reset_log_mirror()
        d._mirror_log("step one")
        d._mirror_log("step two")
        text = mirror.read_text()
        assert "step one" in text and "step two" in text
        d._reset_log_mirror()
        assert "step one" not in mirror.read_text()
