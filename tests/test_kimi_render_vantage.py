"""Cluster J regressions — render.py / vantage.py / execute.py fixes.

Off-Max, pure python: pymxs is stubbed in sys.modules where the bridge needs it.
"""

import contextlib
import os
import sys
import time
from types import SimpleNamespace

import pytest

from maxgaffer.maxbridge import execute as ex
from maxgaffer.maxbridge import render as rd
from maxgaffer.maxbridge import vantage as vt


# ----------------------------------------------------------------- pymxs stub helpers
def _install_pymxs(monkeypatch, rt):
    fake = SimpleNamespace(runtime=rt,
                           undo=lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "pymxs", fake)
    return fake


# ----------------------------------------------------------------- vantage output gate
def test_output_written_rejects_same_stem_sibling(tmp_path):
    """Shot10.png must NOT satisfy an expectation for Shot1.png (old startswith bug)."""
    (tmp_path / "Shot10.png").write_bytes(b"x" * 10)
    assert not vt._output_written(str(tmp_path / "Shot1.png"))


def test_output_written_accepts_exact_and_frame_suffix(tmp_path):
    exact = tmp_path / "Cam01.png"
    assert not vt._output_written(str(exact))
    (tmp_path / "Cam01.0000.png").write_bytes(b"x" * 10)   # Vantage frame suffix
    assert vt._output_written(str(exact))
    (tmp_path / "Cam01.0000.png").unlink()
    exact.write_bytes(b"x" * 10)
    assert vt._output_written(str(exact))


def test_output_written_min_mtime_rejects_stale(tmp_path):
    exact = tmp_path / "Cam01.png"
    exact.write_bytes(b"x" * 10)
    old = time.time() - 3600
    os.utime(exact, (old, old))
    now = time.time()
    assert not vt._output_written(str(exact), min_mtime=now)   # stale → not this run's
    assert vt._output_written(str(exact), min_mtime=old - 5)   # fresh enough → ok
    suffix = tmp_path / "Cam01.0000.png"
    suffix.write_bytes(b"x" * 10)
    os.utime(suffix, (old, old))
    os.remove(exact)
    assert not vt._output_written(str(exact), min_mtime=now)   # stale suffix too


# ----------------------------------------------------------------- vantage CLI batch
CONSOLE_STUB = (
    "import os, sys, time\n"
    "mode, out = sys.argv[1], sys.argv[2]\n"
    "if mode == 'ok':\n"
    "    open(out, 'wb').write(b'png')\n"
    "elif mode == 'suffix':\n"
    "    stem, ext = os.path.splitext(out)\n"
    "    open(stem + '.0000' + ext, 'wb').write(b'png')\n"
    "elif mode == 'sleep':\n"
    "    time.sleep(30)\n"
    "elif mode == 'fail':\n"
    "    sys.exit(3)\n"
)


def _fake_console(monkeypatch, tmp_path):
    script = tmp_path / "fake_console.py"
    script.write_text(CONSOLE_STUB)
    monkeypatch.setattr(vt, "vantage_command",
                        lambda exe, scene, out, w, h, frame=0:
                        [sys.executable, str(script), scene, out])
    return sys.executable      # exists → passes the console_exe check


def _jobs(tmp_path, modes):
    return [{"camera": name, "scene_file": mode,
             "output": str(tmp_path / f"{name}.png")}
            for name, mode in modes]


def test_render_stills_continues_after_failure(monkeypatch, tmp_path):
    exe = _fake_console(monkeypatch, tmp_path)
    jobs = _jobs(tmp_path, [("A", "ok"), ("B", "fail"), ("C", "ok")])
    res = vt.render_stills(jobs, exe, 64, 64)
    assert res == {"A": "ok", "B": "vantage exit 3", "C": "ok"}   # no more break


def test_render_stills_rejects_stale_output(monkeypatch, tmp_path):
    exe = _fake_console(monkeypatch, tmp_path)
    jobs = _jobs(tmp_path, [("A", "noop")])
    stale = tmp_path / "A.png"
    stale.write_bytes(b"yesterday")
    res = vt.render_stills(jobs, exe, 64, 64)
    assert res["A"] == "vantage exit 0 but no output written"      # honest, not "ok"
    assert not stale.exists()                                      # deleted pre-launch


def test_render_stills_accepts_frame_suffix_output(monkeypatch, tmp_path):
    exe = _fake_console(monkeypatch, tmp_path)
    jobs = _jobs(tmp_path, [("A", "suffix")])
    assert vt.render_stills(jobs, exe, 64, 64)["A"] == "ok"


def test_render_stills_cancel_kills_running_console(monkeypatch, tmp_path):
    exe = _fake_console(monkeypatch, tmp_path)
    jobs = _jobs(tmp_path, [("A", "sleep"), ("B", "ok")])
    calls = {"n": 0}

    def should_cancel():                   # flip to True shortly after job A spawns
        calls["n"] += 1
        return calls["n"] > 3

    t0 = time.monotonic()
    res = vt.render_stills(jobs, exe, 64, 64, should_cancel=should_cancel)
    assert time.monotonic() - t0 < 15      # the 30s sleeper was killed, not awaited
    assert res == {"A": "cancelled", "B": "cancelled"}


def test_render_stills_timeout_kills_console(monkeypatch, tmp_path):
    exe = _fake_console(monkeypatch, tmp_path)
    jobs = _jobs(tmp_path, [("A", "sleep"), ("B", "ok")])
    t0 = time.monotonic()
    res = vt.render_stills(jobs, exe, 64, 64, timeout_s=1)
    assert time.monotonic() - t0 < 15
    assert res["A"] == "timeout after 1s"
    assert res["B"] == "ok"                # batch continues after a timeout too


def test_render_stills_job_timeout_is_sane():
    assert vt.DEFAULT_JOB_TIMEOUT_S == 20 * 60     # documented, no longer 3600s


def test_render_stills_missing_console_still_reports_all(tmp_path):
    jobs = _jobs(tmp_path, [("A", "ok"), ("B", "ok")])
    res = vt.render_stills(jobs, str(tmp_path / "nope.exe"), 64, 64)
    assert set(res) == {"A", "B"} and "not found" in res["A"]


# ----------------------------------------------------------------- live link toggle
class _LinkRT:
    """execute() answers per-expression; probes/toggles configurable per scenario."""
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def execute(self, expr):
        self.calls.append(expr)
        val = self.answers.get(expr, RuntimeError(f"undefined: {expr}"))
        if isinstance(val, Exception):
            raise val
        return val

    actionMan = property(lambda self: (_ for _ in ()).throw(RuntimeError("no actionMan")))


def test_live_link_already_active_fires_no_toggle(monkeypatch):
    rt = _LinkRT({"vantageLiveLinkActive()": True})
    _install_pymxs(monkeypatch, rt)
    ok, how = vt.start_live_link()
    assert ok and "already active" in how
    assert not any(c in vt.LIVE_LINK_GLOBALS for c in rt.calls)   # toggle never fired


def test_live_link_inactive_is_started_and_says_so(monkeypatch):
    rt = _LinkRT({"vantageLiveLinkActive()": False, "vantageStartLiveLink()": None})
    _install_pymxs(monkeypatch, rt)
    ok, how = vt.start_live_link()
    assert ok and how.startswith("started via maxscript global")


def test_live_link_undetectable_state_reports_toggle_honestly(monkeypatch):
    rt = _LinkRT({"vantageStartLiveLink()": None})       # probes raise → undetectable
    _install_pymxs(monkeypatch, rt)
    ok, how = vt.start_live_link()
    assert ok
    assert how.startswith("toggled via") and "not detectable" in how
    assert "started" not in how.split("—")[0]            # never claims "started"


def test_live_link_toggle_off_is_reported(monkeypatch):
    fired = {"done": False}

    class RT(_LinkRT):
        def execute(self, expr):
            self.calls.append(expr)
            if expr in vt.LIVE_LINK_PROBES:
                if not fired["done"]:
                    raise RuntimeError("no probe")       # pre-state undetectable
                return False                             # after toggle: link is OFF
            if expr == "vantageStartLiveLink()":
                fired["done"] = True
                return None
            raise RuntimeError("undefined")

    _install_pymxs(monkeypatch, RT({}))
    ok, how = vt.start_live_link()
    assert ok and "now OFF" in how


def test_live_link_no_entry_point_still_honest(monkeypatch):
    _install_pymxs(monkeypatch, _LinkRT({}))
    ok, how = vt.start_live_link()
    assert not ok and "no live-link entry point" in how


# ----------------------------------------------------------------- render_frame bitmap
class _Bitmap:
    def __init__(self):
        self.filename = None


class _RenderRT:
    def __init__(self, save_mode):
        self.renderWidth, self.renderHeight = 640, 480
        self.renderSceneDialog = SimpleNamespace(close=lambda: None)
        self.save_mode = save_mode
        self.closed = []
        self.bm = _Bitmap()

    def Point2(self, w, h):
        return (w, h)

    def render(self, **kw):
        return self.bm

    def save(self, bm):
        if self.save_mode == "raise":
            raise RuntimeError("disk full")
        if self.save_mode == "write":
            with open(bm.filename, "wb") as f:
                f.write(b"img")
        # "noop": writes nothing (silent failure)

    def close(self, bm):
        self.closed.append(bm)


def test_render_frame_closes_bitmap_when_save_throws(monkeypatch, tmp_path):
    rt = _RenderRT("raise")
    _install_pymxs(monkeypatch, rt)
    out = str(tmp_path / "cam.png")
    assert rd.render_frame(object(), out, 64, 64) is None
    assert rt.closed == [rt.bm]                            # no framebuffer leak


def test_render_frame_deletes_stale_target_and_judges_fresh(monkeypatch, tmp_path):
    rt = _RenderRT("noop")                                 # save silently writes nothing
    _install_pymxs(monkeypatch, rt)
    out = tmp_path / "cam.png"
    out.write_bytes(b"yesterday")                          # stale output from last run
    assert rd.render_frame(object(), str(out), 64, 64) is None   # NOT judged by staleness
    assert rt.closed == [rt.bm]
    assert (rt.renderWidth, rt.renderHeight) == (640, 480)       # size restored


def test_render_frame_success_writes_and_closes(monkeypatch, tmp_path):
    rt = _RenderRT("write")
    _install_pymxs(monkeypatch, rt)
    out = str(tmp_path / "cam.png")
    assert rd.render_frame(object(), out, 64, 64) == out
    assert rt.closed == [rt.bm]


# ----------------------------------------------------------------- _coerce hardening
class _CoerceRT:
    def Point3(self, x, y, z):
        return ("P3", x, y, z)

    def color(self, r, g, b):
        return ("color", r, g, b)

    def classOf(self, v):
        return None


@pytest.mark.parametrize("text,expected", [
    ("false", False), ("FALSE", False), ("0", False), ("no", False), ("off", False),
    ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
])
def test_coerce_bool_strings(monkeypatch, text, expected):
    _install_pymxs(monkeypatch, _CoerceRT())
    assert ex._coerce(True, text) is expected


def test_coerce_rejects_garbage_bool_string(monkeypatch):
    _install_pymxs(monkeypatch, _CoerceRT())
    with pytest.raises(ValueError):
        ex._coerce(True, "banana")


def test_coerce_rejects_non_finite(monkeypatch):
    _install_pymxs(monkeypatch, _CoerceRT())
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            ex._coerce(1.0, bad)
    with pytest.raises(ValueError):
        ex._coerce(None, [float("nan"), 0.0, 0.0])


def test_coerce_still_passes_good_values(monkeypatch):
    _install_pymxs(monkeypatch, _CoerceRT())
    assert ex._coerce(True, 0) is False
    assert ex._coerce(5, 2.7) == 2                    # int property truncates
    assert ex._coerce(1.0, 2.5) == 2.5
    assert ex._coerce(None, [1, 2, 3]) == ("color", 1.0, 2.0, 3.0)


# ----------------------------------------------------------------- create_light safety
class _Node:
    def __init__(self, rt):
        self._rt = rt
        rt.nodes.append(self)


class _SunNode(_Node):
    @property
    def target(self):
        return getattr(self, "_target", None)

    @target.setter
    def target(self, v):
        if self._rt.fail_target:
            raise RuntimeError("cannot assign target")
        self._target = v


class _Layer:
    def __init__(self):
        self.added = []

    def addNode(self, n):
        self.added.append(n)


class _ExecRT(_CoerceRT):
    def __init__(self):
        self.nodes = []
        self.deleted = []
        self.layer = _Layer()
        self.fail_target = False
        self.LayerManager = SimpleNamespace(
            getLayerFromName=lambda name: self.layer,
            newLayerFromName=lambda name: self.layer)

    def VRayLight(self):
        return _Node(self)

    def VRaySun(self):
        return _SunNode(self)

    def Targetobject(self):
        return _Node(self)

    def isProperty(self, obj, name):
        return True

    def Name(self, s):
        return s

    def delete(self, n):
        self.deleted.append(n)

    def redrawViews(self):
        pass

    def getNodeByName(self, name, exact=True):
        for n in self.nodes:
            if getattr(n, "name", None) == name:
                return n
        return None


BASIS = {"pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0, "look": [0.0, 200.0, 0.0]}


def test_create_light_rolls_back_orphan_on_failure(monkeypatch):
    rt = _ExecRT()
    _install_pymxs(monkeypatch, rt)
    op = {"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_x"}
    # no "placement" key → the pre-create validation refuses the op BEFORE the ctor
    # runs, so no orphan ever exists — stronger than the old create-then-roll-back
    rep = ex.execute_plan([op], camera=None)
    assert rep["created"] == []
    assert rt.nodes == [] and rt.deleted == []         # nothing created at all
    assert any("op failed" in w for w in rep["warnings"])


def test_create_sun_target_failure_deletes_helper(monkeypatch):
    rt = _ExecRT()
    rt.fail_target = True
    _install_pymxs(monkeypatch, rt)
    monkeypatch.setattr(ex, "camera_basis", lambda cam: BASIS)
    op = {"op": "create_light", "light_type": "VRaySun", "name": "MG_sun",
          "placement": {"bearing_deg": 0, "distance": 100, "height": 0}}
    rep = ex.execute_plan([op], camera=object())
    assert len(rep["created"]) == 1                    # the light itself survived
    assert len(rt.deleted) == 1                        # but the failed target is gone
    assert any("sun target" in w for w in rep["warnings"])


def test_create_sun_target_lands_on_mg_layer(monkeypatch):
    rt = _ExecRT()
    _install_pymxs(monkeypatch, rt)
    monkeypatch.setattr(ex, "camera_basis", lambda cam: BASIS)
    op = {"op": "create_light", "light_type": "VRaySun", "name": "MG_sun",
          "placement": {"bearing_deg": 0, "distance": 100, "height": 0}}
    rep = ex.execute_plan([op], camera=object())
    assert len(rep["created"]) == 1
    assert len(rt.layer.added) == 2                    # light AND its target on-layer
    assert rt.deleted == []


def test_set_op_bool_string_and_nan_end_to_end(monkeypatch):
    rt = _ExecRT()
    _install_pymxs(monkeypatch, rt)
    light = _Node(rt)
    light.name = "MG_key"
    light.enabled = True
    light.multiplier = 1.0
    ops = [
        {"op": "set", "target": "node:MG_key", "prop": "enabled", "value": "false"},
        {"op": "set", "target": "node:MG_key", "prop": "multiplier",
         "value": float("nan")},
        {"op": "set", "target": "node:MG_key", "prop": "enabled", "value": "banana"},
    ]
    rep = ex.execute_plan(ops, camera=None)
    assert light.enabled is False                      # "false" → False, not True
    assert light.multiplier == 1.0                     # NaN rejected, property untouched
    assert len(rep["changes"]) == 1
    assert rep["changes"][0]["before"] is True and rep["changes"][0]["after"] is False
    assert len(rep["warnings"]) == 2                   # NaN + garbage string warned


# ----------------------------------------------------------------- export camera restore
def test_export_vrscene_restores_previous_camera(monkeypatch, tmp_path):
    calls = []

    class RT:
        viewport = SimpleNamespace(
            getCamera=lambda: "prevCam",
            setCamera=lambda cam: calls.append(cam))
        currentTime = SimpleNamespace(frame=5)

        def vrayExportVRScene(self, path, **kw):
            with open(path, "wb") as f:
                f.write(b"vrscene")

    _install_pymxs(monkeypatch, RT())
    monkeypatch.setattr(vt.sc, "set_active_camera", lambda name: True)
    out = str(tmp_path / "shot.vrscene")
    assert vt.export_vrscene(out, "CamA") == out
    assert calls == ["prevCam"]                        # restored after the export
