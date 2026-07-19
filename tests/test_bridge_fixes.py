"""Bridge-layer bugfix regressions (maxbridge/*) — each test encodes a reproduced defect.

pymxs is stubbed as a fake module in sys.modules (the bridge only ever reaches it through
``import pymxs`` inside functions, so a ModuleType with a ``runtime`` namespace suffices).
"""

import contextlib
import importlib.util
import json
import os
import sys
import types

import pytest


# --------------------------------------------------------------------- fake pymxs
class _Node:
    """A scene node: plain attribute bag, like pymxs wrappers."""

    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.name = kw.get("name", "")


class _FakeRT:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.redraws = 0
        self.renderers = _Node(current=_Node())
        self.SceneExposureControl = _Node(exposureControl=None)

    def isProperty(self, obj, name):
        return hasattr(obj, str(name))

    def Name(self, s):
        return s

    def classOf(self, obj):
        return type(obj).__name__

    def Point3(self, x, y, z):
        return _Node(x=x, y=y, z=z)

    def color(self, r, g, b):
        return _Node(r=r, g=g, b=b)

    def redrawViews(self):
        self.redraws += 1

    def delete(self, node):
        self.deleted.append(node)

    def VRayLight(self):
        n = _Node(type=0, multiplier=1.0, enabled=True)
        self.created.append(n)
        return n

    def VRaySun(self):
        n = _Node(intensity_multiplier=1.0, enabled=True)
        self.created.append(n)
        return n

    def getNodeByName(self, name, exact=True):
        return None


@pytest.fixture
def rt(monkeypatch):
    fake_rt = _FakeRT()
    mod = types.ModuleType("pymxs")
    mod.runtime = fake_rt
    mod.undo = lambda *a, **k: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "pymxs", mod)
    return fake_rt


# --------------------------------------------------------------------- execute.py
def test_malformed_create_light_op_leaks_no_orphan(rt):
    """op missing "placement" used to create the node FIRST, then KeyError — an
    untracked light in the scene, on no layer, in no report."""
    from maxgaffer.maxbridge import execute as ex

    report = ex.execute_plan([{"op": "create_light", "light_type": "VRayLight_plane",
                               "name": "MG_key"}])           # no "placement" key
    assert rt.created == []                                  # node never created
    assert report["created"] == []
    assert report["warnings"]                                # …but a warning was recorded


def test_create_light_failure_after_creation_deletes_the_node(rt):
    """If anything after creation fails, the half-built node is deleted, not stranded."""
    from maxgaffer.maxbridge import execute as ex

    class _Unnameable(_Node):
        def __init__(self):
            self.__dict__["name"] = ""

        def __setattr__(self, k, v):
            if k == "name":
                raise RuntimeError("name is controller-locked on this build")
            super().__setattr__(k, v)

    def maker():
        node = _Unnameable()
        rt.created.append(node)
        return node

    rt.VRayLight = maker
    report = ex.execute_plan([{"op": "create_light", "light_type": "VRayLight_plane",
                               "name": "MG_key", "placement": {}}])
    assert rt.created and rt.deleted == rt.created            # cleaned up, not stranded
    assert report["created"] == []
    assert report["warnings"]


def test_create_light_happy_path_still_reports(rt):
    from maxgaffer.maxbridge import execute as ex

    report = ex.execute_plan([{"op": "create_light", "light_type": "VRayLight_plane",
                               "name": "MG_key", "placement": {"bearing_deg": 10,
                                                               "distance": 150}}])
    assert len(rt.created) == 1 and rt.created[0].name == "MG_key"
    assert report["created"][0]["name"] == "MG_key"
    assert report["warnings"] == []


# --------------------------------------------------------------------- draft.py
@pytest.fixture
def draft_env(rt, tmp_path, monkeypatch):
    from maxgaffer.maxbridge import draft as df

    monkeypatch.setattr(df, "SNAPSHOT_PATH", str(tmp_path / "draft_snapshot.json"))
    renderer = _Node(options_progressiveNoiseThreshold=0.01,
                     options_progressiveMaxSubdivs=64)
    rt.renderers.current = renderer
    return df, renderer


def test_draft_snapshot_written_before_first_set_prop(draft_env, monkeypatch):
    """Crash-ordering: a Max crash mid-apply must find the snapshot already on disk."""
    df, renderer = draft_env
    order = []

    real_set_prop = df.set_prop

    def spying_set_prop(obj, names, value):
        order.append(("set_prop", names[0], os.path.exists(df.SNAPSHOT_PATH)))
        return real_set_prop(obj, names, value)

    monkeypatch.setattr(df, "set_prop", spying_set_prop)
    lines = df.apply_draft()
    assert order, "draft props should have been applied"
    assert all(existed for _what, _name, existed in order), \
        "snapshot must exist BEFORE the first renderer mutation"
    assert renderer.options_progressiveNoiseThreshold == pytest.approx(0.05)
    assert any("draft:" in ln for ln in lines)
    df.restore_draft()


def test_draft_roundtrip_restores_originals(draft_env):
    df, renderer = draft_env
    df.apply_draft()
    assert renderer.options_progressiveMaxSubdivs == 12
    assert os.path.exists(df.SNAPSHOT_PATH)
    lines = df.restore_draft()
    assert renderer.options_progressiveNoiseThreshold == pytest.approx(0.01)
    assert renderer.options_progressiveMaxSubdivs == 64       # int stays int
    assert not os.path.exists(df.SNAPSHOT_PATH)
    assert any("restored" in ln for ln in lines)


def test_restore_tolerates_string_valued_snapshot_and_removes_file(draft_env):
    """A hand-edited snapshot with string values used to raise ValueError on the raw
    ':g' format — outside the per-prop try — stranding the file so the 'crash'
    repeated on every launch."""
    df, renderer = draft_env
    with open(df.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump({"options_progressiveNoiseThreshold": "0.02",
                   "options_progressiveMaxSubdivs": "garbage"}, f)
    renderer.options_progressiveNoiseThreshold = 0.05
    lines = df.restore_draft()                                  # must not raise
    assert renderer.options_progressiveNoiseThreshold == pytest.approx(0.02)
    assert not os.path.exists(df.SNAPSHOT_PATH)                 # removed despite the junk
    assert any("restored options_progressiveNoiseThreshold" in ln for ln in lines)
    assert any("could not restore options_progressiveMaxSubdivs" in ln for ln in lines)


def test_restore_removes_unreadable_snapshot(draft_env):
    df, renderer = draft_env
    with open(df.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        f.write("{not json")
    lines = df.restore_draft()
    assert any("unreadable" in ln for ln in lines)
    assert not os.path.exists(df.SNAPSHOT_PATH)


# --------------------------------------------------------------------- apply.py
def test_authored_off_baseline_reads_and_writes_zero(rt):
    """Baseline-0 asymmetry: write pins 0 × factor = 0; read used a phantom 1.0."""
    from maxgaffer.maxbridge import apply as ap

    light = _Node(name="Lamp_A", multiplier=0.0, enabled=True)
    rig = {"sun": None, "dome": None, "groups": {"practicals": [light]}, "notes": []}
    baselines = {"Lamp_A": 0.0}

    st = ap.read_state(rig, baselines)
    assert st.groups["practicals"] == 0.0                     # not 0/1.0, not 1.0

    st.groups["practicals"] = 0.8                             # dimmer up…
    ap.apply_state(rig, baselines, st, undo=False)
    assert light.multiplier == 0.0                            # …authored-off stays off

    st2 = ap.read_state(rig, baselines)
    assert st2.groups["practicals"] == 0.0


def test_missing_baseline_reads_against_1(rt):
    from maxgaffer.maxbridge import apply as ap

    light = _Node(name="NewLight", multiplier=2.5, enabled=True)
    rig = {"sun": None, "dome": None, "groups": {"practicals": [light]}, "notes": []}
    st = ap.read_state(rig, {})
    assert st.groups["practicals"] == pytest.approx(2.5)      # current / 1.0


def test_probe_applies_do_not_redraw_views(rt):
    """undo=False applies are 130+ per deep match — no viewport redraw storm."""
    from maxgaffer.maxbridge import apply as ap

    rig = {"sun": None, "dome": None, "groups": {}, "notes": []}
    ap.apply_state(rig, {}, ap.LightingState(), undo=False)
    ap.apply_state(rig, {}, ap.LightingState(), undo=False)
    assert rt.redraws == 0
    ap.apply_state(rig, {}, ap.LightingState(), undo=True)
    assert rt.redraws == 1


# --------------------------------------------------------------------- exposure.py
def test_read_ev_survives_non_int_gain_type(rt):
    """A string-valued exposure_gain_type raised ValueError out of read_ev → read_state
    died. Non-int must read as not-Target-mode and fall back to the triangle."""
    from maxgaffer.maxbridge.exposure import ExposureHost

    cam = _Node(exposure_value=11.0, exposure_gain_type="Target EV",
                iso=100.0, f_number=8.0, shutter_length_seconds=0.005)

    def _class(obj):
        return "Physical" if obj is cam else type(obj).__name__

    rt.classOf = _class
    host = ExposureHost(cam)
    assert host.kind == "physical_cam"
    assert host.read_ev() is not None                         # no raise, triangle fallback


def test_read_ev_target_mode_still_direct(rt):
    from maxgaffer.maxbridge.exposure import ExposureHost

    cam = _Node(exposure_value=9.5, exposure_gain_type=1)
    rt.classOf = lambda obj: "Physical" if obj is cam else type(obj).__name__
    host = ExposureHost(cam)
    assert host.read_ev() == pytest.approx(9.5)


# --------------------------------------------------------------------- render.py
def test_render_frame_closes_bitmap_when_save_raises(rt, tmp_path):
    from maxgaffer.maxbridge import render as rd

    bm = _Node(filename=None)
    closed = []
    rt.render = lambda **kw: bm
    rt.save = lambda b: (_ for _ in ()).throw(RuntimeError("disk full"))
    rt.close = lambda b: closed.append(b)
    rt.renderSceneDialog = _Node(close=lambda: None)
    rt.renderWidth, rt.renderHeight = 640, 360

    out = rd.render_frame(object(), str(tmp_path / "x.png"), 160, 90)
    assert out is None
    assert closed == [bm]                                     # bitmap not leaked
    assert (rt.renderWidth, rt.renderHeight) == (640, 360)    # size still restored


# --------------------------------------------------------------------- vantage.py
def test_render_stills_continues_after_a_failed_job(tmp_path, monkeypatch):
    from maxgaffer.maxbridge import vantage as vt

    console = tmp_path / "vantage_console.exe"
    console.write_bytes(b"x")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "a.vrscene" in cmd[1]:                             # the FIRST job fails
            raise RuntimeError("console wedged")
        for a in cmd:
            if a.startswith("-outputFile="):
                open(a.split("=", 1)[1], "wb").write(b"png")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vt.subprocess, "run", fake_run)
    jobs = [{"camera": "A", "scene_file": "a.vrscene", "output": str(tmp_path / "a.png")},
            {"camera": "B", "scene_file": "b.vrscene", "output": str(tmp_path / "b.png")},
            {"camera": "C", "scene_file": "c.vrscene", "output": str(tmp_path / "c.png")}]
    results = vt.render_stills(jobs, str(console), 64, 64)
    assert results["A"].startswith("error:")                  # recorded, not raised…
    assert results["B"] == "ok" and results["C"] == "ok"      # …and the batch continued


def test_render_stills_malformed_job_gets_an_error_entry(tmp_path, monkeypatch):
    from maxgaffer.maxbridge import vantage as vt

    console = tmp_path / "vantage_console.exe"
    console.write_bytes(b"x")
    monkeypatch.setattr(vt.subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(returncode=0))
    jobs = [{"camera": "A"},                                  # no scene_file/output
            {"camera": "B", "scene_file": "b.vrscene", "output": str(tmp_path / "b.png")}]
    results = vt.render_stills(jobs, str(console), 64, 64)    # must not raise
    assert set(results) == {"A", "B"}
    assert results["A"].startswith("error:")


def test_render_stills_cancel_marks_remaining_jobs(tmp_path, monkeypatch):
    from maxgaffer.maxbridge import vantage as vt

    console = tmp_path / "vantage_console.exe"
    console.write_bytes(b"x")
    done = []

    def fake_run(cmd, **kw):
        done.append(cmd)
        for a in cmd:
            if a.startswith("-outputFile="):
                open(a.split("=", 1)[1], "wb").write(b"png")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(vt.subprocess, "run", fake_run)
    jobs = [{"camera": c, "scene_file": f"{c}.vrscene",
             "output": str(tmp_path / f"{c}.png")} for c in "ABC"]
    results = vt.render_stills(jobs, str(console), 64, 64,
                               should_cancel=lambda: len(done) >= 1)
    assert results == {"A": "ok", "B": "cancelled", "C": "cancelled"}
    assert len(done) == 1


# --------------------------------------------------------------------- config.py
def test_config_import_survives_unwritable_base(monkeypatch, tmp_path):
    """os.makedirs at import time killed `import config` on an unwritable base."""
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"i am a file, not a directory")      # makedirs → OSError
    monkeypatch.setenv("LOCALAPPDATA", str(blocker / "appdata"))
    spec = importlib.util.spec_from_file_location(
        "maxgaffer_config_isolated",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "maxgaffer", "maxbridge", "config.py"))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "maxgaffer_config_isolated", mod)
    spec.loader.exec_module(mod)                              # must not raise
    assert mod.CONFIG_PATH.endswith("config.json")
    cfg = mod.Config()
    assert cfg.final_render_backend == "vray"                 # constants intact


def test_config_save_creates_the_dir_lazily(monkeypatch, tmp_path):
    from maxgaffer.maxbridge import config as cfgmod

    target = tmp_path / "fresh" / "MaxGaffer" / "config.json"
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(target))
    cfgmod.Config(no_renders=True).save()
    assert json.loads(target.read_text())["no_renders"] is True


def test_sessions_dir_best_effort(monkeypatch, tmp_path):
    from maxgaffer.maxbridge import config as cfgmod

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    d = cfgmod.sessions_dir()
    assert os.path.isdir(d)
