"""Cluster G regressions — config hardening + scene/digest defensive reads.

Off-Max: a stub `pymxs` module is injected into sys.modules (the bridge only ever
touches it through the lazy `_rt()`), so scene/digest logic runs against faithful
fakes of Max nodes, texmaps and collections.
"""

import json
import os
import sys
import types

import pytest

from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import scene as sc
from maxgaffer.maxbridge import digest as dg


# --------------------------------------------------------------------- fakes
class FakePoint3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __sub__(self, other):
        return FakePoint3(self.x - other.x, self.y - other.y, self.z - other.z)


class FakeCam:
    def __init__(self, name, row3=None, cls="Physical"):
        self.name = name
        self._cls = cls
        self._props = []
        self.transform = types.SimpleNamespace(row3=row3 or FakePoint3(0.0, -1.0, 0.0))


class StaleCam:
    """Handle to a deleted node: every attribute read explodes."""
    name = "Ghost"

    @property
    def _cls(self):
        raise RuntimeError("stale handle")

    @property
    def transform(self):
        raise AttributeError("stale handle")


class FakeLight:
    def __init__(self, name, cls="VRayLight"):
        self.name = name
        self._cls = cls
        self._props = ["enabled", "multiplier"]
        self.enabled = True
        self.multiplier = 30.0
        self.layer = types.SimpleNamespace(name="practicals")
        self.pos = FakePoint3(1.0, 2.0, 3.0)


class HostileLight:
    @property
    def _cls(self):
        raise RuntimeError("stale handle")

    @property
    def name(self):
        raise RuntimeError("stale handle")


class FakeSun:
    def __init__(self):
        self.name = "Sun01"
        self._cls = "VRaySun"
        self.target = types.SimpleNamespace(pos=FakePoint3(0.0, 0.0, 0.0))
        self.pos = FakePoint3(0.0, 100.0, 0.0)


class StaleSun:
    name = "SunGone"
    _cls = "VRaySun"
    target = None

    @property
    def pos(self):
        raise AttributeError("stale handle")


class DomeWithTexmapSlot:
    """texmap property that records what the texmap's file prop held AT BIND TIME."""

    def __init__(self):
        self._texmap = None
        self.bind_order = []

    @property
    def texmap(self):
        return self._texmap

    @texmap.setter
    def texmap(self, value):
        self.bind_order.append(getattr(value, "HDRIMapName", None))
        self._texmap = value


class FakeRt:
    def __init__(self):
        self.cameras = []
        self.lights = []
        self.VRayBitmap = None          # set per-test: callable returning a texmap

    def isProperty(self, obj, name):
        return hasattr(obj, name)

    def Name(self, n):
        return n

    def classOf(self, obj):
        return obj._cls

    def getPropNames(self, obj):
        return list(getattr(obj, "_props", []))

    def getNodeByName(self, name, exact=False):
        return ("getNodeByName", name)

    def Point3(self, x, y, z):
        return FakePoint3(x, y, z)


@pytest.fixture()
def fake_max(monkeypatch):
    rt = FakeRt()
    stub = types.ModuleType("pymxs")
    stub.runtime = rt
    monkeypatch.setitem(sys.modules, "pymxs", stub)
    sc._DUPE_WARNED.clear()
    return rt


# --------------------------------------------------------------------- config
def _cfg_file(tmp_path, payload):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return str(p)


def test_unreadable_existing_config_is_loud(tmp_path, monkeypatch, capsys):
    """Silent degradation to full defaults is one of the three hypotheses that looked
    IDENTICAL from the 2026-07-30 TULA transcript, where a hand-set draft_sampler:true
    and a 20 s probe cap sat on disk and neither reached the run. Every other rejection
    in load() is loud; this one being mute is how "my config has no effect" becomes
    unfalsifiable. A MISSING file is a first run — normal, and still silent."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(path))
    cfg = cfgmod.load()
    assert cfg.max_iterations == 5 and cfg.draft_sampler is False   # full defaults
    assert "using DEFAULTS" in capsys.readouterr().out

    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(tmp_path / "never_written.json"))
    assert cfgmod.load().max_iterations == 5
    assert "using DEFAULTS" not in capsys.readouterr().out          # no first-run noise


@pytest.mark.parametrize("payload", [None, [], "str", 42])
def test_load_non_dict_json_is_empty_and_loud(tmp_path, monkeypatch, capsys, payload):
    """P0: valid JSON that is not an object must not AttributeError out of load()."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", _cfg_file(tmp_path, payload))
    cfg = cfgmod.load()
    assert cfg.max_iterations == 5 and cfg.loop_width == 480
    assert "not an object" in capsys.readouterr().out


def test_load_wrong_typed_values_fall_back_per_field(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", _cfg_file(tmp_path, {
        "max_iterations": "five",          # str for int  → default
        "loop_width": 480.0,               # float for int → default
        "loop_height": True,               # bool is NOT an int → default
        "target_score": 90,                # int for float → widened, accepted
        "auto_exposure_control": 1,        # int is NOT a bool → default
        "critic_weights": "x",             # str for dict → default
        "api_key": 42,                     # non-str → default ("")
        "unknown_key": "ignored",
    }))
    cfg = cfgmod.load()
    assert cfg.max_iterations == 5
    assert cfg.loop_width == 480 and type(cfg.loop_width) is int
    assert cfg.loop_height == 270
    assert cfg.target_score == 90.0 and type(cfg.target_score) is float
    assert cfg.auto_exposure_control is True
    assert cfg.critic_weights == {}
    assert cfg.api_key == ""
    assert not hasattr(cfg, "unknown_key")
    out = capsys.readouterr().out
    assert "max_iterations" in out and "keeping default" in out


def test_load_accepts_valid_values_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    path = _cfg_file(tmp_path, {"max_iterations": 8, "target_score": 91.5,
                                "critic_weights": {"key": 0.5},
                                "auto_exposure_control": False,
                                "api_key": "oc_mine"})
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", path)
    cfg = cfgmod.load()
    assert cfg.max_iterations == 8 and cfg.target_score == 91.5
    assert cfg.critic_weights == {"key": 0.5}
    assert cfg.auto_exposure_control is False and cfg.api_key == "oc_mine"


def test_borrow_maxdirector_key_guards_non_dict(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    md = tmp_path / "MaxDirector"
    md.mkdir()
    (md / "config.json").write_text(json.dumps([]), encoding="utf-8")
    assert cfgmod._borrow_maxdirector_key() == ""          # no AttributeError
    assert "not an object" in capsys.readouterr().out
    (md / "config.json").write_text(json.dumps({"api_key": "oc_md"}), encoding="utf-8")
    assert cfgmod._borrow_maxdirector_key() == "oc_md"


def test_appdata_dir_is_lazy_and_sessions_dir_creates(tmp_path, monkeypatch):
    """os.makedirs must NOT happen at path-construction (import) time."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    d = cfgmod._appdata_dir("MaxGaffer_lazy_probe")
    assert not os.path.exists(d)               # path math only, no disk touch
    sd = cfgmod.sessions_dir()
    assert os.path.isdir(sd)


def test_save_creates_parent_dir_lazily(tmp_path, monkeypatch):
    p = tmp_path / "notyet" / "config.json"
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(p))
    cfgmod.Config(api_key="oc_x").save()
    assert json.loads(p.read_text(encoding="utf-8"))["api_key"] == "oc_x"


# --------------------------------------------------------------------- scene
def test_camera_yaw_deg_math(fake_max):
    cam = FakeCam("C", row3=FakePoint3(-1.0, 0.0, 0.0))   # look = +X → east
    assert sc.camera_yaw_deg(cam) == pytest.approx(90.0)


def test_camera_yaw_deg_stale_handle_warns_and_defaults(fake_max, capsys):
    assert sc.camera_yaw_deg(StaleCam()) == 0.0
    assert "camera_yaw_deg" in capsys.readouterr().out


def test_list_cameras_flags_duplicates_once(fake_max, capsys):
    a, b, c = FakeCam("Cam01"), FakeCam("Cam01"), FakeCam("Cam02")
    fake_max.cameras = [a, b, c]
    out = sc.list_cameras()
    assert [e["name"] for e in out] == ["Cam01", "Cam01", "Cam02"]
    assert out[0]["duplicate"] and out[1]["duplicate"]
    assert "duplicate" not in out[2]
    sc.list_cameras()                        # second call must NOT re-shout
    assert capsys.readouterr().out.count("Cam01") == 1


def test_get_camera_deterministic_first_in_scene_order(fake_max):
    a, b = FakeCam("Cam01"), FakeCam("Cam01")
    fake_max.cameras = [a, b]
    assert sc.get_camera("Cam01") is a       # always the first, never arbitrary
    assert sc.get_camera("Nope") == ("getNodeByName", "Nope")   # legacy fallback


def test_read_sun_angles_placeholder_is_loud(fake_max, capsys):
    assert sc.read_sun_angles(StaleSun()) == (0.0, 45.0, 10000.0)   # contract kept
    assert "read_sun_angles" in capsys.readouterr().out


def test_write_sun_angles_refuses_stale_sun(fake_max, capsys):
    """A stale handle must NOT orbit the world origin off fabricated angles."""
    assert sc.write_sun_angles(StaleSun(), 90.0, 30.0) is False
    assert "NOT moved" in capsys.readouterr().out


def test_write_sun_angles_moves_healthy_sun(fake_max):
    sun = FakeSun()                          # at (0,100,0): dist 100, alt 0
    assert sc.write_sun_angles(sun, 90.0, 0.0) is True
    assert sun.pos.x == pytest.approx(100.0)
    assert sun.pos.y == pytest.approx(0.0)
    assert sun.pos.z == pytest.approx(0.0)


def test_set_dome_texture_failure_binds_nothing(fake_max):
    """P1: 'failed' must mean NOTHING changed — no empty texmap left on the dome."""
    fake_max.VRayBitmap = lambda: types.SimpleNamespace()   # no file prop at all
    dome = DomeWithTexmapSlot()
    assert sc.set_dome_texture(dome, r"X:\env.hdr") == "failed"
    assert dome.bind_order == [] and dome.texmap is None


def test_set_dome_texture_sets_file_before_binding(fake_max):
    fake_max.VRayBitmap = lambda: types.SimpleNamespace(HDRIMapName="")
    dome = DomeWithTexmapSlot()
    assert sc.set_dome_texture(dome, r"X:\env.hdr") == "texmap.HDRIMapName"
    assert dome.bind_order == [r"X:\env.hdr"]   # file was set BEFORE the bind
    assert dome.texmap.HDRIMapName == r"X:\env.hdr"


def test_set_dome_texture_existing_texmap(fake_max):
    tex = types.SimpleNamespace(fileName="old.hdr")
    dome = DomeWithTexmapSlot()
    dome._texmap = tex                       # pre-existing texmap: no creation path
    assert sc.set_dome_texture(dome, "new.hdr") == "texmap.fileName"
    assert tex.fileName == "new.hdr" and dome.bind_order == []


# --------------------------------------------------------------------- digest
def test_digest_one_hostile_light_does_not_abort_the_rest(fake_max):
    fake_max.lights = [FakeLight("Good_A"), HostileLight(), FakeLight("Good_B")]
    d = dg.build_digest()
    names = [e["name"] for e in d["lights"]]
    assert names == ["Good_A", "?", "Good_B"]    # hostile node recorded, loop survived
    assert "error" in d["lights"][1]
    assert d["lights"][0]["props"]["multiplier"] == 30.0
    assert d["lights"][2]["props"]["enabled"] is True


def test_digest_surfaces_duplicate_camera_note(fake_max):
    fake_max.cameras = [FakeCam("Cam01"), FakeCam("Cam01")]
    d = dg.build_digest()
    assert len(d["cameras"]) == 2
    for entry in d["cameras"]:
        assert "duplicate" in entry["note"]
