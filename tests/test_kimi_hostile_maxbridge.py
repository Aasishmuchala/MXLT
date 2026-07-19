"""Round-M hostile-mock suite — proves maxbridge/* can NEVER crash 3ds Max.

Threat model (audit): every ``rt.*`` call can return None/undefined or raise at
runtime (deleted nodes, missing V-Ray, locked properties). Every test here runs
the REAL maxbridge code against the hostile fake runtime in ``tests/mock_pymxs.py``
— no source is stubbed out except the pymxs boundary itself.

Assertions come in three flavors:
  * no unhandled exception escapes a public entry point (or only documented
    typed errors);
  * no partial mutation on failure paths — the runtime's ``mutation_log`` must
    show nothing changed (or the documented rollback/restore calls happened);
  * known REAL bugs are reproduced as ``xfail(strict=False)`` — each names the
    exact file:line; when the source is fixed they XPASS (and the marker can go).

Deterministic: FakeMaxRuntime is seeded (CHAOS_SEED) and chaos draws are the
only randomness. Python 3.11 (Max 2026) + 3.12 compatible.
"""

import builtins
import importlib
import json
import math
import sys

import pytest

from tests.mock_pymxs import (CHAOS_SEED, UNDEFINED, FakeMaxRuntime, MockColor,
                              MockNode, MockObject, MockPoint3, MockRtError,
                              MockTextureMap, install)

from maxgaffer.core.genome import LightingState
from maxgaffer.maxbridge import apply as ap
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import digest as dg
from maxgaffer.maxbridge import draft as df
from maxgaffer.maxbridge import execute as ex
from maxgaffer.maxbridge import exposure as exp
from maxgaffer.maxbridge import scene as sc


# --------------------------------------------------------------------- fixture
@pytest.fixture()
def max_rt(monkeypatch, tmp_path):
    """A seeded hostile runtime installed as ``pymxs``; process stays clean."""
    rt = FakeMaxRuntime(seed=CHAOS_SEED)
    monkeypatch.setitem(sys.modules, "pymxs", install(rt))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    sc._DUPE_WARNED.clear()
    return rt


# --------------------------------------------------------------------- builders
def make_cam(rt, name="Cam01", cls="Physical", row3=(0.0, -1.0, 0.0), extra=None):
    tm = sc._rt().transMatrix(sc._rt().Point3(0.0, 0.0, 0.0))  # MockMatrix3 via rt
    tm.row3 = MockPoint3(*row3)
    props = {"exposure_value": 11.0, "exposure_gain_type": 1, "iso": 100,
             "f_number": 8.0, "shutter_length_seconds": 0.005,
             "white_balance_kelvin": 6500.0, "white_balance_type": 1}
    props.update(extra or {})
    cam = MockNode(rt, cls, name, props=props, transform=tm)
    rt.cameras.append(cam)
    return cam


def make_sun(rt, name="Sun01", pos=(0.0, 100.0, 0.0)):
    sun = MockNode(rt, "VRaySun", name,
                   props={"enabled": True, "intensity_multiplier": 1.0,
                          "size_multiplier": 1.0, "turbidity": 3.0,
                          "target": None},
                   pos=pos)
    rt.lights.append(sun)
    return sun


def make_dome(rt, name="Dome01", with_texmap=True, rotation=25.0):
    props = {"type": 1, "enabled": True, "multiplier": 5.0, "useTexmap": False}
    tex = None
    if with_texmap:
        tex = MockObject(rt, "VRayBitmap",
                         {"HDRIMapName": "old.hdr", "horizontalRotation": rotation},
                         superclass=MockTextureMap)
        props["texmap"] = tex
    dome = MockNode(rt, "VRayLight", name, props=props)
    dome._mg["tex_obj"] = tex     # test-side handle, not a Max property
    rt.lights.append(dome)
    return dome


def make_light(rt, name, layer="key", mult=30.0, cls="VRayLight"):
    lt = MockNode(rt, cls, name,
                  props={"type": 0, "enabled": True, "multiplier": mult},
                  layer=layer, pos=(1.0, 2.0, 3.0))
    rt.lights.append(lt)
    return lt


def dome_tex(dome):
    """The test-side texmap handle stored by make_dome (bypasses hostility)."""
    return dome._mg.get("tex_obj")


def basic_rig(rt):
    """sun + dome(with texmap) + two groups + one physical cam + V-Ray EC."""
    sun = make_sun(rt)
    dome = make_dome(rt)
    l1 = make_light(rt, "L1_key", layer="key", mult=30.0)
    l2 = make_light(rt, "L2_fill", layer="0", mult=10.0)   # default layer → practicals
    cam = make_cam(rt)
    ec = rt._makers["vrayCreateVRayExposureControl"]()
    rt.SceneExposureControl.exposureControl = ec
    rt.reset_log()
    return {"sun": sun, "dome": dome, "l1": l1, "l2": l2, "cam": cam, "ec": ec}


def rig_of(nodes):
    return {"sun": nodes["sun"], "dome": nodes["dome"],
            "groups": {"key": [nodes["l1"]], "practicals": [nodes["l2"]]},
            "notes": []}


# ===================================================================== imports
def test_every_maxbridge_module_imports_against_the_mock(max_rt):
    names = ["apply", "config", "controller", "digest", "draft", "execute",
             "exposure", "render", "scene", "vantage"]
    mods = [importlib.import_module("maxgaffer.maxbridge." + n) for n in names]
    assert len(mods) == len(names)
    # the lazy _rt() of every pymxs-touching module resolves to the mock
    for mod in (ap, dg, df, ex, exp, sc):
        assert mod._rt() is max_rt
    from maxgaffer.maxbridge import render as rd
    from maxgaffer.maxbridge import vantage as vt
    assert rd._rt() is max_rt and vt._rt() is max_rt


# ===================================================================== scene.py
def test_list_cameras_empty_scene(max_rt):
    assert sc.list_cameras() == []
    assert sc.get_camera("Nope") is None
    assert sc.set_active_camera("Nope") is False
    assert sc.scene_path() == ""


def test_list_cameras_duplicates_flagged_and_first_wins(max_rt):
    a = make_cam(max_rt, "Cam01")
    b = make_cam(max_rt, "Cam01")
    make_cam(max_rt, "Cam02")
    out = sc.list_cameras()
    assert [c["name"] for c in out] == ["Cam01", "Cam01", "Cam02"]
    assert out[0]["duplicate"] and out[1]["duplicate"]
    assert "duplicate" not in out[2]
    assert sc.get_camera("Cam01") is a or sc.get_camera("Cam01") is b
    assert sc.get_camera("Cam01") is a          # first in scene order, deterministic


def test_list_cameras_target_helpers_filtered_and_stale_camera_skipped(max_rt):
    good = make_cam(max_rt, "Hero")
    helper = MockNode(max_rt, "Targetobject", "Hero.Target")
    max_rt.cameras.append(helper)
    stale = make_cam(max_rt, "Ghost")
    stale.set_stale()
    out = sc.list_cameras()
    assert [c["name"] for c in out] == ["Hero"]
    assert sc.get_camera("Hero") is good


def test_cameras_collection_breaking_mid_iteration_does_not_raise(max_rt):
    make_cam(max_rt, "A")
    make_cam(max_rt, "B")
    max_rt.cameras.fail_after = 1
    out = sc.list_cameras()                      # degrades, never raises
    assert isinstance(out, list)
    assert sc.get_camera("A") is None or sc.get_camera("A").get_raw("name") == "A"


def test_camera_yaw_deg_math_and_hostile_transforms(max_rt, capsys):
    cam = make_cam(max_rt, "E", row3=(-1.0, 0.0, 0.0))   # look = +X → east
    assert sc.camera_yaw_deg(cam) == pytest.approx(90.0)
    stale = make_cam(max_rt, "S")
    stale.set_stale()
    assert sc.camera_yaw_deg(stale) == 0.0       # loud placeholder, no raise
    assert "camera_yaw_deg" in capsys.readouterr().out
    corrupt = make_cam(max_rt, "C")
    corrupt._mg["props"]["transform"] = "not a matrix"
    assert sc.camera_yaw_deg(corrupt) == 0.0     # bad TM values degrade too


def test_classify_rig_empty_scene(max_rt):
    rig = sc.classify_rig()
    assert rig["sun"] is None and rig["dome"] is None
    assert rig["groups"] == {} and rig["sky_env"] is False
    assert any("no VRaySun" in n for n in rig["notes"])
    assert any("no VRayLight dome" in n for n in rig["notes"])


def test_classify_rig_missing_vray_classes_groups_standard_lights(max_rt):
    make_light(max_rt, "Photo", cls="Free_Light")
    max_rt.environmentMap = MockObject(max_rt, "BitmapTexture", {"fileName": "x.jpg"})
    rig = sc.classify_rig()
    assert rig["sun"] is None and rig["dome"] is None
    assert "key" in rig["groups"]                # standard light still joins a board
    assert rig["sky_env"] is False               # non-VRaySky env map read safely


def test_classify_rig_sky_env_and_stale_light(max_rt):
    max_rt.environmentMap = MockObject(max_rt, "VRaySky", {})
    stale = make_light(max_rt, "Gone")
    stale.set_stale()
    keep = make_light(max_rt, "Keep", layer="sconces")
    rig = sc.classify_rig()
    assert rig["sky_env"] is True
    assert rig["groups"].get("sconces") == [keep]  # stale node skipped, survivor kept


def test_classify_rig_duplicate_sun_and_dome_noted(max_rt):
    s1 = make_sun(max_rt, "Sun_A")
    make_sun(max_rt, "Sun_B")
    d1 = make_dome(max_rt, "Dome_A")
    make_dome(max_rt, "Dome_B")
    rig = sc.classify_rig()
    assert rig["sun"] is s1 and rig["dome"] is d1     # first wins
    assert any("extra VRaySun" in n for n in rig["notes"])
    assert any("extra dome" in n for n in rig["notes"])


def test_sun_stale_read_placeholder_write_refused(max_rt, capsys):
    sun = make_sun(max_rt)
    sun.set_stale()
    assert sc.read_sun_angles(sun) == (0.0, 45.0, 10000.0)   # loud placeholder
    assert sc.write_sun_angles(sun, 90.0, 30.0) is False     # refuses to orbit origin
    out = capsys.readouterr().out
    assert "read_sun_angles" in out and "NOT moved" in out


def test_write_sun_angles_healthy_geometry(max_rt):
    sun = make_sun(max_rt, pos=(0.0, 100.0, 0.0))     # dist 100, alt 0
    assert sc.write_sun_angles(sun, 90.0, 0.0) is True
    pos = sun.get_raw("pos")
    assert pos.x == pytest.approx(100.0) and pos.y == pytest.approx(0.0)


def test_dome_rotation_texmap_path_and_failure_fallback(max_rt):
    dome = make_dome(max_rt, rotation=25.0)
    assert sc.read_dome_rotation(dome) == pytest.approx(25.0)
    how = sc.write_dome_rotation(dome, 45.0)
    assert how == "texmap.horizontalRotation"
    assert dome_tex(dome).get_raw("horizontalRotation") == pytest.approx(45.0)
    # texmap rotation prop locked, node path broken too → 'failed', no raise
    dome_tex(dome).arm_set("horizontalRotation")
    dome.arm_set("transform")
    assert sc.write_dome_rotation(dome, 90.0) == "failed"


def test_set_dome_texture_creation_success_sets_file_before_binding(max_rt):
    dome = make_dome(max_rt, with_texmap=False)
    max_rt.reset_log()
    how = sc.set_dome_texture(dome, r"X:\env\sky.hdr")
    assert how == "texmap.HDRIMapName"
    tex = dome.get_raw("texmap")
    assert tex is not None and tex.get_raw("HDRIMapName") == r"X:\env\sky.hdr"
    assert dome.get_raw("useTexmap") is True
    sets = max_rt.sets()
    file_idx = next(i for i, e in enumerate(sets)
                    if e[2] == "HDRIMapName" and e[1] is tex)
    bind_idx = next(i for i, e in enumerate(sets)
                    if e[1] is dome and e[2] == "texmap")
    assert file_idx < bind_idx                  # file set BEFORE the bind


def test_set_dome_texture_missing_vray_classes_fails_clean(max_rt):
    max_rt.remove_maker("VRayBitmap")
    max_rt.remove_maker("VRayHDRI")
    dome = make_dome(max_rt, with_texmap=False)
    max_rt.reset_log()
    assert sc.set_dome_texture(dome, r"X:\env.hdr") == "failed"
    assert dome.get_raw("texmap") is None
    assert max_rt.sets() == []                  # ZERO mutations on failure


def test_set_dome_texture_unwritable_file_prop_binds_nothing(max_rt):
    """The audit's P1: 'failed' must never leave an empty texmap on the dome."""
    def barren_texmap():
        return MockObject(max_rt, "VRayBitmap", {}, superclass=MockTextureMap)
    max_rt.add_maker("VRayBitmap", barren_texmap)
    max_rt.add_maker("VRayHDRI", barren_texmap)
    dome = make_dome(max_rt, with_texmap=False)
    max_rt.reset_log()
    assert sc.set_dome_texture(dome, r"X:\env.hdr") == "failed"
    assert dome.get_raw("texmap") is None       # nothing bound
    assert max_rt.sets(dome, "texmap") == []    # no bind recorded


def test_set_dome_texture_bind_failure_leaves_dome_untouched(max_rt):
    dome = make_dome(max_rt, with_texmap=False)
    dome.arm_set("texmap")                       # locked property at bind time
    max_rt.reset_log()
    assert sc.set_dome_texture(max_rt.lights[-1] if False else dome,
                               r"X:\env.hdr") == "failed"
    assert dome.get_raw("texmap") is None        # bind never landed
    assert max_rt.sets(dome, "texmap") == []


def test_set_dome_texture_existing_texmap_set_failure_keeps_old_file(max_rt):
    dome = make_dome(max_rt, with_texmap=True)
    tex = dome_tex(dome)
    tex.arm_set("HDRIMapName")                   # read-only file spinner
    max_rt.reset_log()
    assert sc.set_dome_texture(dome, "new.hdr") == "failed"
    assert tex.get_raw("HDRIMapName") == "old.hdr"   # original value intact


# ===================================================================== digest.py
def test_digest_empty_scene(max_rt):
    d = dg.build_digest()
    assert d["lights"] == [] and d["cameras"] == []
    assert d["environment"]["class"] == "none"
    assert d["exposure"]["class"] == "none"
    assert d["stats"]["lights"] == 0


def test_digest_skips_one_hostile_light_and_continues(max_rt):
    make_light(max_rt, "Good_A", mult=30.0)
    hostile = make_light(max_rt, "Evil")
    hostile.set_stale()                          # deleted-node handle mid-scene
    make_light(max_rt, "Good_B", mult=5.0)
    d = dg.build_digest()
    names = [e["name"] for e in d["lights"]]
    assert names == ["Good_A", "?", "Good_B"]    # hostile recorded, loop survived
    assert "error" in d["lights"][1]
    assert d["lights"][0]["props"]["multiplier"] == 30.0
    assert d["lights"][2]["layer"] == "key"


def test_digest_all_globals_hostile_still_returns_full_shape(max_rt):
    max_rt.arm_global("environmentMap")
    max_rt.renderers.arm_get("current")
    max_rt.SceneExposureControl.arm_get("exposureControl")
    max_rt.lights.fail_after = 1
    make_light(max_rt, "A")
    make_light(max_rt, "B")
    d = dg.build_digest()
    assert set(d) == {"renderer", "environment", "exposure",
                      "lights", "cameras", "stats"}
    assert d["lights"] == []                     # broken collection → empty, not crash


def test_digest_normalizes_colors_points_and_texmaps(max_rt):
    lt = make_light(max_rt, "L")
    lt._mg["props"]["tint"] = max_rt.color(255.0, 128.0, 0.0)
    lt._mg["props"]["tex"] = MockObject(max_rt, "VRayBitmap", {"HDRIMapName": "e.hdr"},
                                        superclass=MockTextureMap)
    d = dg.build_digest()
    props = d["lights"][0]["props"]
    assert props["tint"] == [255.0, 128.0, 0.0]
    assert props["pos"] == [1.0, 2.0, 3.0]
    assert props["tex"] == "<map:VRayBitmap>"


def test_camera_basis_healthy_and_stale(max_rt):
    cam = make_cam(max_rt, row3=(0.0, -1.0, 0.0))
    basis = dg.camera_basis(cam)
    assert basis["yaw_deg"] == pytest.approx(0.0)
    assert len(basis["pos"]) == 3 and len(basis["look"]) == 3
    cam.set_stale()
    assert dg.camera_basis(cam) is None


# ===================================================================== apply.py
def test_capture_baselines_empty_and_zero_and_stale(max_rt):
    assert dict(ap.capture_baselines({"groups": {}})) == {}
    zero = make_light(max_rt, "Dimmed", mult=0.0)
    stale = make_light(max_rt, "Gone")
    stale.set_stale()
    good = make_light(max_rt, "Key", mult=30.0)
    rig = {"groups": {"practicals": [zero, stale, good]}}
    fresh = ap.capture_baselines(rig)
    assert dict(fresh) == {"Key": 30.0}          # 0-poison + stale both refused
    assert any("Dimmed" in n and "forget_baseline" in n for n in fresh.notes)


def test_capture_baselines_armed_read_and_duplicate_names(max_rt):
    armed = make_light(max_rt, "Locked", mult=99.0)
    armed.arm_get("multiplier")                  # locked property on read
    a = make_light(max_rt, "Dupe", mult=10.0)
    b = make_light(max_rt, "Dupe", mult=20.0)
    rig = {"groups": {"g": [armed, a, b]}}
    fresh = ap.capture_baselines(rig)
    assert fresh["Locked"] == pytest.approx(1.0)  # unreadable → default, no raise
    assert fresh["Dupe"] == pytest.approx(20.0)   # name-keyed: last wins, no crash


def test_read_state_empty_rig(max_rt):
    st = ap.read_state({"sun": None, "dome": None, "groups": {}}, {})
    assert isinstance(st, LightingState)         # nothing to read, nothing raised


def test_read_state_healthy_rig_full_mapping(max_rt):
    n = basic_rig(max_rt)
    st = ap.read_state(rig_of(n), {"L1_key": 30.0, "L2_fill": 10.0}, n["cam"])
    assert st.get("sun.azimuth_deg") == pytest.approx(0.0)
    assert st.get("sun.altitude_deg") == pytest.approx(0.0)
    assert st.get("sun.enabled") == pytest.approx(1.0)
    assert st.get("sun.intensity") == pytest.approx(1.0)
    assert st.get("dome.enabled") == pytest.approx(1.0)
    assert st.get("dome.rotation_deg") == pytest.approx(25.0)
    assert st.get("dome.intensity") == pytest.approx(5.0)
    assert st.groups["key"] == pytest.approx(1.0)
    assert st.groups["practicals"] == pytest.approx(1.0)
    assert st.get("exposure.ev") == pytest.approx(0.0)       # from the V-Ray EC
    assert st.get("exposure.wb_kelvin") == pytest.approx(6500.0)


def test_read_state_stale_sun_uses_loud_placeholder(max_rt, capsys):
    n = basic_rig(max_rt)
    n["sun"].set_stale()                         # deleted mid-session, rig cached
    st = ap.read_state(rig_of(n), {}, n["cam"])
    assert st.get("sun.azimuth_deg") == pytest.approx(0.0)   # placeholder, not crash
    assert st.get("sun.altitude_deg") == pytest.approx(45.0)
    assert "read_sun_angles" in capsys.readouterr().out


def test_apply_state_healthy_full_write(max_rt):
    n = basic_rig(max_rt)
    max_rt.reset_log()
    st = LightingState()
    st.set("sun.azimuth_deg", 90.0)
    st.set("sun.altitude_deg", 30.0)
    st.set("sun.intensity", 2.0)
    st.set("dome.enabled", 0.0)
    st.set("dome.intensity", 3.0)
    st.set("dome.rotation_deg", 45.0)
    st.groups["key"] = 0.5
    st.set("exposure.ev", 12.0)
    st.set("exposure.wb_kelvin", 4300.0)
    warnings = ap.apply_state(rig_of(n), {"L1_key": 30.0}, st, n["cam"])
    assert warnings == []
    assert max_rt.sets(n["sun"], "intensity_multiplier")[0][4] == pytest.approx(2.0)
    assert max_rt.sets(n["dome"], "enabled")[0][4] is False
    assert max_rt.sets(n["dome"], "multiplier")[0][4] == pytest.approx(3.0)
    assert max_rt.sets(dome_tex(n["dome"]), "horizontalRotation")[0][4] == \
        pytest.approx(45.0)
    assert max_rt.sets(n["l1"], "multiplier")[0][4] == pytest.approx(15.0)  # 30×0.5
    assert max_rt.sets(n["ec"], "ev")[0][4] == pytest.approx(12.0)
    assert max_rt.sets(n["ec"], "temperature")[0][4] == pytest.approx(4300.0)
    labels = [e[1] for e in max_rt.mutation_log if e[0] == "undo_enter"]
    assert "MaxGaffer lighting" in labels
    exits = [e for e in max_rt.mutation_log
             if e[0] == "undo_exit" and e[1] == "MaxGaffer lighting"]
    assert exits and exits[-1][2] == "ok"


def test_apply_state_mid_write_failures_warn_and_isolate(max_rt):
    """One locked property must not stop the other writes — and nothing raises."""
    n = basic_rig(max_rt)
    n["sun"].arm_set("intensity_multiplier")     # locked spinner
    n["l1"].arm_set("multiplier")                # locked light
    max_rt.reset_log()
    st = LightingState()
    st.set("sun.intensity", 2.0)
    st.set("sun.turbidity", 4.0)
    st.groups["key"] = 0.5
    warnings = ap.apply_state(rig_of(n), {"L1_key": 30.0}, st, n["cam"])
    assert any("sun.intensity" in w for w in warnings)
    assert any("L1_key" in w and "no multiplier" in w for w in warnings)
    assert max_rt.sets(n["sun"], "intensity_multiplier") == []      # never landed
    assert max_rt.sets(n["sun"], "turbidity")[0][4] == pytest.approx(4.0)  # others did
    assert max_rt.sets(n["l1"], "multiplier") == []
    exits = [e for e in max_rt.mutation_log
             if e[0] == "undo_exit" and e[1] == "MaxGaffer lighting"]
    assert exits and exits[-1][2] == "ok"        # failures never escape the record


def test_apply_state_zero_baseline_is_never_poisoned(max_rt):
    """A 0.0 baseline in the session (legacy poison) behaves as authored 1.0 in
    BOTH directions — no divide-by-0 ghost factor, no 0-write."""
    n = basic_rig(max_rt)
    st = ap.read_state(rig_of(n), {"L1_key": 0.0}, n["cam"])
    # 30.0/1.0 — the poisoned 0 baseline reads as authored 1.0, never a ÷0 ghost
    assert st.groups["key"] == pytest.approx(30.0)
    max_rt.reset_log()
    st2 = LightingState()
    st2.groups["key"] = 0.5
    warnings = ap.apply_state(rig_of(n), {"L1_key": 0.0}, st2, n["cam"])
    assert warnings == []
    assert max_rt.sets(n["l1"], "multiplier")[0][4] == pytest.approx(0.5)  # 1.0×0.5


def test_apply_state_empty_rig_and_state(max_rt):
    warnings = ap.apply_state({"sun": None, "dome": None, "groups": {}}, {},
                              LightingState())
    assert warnings == []


def test_apply_state_junk_values_that_ARE_isolated(max_rt):
    """The params the audit hardened (sun.intensity, group factors) downgrade to
    warnings even against the real mock — contrast with the xfail bugs below."""
    n = basic_rig(max_rt)
    st = LightingState()
    st.values["sun.intensity"] = "junk"          # raw write past the clamps
    st.groups["key"] = "loud"
    warnings = ap.apply_state(rig_of(n), {"L1_key": 30.0}, st, n["cam"])
    assert any("sun.intensity" in w and "non-numeric" in w for w in warnings)
    assert any("group.key" in w and "non-numeric" in w for w in warnings)


def test_apply_state_junk_dome_rotation_without_texmap_warns(max_rt):
    """The dome rotation NODE fallback is fault-isolated — unlike the texmap path
    (see the xfail below); this locks the documented inconsistency."""
    n = basic_rig(max_rt)
    n["dome"] = make_dome(max_rt, name="BareDome", with_texmap=False)
    rig = rig_of(n)
    st = LightingState()
    st.values["dome.rotation_deg"] = "junk"
    warnings = ap.apply_state(rig, {}, st, n["cam"])
    assert any("dome.rotation_deg" in w for w in warnings)


# ===================================================================== exposure.py
def test_host_detection_no_ec_no_camera(max_rt):
    host = exp.ExposureHost(None)
    assert host.kind == "none"
    assert host.read_ev() is None and host.read_wb_kelvin() is None
    assert host.write_ev(11.0) is False and host.write_wb_kelvin(4300.0) is False
    assert host.describe()["kind"] == "none"


def test_host_detection_vray_ec_wins(max_rt):
    n = basic_rig(max_rt)
    host = exp.ExposureHost(n["cam"])
    assert host.kind == "exposure_control"
    assert host.read_ev() == pytest.approx(0.0)
    assert host.read_wb_kelvin() == pytest.approx(6500.0)
    max_rt.reset_log()
    assert host.write_ev(12.0) is True
    assert max_rt.sets(n["ec"], "ev")[0][4] == pytest.approx(12.0)
    assert host.write_wb_kelvin(4300.0) is True
    assert max_rt.sets(n["ec"], "temperature")[0][4] == pytest.approx(4300.0)


def test_host_detection_native_ec_falls_through_to_physical_cam(max_rt):
    native = MockObject(max_rt, "PhotographicExposureControl", {"ev": 1.0})
    max_rt.SceneExposureControl._mg["props"]["exposureControl"] = native
    cam = make_cam(max_rt)
    host = exp.ExposureHost(cam)
    assert host.kind == "physical_cam"           # native EC is NOT a host
    assert host.read_ev() == pytest.approx(11.0)  # Target-EV direct read
    plain = MockNode(max_rt, "TargetCamera", "Plain")
    assert exp.ExposureHost(plain).kind == "none"


def test_physical_cam_manual_mode_ev_math(max_rt):
    cam = make_cam(max_rt, extra={"exposure_value": None})
    cam._mg["props"].pop("exposure_value")       # legacy/manual camera
    host = exp.ExposureHost(cam)
    assert host.kind == "physical_cam"
    expected = math.log2((8.0 * 8.0) / 0.005)    # iso 100 → EV100
    assert host.read_ev() == pytest.approx(expected)
    max_rt.reset_log()
    assert host.write_ev(expected - 1.0) is True  # one stop brighter: ISO doubles
    assert max_rt.sets(cam, "iso")[0][4] == pytest.approx(200.0)


def test_physical_cam_wb_kelvin_and_color_only_host(max_rt):
    cam = make_cam(max_rt)
    host = exp.ExposureHost(cam)
    assert host.read_wb_kelvin() == pytest.approx(6500.0)
    max_rt.reset_log()
    assert host.write_wb_kelvin(4300.0) is True
    assert max_rt.sets(cam, "white_balance_kelvin")[0][4] == pytest.approx(4300.0)
    assert max_rt.sets(cam, "white_balance_type")[0][4] == 1
    # color-swatch-only host: kelvin spinner absent → illuminant color written
    swatch = make_cam(max_rt, name="Swatch")
    for prop in ("white_balance_kelvin", "exposure_value", "exposure_gain_type"):
        swatch._mg["props"].pop(prop)
    swatch._mg["props"]["white_balance_custom"] = None
    host2 = exp.ExposureHost(swatch)
    max_rt.reset_log()
    assert host2.write_wb_kelvin(4300.0) is True
    written = max_rt.sets(swatch, "white_balance_custom")[0][4]
    assert isinstance(written, MockColor)
    assert max_rt.sets(swatch, "white_balance_type")[0][4] == 2


def test_ensure_exposure_control_creates_on_empty_slot_undo_wrapped(max_rt):
    max_rt.reset_log()
    msg = exp.ensure_exposure_control()
    assert msg is not None and "slot was empty" in msg
    ec = max_rt.SceneExposureControl.get_raw("exposureControl")
    assert ec is not None and max_rt.classOf(ec) == "VRayExposureControl"
    assert max_rt.ec_assignments()                # the assignment was logged
    labels = [e[1] for e in max_rt.mutation_log if e[0] == "undo_enter"]
    assert "MaxGaffer exposure control" in labels
    assert exp.ensure_exposure_control() is None  # already there → no-op


def test_ensure_exposure_control_never_clobbers_native_ec(max_rt):
    native = MockObject(max_rt, "PhotographicExposureControl", {"ev": 1.0})
    max_rt.SceneExposureControl._mg["props"]["exposureControl"] = native
    max_rt.reset_log()
    msg = exp.ensure_exposure_control()
    assert msg is not None and "non-V-Ray exposure control" in msg
    assert "PhotographicExposureControl" in msg
    assert max_rt.created("VRayExposureControl") == []
    assert max_rt.ec_assignments() == []          # untouched
    assert max_rt.SceneExposureControl.get_raw("exposureControl") is native


def test_ensure_exposure_control_maker_missing_and_assignment_locked(max_rt):
    max_rt.remove_maker("vrayCreateVRayExposureControl")   # V-Ray not installed
    assert exp.ensure_exposure_control() is None
    assert max_rt.SceneExposureControl.get_raw("exposureControl") is None
    # maker back, but the scene slot is locked → still no raise, no assignment
    max_rt._register_default_makers()
    max_rt.SceneExposureControl._mg["props"]["exposureControl"] = None
    max_rt.SceneExposureControl.arm_set("exposureControl")
    max_rt.reset_log()
    assert exp.ensure_exposure_control() is None
    assert max_rt.SceneExposureControl.get_raw("exposureControl") is None
    assert max_rt.ec_assignments() == []


def test_exposure_slot_reading_undefined_is_conservative(max_rt):
    """An `undefined` EC slot (real Max spelling of 'empty-ish') must never be
    clobbered and never crash — the bridge refuses to classify it as V-Ray."""
    max_rt.SceneExposureControl._mg["props"]["exposureControl"] = UNDEFINED
    host = exp.ExposureHost(None)
    assert host.kind == "none"                   # not misread as a V-Ray EC
    max_rt.reset_log()
    msg = exp.ensure_exposure_control()
    assert msg is not None                       # conservative: no creation claim
    assert max_rt.created("VRayExposureControl") == []
    assert max_rt.SceneExposureControl.get_raw("exposureControl") is UNDEFINED


# ===================================================================== draft.py
@pytest.fixture()
def draft_env(max_rt, monkeypatch, tmp_path):
    snap = tmp_path / "draft_snapshot.json"
    monkeypatch.setattr(df, "SNAPSHOT_PATH", str(snap))
    renderer = MockObject(max_rt, "VRayRenderer", {
        "options_progressiveNoiseThreshold": 0.01,
        "options_progressiveMaxSubdivs": 24,
        "options_progressiveTimeLimit": 2.0,
        "options_maxSubdivs": 16,
    })
    max_rt.renderers._mg["props"]["current"] = renderer
    max_rt.reset_log()
    return renderer, snap


def test_apply_draft_no_renderer_and_no_known_props(max_rt, monkeypatch, tmp_path):
    monkeypatch.setattr(df, "SNAPSHOT_PATH", str(tmp_path / "snap.json"))
    lines = df.apply_draft()                     # renderers.current is None
    assert any("no current renderer" in ln for ln in lines)
    bare = MockObject(max_rt, "VRayRenderer", {"unrelated": 1})
    max_rt.renderers._mg["props"]["current"] = bare
    lines = df.apply_draft()
    assert any("nothing changed" in ln for ln in lines)
    assert max_rt.sets(bare) == []


def test_apply_draft_snapshot_written_before_any_mutation(max_rt, draft_env,
                                                          monkeypatch):
    renderer, snap = draft_env
    real_set = df.set_prop
    seen = []

    def guarded_set(obj, names, value):
        assert snap.exists(), "set_prop ran before the crash-safe snapshot"
        seen.append(names[0])
        return real_set(obj, names, value)

    monkeypatch.setattr(df, "set_prop", guarded_set)
    lines = df.apply_draft()
    assert len(seen) == 4                        # all four sampler rows applied
    saved = json.loads(snap.read_text(encoding="utf-8"))
    assert saved["options_progressiveNoiseThreshold"] == pytest.approx(0.01)
    assert saved["options_progressiveMaxSubdivs"] == 24
    assert renderer.get_raw("options_progressiveMaxSubdivs") == 12  # int preserved
    assert renderer.get_raw("options_progressiveNoiseThreshold") == pytest.approx(0.05)
    assert any("→" in ln for ln in lines)


def test_apply_draft_snapshot_write_failure_zero_mutations(max_rt, draft_env,
                                                           monkeypatch):
    renderer, _ = draft_env

    def bad_open(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(builtins, "open", bad_open)
    lines = df.apply_draft()
    assert any("ABORTED" in ln for ln in lines)
    assert max_rt.sets(renderer) == []           # nothing touched, honest abort


def test_apply_draft_mid_write_failure_honest_log_snapshot_intact(max_rt, draft_env):
    renderer, snap = draft_env
    renderer.arm_set("options_progressiveMaxSubdivs")   # locked mid-write
    lines = df.apply_draft()
    assert sum("would not take the draft value" in ln for ln in lines) == 1
    assert renderer.get_raw("options_progressiveMaxSubdivs") == 24   # untouched
    assert renderer.get_raw("options_progressiveNoiseThreshold") == pytest.approx(0.05)
    saved = json.loads(snap.read_text(encoding="utf-8"))
    assert saved["options_progressiveMaxSubdivs"] == 24              # true original


def test_apply_draft_pending_snapshot_restores_before_reapplying(max_rt, draft_env):
    renderer, snap = draft_env
    snap.write_text(json.dumps({"options_progressiveNoiseThreshold": 0.5}),
                    encoding="utf-8")
    lines = df.apply_draft()
    assert any("restored options_progressiveNoiseThreshold" in ln for ln in lines)
    saved = json.loads(snap.read_text(encoding="utf-8"))
    assert saved["options_progressiveNoiseThreshold"] == pytest.approx(0.5)


def test_restore_draft_round_trip_and_file_cleared(max_rt, draft_env):
    renderer, snap = draft_env
    df.apply_draft()
    max_rt.reset_log()
    lines = df.restore_draft()
    assert renderer.get_raw("options_progressiveNoiseThreshold") == pytest.approx(0.01)
    assert renderer.get_raw("options_progressiveMaxSubdivs") == 24  # int re-coerced
    assert any("restored options_progressiveMaxSubdivs → 24" in ln for ln in lines)
    assert not snap.exists()


def test_restore_draft_mid_restore_failures_isolated_and_file_cleared(max_rt,
                                                                      draft_env):
    renderer, snap = draft_env
    df.apply_draft()
    renderer.arm_set("options_progressiveMaxSubdivs")   # locked at restore time
    renderer._mg["props"].pop("options_maxSubdivs")     # prop gone after renderer swap
    max_rt.reset_log()
    lines = df.restore_draft()
    assert any("could not restore options_progressiveMaxSubdivs" in ln
               for ln in lines)
    assert any("could not restore options_maxSubdivs" in ln for ln in lines)
    assert renderer.get_raw("options_progressiveNoiseThreshold") == pytest.approx(0.01)
    assert not snap.exists()                     # always cleared, never stranded


def test_restore_draft_renderer_gone_and_garbage_file(max_rt, draft_env):
    renderer, snap = draft_env
    df.apply_draft()
    max_rt.renderers._mg["props"]["current"] = None     # renderer vanished
    lines = df.restore_draft()
    assert lines == [] and not snap.exists()            # silent, cleared
    snap.write_text("{ not json", encoding="utf-8")
    lines = df.restore_draft()
    assert any("unreadable" in ln for ln in lines) and not snap.exists()
    assert df.restore_draft() == []                     # no snapshot → no-op


# ===================================================================== execute.py
def _plan_ops():
    return [
        {"op": "set", "target": "renderer",
         "prop": "options_progressiveNoiseThreshold", "value": 0.5, "why": "test"},
        {"op": "set", "target": "node:L1_key", "prop": "multiplier", "value": 12.0},
        {"op": "set", "target": "node:Ghost", "prop": "multiplier", "value": 1.0},
        {"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_Fill",
         "placement": {"bearing_deg": 45.0, "distance": 300.0, "height": 100.0},
         "props": {"multiplier": 7.0}, "why": "fill"},
    ]


def test_execute_plan_fault_isolated_set_and_create(max_rt):
    renderer, _ = None, None
    renderer = MockObject(max_rt, "VRayRenderer",
                          {"options_progressiveNoiseThreshold": 0.01})
    max_rt.renderers._mg["props"]["current"] = renderer
    make_light(max_rt, "L1_key")
    max_rt.reset_log()
    report = ex.execute_plan(_plan_ops(), camera=None)
    assert len(report["changes"]) == 2           # renderer + node:L1_key
    assert any("node:Ghost" in w and "vanished" in w for w in report["warnings"])
    assert report["created"][0]["name"] == "MG_Fill"
    assert renderer.get_raw("options_progressiveNoiseThreshold") == pytest.approx(0.5)
    created_node = max_rt.created("VRayLight")[0][2]
    assert created_node.get_raw("name") == "MG_Fill"
    assert created_node.get_raw("multiplier") == pytest.approx(7.0)
    assert created_node.get_raw("type") == 0     # plane preset
    layer = max_rt.LayerManager._layers.get("MG_lights")
    assert layer is not None and created_node in layer.nodes
    exits = [e for e in max_rt.mutation_log if e[0] == "undo_exit"]
    assert exits and exits[-1][2] == "ok"


def test_execute_plan_missing_vray_classes_and_unknown_types(max_rt):
    max_rt.remove_maker("VRayLight")
    max_rt.remove_maker("VRaySun")
    ops = [{"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_A",
            "placement": {}},
           {"op": "create_light", "light_type": "NoSuchLight", "name": "MG_B",
            "placement": {}},
           {"op": "definitely_not_an_op"}]
    report = ex.execute_plan(ops)
    assert any("class unavailable" in w for w in report["warnings"])
    assert any("op failed" in w for w in report["warnings"])   # both bad ops isolated
    assert report["created"] == []
    assert max_rt.deletes() == []                # nothing created → nothing leaked


def test_execute_plan_mid_create_failure_rolls_back_the_orphan(max_rt):
    """A ctor that succeeds then a mid-setup failure must DELETE the orphan node —
    the mutation log shows create→delete, and the node is a stale handle after."""
    def hostile_maker():
        node = MockNode(max_rt, "VRayLight", "VRayLight_auto",
                        props={"type": 0, "multiplier": 1.0})
        node.arm_set("name")                     # the very first setup write fails
        max_rt.mutation_log.append(("create", "VRayLight", node))
        return node
    max_rt.add_maker("VRayLight", hostile_maker)
    ops = [{"op": "create_light", "light_type": "VRayLight_plane", "name": "MG_X",
            "placement": {}}]
    max_rt.reset_log()
    report = ex.execute_plan(ops)
    assert any("node removed" in w for w in report["warnings"])
    assert report["created"] == []
    creates = max_rt.created("VRayLight")
    assert len(creates) == 1 and max_rt.deletes(creates[0][2])  # rolled back
    assert creates[0][2].stale                                  # deleted handle


# ===================================================================== config.py + controller.py
def _write_config(tmp_path, monkeypatch, raw):
    p = tmp_path / "config.json"
    p.write_bytes(raw)
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(p))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))   # keep the MaxDirector key borrow off the real box
    return str(p)


@pytest.mark.parametrize("raw", [
    b"null", b"[]", b'"str"', b"42",            # valid JSON, not an object
    b"{ definitely not json",                   # corrupt bytes
])
def test_config_load_hostile_payloads_never_raise(tmp_path, monkeypatch, raw):
    _write_config(tmp_path, monkeypatch, raw)
    cfg = cfgmod.load()                          # must NOT raise
    assert cfg.max_iterations == 5               # defaults survive the garbage
    assert cfg.loop_width == 480 and type(cfg.loop_width) is int
    assert cfg.api_key == ""
    assert cfg.critic_weights == {}
    assert cfg.auto_exposure_control is True
    assert cfg.target_score == 82.0              # the untouched default


def test_config_load_wrong_typed_fields_fall_back_per_field(tmp_path, monkeypatch):
    _write_config(tmp_path, monkeypatch,
                  b'{"max_iterations": "five", "loop_width": 480.0, "api_key": 42,'
                  b' "critic_weights": "x", "auto_exposure_control": 1,'
                  b' "target_score": 90}')
    cfg = cfgmod.load()
    assert cfg.max_iterations == 5               # str for int → default
    assert cfg.loop_width == 480                 # float for int → default
    assert cfg.api_key == ""                     # non-str → default
    assert cfg.critic_weights == {}              # str for dict → default
    assert cfg.auto_exposure_control is True     # int is NOT a bool → default
    assert cfg.target_score == 90.0              # int for float → widened, accepted


def test_config_load_unreadable_path_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(cfgmod, "CONFIG_PATH", str(tmp_path))   # a DIRECTORY
    cfg = cfgmod.load()
    assert cfg.model == "claude-opus-4-8"


def _controller(max_rt, monkeypatch, tmp_path, cfg=None):
    from maxgaffer.maxbridge import controller as cm
    monkeypatch.setattr(cm.cfgmod, "sessions_dir", lambda: str(tmp_path))
    return cm.Controller(cfg or cfgmod.Config()), cm


def test_controller_construction_and_scene_ops_against_hostile_mock(max_rt,
                                                                    monkeypatch,
                                                                    tmp_path):
    _write_config(tmp_path, monkeypatch, b'{"max_iterations": "five"}')
    cfg = cfgmod.load()                          # hostile file, safe object
    c, cm = _controller(max_rt, monkeypatch, tmp_path, cfg)
    n = basic_rig(max_rt)
    rig = c.rig()
    assert rig["sun"] is n["sun"] and rig["dome"] is n["dome"]
    assert c._baselines == {"L1_key": 30.0, "L2_fill": 10.0}
    cams = c.cameras()
    assert [e["name"] for e in cams] == ["Cam01"]
    st = c.read_state("Cam01")
    assert st.get("dome.rotation_deg") == pytest.approx(25.0)
    st.set("dome.rotation_deg", 60.0)
    assert c.apply_state(st, "Cam01") == []
    assert dome_tex(n["dome"]).get_raw("horizontalRotation") == pytest.approx(60.0)
    assert c.restore_pre_match("Cam01") is False  # nothing to restore, no raise


def test_controller_rig_rescan_after_dim_never_poisons_baselines(max_rt,
                                                                 monkeypatch,
                                                                 tmp_path):
    """The end-to-end 0-poison scenario: MaxGaffer dims a group to 0, the rig is
    re-scanned, and the authored baseline must SURVIVE (capture refuses the 0)."""
    c, cm = _controller(max_rt, monkeypatch, tmp_path)
    n = basic_rig(max_rt)
    c.rig()
    assert c._baselines["L1_key"] == pytest.approx(30.0)
    n["l1"]._mg["props"]["multiplier"] = 0.0     # the match dimmed the group
    n["l1"]._mg["props"]["name"] = "Key_renamed"  # …and the artist renamed it
    c.rig(refresh=True)
    assert c._baselines["L1_key"] == pytest.approx(30.0)   # never overwritten
    assert "Key_renamed" not in c._baselines               # 0 capture refused


def test_controller_set_dome_hdri_failure_returns_failed_not_raise(max_rt,
                                                                   monkeypatch,
                                                                   tmp_path):
    c, cm = _controller(max_rt, monkeypatch, tmp_path)
    make_dome(max_rt, with_texmap=False)
    barren = lambda: MockObject(max_rt, "VRayBitmap", {}, superclass=MockTextureMap)
    max_rt.add_maker("VRayBitmap", barren)
    max_rt.add_maker("VRayHDRI", barren)
    c.rig(refresh=True)
    assert c.set_dome_hdri(r"X:\env.hdr") == "failed"   # controller stays honest
    dome = c.rig()["dome"]
    assert dome.get_raw("texmap") is None               # …and nothing was bound


# ===================================================================== chaos mode
def test_chaos_write_paths_no_escape_and_mutation_invariants(max_rt, tmp_path,
                                                             monkeypatch):
    """Seeded random failures on rt services + property WRITES (locked props,
    mid-write crashes). Public entry points must not raise, and the mutation
    log must keep its invariants: failed dome-texture ⇒ nothing bound; any
    draft mutation ⇒ a snapshot file exists; execute deletes only roll back.

    The scene carries NO exposure control: an EC object whose classOf read
    fails escapes the host chain — that is KNOWN BUG #5 (exposure.py:71),
    reproduced deterministically in test_known_bug_exposure_host_unreadable_ec.
    """
    rt = max_rt
    n = basic_rig(rt)                            # build the scene BEFORE chaos
    rt.SceneExposureControl._mg["props"]["exposureControl"] = None   # see above
    rt.chaos_rt = 0.20
    rt.chaos_set = 0.20
    rig = rig_of(n)
    baselines = {"L1_key": 30.0, "L2_fill": 10.0}
    monkeypatch.setattr(df, "SNAPSHOT_PATH", str(tmp_path / "snap.json"))
    renderer = MockObject(rt, "VRayRenderer",
                          {"options_progressiveNoiseThreshold": 0.01,
                           "options_progressiveMaxSubdivs": 24})
    rt.renderers._mg["props"]["current"] = renderer
    for i in range(30):
        rt.reset_log()
        st = LightingState()
        st.set("sun.azimuth_deg", float(i % 360))
        st.set("sun.altitude_deg", 30.0)
        st.set("sun.intensity", 1.5)
        st.set("dome.rotation_deg", float(i * 7 % 360))
        st.set("dome.intensity", 2.0)
        st.groups["key"] = 0.5
        st.set("exposure.ev", 11.0)
        st.set("exposure.wb_kelvin", 4300.0)
        warnings = ap.apply_state(rig, baselines, st, n["cam"])
        assert isinstance(warnings, list)
        exits = [e for e in rt.mutation_log if e[0] == "undo_exit"]
        assert exits and exits[-1][2] == "ok"    # nothing escaped the record
        # dome texture: fresh dome each round; failure ⇒ zero binding
        dome = make_dome(rt, name="ChaosDome{0}".format(i), with_texmap=False)
        how = sc.set_dome_texture(dome, r"X:\env\c{0}.hdr".format(i))
        bound = dome.get_raw("texmap")
        if how == "failed":
            assert bound is None or bound is UNDEFINED
        else:
            assert bound is not None
            sets = rt.sets()
            file_idx = next(j for j, e in enumerate(sets)
                            if e[1] is bound and e[2] == "HDRIMapName")
            bind_idx = next(j for j, e in enumerate(sets)
                            if e[1] is dome and e[2] == "texmap")
            assert file_idx < bind_idx           # file first, bind second
        # draft: any renderer mutation implies the snapshot file exists FIRST
        df.apply_draft()
        if rt.sets(renderer):
            assert (tmp_path / "snap.json").exists()
        df.restore_draft()
        assert not (tmp_path / "snap.json").exists()
        # execute: every delete must roll back a create in the same plan
        report = ex.execute_plan(
            [{"op": "create_light", "light_type": "VRayLight_plane",
              "name": "MG_Chaos{0}".format(i), "placement": {}}])
        created_ids = {id(e[2]) for e in rt.created("VRayLight")}
        for d in rt.deletes():
            assert id(d[1]) in created_ids       # deletes are rollbacks only
        assert isinstance(report["warnings"], list)


def test_chaos_exposure_control_creation_no_escape(max_rt):
    """ensure_exposure_control under chaos on an EMPTY slot: creation, locked
    assignment, maker failure — every round degrades to a log line or None."""
    rt = max_rt
    rt.chaos_rt = 0.30
    rt.chaos_set = 0.30
    for _ in range(30):
        # force an empty slot (raw write, bypasses hostility) each round —
        # an EC already IN the slot hits KNOWN BUG #5 (exposure.py:71) instead
        rt.SceneExposureControl._mg["props"]["exposureControl"] = None
        msg = exp.ensure_exposure_control()
        assert msg is None or isinstance(msg, str)


def test_chaos_read_paths_no_escape(max_rt):
    """Seeded random failures on rt services + property READS against a
    read-only scene. Enumeration and state reads must degrade, never raise.
    No EC in the slot — classOf on a hostile EC object is KNOWN BUG #5
    (exposure.py:71), covered by its own deterministic xfail repro."""
    rt = max_rt
    n = basic_rig(rt)                            # build the scene BEFORE chaos
    rt.SceneExposureControl._mg["props"]["exposureControl"] = None   # see above
    rt.chaos_rt = 0.25
    rt.chaos_get = 0.25
    rig = rig_of(n)
    baselines = {"L1_key": 30.0, "L2_fill": 10.0}
    for _ in range(40):
        cams = sc.list_cameras()
        assert isinstance(cams, list)
        cam_node = sc.get_camera("Cam01")        # ONE call per round (chaos varies)
        assert cam_node is None or cam_node.get_raw("name") == "Cam01"
        out = sc.classify_rig()
        assert set(out) == {"sun", "dome", "sky_env", "groups", "notes"}
        d = dg.build_digest()
        assert set(d) == {"renderer", "environment", "exposure",
                          "lights", "cameras", "stats"}
        st = ap.read_state(rig, baselines, n["cam"])
        assert isinstance(st, LightingState)
        fresh = ap.capture_baselines(rig)
        assert isinstance(fresh, dict)
        desc = exp.ExposureHost(n["cam"]).describe()
        assert desc["kind"] in ("exposure_control", "physical_cam", "none")
        rot = sc.read_dome_rotation(n["dome"])
        assert isinstance(rot, float)
        yaw = sc.camera_yaw_deg(n["cam"])
        assert 0.0 <= yaw < 360.0
        basis = dg.camera_basis(n["cam"])
        assert basis is None or set(basis) == {"pos", "yaw_deg", "look"}


def test_chaos_vantage_and_render_entry_points(max_rt):
    from maxgaffer.maxbridge import render as rd
    from maxgaffer.maxbridge import vantage as vt
    rt = max_rt
    rt.chaos_rt = 0.30
    cam = make_cam(rt)
    for _ in range(20):
        ok, how = vt.start_live_link()
        assert isinstance(ok, bool) and isinstance(how, str)
        out = rd.render_frame(cam, str(max_rt and "loop.png"), 64, 64)
        assert out is None or isinstance(out, str)
        assert rd.transcode_to_png("ref.exr", "ref.png") is None


# ===================================================================== KNOWN REAL BUGS
# Reproduced here as xfail(strict=False): the suite stays green, each failure
# names the exact source line, and a future fix turns these into XPASS.
# NOTHING in maxbridge was modified by this suite.

_BUG1 = ("REAL BUG maxbridge/exposure.py:151 — read_ev(): int(gain_type) is the "
         "only unguarded conversion in the file; a hostile exposure_gain_type "
         "(junk string / undefined) raises ValueError/TypeError out of an "
         "Optional[float] API, escaping apply.read_state (apply.py:135) and "
         "write_ev's legacy fallback (exposure.py:184) on Max's main thread")

_BUG2A = ("REAL BUG maxbridge/apply.py:171-173 + scene.py:308 — sun.azimuth_deg/"
          "sun.altitude_deg bypass _state_float: raw junk in the public "
          "state.values dict reaches math.radians() outside any guard and "
          "raises TypeError inside the undo record")

_BUG2B = ("REAL BUG maxbridge/apply.py:198 + scene.py:347 — dome.rotation_deg "
          "bypasses _state_float: write_dome_rotation computes "
          "float(degrees_ % 360.0) OUTSIDE any try on the texmap path "
          "(the node fallback IS guarded — inconsistent isolation)")

_BUG2C = ("REAL BUG maxbridge/apply.py:217-221 + exposure.py:177/183/216 — "
          "exposure.ev / exposure.wb_kelvin bypass _state_float: float() of a "
          "raw junk state value raises ValueError inside write_ev/"
          "write_wb_kelvin, escaping apply_state's undo record")

_BUG3 = ("REAL BUG maxbridge/apply.py:213 — the 'has no multiplier' warning "
         "does getattr(lt, 'name', '?') on the light whose set_prop just "
         "failed; on a stale (deleted) node .name raises RuntimeError, which "
         "getattr's default does NOT catch — the fault-isolation path itself "
         "crashes apply_state on exactly the deleted-node scenario it guards")

_BUG5 = ("REAL BUG maxbridge/exposure.py:71 — _find_exposure_control calls "
         "str(rt.classOf(ec)) UNGUARDED on whatever the scene slot holds; a "
         "stale/corrupt exposure-control object (deleted EC, hostile plugin) "
         "raises RuntimeError out of ExposureHost.__init__, escaping "
         "apply.read_state (apply.py:134), apply_state (apply.py:216) and "
         "ensure_exposure_control (exposure.py:84) on Max's main thread. "
         "Found by the seeded chaos sweep (chaos_rt on classOf)")


@pytest.mark.xfail(reason=_BUG1, strict=False)
def test_known_bug_read_ev_hostile_gain_type_never_raises(max_rt):
    junk_cam = make_cam(max_rt, "JunkGain",
                        extra={"exposure_gain_type": "target EV (junk)"})
    ev = exp.ExposureHost(junk_cam).read_ev()    # contract: Optional[float]
    assert ev is None or isinstance(ev, float)
    undef_cam = make_cam(max_rt, "UndefGain")
    undef_cam.set_undefined("exposure_gain_type")
    ev2 = exp.ExposureHost(undef_cam).read_ev()
    assert ev2 is None or isinstance(ev2, float)


@pytest.mark.xfail(reason=_BUG2A, strict=False)
def test_known_bug_apply_junk_sun_angles_warns_not_raises(max_rt):
    n = basic_rig(max_rt)
    st = LightingState()
    st.values["sun.azimuth_deg"] = "junk"        # raw write into the public dict
    warnings = ap.apply_state(rig_of(n), {}, st, n["cam"])
    assert any("sun.azimuth_deg" in w for w in warnings)
    st2 = LightingState()
    st2.values["sun.altitude_deg"] = ["not-a-number"]
    warnings2 = ap.apply_state(rig_of(n), {}, st2, n["cam"])
    assert any("sun.altitude_deg" in w for w in warnings2)


@pytest.mark.xfail(reason=_BUG2B, strict=False)
def test_known_bug_apply_junk_dome_rotation_texmap_path_warns_not_raises(max_rt):
    n = basic_rig(max_rt)                        # dome HAS a texmap → unguarded path
    st = LightingState()
    st.values["dome.rotation_deg"] = "junk"
    warnings = ap.apply_state(rig_of(n), {}, st, n["cam"])
    assert any("dome.rotation_deg" in w for w in warnings)


@pytest.mark.xfail(reason=_BUG2C, strict=False)
def test_known_bug_apply_junk_exposure_values_warn_not_raise(max_rt):
    n = basic_rig(max_rt)                        # V-Ray EC host is writable
    st = LightingState()
    st.values["exposure.ev"] = "junk"
    warnings = ap.apply_state(rig_of(n), {}, st, n["cam"])
    assert any("exposure.ev" in w for w in warnings)
    st2 = LightingState()
    st2.values["exposure.wb_kelvin"] = "junk"
    warnings2 = ap.apply_state(rig_of(n), {}, st2, n["cam"])
    assert any("exposure.wb_kelvin" in w for w in warnings2)


@pytest.mark.xfail(reason=_BUG3, strict=False)
def test_known_bug_apply_stale_group_light_warns_not_raises(max_rt):
    n = basic_rig(max_rt)
    n["l1"].set_stale()                          # deleted mid-session, rig cached
    st = LightingState()
    st.groups["key"] = 0.5
    warnings = ap.apply_state(rig_of(n), {"L1_key": 30.0}, st, n["cam"])
    assert any("key" in w for w in warnings)     # a warning, never a crash


def test_classify_rig_duplicate_sun_hostile_name(max_rt):
    """FIXED (was _BUG4): duplicate-sun/dome notes now read names via _node_name."""
    make_sun(max_rt, "Sun_A")
    evil = make_sun(max_rt, "Sun_B")
    evil.arm_get("name")                         # class readable, name is not
    rig = sc.classify_rig()                      # must note-and-continue
    assert rig["sun"] is not None


@pytest.mark.xfail(reason=_BUG5, strict=False)
def test_known_bug_exposure_host_unreadable_ec_class(max_rt):
    """A stale (deleted) exposure-control object still referenced by the scene
    slot: host detection must treat it as 'no usable host', never raise."""
    ec = MockObject(max_rt, "VRayExposureControl", {"ev": 11.0})
    max_rt.SceneExposureControl._mg["props"]["exposureControl"] = ec
    ec.set_stale()                               # deleted EC, slot still points at it
    host = exp.ExposureHost(None)                # contract: kind 'none', no raise
    assert host.kind == "none"
    st = ap.read_state({"sun": None, "dome": None, "groups": {}}, {})
    assert isinstance(st, LightingState)
    msg = exp.ensure_exposure_control()          # must not raise either
    assert msg is None or isinstance(msg, str)
