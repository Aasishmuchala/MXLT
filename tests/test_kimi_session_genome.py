"""Cluster-A audit regressions — corrupt/non-finite sidecar hardening for the genome
and the session store.

Pins the data-loss guards: a corrupt sidecar is quarantined (renamed to a timestamped
.corrupt backup) and auto-save is blocked until an explicit save, NaN/Infinity and
non-numeric values are coerced or dropped per-key instead of crashing the load, group
diffs compare against the neutral multiplier, and 0.0 baselines are declined at
adoption. Pure python — runs off-Max.
"""

import json
import logging

import pytest

from maxgaffer.core.genome import LightingState, clamp
from maxgaffer.core.session import FORMAT_VERSION, Session, preset_dumps, preset_loads


# --------------------------------------------------------------------------- clamp
def test_clamp_coerces_non_finite_and_non_numeric():
    # wrap params fall back to 0° (math.fmod(inf) used to raise ValueError here)
    assert clamp("sun.azimuth_deg", 1e999) == 0.0          # json parses 1e999 to inf
    assert clamp("sun.azimuth_deg", float("nan")) == 0.0
    assert clamp("dome.rotation_deg", float("-inf")) == 0.0
    # bounded params fall back to spec.lo — wrong but bounded, never a crash
    assert clamp("sun.intensity", 1e999) == 0.05
    assert clamp("sun.intensity", "warm") == 0.05
    assert clamp("sun.intensity", None) == 0.05
    assert clamp("exposure.ev", float("nan")) == -4.0


def test_clamp_normal_behavior_unchanged():
    assert clamp("sun.altitude_deg", 500) == 88.0
    assert clamp("sun.altitude_deg", -90) == -4.0
    assert clamp("sun.azimuth_deg", 370) == 10.0
    assert clamp("sun.azimuth_deg", -10) == 350.0
    with pytest.raises(KeyError):
        clamp("nonsense.param", 1)


# ----------------------------------------------------------------------- from_dict
def test_from_dict_drops_bad_values_keeps_good(caplog):
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.genome"):
        st = LightingState.from_dict({
            "values": {"sun.azimuth_deg": 1e999, "sun.intensity": "bright",
                       "sun.turbidity": None, "sun.altitude_deg": 30.0},
            "groups": {"spots": float("nan"), "practicals": 0.5},
        })
    # bad keys dropped per-key, good keys survive — one corrupt value can't kill the rest
    assert st.values == {"sun.altitude_deg": 30.0}
    assert st.groups == {"practicals": 0.5}
    assert "sun.azimuth_deg" in caplog.text and "spots" in caplog.text


def test_loaded_nan_does_not_roundtrip(tmp_path):
    # Python's json accepts NaN literals — a poisoned sidecar must come back clean
    p = tmp_path / "scene.maxgaffer.json"
    p.write_text('{"cameras": {"cam": {"state": {"values": {"sun.azimuth_deg": NaN,'
                 '"sun.altitude_deg": 12.0}}}}}')
    s = Session.load(str(p))
    st = s.entry("cam").state
    assert st is not None
    assert "sun.azimuth_deg" not in st.values           # dropped, not stored as nan
    assert st.get("sun.altitude_deg") == 12.0
    assert s.save()                                     # not a corrupt load → saves fine
    assert "NaN" not in p.read_text()                   # and the round-trip stays clean


def test_preset_loads_survives_corrupt_values():
    d = json.loads(preset_dumps(LightingState(), "x"))
    d["state"]["values"]["sun.azimuth_deg"] = 1e999     # Infinity on the wire
    d["state"]["values"]["sun.altitude_deg"] = 25.0
    st = preset_loads(json.dumps(d))
    assert st is not None
    assert "sun.azimuth_deg" not in st.values
    assert st.get("sun.altitude_deg") == 25.0


# ----------------------------------------------------------------------------- diff
def test_diff_missing_group_uses_neutral_multiplier():
    a, b = LightingState(), LightingState()
    b.groups["NewLayer"] = 1.0
    assert a.diff(b) == {}          # a layer appearing at its neutral value is NO change
    b.groups["NewLayer"] = 0.5
    assert a.diff(b) == {"group.NewLayer": (1.0, 0.5)}   # real change vs the 1.0 sentinel
    # fixed params keep the 0.0 sentinel and both-sided keys are untouched
    a.set("sun.altitude_deg", 30.0)
    b.set("sun.altitude_deg", 50.0)
    assert a.diff(b)["sun.altitude_deg"] == (30.0, 50.0)


# --------------------------------------------------------------- corrupt quarantine
def test_corrupt_sidecar_quarantined_and_autosave_blocked(tmp_path, caplog):
    p = tmp_path / "scene.maxgaffer.json"
    p.write_text("{not json")
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.session"):
        s = Session.load(str(p), now_fn=lambda: "2026-07-16T12:00:00")
    assert s.cameras == {}                              # old contract: empty session
    assert s.settings["apply_on_select"] is True
    assert not p.exists()                               # moved aside, not deleted
    backups = list(tmp_path.glob("*.corrupt"))
    assert len(backups) == 1
    assert backups[0].read_text() == "{not json"        # recoverable bytes preserved
    assert "20260716T120000" in backups[0].name         # timestamped
    assert "MaxGaffer" in caplog.text                   # loud

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.session"):
        assert s.save() is False                        # auto-save may NOT overwrite
    assert not p.exists()                               # nothing written back
    assert "BLOCKED" in caplog.text

    assert s.save(force=True) is True                   # explicit save breaks the seal
    assert s.save() is True                             # and re-arms normal saving
    assert backups[0].exists()                          # the backup survives all of it


@pytest.mark.parametrize("text", ["[]", "null", "42", '"str"'])
def test_non_dict_top_level_quarantined(tmp_path, text):
    p = tmp_path / "scene.maxgaffer.json"
    p.write_text(text)
    s = Session.load(str(p))                            # used to raise AttributeError
    assert s.cameras == {}
    assert len(list(tmp_path.glob("*.corrupt"))) == 1
    assert s.save() is False


def test_one_bad_camera_does_not_kill_the_load(tmp_path, caplog):
    p = tmp_path / "scene.maxgaffer.json"
    p.write_text(json.dumps({"cameras": {
        "good": {"reference": "r.jpg",
                 "state": {"values": {"sun.intensity": "bright", "sun.altitude_deg": 10.0}}},
        "broken": {"locks": 5},                          # int locks: junk field
    }}))
    s = Session.load(str(p))
    # the union hardening ABSORBS junk fields instead of discarding the camera: both
    # cameras survive the load, and the int locks neutralize to an empty set (less
    # data loss than the old skip-the-whole-entry behavior; the skip path remains as
    # a belt for from_dict raises)
    assert set(s.cameras) == {"good", "broken"}
    assert s.entry("broken").locks == set()
    st = s.entry("good").state                          # and inside an entry, only the
    assert "sun.intensity" not in st.values             # bad KEY is dropped — the camera
    assert st.get("sun.altitude_deg") == 10.0           # and its good values survive


# ------------------------------------------------------------------ format version
def test_newer_format_version_warns_and_blocks_autosave(tmp_path, caplog):
    p = tmp_path / "scene.maxgaffer.json"
    p.write_text(json.dumps({"version": FORMAT_VERSION + 1,
                             "cameras": {"cam": {"reference": "r.jpg"}}}))
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.session"):
        s = Session.load(str(p))
    assert list(s.cameras) == ["cam"]                   # best-effort load still happens
    assert "NEWER" in caplog.text
    assert p.exists()                                   # newer file is left in place
    assert s.save() is False                            # ... never silently downgraded
    assert s.save(force=True) is True                   # explicit save is the escape hatch
    assert json.loads(p.read_text())["version"] == FORMAT_VERSION


def test_current_format_loads_and_saves_normally(tmp_path):
    p = str(tmp_path / "scene.maxgaffer.json")
    s = Session(p, now_fn=lambda: "t")
    s.set_reference("cam", "r.jpg")
    assert s.save()
    s2 = Session.load(p)
    assert s2.save()                                    # no protection on a clean load


# ------------------------------------------------------------------------- baselines
def test_adopt_baselines_declines_zero_and_non_finite(caplog):
    s = Session()
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.session"):
        added = s.adopt_baselines({"Dimmed": 0.0, "Weird": float("nan"),
                                   "AlsoWeird": 1e999, "Authored": 30.0})
    assert added == ["Authored"]
    assert s.baselines == {"Authored": 30.0}            # 0-poisoning structurally refused
    assert "Dimmed" in caplog.text and "forget_baseline" in caplog.text
    # a declined 0.0 does NOT mark the light as seen — a later real value still adopts
    assert s.adopt_baselines({"Dimmed": 12.0}) == ["Dimmed"]
    # known baselines are still never overwritten (SPEC adopt-once law)
    assert s.adopt_baselines({"Authored": 0.0}) == []
    assert s.baselines["Authored"] == 30.0


# ------------------------------------------------------------------------------ save
def test_save_is_atomic_and_leaves_no_tmp(tmp_path):
    p = tmp_path / "scene.maxgaffer.json"
    s = Session(str(p), now_fn=lambda: "t")
    s.set_reference("cam", "r.jpg")
    assert s.save()
    assert not list(tmp_path.glob("*.tmp"))             # tmp always replaced away
    assert json.loads(p.read_text())["cameras"]["cam"]["reference"] == "r.jpg"


def test_save_failure_logs_and_returns_false(tmp_path, caplog):
    p = str(tmp_path / "missing_dir" / "scene.maxgaffer.json")
    s = Session(p)
    s.set_reference("cam", "r.jpg")
    with caplog.at_level(logging.WARNING, logger="maxgaffer.core.session"):
        assert s.save() is False                        # OSError surfaced, not swallowed
    assert "save failed" in caplog.text
