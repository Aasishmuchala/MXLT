"""Off-box coverage for ``maxbridge.scene.report_aliases`` — the alias self-report.

``report_aliases`` walks every logical rig parameter through ``matched_prop`` (the read-only
twin of ``set_prop``) and surfaces the FIRST candidate property that actually exists on the
target object, firing no writes. The on-box checklist reads this to confirm which drifting
V-Ray build alias won. These tests run the REAL scene code against the hostile fake runtime
in ``tests/mock_pymxs.py`` — the pymxs boundary is the only thing stubbed.

They pin the contract that matters:
  * the reported name is the first EXISTING candidate (first-candidate-wins), and later
    candidates — including the report-only ``*_REPORT`` tails (``power`` / ``visibility_range``
    / ``atmo_height``) — are reached when the earlier ones are absent;
  * an unmatched logical prop, an absent target object, and a texmap-less dome all report
    ``None`` rather than raising, and all 12 keys are ALWAYS present;
  * the report is read-only (no mutation_log entry, ``matched_prop`` never writes) and matches
    exactly what ``set_prop`` would target;
  * the shared write tuples (LIGHT_MULT / FOG_DISTANCE / FOG_HEIGHT) stay byte-identical — the
    ``*_REPORT`` analogs live apart, so apply.py's capture/set path is untouched (G3);
  * off-Max → ``{}`` and ``record`` leaves ``LAST_ALIASES`` exactly as intended.

Stdlib + FakeMaxRuntime only; deterministic. Python 3.11 (Max 2026) compatible.
"""

import sys

import pytest

from tests.mock_pymxs import (CHAOS_SEED, FakeMaxRuntime, MockNode, MockObject,
                              MockTextureMap, install)

from maxgaffer.maxbridge import scene as sc


#: Every key report_aliases must ALWAYS return, whatever the rig config.
EXPECTED_KEYS = {
    "sun.intensity", "sun.size", "sun.turbidity", "sun.enabled",
    "dome.enabled", "dome.intensity", "dome.tex_on", "dome.tex_rotation", "dome.tex_file",
    "atmosphere.enabled", "atmosphere.distance", "atmosphere.height",
}


# --------------------------------------------------------------------- fixtures
@pytest.fixture()
def rt(monkeypatch):
    """A seeded fake runtime installed as ``pymxs``; the process stays clean."""
    r = FakeMaxRuntime(seed=CHAOS_SEED)
    monkeypatch.setitem(sys.modules, "pymxs", install(r))
    return r


@pytest.fixture(autouse=True)
def _preserve_last_aliases():
    """report_aliases(record=True) reassigns a module global — restore it around each test
    so nothing leaks into sibling tests in a full run."""
    saved = sc.LAST_ALIASES
    try:
        yield
    finally:
        sc.LAST_ALIASES = saved


# --------------------------------------------------------------------- builders
def make_sun(rt, **props):
    """A VRaySun-role node carrying exactly the given candidate properties."""
    return MockNode(rt, "VRaySun", "Sun01", props=dict(props))


def make_tex(rt, **props):
    """A dome HDRI texmap (VRayBitmap/VRayHDRI stand-in) with the given file/rotation aliases."""
    return MockObject(rt, "VRayBitmap", dict(props), superclass=MockTextureMap)


def make_dome(rt, tex=None, **props):
    """A VRayLight dome-role node; ``tex`` (when given) is bound as the ``texmap`` property."""
    bag = dict(props)
    if tex is not None:
        bag["texmap"] = tex
    return MockNode(rt, "VRayLight", "Dome01", props=bag)


def make_atmo(rt, cls="VRayEnvironmentFog", **props):
    """An atmospheric effect (environment fog / aerial perspective) with the given aliases."""
    return MockObject(rt, cls, dict(props))


# ===================================================================== key invariants
@pytest.mark.parametrize("rig_factory", [
    lambda rt: {},
    lambda rt: {"sun": None, "dome": None, "atmosphere": None},
    lambda rt: {"sun": make_sun(rt, intensity_multiplier=1.0)},
    lambda rt: {
        "sun": make_sun(rt, intensity_multiplier=1.0, size_multiplier=1.0,
                        turbidity=3.0, enabled=True),
        "dome": make_dome(rt, tex=make_tex(rt, HDRIMapName="a.hdr", horizontalRotation=25.0),
                          type=1, enabled=True, multiplier=5.0, useTexmap=True),
        "atmosphere": make_atmo(rt, enabled=True, fog_distance=25000.0, fog_height=1200.0),
    },
])
def test_all_twelve_keys_always_present(rt, rig_factory):
    """Whatever the config — empty rig, all-None targets, partial, or full — every one of the
    12 logical keys is present, no more and no fewer."""
    aliases = sc.report_aliases(rig_factory(rt))
    assert set(aliases) == EXPECTED_KEYS


# ===================================================================== first-candidate wins
def test_full_rig_reports_first_existing_candidate(rt):
    """The canonical build: every object carries its FIRST candidate, so the report echoes the
    leading name in each tuple."""
    tex = make_tex(rt, HDRIMapName="env.hdr", horizontalRotation=25.0)
    rig = {
        "sun": make_sun(rt, intensity_multiplier=1.0, size_multiplier=1.0,
                        turbidity=3.0, enabled=True),
        "dome": make_dome(rt, tex=tex, type=1, enabled=True, multiplier=5.0, useTexmap=False),
        "atmosphere": make_atmo(rt, enabled=True, fog_distance=25000.0, fog_height=1200.0),
    }
    assert sc.report_aliases(rig) == {
        "sun.intensity": "intensity_multiplier",
        "sun.size": "size_multiplier",
        "sun.turbidity": "turbidity",
        "sun.enabled": "enabled",
        "dome.enabled": "enabled",
        "dome.intensity": "multiplier",
        "dome.tex_on": "useTexmap",
        "dome.tex_rotation": "horizontalRotation",
        "dome.tex_file": "HDRIMapName",
        "atmosphere.enabled": "enabled",
        "atmosphere.distance": "fog_distance",
        "atmosphere.height": "fog_height",
    }


# ===================================================================== later-candidate fallthrough
def test_later_candidates_and_report_only_tails_are_reached(rt):
    """When the leading candidate is absent the report walks on to the real alias — including
    the report-only ``*_REPORT`` tails (VRayIES ``power``, VRayAerialPerspective
    ``visibility_range`` / ``atmo_height``), proving those extra names are reachable."""
    tex = make_tex(rt, bitmap_filename="b.hdr", tex_hrotation=10.0)   # last file/rot candidates
    rig = {
        "sun": make_sun(rt, intensity=2.0, sizeMultiplier=1.0, on=True),   # no turbidity
        "dome": make_dome(rt, tex=tex, on=True, power=1700.0, use_texture=True),
        "atmosphere": make_atmo(rt, "VRayAerialPerspective", active=True,
                                visibility_range=500.0, atmo_height=50.0),
    }
    aliases = sc.report_aliases(rig)
    assert aliases["sun.intensity"] == "intensity"          # 3rd SUN_INTENSITY candidate
    assert aliases["sun.size"] == "sizeMultiplier"          # 2nd SUN_SIZE candidate
    assert aliases["sun.turbidity"] is None                 # no candidate present → None
    assert aliases["sun.enabled"] == "on"                   # 2nd LIGHT_ON candidate
    assert aliases["dome.enabled"] == "on"
    assert aliases["dome.intensity"] == "power"             # LIGHT_MULT_REPORT tail
    assert aliases["dome.tex_on"] == "use_texture"          # 3rd DOME_TEX_ON candidate
    assert aliases["dome.tex_rotation"] == "tex_hrotation"  # last DOME_TEX_ROT candidate
    assert aliases["dome.tex_file"] == "bitmap_filename"    # last DOME_TEX_FILE candidate
    assert aliases["atmosphere.enabled"] == "active"        # 3rd FOG_ON candidate
    assert aliases["atmosphere.distance"] == "visibility_range"  # FOG_DISTANCE_REPORT tail
    assert aliases["atmosphere.height"] == "atmo_height"    # FOG_HEIGHT_REPORT tail


# ===================================================================== None when unmatched
def test_present_object_with_no_matching_candidate_reports_none(rt):
    """An object that exists but carries none of a row's candidates reports ``None`` for that
    row while its OTHER rows still resolve."""
    sun = make_sun(rt, enabled=True)                     # only sun.enabled is matchable
    dome = make_dome(rt, multiplier=5.0)                 # only dome.intensity is matchable
    atmo = make_atmo(rt, fog_distance=1000.0)            # only atmosphere.distance is matchable
    aliases = sc.report_aliases({"sun": sun, "dome": dome, "atmosphere": atmo})
    assert aliases["sun.intensity"] is None
    assert aliases["sun.size"] is None
    assert aliases["sun.turbidity"] is None
    assert aliases["sun.enabled"] == "enabled"
    assert aliases["dome.enabled"] is None
    assert aliases["dome.intensity"] == "multiplier"
    assert aliases["atmosphere.enabled"] is None
    assert aliases["atmosphere.distance"] == "fog_distance"
    assert aliases["atmosphere.height"] is None


def test_absent_objects_report_none_for_their_rows(rt):
    """rig.get returning None for sun / dome / atmosphere leaves every one of their rows None —
    including the two texmap rows, which resolve through a None dome without raising."""
    aliases = sc.report_aliases({"sun": None, "dome": None, "atmosphere": None})
    assert set(aliases) == EXPECTED_KEYS
    assert all(v is None for v in aliases.values())


def test_empty_rig_reports_all_none(rt):
    """A rig missing every key (bare ``{}``) still yields all 12 keys, each None."""
    aliases = sc.report_aliases({})
    assert set(aliases) == EXPECTED_KEYS
    assert all(v is None for v in aliases.values())


# ===================================================================== dome texmap resolution
def test_dome_without_texmap_nulls_only_the_tex_rows(rt):
    """A dome carrying no ``texmap`` leaves the two tex_* rows None (resolved via
    get_prop(dome, ("texmap",)) → None) while dome.enabled / .intensity / .tex_on — read off
    the dome itself — still resolve."""
    dome = make_dome(rt, type=1, enabled=True, multiplier=5.0, useTexmap=False)  # no texmap
    aliases = sc.report_aliases({"dome": dome})
    assert aliases["dome.tex_rotation"] is None
    assert aliases["dome.tex_file"] is None
    assert aliases["dome.enabled"] == "enabled"
    assert aliases["dome.intensity"] == "multiplier"
    assert aliases["dome.tex_on"] == "useTexmap"
    assert set(aliases) == EXPECTED_KEYS


def test_stale_targets_report_none_and_never_raise(rt):
    """Deleted-node handles (every access raises inside the mock) degrade to all-None rows —
    record-don't-raise — with the full key set intact."""
    sun = make_sun(rt, intensity_multiplier=1.0)
    sun.set_stale()
    dome = make_dome(rt, tex=make_tex(rt, HDRIMapName="x.hdr"), type=1, multiplier=5.0)
    dome.set_stale()
    aliases = sc.report_aliases({"sun": sun, "dome": dome, "atmosphere": None})
    assert set(aliases) == EXPECTED_KEYS
    assert all(v is None for v in aliases.values())


# ===================================================================== read-only guarantee
def test_report_fires_no_writes(rt):
    """report_aliases is pure introspection: after a full-rig report the mutation_log is empty
    — not one property was set."""
    tex = make_tex(rt, HDRIMapName="env.hdr", horizontalRotation=25.0)
    rig = {
        "sun": make_sun(rt, intensity_multiplier=1.0, enabled=True),
        "dome": make_dome(rt, tex=tex, type=1, enabled=True, multiplier=5.0, useTexmap=True),
        "atmosphere": make_atmo(rt, enabled=True, fog_distance=25000.0, fog_height=1200.0),
    }
    rt.reset_log()
    sc.report_aliases(rig)
    assert rt.mutation_log == []


# ===================================================================== record / LAST_ALIASES
def test_record_default_false_leaves_last_aliases_untouched(rt):
    """The default (record=False) never publishes into the module snapshot."""
    sc.LAST_ALIASES = {"sentinel": True}
    sc.report_aliases({"dome": make_dome(rt, enabled=True)})
    assert sc.LAST_ALIASES == {"sentinel": True}


def test_record_true_publishes_the_returned_dict(rt):
    """record=True stores the exact returned dict into LAST_ALIASES (same object)."""
    out = sc.report_aliases({"dome": make_dome(rt, enabled=True)}, record=True)
    assert sc.LAST_ALIASES is out
    assert set(sc.LAST_ALIASES) == EXPECTED_KEYS


# ===================================================================== off-Max degradation
def test_off_max_returns_empty_and_never_records(monkeypatch):
    """With no pymxs installed, report_aliases returns ``{}`` and — because it bails before the
    record block — leaves LAST_ALIASES untouched even when record=True is requested."""
    monkeypatch.delitem(sys.modules, "pymxs", raising=False)
    sc.LAST_ALIASES = {"keep": 1}
    assert sc.report_aliases({"sun": None}, record=True) == {}
    assert sc.LAST_ALIASES == {"keep": 1}


# ===================================================================== G3 invariant: tuples apart
def test_shared_write_tuples_stay_byte_identical():
    """The report-only ``*_REPORT`` tuples are the shared write tuples PLUS a class-specific
    tail — the shared tuples themselves are byte-identical, so apply.py's capture/set path is
    never widened by an alias the report needs (G3)."""
    assert sc.LIGHT_MULT == ("multiplier", "intensity")
    assert sc.FOG_DISTANCE == ("fog_distance", "fogDistance", "distance")
    assert sc.FOG_HEIGHT == ("fog_height", "fogHeight", "height")
    assert sc.LIGHT_MULT_REPORT == sc.LIGHT_MULT + ("power",)
    assert sc.FOG_DISTANCE_REPORT == sc.FOG_DISTANCE + ("visibility_range",)
    assert sc.FOG_HEIGHT_REPORT == sc.FOG_HEIGHT + ("atmo_height",)


# ===================================================================== get_prop / set_prop / matched_prop
def test_matched_prop_tracks_the_set_prop_write_target(rt):
    """matched_prop reports exactly the name set_prop would write, and writes nothing itself."""
    obj = MockNode(rt, "VRayLight", "L", props={"intensity": 2.0})   # multiplier absent
    rt.reset_log()
    name = sc.matched_prop(obj, sc.LIGHT_MULT)
    assert name == "intensity"
    assert rt.mutation_log == []                                     # pure read
    used = sc.set_prop(obj, sc.LIGHT_MULT, 9.0)
    assert used == name                                              # same target
    assert obj.get_raw("intensity") == 9.0


def test_get_prop_and_set_prop_unchanged_when_no_candidate_exists(rt):
    """No candidate present: get_prop returns its default, set_prop returns None and mutates
    nothing (the pre-existing contract this wave must not disturb)."""
    obj = MockNode(rt, "VRayLight", "L", props={"intensity": 2.0})
    assert sc.get_prop(obj, ("multiplier", "intensity")) == 2.0     # first existing wins
    assert sc.get_prop(obj, ("nope", "nada"), default=-1) == -1
    rt.reset_log()
    assert sc.set_prop(obj, ("nope", "nada"), 5.0) is None
    assert rt.mutation_log == []
    assert sc.matched_prop(obj, ("nope", "nada")) is None


def test_matched_prop_reports_the_write_target_get_prop_skips(rt):
    """Documented divergence: a candidate whose getattr raises (armed) is the WRITE target —
    matched_prop reports it (isProperty-only), while get_prop keeps its getattr inside the try
    and falls through to the next readable candidate."""
    obj = MockNode(rt, "VRayLight", "L", props={"multiplier": 5.0, "intensity": 2.0})
    obj.arm_get("multiplier")                                        # isProperty True, getattr raises
    assert sc.matched_prop(obj, sc.LIGHT_MULT) == "multiplier"       # tracks the write target
    assert sc.get_prop(obj, sc.LIGHT_MULT) == 2.0                    # skips armed, reads next
