"""Multi-reference storage — the frozen Section A contract, driven off-box.

Every camera may carry several references (a hero angle plus supporting views); the FIRST
entry is always the PRIMARY and its columns mirror the legacy single-reference fields for
backward compatibility.  These tests lock the load-bearing invariants:

  * ``RefEntry`` round-trips through to_dict/from_dict with defensive coercion;
  * ``set_reference`` is byte-for-byte the old single-reference API (same-ref no reset,
    new-ref / in-place-change reset, clear yields the ``"|missing"`` signature);
  * ``add_reference`` / ``set_references`` / ``remove_reference`` behave per the contract;
  * the SYMMETRY guard — a legacy-only entry built by direct ``e.reference = ...`` assignment
    serializes IDENTICALLY to its post-load form (the exact fuzz case at
    test_kimi_fuzz_integration.py:708);
  * the legacy trio + semantics + score ALWAYS mirror ``references[0]`` after ALL SIX
    mutation paths (set / add / set_references / remove / load-re-resolution / relink);
  * an OLD v2 single-reference sidecar loads and synthesizes a 1-item primary list;
  * v3 save/load round-trips and a NEWER sidecar is protected from silent downgrade;
  * ``relink_references`` repairs (and recomputes the relative of) EVERY reference;
  * the controller FUSES several references' semantic reads via ``consolidate_analyses`` and
    suppresses the single-image "samples disagreed" contest warning when n_references > 1,
    while the single-reference path stays untouched.

No runtime Max: the Session tests are pure, and the controller tests fake the semantic call
at the seam (the same fake-scene pattern as test_controller_fixes.py / test_ui_dock.py).
"""

import json
import os
import shutil

import pytest

from maxgaffer.core import consensus
from maxgaffer.core.scenarios import DEFAULT_SEMANTICS
from maxgaffer.core.session import (CameraEntry, FORMAT_VERSION, RefEntry, Session,
                                    reference_signature)
from maxgaffer.maxbridge import config as cfgmod
from maxgaffer.maxbridge import controller as ctl


REF_KEYS = {"path", "relative", "signature", "role", "semantics", "score"}
MISSING = reference_signature("")          # the canonical "no primary" signature: "|missing"


def assert_mirror(e: CameraEntry) -> None:
    """The frozen cross-cutting invariant: the legacy reference / reference_relative /
    reference_signature / semantics / score columns ALWAYS mirror references[0]; when the
    list is empty the legacy columns clear to ('' / '' / '|missing' / {} / None)."""
    refs = e.references
    if refs:
        p = refs[0]
        assert p.role == "primary"
        assert e.reference == p.path
        assert e.reference_relative == p.relative
        assert e.reference_signature == p.signature
        assert e.semantics == p.semantics
        assert e.score == p.score
    else:
        assert e.reference == ""
        assert e.reference_relative == ""
        assert e.reference_signature == MISSING
        assert e.semantics == {}
        assert e.score is None


# --------------------------------------------------------------------------- RefEntry
def test_refentry_roundtrip_exact_keys():
    r = RefEntry(path="D:/refs/hero.exr", relative="refs/hero.exr",
                 signature="sig|1|2", role="angle_1",
                 semantics={"time_of_day": "dusk"}, score=71.5)
    d = r.to_dict()
    assert set(d) == REF_KEYS
    back = RefEntry.from_dict(d)
    assert back.to_dict() == d


def test_refentry_from_dict_defensive_coercion():
    # str-coercion of path/relative/signature, missing role → "primary"
    r = RefEntry.from_dict({"path": 42, "relative": None, "signature": 7})
    assert r.path == "42" and r.relative == "" and r.signature == "7"
    assert r.role == "primary"
    # non-dict semantics → {}, and a non-finite / non-numeric score → None (like CameraEntry)
    assert RefEntry.from_dict({"semantics": ["nope"]}).semantics == {}
    assert RefEntry.from_dict({"score": float("inf")}).score is None
    assert RefEntry.from_dict({"score": "88"}).score is None
    assert RefEntry.from_dict({"score": 88}).score == 88.0
    # a blank role also normalizes to the primary default
    assert RefEntry.from_dict({"role": ""}).role == "primary"


# --------------------------------------------------------------- set_reference parity
def test_set_reference_new_ref_resets_and_mirrors(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "a.jpg")
    e = s.entry("cam")
    e.semantics = {"time_of_day": "night"}
    e.score = 90.0
    e.references[0].semantics = e.semantics       # keep the mirror coherent for the check
    e.references[0].score = e.score
    assert_mirror(e)

    s.set_reference("cam", "b.jpg")               # a genuinely new reference → reset
    assert e.semantics == {} and e.score is None
    assert e.reference == "b.jpg"
    assert len(e.references) == 1 and e.references[0].path == "b.jpg"
    assert_mirror(e)


def test_set_reference_same_ref_preserves_analysis(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "b.jpg")
    s.entry("cam").semantics = {"k": 1}
    s.entry("cam").references[0].semantics = {"k": 1}
    s.set_reference("cam", "b.jpg")               # SAME ref → no reset
    assert s.entry("cam").semantics == {"k": 1}
    assert_mirror(s.entry("cam"))


def test_set_reference_in_place_change_invalidates(tmp_path):
    ref = tmp_path / "reference.jpg"
    ref.write_bytes(b"first")
    s = Session()
    s.set_reference("cam", str(ref))
    e = s.entry("cam")
    e.semantics = {"time_of_day": "day"}
    e.score = 88.0
    e.references[0].semantics = e.semantics
    e.references[0].score = e.score

    ref.write_bytes(b"a wholly different image payload")   # same path, new pixels
    s.set_reference("cam", str(ref))
    assert e.semantics == {} and e.score is None
    assert e.reference_signature and e.reference_signature != MISSING
    assert_mirror(e)


def test_set_reference_clear_yields_missing_signature(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "a.jpg")
    s.set_reference("cam", "")                      # clear
    e = s.entry("cam")
    assert e.references == []
    assert e.reference == "" and e.reference_relative == ""
    assert e.reference_signature == MISSING        # reproduced, NOT forced to ""
    assert e.semantics == {} and e.score is None
    assert_mirror(e)


# ---------------------------------------------------------------------- add_reference
def test_add_reference_appends_without_disturbing_primary(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "hero.jpg")
    s.entry("cam").semantics = {"time_of_day": "dusk"}
    s.entry("cam").references[0].semantics = {"time_of_day": "dusk"}
    s.add_reference("cam", "angle.jpg")
    e = s.entry("cam")
    assert [r.path for r in e.references] == ["hero.jpg", "angle.jpg"]
    assert e.references[1].role == "angle_1"       # default role = angle_<len at append>
    assert e.references[0].role == "primary"
    assert e.semantics == {"time_of_day": "dusk"}  # primary analysis untouched
    assert_mirror(e)


def test_add_reference_noop_on_empty_and_duplicate(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "hero.jpg")
    s.add_reference("cam", "")                      # empty → no-op
    s.add_reference("cam", "angle.jpg")
    s.add_reference("cam", "angle.jpg")            # duplicate signature → no-op
    e = s.entry("cam")
    assert [r.path for r in e.references] == ["hero.jpg", "angle.jpg"]


def test_add_reference_on_empty_camera_becomes_primary(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.add_reference("cam", "solo.jpg", role="whatever")
    e = s.entry("cam")
    assert len(e.references) == 1
    assert e.references[0].path == "solo.jpg"
    assert e.references[0].role == "primary"       # first-ever ref is always the primary
    assert_mirror(e)


def test_add_reference_custom_role(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "hero.jpg")
    s.add_reference("cam", "left.jpg", role="left_flank")
    assert s.entry("cam").references[1].role == "left_flank"


# --------------------------------------------------------------------- set_references
def test_set_references_replaces_and_forces_primary(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_references("cam", ["a.jpg", {"path": "b.jpg", "role": "side"}, "c.jpg"])
    e = s.entry("cam")
    assert [r.path for r in e.references] == ["a.jpg", "b.jpg", "c.jpg"]
    assert e.references[0].role == "primary"       # first item forced primary
    assert e.references[1].role == "side"          # explicit role kept
    assert e.references[2].role == "angle_2"       # default role for the rest
    assert_mirror(e)


def test_set_references_dedups_by_signature(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_references("cam", ["dup.jpg", "dup.jpg", "other.jpg"])
    assert [r.path for r in s.entry("cam").references] == ["dup.jpg", "other.jpg"]


def test_set_references_empty_clears(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "a.jpg")
    s.set_references("cam", [])                     # equivalent to set_reference("")
    e = s.entry("cam")
    assert e.references == []
    assert e.reference_signature == MISSING
    assert_mirror(e)


def test_set_references_reset_gate_matches_set_reference(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "a.jpg")
    s.entry("cam").semantics = {"cached": True}
    s.entry("cam").score = 61.0
    s.entry("cam").references[0].semantics = {"cached": True}
    s.entry("cam").references[0].score = 61.0

    # unchanged primary → preserve the cached analysis / score
    s.set_references("cam", ["a.jpg", "b.jpg"])
    assert s.entry("cam").semantics == {"cached": True}
    assert s.entry("cam").score == 61.0

    # a new primary → reset (same gate as set_reference)
    s.set_references("cam", ["b.jpg", "a.jpg"])
    assert s.entry("cam").semantics == {}
    assert s.entry("cam").score is None
    assert_mirror(s.entry("cam"))


# ------------------------------------------------------------------- remove_reference
def test_remove_reference_promotes_next_primary(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_references("cam", ["hero.jpg", "angle.jpg"])
    assert s.remove_reference("cam", 0) is True    # drop the primary
    e = s.entry("cam")
    assert [r.path for r in e.references] == ["angle.jpg"]
    assert e.references[0].role == "primary"       # the survivor is re-mirrored as primary
    assert e.reference == "angle.jpg"
    assert_mirror(e)


def test_remove_reference_by_path_and_signature(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_references("cam", ["hero.jpg", "angle.jpg", "third.jpg"])
    sig = s.entry("cam").references[2].signature
    assert s.remove_reference("cam", "angle.jpg") is True     # by path
    assert s.remove_reference("cam", sig) is True             # by signature
    assert [r.path for r in s.entry("cam").references] == ["hero.jpg"]


def test_remove_reference_misses_return_false(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_references("cam", ["hero.jpg"])
    assert s.remove_reference("cam", 5) is False              # out-of-range index
    assert s.remove_reference("cam", "nope.jpg") is False     # unknown path
    assert s.remove_reference("cam", True) is False           # bool is never an index
    assert s.remove_reference("ghost", 0) is False            # unknown camera


def test_remove_last_reference_clears_legacy_columns(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    s.set_reference("cam", "solo.jpg")
    assert s.remove_reference("cam", 0) is True
    e = s.entry("cam")
    assert e.references == []
    assert e.reference_signature == MISSING
    assert_mirror(e)


# --------------------------------------------------------------------- references read
def test_references_accessor_synthesizes_legacy_primary(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    e = s.entry("cam")
    e.reference = "legacy.jpg"                      # legacy-only entry (references still [])
    e.reference_relative = "refs/legacy.jpg"
    e.reference_signature = "legacy|1|2"
    refs = s.references("cam")
    assert len(refs) == 1 and refs[0].path == "legacy.jpg"
    assert refs[0].role == "primary"
    assert e.references == []                       # read accessor must NOT mutate the entry


def test_references_accessor_unknown_camera_is_empty(tmp_path):
    s = Session(str(tmp_path / "x.json"))
    assert s.references("nobody") == []


# ------------------------------------------------------------------ SYMMETRY guard A3
def test_to_dict_symmetric_for_legacy_only_entry():
    """The exact fuzz case (test_kimi_fuzz_integration.py:708): an in-memory entry built by
    direct attribute assignment (references still []) serializes IDENTICALLY to its post-load
    form, so any captured-before-save == after-load equality check stays green."""
    e = CameraEntry(camera_name="Cam")
    e.reference = "refs/hero.png"
    e.reference_relative = "refs/hero.png"
    e.reference_signature = "refs/hero.png|1234|5678"
    e.semantics = {"scene_type": "interior", "note": "z"}
    e.score = 77.25
    before = e.to_dict()
    # to_dict and from_dict normalize to the SAME canonical shape
    assert CameraEntry.from_dict(before).to_dict() == before
    # and the synthesized primary is present in the serialization
    assert before["references"] and before["references"][0]["role"] == "primary"
    assert before["references"][0]["path"] == "refs/hero.png"


def test_symmetry_survives_real_session_save_load(tmp_path):
    """The fuzz guard end-to-end: legacy-style direct assignment, saved through a real
    Session and reloaded, must be byte-identical (references==[] before save)."""
    path = str(tmp_path / "sym.maxgaffer.json")
    s = Session(path, now_fn=lambda: "2026-07-22T00:00:00")
    e = s.entry("Cam")
    e.reference = "refs/only.png"
    e.reference_relative = "refs/only.png"
    e.reference_signature = "refs/only.png|9|9"
    e.semantics = {"scene_type": "interior"}
    e.score = 42.0
    assert e.references == []                        # the exact pre-save shape
    before = e.to_dict()
    assert s.save()

    s2 = Session.load(path)
    assert not s2._protect_existing
    assert s2.entry("Cam").to_dict() == before


# --------------------------------------------------- exhaustive mirror after 6 paths
def test_mirror_invariant_after_every_mutation_path(tmp_path):
    """The legacy trio + semantics + score mirror references[0] after set / add /
    set_references / remove — the four in-memory mutation paths (load + relink are covered
    by their own move / relink tests below)."""
    s = Session(str(tmp_path / "m.json"))
    s.set_reference("cam", "a.jpg");                assert_mirror(s.entry("cam"))
    s.add_reference("cam", "b.jpg");                assert_mirror(s.entry("cam"))
    s.set_references("cam", ["c.jpg", "d.jpg"]);    assert_mirror(s.entry("cam"))
    s.remove_reference("cam", 0);                   assert_mirror(s.entry("cam"))
    s.remove_reference("cam", 0);                   assert_mirror(s.entry("cam"))  # → empty


# ------------------------------------------------- backward-compat v2 sidecar loading
def test_v2_sidecar_synthesizes_one_primary(tmp_path):
    """An OLD single-reference sidecar (no ``references`` array) loads and synthesizes a
    1-item primary list; the legacy trio is self-healed to references[0]."""
    path = tmp_path / "old.maxgaffer.json"
    path.write_text(json.dumps({
        "version": 2,
        "cameras": {"Cam": {"camera_name": "Cam", "reference": "look.jpg",
                            "reference_relative": "refs/look.jpg",
                            "reference_signature": "look.jpg|1|2",
                            "semantics": {"time_of_day": "noon"}, "score": 66.0}},
        "settings": {}, "baselines": {}}), encoding="utf-8")
    s = Session.load(str(path))
    assert not s._protect_existing                   # v2 < v3 loads freely
    e = s.cameras["Cam"]
    assert len(e.references) == 1
    assert e.references[0].path == "look.jpg"
    assert e.references[0].role == "primary"
    assert e.references[0].semantics == {"time_of_day": "noon"}
    assert e.references[0].score == 66.0
    assert_mirror(e)


def test_v2_sidecar_with_no_reference_loads_empty(tmp_path):
    path = tmp_path / "old_empty.maxgaffer.json"
    path.write_text(json.dumps({
        "version": 2,
        "cameras": {"Cam": {"camera_name": "Cam"}},
        "settings": {}, "baselines": {}}), encoding="utf-8")
    e = Session.load(str(path)).cameras["Cam"]
    # A pristine never-referenced entry stays empty; its on-disk signature ("") is left as-is
    # (only a clearing MUTATION reproduces the "|missing" sentinel).
    assert e.references == []
    assert e.reference == "" and e.semantics == {} and e.score is None


# ------------------------------------------------------- v3 round-trip + downgrade guard
def test_v3_multi_reference_roundtrip(tmp_path):
    path = str(tmp_path / "v3.maxgaffer.json")
    s = Session(path)
    s.set_references("Cam", ["hero.jpg", {"path": "angle.jpg", "role": "side"}])
    assert s.save()

    on_disk = json.loads(open(path, encoding="utf-8").read())
    assert on_disk["version"] == FORMAT_VERSION == 3

    s2 = Session.load(path)
    e = s2.cameras["Cam"]
    assert [(r.path, r.role) for r in e.references] == [
        ("hero.jpg", "primary"), ("angle.jpg", "side")]
    assert_mirror(e)


def test_newer_sidecar_is_protected_from_downgrade(tmp_path):
    path = str(tmp_path / "future.maxgaffer.json")
    open(path, "w", encoding="utf-8").write(json.dumps({
        "version": FORMAT_VERSION + 1,
        "cameras": {"Cam": {"camera_name": "Cam", "reference": "a.jpg"}},
        "settings": {}, "baselines": {}}))
    s = Session.load(path)
    assert s._protect_existing                        # newer format → guarded
    assert "Cam" in s.cameras                          # still loaded best-effort
    assert s.save() is False                            # auto-save blocked
    assert s.save(force=True) is True                  # explicit save overwrites


# --------------------------------------------------- load re-resolution across all refs
def test_load_re_resolves_every_reference_after_project_move(tmp_path):
    """Every RefEntry's path is re-resolved from its portable relative after a folder move,
    then the FULL legacy trio re-mirrors to references[0]."""
    source = tmp_path / "source"
    (source / "refs").mkdir(parents=True)
    hero = source / "refs" / "hero.jpg"; hero.write_bytes(b"h")
    angle = source / "refs" / "angle.jpg"; angle.write_bytes(b"a")
    sidecar = source / "scene.maxgaffer.json"
    s = Session(str(sidecar))
    s.set_references("Cam", [str(hero), str(angle)], "42")
    assert s.save()

    moved = tmp_path / "moved"
    shutil.move(str(source), str(moved))
    loaded = Session.load(str(moved / "scene.maxgaffer.json"))
    e = loaded.entry("Cam", "42")
    assert e.references[0].path == str(moved / "refs" / "hero.jpg")
    assert e.references[1].path == str(moved / "refs" / "angle.jpg")
    assert os.path.isfile(e.references[0].path)
    assert os.path.isfile(e.references[1].path)
    assert_mirror(e)


# ------------------------------------------------------------------------- relink
def test_relink_repairs_every_reference_and_recomputes_relative(tmp_path):
    """relink_references iterates EVERY reference, repairing its path + signature AND
    recomputing its portable relative (the old single-reference path did not refresh the
    relative — the per-RefEntry rewrite must not regress this)."""
    proj = tmp_path / "proj"; proj.mkdir()
    sidecar = proj / "scene.maxgaffer.json"
    # originals live OUTSIDE the project → no expressible relative at bind time
    orig = tmp_path / "elsewhere"; orig.mkdir()
    hero = orig / "hero.jpg"; hero.write_bytes(b"h")
    angle = orig / "angle.jpg"; angle.write_bytes(b"a")
    s = Session(str(sidecar))
    s.set_references("Cam", [str(hero), str(angle)])
    assert s.entry("Cam").references[0].relative == ""   # not project-relative yet
    assert s.entry("Cam").references[1].relative == ""

    # the files vanish from their bound paths and reappear UNDER the project folder
    shutil.rmtree(str(orig))
    found = proj / "relinked"; found.mkdir()
    (found / "hero.jpg").write_bytes(b"h")
    (found / "angle.jpg").write_bytes(b"a")

    changed = s.relink_references([str(proj)])
    assert changed == {"Cam": str((found / "hero.jpg"))}   # keyed name → new PRIMARY path
    e = s.entry("Cam")
    assert e.references[0].path == str(found / "hero.jpg")
    assert e.references[1].path == str(found / "angle.jpg")   # the ANGLE was repaired too
    # relative recomputed from "" to a real project-relative path (no regression)
    assert e.references[0].relative and e.references[0].relative != ""
    assert e.references[1].relative and e.references[1].relative != ""
    assert e.references[0].signature == reference_signature(str(found / "hero.jpg"))
    assert_mirror(e)


# ================================================================ controller: fusion
def _controller(tmp_path, monkeypatch, samples=1):
    """A real Controller wired off-box: a stable scene path, no gateway, no pymxs."""
    cfg = cfgmod.Config(no_renders=True, analyze_samples=samples)
    c = ctl.Controller(cfg)
    monkeypatch.setattr(ctl.sc, "scene_path", lambda: str(tmp_path / "t.max"))
    monkeypatch.setattr(ctl.sc, "list_cameras", lambda: [])   # _camera_id → "" (bare-name)
    return c


def test_controller_fuses_multiple_references(tmp_path, monkeypatch):
    """analyze_reference on a camera with >1 reference FUSES the per-reference reads through
    consolidate_analyses (Route A denoising) and suppresses the single-image contest flag."""
    hero = tmp_path / "hero.jpg"; hero.write_bytes(b"h")
    angle = tmp_path / "angle.jpg"; angle.write_bytes(b"a")
    c = _controller(tmp_path, monkeypatch)
    c.session.set_references("Cam", [str(hero), str(angle)])
    c._last_analyze_agreement = 0.4        # a stale contest flag from an earlier run

    analyzed = []

    def fake_samples(path):
        analyzed.append(path)
        return {"time_of_day": "golden_hour", "sun_bearing_deg": 100.0,
                "sun_active": True, "wb_kelvin_estimate": 3800.0, "confidence": 0.8}

    monkeypatch.setattr(c, "_analyze_samples", fake_samples)
    real_consolidate = consensus.consolidate_analyses
    fused_counts = []

    def spy(samples):
        fused_counts.append(len(samples))
        return real_consolidate(samples)

    monkeypatch.setattr(ctl.consensus, "consolidate_analyses", spy)

    out = c.analyze_reference("Cam")
    assert fused_counts == [2]                          # both references fed to the fusion
    assert set(analyzed) == {str(hero), str(angle)}    # each view analyzed once
    assert c._last_analyze_agreement is None           # contest warning suppressed
    assert "consensus_agreement" not in out            # the cross-ref scatter stays private
    assert out.get("time_of_day") == "golden_hour"
    # the fused read is mirrored onto the primary (Route A: the solve/critic/LLM see this)
    assert c.session.entry("Cam").semantics == out


def test_controller_multi_reference_caches_fused_read(tmp_path, monkeypatch):
    hero = tmp_path / "hero.jpg"; hero.write_bytes(b"h")
    angle = tmp_path / "angle.jpg"; angle.write_bytes(b"a")
    c = _controller(tmp_path, monkeypatch)
    c.session.set_references("Cam", [str(hero), str(angle)])

    calls = {"n": 0}

    def fake_samples(path):
        calls["n"] += 1
        return {"time_of_day": "noon", "sun_bearing_deg": 10.0, "sun_active": True,
                "wb_kelvin_estimate": 6000.0, "confidence": 0.9}

    monkeypatch.setattr(c, "_analyze_samples", fake_samples)
    first = c.analyze_reference("Cam")
    after_first = calls["n"]
    second = c.analyze_reference("Cam")               # nothing stale → served from cache
    assert second == first
    assert calls["n"] == after_first                   # no re-analysis on the cached hit


def test_controller_single_reference_path_unchanged(tmp_path, monkeypatch):
    """A single reference (the default) never enters the fusion branch — it drives the
    original inline ANALYZE body and returns the consolidated single-image semantics."""
    hero = tmp_path / "hero.jpg"; hero.write_bytes(b"h")
    c = _controller(tmp_path, monkeypatch)
    c.session.set_reference("Cam", str(hero))

    fused = {"called": False}
    monkeypatch.setattr(c, "_analyze_multi_reference",
                        lambda e: fused.__setitem__("called", True) or {})
    monkeypatch.setattr(c, "_image_block", lambda path: {"type": "image"})
    monkeypatch.setattr(c, "_semantic_call",
                        lambda system, messages, max_tokens: json.dumps(DEFAULT_SEMANTICS))

    out = c.analyze_reference("Cam")
    assert fused["called"] is False                    # fusion NOT taken for one reference
    assert out == DEFAULT_SEMANTICS                     # consolidated single-sample read
    assert "consensus_agreement" not in out
    assert c._last_analyze_agreement is None            # N=1 unanimous → no contest flag


def test_controller_references_dict_shape(tmp_path, monkeypatch):
    hero = tmp_path / "hero.jpg"; hero.write_bytes(b"h")
    c = _controller(tmp_path, monkeypatch)
    c.session.set_references("Cam", [str(hero), {"path": "angle.jpg", "role": "side"}])
    c.session.entry("Cam").semantics = {"time_of_day": "dusk"}
    c.session.entry("Cam").references[0].semantics = {"time_of_day": "dusk"}
    rows = c.references("Cam")
    assert [r["role"] for r in rows] == ["primary", "side"]
    assert set(rows[0]) == {"path", "relative", "signature", "role", "score", "has_semantics"}
    assert rows[0]["has_semantics"] is True
    assert rows[1]["has_semantics"] is False
