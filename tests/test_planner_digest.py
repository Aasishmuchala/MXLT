"""Scene-aware agent mode — digest formatting and the plan validator's grounding."""

import json

import pytest

from maxgaffer.core import scenedigest
from maxgaffer.core.parse import ParseError
from maxgaffer.core.planner import MAX_OPS, describe_plan, validate_plan

RAW = {
    "renderer": {"class": "V_Ray_7", "props": {
        "options_progressiveNoiseThreshold": 0.01, "environment_gi_on": True,
        "system_lowThreadPriority": False}},
    "environment": {"class": "VRaySky", "props": {"sun_turbidity": 3.0}},
    "exposure": {"class": "VRay_Exposure_Control", "props": {"ev": 12.0,
                                                             "temperature": 6500.0}},
    "lights": [
        {"name": "VRaySun001", "class": "VRaySun", "layer": "0", "pos": [10, 10, 500],
         "props": {"enabled": True, "turbidity": 3.0, "intensity_multiplier": 1.0}},
        {"name": "Spot_A", "class": "VRayLight", "layer": "practicals",
         "pos": [0, 0, 100], "props": {"on": True, "multiplier": 30.0}},
    ],
    "cameras": [{"name": "PhysCam_Hero", "class": "Physical", "yaw_deg": 90.0,
                 "pos": [0, -300, 150], "props": {"exposure_value": 12.0}}],
    "stats": {"objects": 1200, "scene": "tula.max"},
}
CAT = scenedigest.catalog(RAW)


def plan_reply(ops):
    return json.dumps({"read": "scene is midday, ref is dusk", "ops": ops,
                       "expects": "warm dusk"})


# --------------------------------------------------------------------------- digest
def test_catalog_targets_and_props():
    assert CAT["renderer"] == {"options_progressiveNoiseThreshold", "environment_gi_on",
                               "system_lowThreadPriority"}
    assert "node:VRaySun001" in CAT and "turbidity" in CAT["node:VRaySun001"]
    assert "node:PhysCam_Hero" in CAT and "exposure_value" in CAT["node:PhysCam_Hero"]


def test_digest_text_sections_and_truncation():
    text = scenedigest.to_text(RAW)
    for marker in ("RENDERER", "ENVIRONMENT", "EXPOSURE", "LIGHTS (2)", "CAMERAS (1)",
                   "VRaySun001", "practicals", "SCENE"):
        assert marker in text
    tiny = scenedigest.to_text(RAW, max_chars=200)
    assert len(tiny) <= 200 and "truncated" in tiny


def test_priority_props_survive_trimming():
    big = {"renderer": {"class": "R", "props": {f"zz_obscure_{i}": i for i in range(80)}},
           "lights": [], "cameras": []}
    big["renderer"]["props"]["environment_gi_on"] = True
    text = scenedigest.to_text(big)
    assert "environment_gi_on" in text          # lighting-relevant name ranked first
    assert "more settable properties exist" in text


# --------------------------------------------------------------------------- planner
def test_plan_grounding_drops_hallucinations():
    ops, rejected, meta = validate_plan(plan_reply([
        {"op": "set", "target": "node:VRaySun001", "prop": "turbidity", "value": 5.5,
         "why": "haze"},
        {"op": "set", "target": "node:VRaySun001", "prop": "mood", "value": 1,
         "why": "hallucinated prop"},
        {"op": "set", "target": "node:Ghost", "prop": "on", "value": True, "why": "x"},
        {"op": "set", "target": "renderer", "prop": "environment_gi_on",
         "value": "not-a-list-but-fine", "why": "string ok"},
        {"op": "teleport", "target": "renderer", "prop": "x", "value": 1, "why": "x"},
    ]), CAT)
    assert [o["prop"] for o in ops if o["op"] == "set"] == ["turbidity",
                                                            "environment_gi_on"]
    assert any("mood" in r for r in rejected)
    assert any("Ghost" in r for r in rejected)
    assert any("teleport" in r for r in rejected)
    assert meta["read"].startswith("scene is")


def test_create_light_validation_and_placement_clamp():
    ops, rejected, _ = validate_plan(plan_reply([
        {"op": "create_light", "light_type": "VRayLight_plane", "name": "window_fill",
         "placement": {"bearing_deg": -700, "distance": 2, "height": 120},
         "aim_at_camera_target": True, "props": {"multiplier": 8, "color": [255, 230, 200],
                                                 "junk": {"nested": 1}}, "why": "fill"},
        {"op": "create_light", "light_type": "LaserCannon", "name": "MG_no",
         "placement": {"bearing_deg": 0, "distance": 100, "height": 0}, "why": "x"},
        {"op": "create_light", "light_type": "VRaySun", "name": "MG_sun2",
         "at_node": "Spot_A", "why": "sun at practical"},
        {"op": "create_light", "light_type": "VRaySun", "name": "MG_bad",
         "why": "no placement"},
    ]), CAT)
    assert len(ops) == 2
    plane = ops[0]
    assert plane["name"] == "MG_window_fill"                 # prefix enforced
    assert plane["placement"]["bearing_deg"] == -180.0       # clamped
    assert plane["placement"]["distance"] == 10.0            # clamped to floor
    assert plane["props"] == {"multiplier": 8, "color": [255, 230, 200]}  # junk dropped
    assert ops[1]["placement"] == {"at_node": "Spot_A"}
    assert any("LaserCannon" in r for r in rejected)
    assert any("no placement" in r or "needs placement" in r for r in rejected)


def test_create_light_physical_placement_is_canonical_and_clamped():
    ops, rejected, _ = validate_plan(plan_reply([
        {"op": "create_light", "light_type": "VRayLight_plane", "name": "metric",
         "placement": {"bearing_deg": 25, "distance_m": 2.5, "height_m": 1.2},
         "why": "portable placement"},
        {"op": "create_light", "light_type": "VRayLight_sphere", "name": "mixed",
         "placement": {"bearing_deg": 0, "distance_m": 2, "height": 50},
         "why": "mixed schemas must not pass"},
    ]), CAT)
    assert len(ops) == 1
    assert ops[0]["placement"] == {"bearing_deg": 25.0, "distance_m": 2.5,
                                    "height_m": 1.2}
    assert "2.50m" in describe_plan(ops)[0]
    assert any("height_m" in note for note in rejected)


def test_plan_caps_at_max_ops_and_requires_json():
    many = [{"op": "set", "target": "exposure", "prop": "ev", "value": i, "why": "x"}
            for i in range(20)]
    ops, rejected, _ = validate_plan(plan_reply(many), CAT)
    assert len(ops) == MAX_OPS
    assert any("truncated" in r for r in rejected)
    with pytest.raises(ParseError):
        validate_plan("I would rather describe it in prose.", CAT)


def test_duplicate_created_names_rejected():
    ops, rejected, _ = validate_plan(plan_reply([
        {"op": "create_light", "light_type": "VRayLight_sphere", "name": "MG_a",
         "placement": {"bearing_deg": 0, "distance": 100, "height": 50}, "why": "a"},
        {"op": "create_light", "light_type": "VRayLight_sphere", "name": "MG_a",
         "placement": {"bearing_deg": 10, "distance": 100, "height": 50}, "why": "dup"},
    ]), CAT)
    assert len(ops) == 1
    assert any("already exists" in r for r in rejected)


def test_describe_plan_lines():
    ops, _, _ = validate_plan(plan_reply([
        {"op": "set", "target": "exposure", "prop": "ev", "value": 10.5, "why": "brighten"},
        {"op": "create_light", "light_type": "VRayIES", "name": "MG_sconce",
         "at_node": "Spot_A", "why": "practical glow"},
    ]), CAT)
    lines = describe_plan(ops)
    assert lines[0].startswith("set  exposure · ev → 10.5")
    assert "MG_sconce" in lines[1] and "at node Spot_A" in lines[1]
