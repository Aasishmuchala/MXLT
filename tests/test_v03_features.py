"""v0.3.0 completion features — overcast dim mode, presets, API surface, config."""

import json

from maxgaffer.core.genome import LightingState
from maxgaffer.core.rules import initial_state
from maxgaffer.core.session import preset_dumps, preset_loads


def rig_state():
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 0.0, "sun.altitude_deg": 45.0,
                 "sun.intensity": 1.0, "sun.size": 1.0, "sun.turbidity": 3.0,
                 "dome.enabled": 1, "dome.intensity": 1.0,
                 "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    return st


def overcast_semantics():
    return {"scene_type": "exterior", "time_of_day": "overcast_day", "sky": "overcast",
            "sun_active": True, "sun_bearing_deg": 0.0, "sun_altitude_band": "na",
            "light_quality": "soft", "wb_kelvin_estimate": 6800.0, "practicals_on": False,
            "atmosphere": "none", "contrast_character": "airy", "key_notes": "",
            "confidence": 0.9}


# --------------------------------------------------------------- overcast sun modes
def test_overcast_disable_mode_default():
    st, _ = initial_state(overcast_semantics(), rig_state(), camera_yaw_deg=0.0)
    assert st.get("sun.enabled") == 0


def test_overcast_dim_mode_keeps_sun_alive():
    st, why = initial_state(overcast_semantics(), rig_state(), camera_yaw_deg=0.0,
                            overcast_sun_mode="dim")
    assert st.get("sun.enabled") == 1
    assert st.get("sun.intensity") == 0.05          # genome floor
    assert st.get("sun.size") == 12.0
    assert any("VRaySky" in w for w in why)


def test_dim_mode_ignored_at_night():
    sem = overcast_semantics()
    sem.update(time_of_day="night", sky="night", sun_active=False)
    st, _ = initial_state(sem, rig_state(), camera_yaw_deg=0.0, overcast_sun_mode="dim")
    assert st.get("sun.enabled") == 0               # night is night, dim or not


# --------------------------------------------------------------- presets
def test_preset_roundtrip_and_rejection():
    st = rig_state()
    st.groups["practicals"] = 0.5
    text = preset_dumps(st, name="dusk_hero", now="2026-07-16")
    st2 = preset_loads(text)
    assert st2 is not None and st2.diff(st) == {}
    assert preset_loads("not json") is None
    assert preset_loads(json.dumps({"random": "json"})) is None
    # out-of-bounds values in a hand-edited preset are re-clamped by the genome
    tampered = json.loads(text)
    tampered["state"]["values"]["sun.altitude_deg"] = 500
    assert preset_loads(json.dumps(tampered)).get("sun.altitude_deg") == 88.0


# --------------------------------------------------------------- API surface (off-Max)
def test_api_imports_and_exposes_contract():
    """The MaxDirector integration point must import cleanly with no pymxs present."""
    import inspect

    from maxgaffer import api

    assert set(api.__all__) == {"match_camera", "match_all_cameras", "apply_camera_state",
                                "render_cameras", "export_vrscenes_for_vantage",
                                "get_controller",
                                "seed_dome", "scenario_board", "adopt_scenario",
                                "bake_lighting_animation",
                                # additive reference-management + fairness API (multi-reference
                                # support + single-reference fairness gate)
                                "add_reference", "list_references", "remove_reference",
                                "assess_reference_fairness"}
    sig = inspect.signature(api.match_camera)
    assert list(sig.parameters) == ["camera_name", "reference_path", "log",
                                    "should_cancel", "locks", "sweep", "deep",
                                    "quality_profile", "config_overrides",
                                    "references"]


# --------------------------------------------------------------- config completeness
def test_config_reflects_verified_stack():
    from maxgaffer.maxbridge.config import Config

    cfg = Config()
    assert cfg.draft_sampler is False               # opt-in, never default-on
    # VRaySky auto-binds to "the first enabled VRaySun" (doc-verified) → dim by default
    assert cfg.overcast_sun_mode == "dim"
    assert cfg.keep_runs == 10
    # stock Vantage 3.x has no render CLI (Chaos support-confirmed) → V-Ray backend
    assert cfg.final_render_backend == "vray"
    assert cfg.vantage_exe.endswith("vantage.exe")
    assert cfg.auto_exposure_control is True


def test_shutter_seconds_units():
    """Native Physical stores a DURATION (shutter_length_seconds); legacy VRayPhysical
    stores a SPEED (shutter_speed, 1/s) — mixing them up is a silent 4-6 stop EV error."""
    from maxgaffer.maxbridge.exposure import shutter_seconds

    assert shutter_seconds("shutter_length_seconds", 0.005) == 0.005
    assert abs(shutter_seconds("shutter_speed", 200.0) - 0.005) < 1e-12
    assert shutter_seconds("shutter_speed", 0.0) > 0          # guarded against div-zero
