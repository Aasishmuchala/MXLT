"""Lighting-TRANSFER score — the honest cross-domain metric.

The critic scores pixels, which is unanswerable when the reference is a photo of a
different building: MXLT scored a real dawn plate at 75 because a CG room cannot produce
that photograph's pixels, not because the lighting was wrong. transfer.score asks the
question an artist actually asks — is my sun where that photo's sun is, at that colour
temperature, hardness and haze — by comparing the recovered rig to the reference's own
ANALYZE reading.
"""
import pytest

from maxgaffer.core import transfer
from maxgaffer.core.genome import LightingState


def rig(**kw):
    st = LightingState()
    for k, v in kw.items():
        st.set(k.replace("__", "."), v)
    return st


GOLDEN = {"sun_active": True, "sun_bearing_deg": 40.0, "sun_altitude_band": "golden",
          "wb_kelvin_estimate": 4000.0, "light_quality": "hard", "atmosphere": "light_haze"}


def test_a_faithful_transfer_scores_high():
    """Camera yaw 65 + bearing 40 = azimuth 105; the golden band centres on 8 degrees."""
    st = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
             sun__size=1.0, exposure__wb_kelvin=4000.0, atmosphere__distance_m=250.0)
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["score"] > 95.0, out
    assert set(out["measurable"]) >= {"direction", "elevation", "warmth", "presence"}


def test_the_sunless_impersonation_is_caught():
    """The exact failure the pixel critic scored ~92: sun switched off against a sunlit
    reference. Transfer must call that what it is."""
    st = rig(sun__enabled=0.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
             exposure__wb_kelvin=4000.0)
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["parts"]["presence"] == 0.0
    assert any("sun is OFF" in n for n in out["notes"])


def test_a_sun_in_the_wrong_quadrant_scores_low():
    st = rig(sun__enabled=1.0, sun__azimuth_deg=285.0, sun__altitude_deg=9.0,
             exposure__wb_kelvin=4000.0)          # 180 degrees out
    out = transfer.score(GOLDEN, st, camera_yaw_deg=65.0)
    assert out["parts"]["direction"] == 0.0
    assert out["score"] < 70.0


def test_elevation_uses_the_BAND_not_a_point():
    """A band is a range: 'golden' centres on 8 degrees with 7 degrees of slack, so 13
    degrees is still a faithful transfer."""
    inside = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=13.0,
                 exposure__wb_kelvin=4000.0)
    assert transfer.score(GOLDEN, inside, 65.0)["parts"]["elevation"] == 1.0
    outside = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=55.0,
                  exposure__wb_kelvin=4000.0)
    assert transfer.score(GOLDEN, outside, 65.0)["parts"]["elevation"] < 0.6


def test_warmth_is_measured_in_mireds_not_kelvin():
    """Equal Kelvin gaps are NOT equally visible. 4000 to 5000 K is an obvious shift;
    9000 to 10000 K is barely perceptible — mireds capture that, Kelvin does not."""
    warm_miss = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
                    exposure__wb_kelvin=5000.0)
    cool_ref = dict(GOLDEN, wb_kelvin_estimate=9000.0)
    cool_miss = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0,
                    exposure__wb_kelvin=10000.0)
    warm_part = transfer.score(GOLDEN, warm_miss, 65.0)["parts"]["warmth"]
    cool_part = transfer.score(cool_ref, cool_miss, 65.0)["parts"]["warmth"]
    assert cool_part > warm_part      # same 1000 K gap, far less visible when cool


def test_absent_inputs_are_skipped_never_scored_as_agreement():
    bare = rig(sun__enabled=1.0, sun__azimuth_deg=105.0, sun__altitude_deg=9.0)
    out = transfer.score({"sun_active": True, "sun_bearing_deg": 40.0,
                          "sun_altitude_band": "golden"}, bare, 65.0)
    assert "warmth" not in out["parts"]        # no estimate and no wb on the rig
    assert "atmosphere" not in out["parts"]
    assert out["score"] > 0


def test_degrades_on_junk_rather_than_raising():
    assert transfer.score(None, None)["score"] == 0.0
    assert transfer.score({}, rig())["score"] == 0.0
    assert transfer.score({"sun_bearing_deg": float("nan")}, rig())["score"] == 0.0


# ------------------------------------------------------------------ objective blending
def test_blend_pulls_the_objective_toward_semantic_agreement():
    """Pixels cannot pin sun direction on an interior — measured on-box 2026-07-25, a
    64-degree azimuth error still scored 90.92. Blending the ANALYZE reading in is what
    makes a structurally wrong rig score lower than a right one."""
    from maxgaffer.core.director import Hooks, MatchConfig, blend_transfer
    from maxgaffer.core.genome import LightingState

    st = LightingState()
    hooks = Hooks(apply=lambda s: None, render=lambda t: None, stats=lambda p: None,
                  llm_deltas=lambda c: "", transfer=lambda s: 0.40)
    cfg = MatchConfig(transfer_weight=0.25)
    # a 90 pixel score with poor semantic agreement is pulled down
    assert blend_transfer(90.0, st, hooks, cfg) == pytest.approx(0.75 * 90 + 0.25 * 40)
    # perfect agreement leaves a good score essentially intact
    hooks.transfer = lambda s: 1.0
    assert blend_transfer(90.0, st, hooks, cfg) == pytest.approx(0.75 * 90 + 0.25 * 100)


def test_blend_is_inert_without_a_hook_or_weight():
    from maxgaffer.core.director import Hooks, MatchConfig, blend_transfer
    from maxgaffer.core.genome import LightingState

    st = LightingState()
    no_hook = Hooks(apply=lambda s: None, render=lambda t: None, stats=lambda p: None,
                    llm_deltas=lambda c: "")
    assert blend_transfer(90.0, st, no_hook, MatchConfig(transfer_weight=0.25)) == 90.0
    with_hook = Hooks(apply=lambda s: None, render=lambda t: None, stats=lambda p: None,
                      llm_deltas=lambda c: "", transfer=lambda s: 0.0)
    assert blend_transfer(90.0, st, with_hook, MatchConfig()) == 90.0   # weight 0
    # a hook that raises or returns None must never break scoring
    boom = Hooks(apply=lambda s: None, render=lambda t: None, stats=lambda p: None,
                 llm_deltas=lambda c: "", transfer=lambda s: 1 / 0)
    assert blend_transfer(90.0, st, boom, MatchConfig(transfer_weight=0.25)) == 90.0


# ------------------------------------------------ the rig must not delete its own criteria
def _sunlit_ref():
    return {"sun_active": True, "sun_bearing_deg": -60.0, "sun_altitude_band": "golden",
            "wb_kelvin_estimate": 4800.0, "light_quality": "hard",
            "atmosphere": "light_haze"}


def _rig(**kw):
    st = LightingState()
    base = {"sun.enabled": 1.0, "sun.azimuth_deg": 105.0, "sun.altitude_deg": 10.0,
            "sun.size": 1.0, "exposure.wb_kelvin": 4800.0, "atmosphere.distance_m": 250.0}
    base.update(kw)
    for k, v in base.items():
        st.set(k, v)
    return st


def test_switching_the_sun_off_cannot_buy_a_better_transfer_score():
    """Renormalising over MEASURABLE parts is right when the REFERENCE is silent and an
    exploit when the CANDIDATE decides what is measurable. Measured on-box 2026-07-25:
    direction, elevation and hardness are all gated on the rig's sun being on, so switching
    it off removed the three criteria the search was failing and left only ones it already
    satisfied — sun-off scored 83.33 against 79.09 for aiming 64 degrees wrong. Blended into
    the objective at 0.25 that paid the search to go sunless and it did, finishing at 80.35
    with sun.enabled = 0. Giving up must never score better than trying and missing."""
    ref = _sunlit_ref()
    aimed = transfer.score(ref, _rig(), 165.0)["score"]
    missed = transfer.score(ref, _rig(**{"sun.azimuth_deg": 41.0}), 165.0)["score"]
    off = transfer.score(ref, _rig(**{"sun.enabled": 0.0}), 165.0)["score"]
    assert aimed > missed > off, (aimed, missed, off)
    assert off < 40.0, "a sunless rig for a sunlit reference must score badly, not decently"


def test_sun_criteria_are_scored_zero_not_skipped_when_the_sun_is_off():
    got = transfer.score(_sunlit_ref(), _rig(**{"sun.enabled": 0.0}), 165.0)
    for part in ("direction", "elevation", "hardness"):
        assert part in got["measurable"], f"{part} vanished from the denominator"
        assert got["parts"][part] == 0.0


def test_reference_silence_still_renormalises():
    """The other half of the rule: when the REFERENCE carries no reading there is genuinely
    nothing to compare, and those parts must stay out of the denominator rather than score
    zero against a value nobody supplied."""
    bare = {"sun_active": True}          # no bearing, no band, no quality, no wb, no haze
    got = transfer.score(bare, _rig(), 165.0)
    assert got["measurable"] == ["presence"]
    assert got["score"] == 100.0


def test_a_sunless_reference_does_not_demand_sun_criteria():
    dull = {"sun_active": False, "wb_kelvin_estimate": 6500.0, "atmosphere": "none"}
    got = transfer.score(dull, _rig(**{"sun.enabled": 0.0, "atmosphere.distance_m": 4000.0,
                                       "exposure.wb_kelvin": 6500.0}), 165.0)
    assert "direction" not in got["measurable"]     # no bearing was read to compare against
    assert got["parts"]["presence"] == 1.0
    assert got["score"] == 100.0


# ------------------------------------------------ trust the reading as far as it agrees
def test_circular_median_survives_one_wild_sample():
    """ANALYZE's bearing was consolidated with a circular MEAN, and a mean has no outlier
    rejection: [60, 70, -50] averages to 35 degrees when two of three samples agree within
    ten. That injected error is the size of the sun-placement misses measured on-box, and
    it matters far more now the bearing carries a quarter of the objective."""
    from maxgaffer.core import consensus

    assert consensus._circular_mean_deg([60.0, 70.0, -50.0]) == pytest.approx(35.0, abs=1)
    assert consensus._circular_median_deg([60.0, 70.0, -50.0]) == pytest.approx(60.0, abs=1)
    # and the wrap the original docstring rightly worried about still behaves
    assert abs(consensus._circular_median_deg([-170.0, 170.0, 0.0])) > 150.0


def test_consensus_reports_bearing_scatter_and_is_not_masked_by_the_clock():
    """Agreement was measured on time_of_day ALONE, so the bearing could scatter 130 degrees
    across reads of one image while every sample said 'morning' — agreement 1.0, no warning,
    and nothing downstream knew the direction was a coin flip."""
    from maxgaffer.core import consensus

    def sample(bearing):
        return {"time_of_day": "morning", "sky": "clear", "sun_active": True,
                "sun_bearing_deg": bearing, "wb_kelvin_estimate": 5500.0,
                "confidence": 0.8}

    scattered = consensus.consolidate_analyses([sample(45.0), sample(-52.5), sample(77.6)])
    assert scattered["sun_bearing_spread_deg"] > 40.0
    assert scattered["sun_bearing_agreement"] < 0.3
    assert scattered["consensus_agreement"] < 0.3, "unanimous clock masked a contested sun"

    agreed = consensus.consolidate_analyses([sample(64.0), sample(65.0), sample(64.5)])
    assert agreed["sun_bearing_spread_deg"] < 5.0
    assert agreed["sun_bearing_agreement"] > 0.9
    assert agreed["consensus_agreement"] > 0.9


def test_transfer_weight_scales_with_bearing_agreement_but_never_to_zero():
    import inspect

    from maxgaffer.core.director import TRANSFER_WEIGHT
    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    assert "TRANSFER_WEIGHT * bearing_trust" in src
    assert "max(0.25," in src, ("the floor must not be zero — pixels are BLIND to sun "
                               "direction, so a contested reading still beats none")
    assert TRANSFER_WEIGHT == 0.25


def test_the_sweep_measurement_updates_the_belief_the_loop_defends():
    """ANALYZE estimates the bearing from ONE image — a hard absolute judgement, and
    measurably unreliable (four reads of one reference: 45.0, -52.5, 77.6, 64.9; on A2 the
    sign came out backwards and put the sun 19.4 degrees off, exactly twice the 9.7 it
    reported). The sweep renders candidate directions in the real scene and picks
    comparatively, which is a far easier judgement. Reading is the prior, sweep is the
    measurement, transfer defends the result — otherwise the objective spends the match
    pulling the sun back off the swept direction and onto the noisy estimate."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    sweep_at = src.index('start.set("sun.azimuth_deg", az)')
    tail = src[sweep_at:sweep_at + 1800]
    assert 'sem_live["sun_bearing_deg"]' in tail, "sweep result never reaches the objective"
    assert 'sem_live["sun_bearing_agreement"] = 1.0' in tail


def test_the_sweep_never_writes_its_answer_into_the_cached_reading():
    """e.semantics is the record of what ANALYZE actually read. Laundering a sweep-derived
    bearing into it would corrupt every later run of the camera — and would make a cached
    read look unanimous when it never was."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    assert "sem_live = dict(semantics)" in src, "the sweep must mutate a COPY"
    sweep_at = src.index('start.set("sun.azimuth_deg", az)')
    tail = src[sweep_at:sweep_at + 1800]
    assert 'semantics["sun_bearing_deg"]' not in tail
    assert "e.semantics[" not in tail
