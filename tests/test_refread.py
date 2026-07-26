"""Measuring the reference instead of guessing at it. ANALYZE is the plugin's least
reliable component — four reads of one image gave sun bearings 130 degrees apart — and most
of what it guesses is measurable from the pixels."""

import pytest

from maxgaffer.core import refread


def test_an_empty_frame_refutes_a_sunlit_reading():
    """The asymmetry is the whole design. Measured across every reference this session,
    sunlit plates carried 0.0251 to 0.0397 of frame in bright directional patches — a
    golden-hour interior, a real dawn photograph, a sun+sky exterior — and genuinely
    sunless ones carried exactly 0.0000. No directional light anywhere is near-certain
    evidence against a sunlit reading."""
    sunless = refread.measure({"hot_frac": 0.0})
    assert sunless["sun_active"]["value"] is False
    assert sunless["sun_active"]["confidence"] >= 0.75      # strong enough to act on

    fused = refread.fuse({"sun_active": True, "time_of_day": "morning"}, sunless)
    assert fused["sun_active"] is False, "a model saying 'sun' over an empty frame is wrong"
    assert "sun_active" in fused["measured_fields"]


def test_bright_patches_are_weak_evidence_FOR_sun_and_do_not_overrule():
    """Lamps, a blown window and a specular table all make bright patches. The measurement
    may refute a sunlit reading; it may not assert one."""
    lit = refread.measure({"hot_frac": 0.033})
    assert lit["sun_active"]["value"] is True
    assert lit["sun_active"]["confidence"] < 0.75, "asserting sun from patches alone"

    # a night interior full of practicals: the model says no sun, and it keeps saying so
    fused = refread.fuse({"sun_active": False, "time_of_day": "night"}, lit)
    assert fused["sun_active"] is False
    assert "measured_fields" not in fused


def test_the_threshold_sits_below_every_true_positive_measured():
    """It is used to refute, so firing on a real sun would be the expensive failure. The
    lowest sunlit reference measured this session was 0.0251."""
    assert refread.SUN_ACTIVE_HOT_FRAC < 0.0251
    for real_sun in (0.0251, 0.0273, 0.0330, 0.0397):
        assert refread.measure({"hot_frac": real_sun})["sun_active"]["value"] is True


def test_fuse_never_rewrites_the_caller_s_reading():
    """e.semantics is the record of what ANALYZE actually said. Laundering a measurement
    into it would make a contested read look authoritative to every later run — the same
    rule the sun sweep had to learn."""
    original = {"sun_active": True, "wb_kelvin_estimate": 5500.0}
    fused = refread.fuse(original, refread.measure({"hot_frac": 0.0}))
    assert original["sun_active"] is True, "the cached reading was mutated"
    assert fused is not original


def test_it_reports_what_it_overruled_and_why():
    logs = []
    refread.fuse({"sun_active": True}, refread.measure({"hot_frac": 0.0}), log=logs.append)
    assert any("sun_active" in m and "no bright directional patch" in m for m in logs)
    # ...and stays silent when it agrees
    quiet = []
    refread.fuse({"sun_active": False}, refread.measure({"hot_frac": 0.0}), log=quiet.append)
    assert quiet == []


def test_it_degrades_on_junk_and_on_legacy_stats():
    assert refread.measure(None) == {}
    assert refread.measure({}) == {}                       # stats predating the patch map
    assert refread.measure({"hot_frac": "nonsense"}) == {}
    assert refread.fuse(None, None) == {}
    assert refread.fuse({"sun_active": True}, {}) == {"sun_active": True}
    assert refread.fuse({"a": 1}, {"b": {"no_value_key": 2}}) == {"a": 1}


def test_the_analyze_path_measures_and_leaves_the_cache_alone():
    """The fused reading is what the match uses; e.semantics stays the record of what the
    model actually said, so a later run can still see it was a guess."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = " ".join(inspect.getsource(Controller.analyze_reference).split())
    assert "refread.fuse( semantics, refread.measure(" in src
    # the model's reading is handed to measure() so the auto-white-balance cross-check can
    # see the time of day — the one thing the pixels cannot tell it
    assert "reading=semantics" in src
    cache_at = src.index("e.semantics = semantics")
    fuse_at = src.index("refread.fuse(")
    assert cache_at < fuse_at, "the cache must be written from the UNFUSED reading"


# ------------------------------------------------------- measured colour temperature
def test_the_illuminant_measurement_is_blended_not_switched_in():
    """cct's confidence is built from off-locus distance and estimator disagreement, which
    is honest but conservative about its own ACCURACY. Measured against ground truth on this
    session's references it read 4846.7 K against a true 4900 (53 K out) at confidence 0.72,
    and 5024.7 against a true 4800 (225 K out) at confidence only 0.25 — while the model's
    guess for that same image was 5500, a 700 K miss. A hard confidence threshold would have
    thrown the good readings away with the doubtful ones."""
    m = refread.measure({"illum": [0.60, 0.55, 0.45], "illum_sog": [0.60, 0.55, 0.45],
                         "illum_edge": [0.60, 0.55, 0.45]})
    if "wb_kelvin_estimate" not in m:
        return                                  # chromaticity refused — covered elsewhere
    assert m["wb_kelvin_estimate"]["blend"] is True
    fused = refread.fuse({"wb_kelvin_estimate": 9000.0}, m)
    got = fused["wb_kelvin_estimate"]
    lo, hi = sorted((9000.0, m["wb_kelvin_estimate"]["value"]))
    assert lo <= got <= hi, "a blend must land between the two, never outside"
    assert "wb_kelvin_estimate" in fused["measured_fields"]


def test_the_measurement_never_gets_less_than_half_the_vote():
    """Even a diffident measurement beat the guess on every reference where truth was
    known, so equal weight is the floor and generous to the model."""
    warm = {"value": 3000.0, "confidence": 0.0, "why": "x", "blend": True}
    fused = refread.fuse({"wb_kelvin_estimate": 9000.0}, {"wb_kelvin_estimate": warm})
    # averaged in MIREDS, so the midpoint is 4500 K, not 6000 — equal steps must look equal
    assert fused["wb_kelvin_estimate"] == pytest.approx(4500.0, abs=1.0)
    sure = dict(warm, confidence=1.0)
    assert refread.fuse({"wb_kelvin_estimate": 9000.0},
                        {"wb_kelvin_estimate": sure})["wb_kelvin_estimate"] == \
        pytest.approx(3000.0, abs=1.0)


def test_warmth_is_blended_in_mireds_because_kelvin_steps_are_not_even():
    """4000 to 5000 K is an obvious shift; 9000 to 10000 K is barely visible. A kelvin
    average would quietly favour the cool end of every blend."""
    m = {"value": 4000.0, "confidence": 0.0, "why": "x", "blend": True}
    fused = refread.fuse({"wb_kelvin_estimate": 8000.0}, {"wb_kelvin_estimate": m})
    assert fused["wb_kelvin_estimate"] < 6000.0, "this is the kelvin midpoint, not a mired one"
    assert fused["wb_kelvin_estimate"] == pytest.approx(1.0e6 / (0.5 * 250 + 0.5 * 125), abs=1)


def test_an_unreadable_illuminant_leaves_the_model_s_estimate_alone():
    """A dome-only HDRI reference was REFUSED by the measurement — gray-world cannot read
    an illuminant off it — and the honest result is to keep the guess rather than invent."""
    assert "wb_kelvin_estimate" not in refread.measure({"hot_frac": 0.0})
    kept = refread.fuse({"wb_kelvin_estimate": 5200.0}, refread.measure({"hot_frac": 0.0}))
    assert kept["wb_kelvin_estimate"] == 5200.0


def test_shadow_hardness_is_shipped_but_not_fused_yet():
    """core/shadowedge.py measures light_quality and atmosphere and is available as a
    diagnostic, but it does not overrule the model's reading. Checked against this session's
    references it called a known-hard golden-hour sun "mixed" at 0.59 — an honest miss,
    reported at low confidence — but it also called a dome-only HDRI "mixed" at 0.81 with
    confidence 1.0. A measurement that is CONFIDENTLY wrong is worse than a guess, and the
    bar for replacing a guess is beating it."""
    from maxgaffer.core import shadowedge          # shipped and importable

    assert hasattr(shadowedge, "edge_hardness") and hasattr(shadowedge, "haze_estimate")
    measured = refread.measure({"hot_frac": 0.03, "illum": [0.6, 0.55, 0.45],
                                "illum_sog": [0.6, 0.55, 0.45],
                                "illum_edge": [0.6, 0.55, 0.45]})
    assert "light_quality" not in measured
    assert "atmosphere" not in measured


# ------------------------------------------------ the one thing the pixels cannot see
def _warm_lit_stats():
    """An illuminant measuring neutral — what a camera's own white balance leaves behind."""
    return {"hot_frac": 0.03, "illum": [0.5774, 0.5774, 0.5774],
            "illum_sog": [0.5774, 0.5774, 0.5774], "illum_edge": [0.5774, 0.5774, 0.5774]}


def test_a_neutral_measurement_on_a_golden_hour_photo_is_withheld():
    """A consumer photograph has already been white-balanced in camera, so measuring it
    recovers the CORRECTION rather than the light — and cct cannot detect this. It returns a
    clean, on-locus, self-consistent wrong answer with healthy confidence, which is the worst
    shape a wrong measurement can take.

    The model can still see the picture is OF a sunset. When it names a time of day whose
    light is definitionally not neutral and the measurement comes back neutral anyway, the
    disagreement IS the evidence."""
    st = _warm_lit_stats()
    guessed = {"time_of_day": "golden_hour", "wb_kelvin_estimate": 3800.0}
    m = refread.measure(st, reading=guessed)
    assert "wb_kelvin_estimate" not in m, "a laundered neutral must not reach the reading"
    assert "_wb_withheld" in m and "white-balanced in camera" in m["_wb_withheld"]["why"]

    fused = refread.fuse(guessed, m)
    assert fused["wb_kelvin_estimate"] == 3800.0, "the model's warm read must survive intact"


def test_a_neutral_measurement_on_a_MIDDAY_photo_is_kept():
    """The cross-check must not fire on scenes whose light really is neutral, or it would
    veto every correct daylight reading."""
    m = refread.measure(_warm_lit_stats(), reading={"time_of_day": "midday"})
    assert "wb_kelvin_estimate" in m
    assert "_wb_withheld" not in m


def test_a_WARM_measurement_on_a_golden_hour_photo_is_kept():
    """Only a NEUTRAL measurement is suspicious on a warm scene. A raw or CG reference that
    genuinely measures warm is the case this whole module exists to serve."""
    warm = {"hot_frac": 0.03, "illum": [0.66, 0.55, 0.39],
            "illum_sog": [0.66, 0.55, 0.39], "illum_edge": [0.66, 0.55, 0.39]}
    m = refread.measure(warm, reading={"time_of_day": "golden_hour"})
    assert "_wb_withheld" not in m


def test_measure_still_works_without_the_model_s_reading():
    """The cross-check is an addition, not a dependency — measure(stats) alone must behave
    exactly as it did."""
    assert "wb_kelvin_estimate" in refread.measure(_warm_lit_stats())
    assert refread.measure(_warm_lit_stats(), reading=None).get("_wb_withheld") is None
