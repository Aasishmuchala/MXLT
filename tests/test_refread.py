"""Measuring the reference instead of guessing at it. ANALYZE is the plugin's least
reliable component — four reads of one image gave sun bearings 130 degrees apart — and most
of what it guesses is measurable from the pixels."""

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

    src = inspect.getsource(Controller.analyze_reference)
    assert "refread.fuse(semantics, refread.measure(" in src
    cache_at = src.index("e.semantics = semantics")
    fuse_at = src.index("refread.fuse(")
    assert cache_at < fuse_at, "the cache must be written from the UNFUSED reading"
