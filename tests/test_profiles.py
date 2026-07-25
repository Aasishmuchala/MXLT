import pytest

from maxgaffer.core.profiles import resolve_profile


def test_standard_profile_preserves_user_baseline():
    """Standard still honours the user's resolution, iterations and target verbatim — but
    it now carries a MODEST finisher. It used to have none, so a default match handed back
    whatever the loop happened to land on; Fast remains the no-polish option."""
    p = resolve_profile("standard", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert (p.loop_width, p.loop_height, p.max_iterations, p.target_score) == (
        480, 270, 5, 82)
    assert (p.sweep_width, p.sweep_height) == (240, 135)
    assert p.polish and (p.polish_rounds, p.polish_max_probes) == (8, 120)
    assert not resolve_profile("fast", loop_width=480, loop_height=270,
                               max_iterations=5, sweep_count=8,
                               target_score=82).polish


def test_fast_profile_only_reduces_cost():
    p = resolve_profile("fast", loop_width=480, loop_height=270,
                        max_iterations=9, sweep_count=12, target_score=95)
    assert (p.loop_width, p.loop_height) == (320, 180)
    assert p.max_iterations == 3 and p.sweep_count == 4 and p.target_score == 78
    assert p.worst_case_renders == 7


def test_hero_profile_has_strict_finite_render_cap():
    # polish budget deliberately raised 48 -> 160 (2026-07-24: the 48-probe cap was
    # measured exhausted with gains still coming, and the axis list is now dynamic —
    # groups + fog — so the budget covers more parameters). The cap stays FINITE and
    # explicit; that property, not the old constant, is what this test locks.
    p = resolve_profile("hero", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert p.polish and p.polish_max_probes == 500 and p.target_score == 99
    assert p.worst_case_renders == 8 + 6 + 500


def test_deep_is_hero_alias_and_unknown_is_rejected():
    p = resolve_profile("deep", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert p.name == "hero"
    with pytest.raises(ValueError, match="unknown match profile"):
        resolve_profile("infinite", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)


def test_stall_patience_scales_with_profile_depth():
    """The loop is where GEOMETRY is solved (polish only refines the basin it is handed).
    Measured 2026-07-25: MatchConfig's default patience of 2 ended hero runs after 3 of
    their 10 iterations on a single dip, leaving 70% of the loop budget unused."""
    kw = dict(loop_width=480, loop_height=270, max_iterations=10, sweep_count=8,
              target_score=82)
    assert resolve_profile("fast", **kw).stall_patience == 2
    assert resolve_profile("standard", **kw).stall_patience == 3
    assert resolve_profile("hero", **kw).stall_patience == 5
