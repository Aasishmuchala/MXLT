import pytest

from maxgaffer.core.profiles import resolve_profile


def test_standard_profile_preserves_user_baseline():
    p = resolve_profile("standard", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert (p.loop_width, p.loop_height, p.max_iterations, p.target_score) == (
        480, 270, 5, 82)
    assert (p.sweep_width, p.sweep_height) == (240, 135)
    assert not p.polish


def test_fast_profile_only_reduces_cost():
    p = resolve_profile("fast", loop_width=480, loop_height=270,
                        max_iterations=9, sweep_count=12, target_score=95)
    assert (p.loop_width, p.loop_height) == (320, 180)
    assert p.max_iterations == 3 and p.sweep_count == 4 and p.target_score == 78
    assert p.worst_case_renders == 7


def test_hero_profile_has_strict_sub_100_render_cap():
    p = resolve_profile("hero", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert p.polish and p.polish_max_probes == 48 and p.target_score == 99
    assert p.worst_case_renders == 62


def test_deep_is_hero_alias_and_unknown_is_rejected():
    p = resolve_profile("deep", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
    assert p.name == "hero"
    with pytest.raises(ValueError, match="unknown match profile"):
        resolve_profile("infinite", loop_width=480, loop_height=270,
                        max_iterations=5, sweep_count=8, target_score=82)
