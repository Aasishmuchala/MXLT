"""Dynamic polish axes — groups + fog join the coordinate line search (2026-07-24).

The archetype matrix measured two axis gaps: light-GROUP recovery rode on the LLM alone
(an aerial city never re-lit its 4 zones), and the fog solve could land in the right
haze bucket but never fine-tune. polish_axes() now appends one log2 axis per group on
the state plus atmosphere.distance_m when present, and run_polish gates membership with
the group-aware _has_axis. These tests drive run_polish over a synthetic, render-free
landscape (critic.score monkeypatched onto the applied state) to prove the new axes are
actually climbed.
"""
import pytest

from maxgaffer.core import director
from maxgaffer.core.director import (POLISH_PARAMS, _has_axis, polish_axes, run_polish,
                                     Hooks, MatchConfig)
from maxgaffer.core.genome import GROUP_PREFIX, LightingState


# ------------------------------------------------------------------ axis composition
def test_polish_axes_static_only_without_groups_or_fog():
    st = LightingState()
    st.set("exposure.ev", 10.0)
    assert polish_axes(st) == tuple(POLISH_PARAMS)


def test_polish_axes_appends_groups_and_fog():
    st = LightingState()
    st.set("exposure.ev", 10.0)
    st.set("atmosphere.distance_m", 200.0)
    st.groups["practicals"] = 0.4
    st.groups["accents"] = 1.0
    axes = polish_axes(st)
    keys = [k for k, _s, _l, _f in axes]
    assert keys[:len(POLISH_PARAMS)] == [k for k, _s, _l, _f in POLISH_PARAMS]
    assert GROUP_PREFIX + "accents" in keys and GROUP_PREFIX + "practicals" in keys
    assert "atmosphere.distance_m" in keys
    # dynamic axes are log2 (dimmer factors / fog distance are perceptually log)
    for k, _s, is_log, _f in axes:
        if k.startswith(GROUP_PREFIX) or k == "atmosphere.distance_m":
            assert is_log


def test_has_axis_is_group_aware():
    st = LightingState()
    st.set("exposure.ev", 10.0)
    st.groups["practicals"] = 1.0
    assert _has_axis(st, "exposure.ev")
    assert _has_axis(st, GROUP_PREFIX + "practicals")
    assert not _has_axis(st, GROUP_PREFIX + "nope")
    assert not _has_axis(st, "sun.azimuth_deg")


def test_new_compensation_pairs_present():
    assert ("exposure.ev", "sun.intensity") in director._POLISH_PAIRS
    assert ("sun.turbidity", "exposure.wb_kelvin") in director._POLISH_PAIRS


# ------------------------------------------------------------------ synthetic climb
class _World:
    """Render-free polish world: hooks.apply records the candidate; the patched
    critic.score scores THAT state directly, so run_polish's climb is deterministic."""

    def __init__(self, monkeypatch, score_fn):
        self.applied = None
        self.hooks = Hooks(
            apply=self._apply, render=lambda tag: "probe.png",
            stats=lambda path: {"ok": 1}, llm_deltas=lambda ctx: "",
            log=lambda m: None, should_cancel=lambda: False)

        class _V:
            def __init__(self, score):
                self.score = score
                self.components = {}

        monkeypatch.setattr(director.critic, "score",
                            lambda ref, cur, weights=None: _V(score_fn(self.applied)))

    def _apply(self, state):
        self.applied = state.copy()


def _cfg(**kw):
    base = dict(polish=True, polish_rounds=8, polish_min_gain=0.03,
                polish_stop_at=99.5, polish_round_eps=0.2, polish_max_probes=120)
    base.update(kw)
    return MatchConfig(**base)


def test_run_polish_climbs_a_group_axis(monkeypatch):
    """Score peaks when group 'practicals' == 1.0; it starts scrambled at 0.06. The old
    static-axis polish could NEVER move it — the dynamic axis must climb it."""
    def score_fn(state):
        g = state.get(GROUP_PREFIX + "practicals") if state else 0.0
        return 90.0 - 40.0 * abs(g - 1.0)

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.groups["practicals"] = 0.06
    best, score, probes, _conv, _proven = run_polish(st, score_fn(st), {"grid": [0] * 9},
                                                     w.hooks, _cfg())
    assert best.groups["practicals"] == pytest.approx(1.0, abs=0.25)
    assert score > 85.0
    assert probes > 0


def test_run_polish_climbs_the_fog_axis(monkeypatch):
    """Score peaks at atmosphere.distance_m == 20 (dense haze); starts at 320 (clear).
    A log2 axis must walk it down across four octaves."""
    import math

    def score_fn(state):
        d = state.get("atmosphere.distance_m", 320.0) if state else 320.0
        return 95.0 - 12.0 * abs(math.log2(max(1e-3, d / 20.0)))

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.set("atmosphere.distance_m", 320.0)
    best, score, probes, _conv, _proven = run_polish(st, score_fn(st), {"grid": [0] * 9},
                                                     w.hooks, _cfg())
    assert best.get("atmosphere.distance_m") == pytest.approx(20.0, rel=0.6)
    assert score > 88.0


def test_run_polish_respects_group_locks(monkeypatch):
    def score_fn(state):
        g = state.get(GROUP_PREFIX + "practicals") if state else 0.0
        return 90.0 - 40.0 * abs(g - 1.0)

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.groups["practicals"] = 0.06
    best, _score, _probes, _c, _p = run_polish(
        st, score_fn(st), {"grid": [0] * 9}, w.hooks, _cfg(),
        locks={GROUP_PREFIX + "practicals"})
    assert best.groups["practicals"] == pytest.approx(0.06)
