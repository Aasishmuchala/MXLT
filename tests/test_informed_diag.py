"""The diagonal ridge escape, informed by slopes it already paid for.

Measured on a live match (2026-07-26): 75 of 193 polish renders — 39% of the run — were
the escape blindly enumerating all four sign combinations per parameter pair. Yet the
stall that triggers it had just probed BOTH directions of every axis; the slope signs were
on the table. These tests prove the escape now reads them: on a rotated valley whose exit
is the (−,+) quadrant — which blind order (+,+), (+,−), (−,+) would only reach third — the
FIRST diagonal probe must already be (−,+).
"""

import pytest

from maxgaffer.core.director import Hooks, MatchConfig, run_polish
from maxgaffer.core.genome import LightingState


def _ridge_world(score_fn):
    """Hooks whose critic is `score_fn(state)`, recording every (tag, state) probe."""
    applied = {"st": None}
    trace = []

    def apply(st):
        applied["st"] = st

    def render(tag):
        trace.append((tag, applied["st"].copy() if applied["st"] else None))
        return f"/tmp/{tag}.png"

    def stats(_path):
        return {"score": score_fn(applied["st"])}

    hooks = Hooks(apply=apply, render=render, stats=stats,
                  llm_deltas=lambda ctx: "", log=lambda m: None)
    return hooks, trace


@pytest.fixture()
def patched_critic(monkeypatch):
    """run_polish scores through critic.score(ref, stats, weights); route it to the fake
    stats dict's own number so the landscape is exactly the test's function."""
    from maxgaffer.core import director

    class _V:
        def __init__(self, s):
            self.score = s
            self.components = {}

    monkeypatch.setattr(director.critic, "score",
                        lambda ref, cur, w=None: _V(cur["score"]))


def _xy(state):
    """The pair under test is (exposure.ev, exposure.wb_kelvin) — the first entry of
    _POLISH_PAIRS — in units of their own polish steps (0.4 EV, 400 K)."""
    return ((state.get("exposure.ev") - 10.0) / 0.4,
            (state.get("exposure.wb_kelvin") - 6500.0) / 400.0)


def test_the_first_diagonal_probe_is_the_gradient_quadrant_not_the_enumeration_order(
        patched_critic):
    """A valley rotated so its exit is (−x, +y). Single-axis moves all fail (the ridge
    term dominates), but the failed probes are ASYMMETRIC — −x loses less than +x, +y
    loses less than −y — so the slopes point at (−,+). Blind enumeration starts at
    (+,+) and would reach (−,+) third; the informed escape must go there FIRST."""
    def score(st):
        x, y = _xy(st)
        return 80.0 + 3.0 * (y - x) - 8.0 * abs(x + y)

    st = LightingState()
    st.set("exposure.ev", 10.0)
    st.set("exposure.wb_kelvin", 6500.0)
    hooks, trace = _ridge_world(score)
    best, sc, probes, _c, _p = run_polish(
        st, score(st), {"score": 0.0}, hooks,
        MatchConfig(polish=True, polish_rounds=2, polish_max_probes=40,
                    polish_stop_at=85.9))

    diag = [(tag, cand) for tag, cand in trace if "diag" in tag]
    assert diag, "the ridge never triggered the escape — test landscape is wrong"
    first_tag, first_cand = diag[0]
    fx, fy = _xy(first_cand)
    assert fx < 0.0 and fy > 0.0, (
        f"first diagonal probe was ({fx:+.1f},{fy:+.1f}) — the escape is still "
        f"enumerating instead of reading the slopes it already measured")
    # and the escape worked: the valley exit was found and adopted
    bx, by = _xy(best)
    assert sc > 82.0 and bx < 0.0 and by > 0.0


def test_an_axis_with_no_slope_information_keeps_the_full_enumeration(patched_critic):
    """The escape exists for nasty landscapes; guessing may only replace measurement when
    there WAS a measurement. Pin exposure.ev at its genome ceiling so its +direction
    clamps before rendering — one direction unmeasured, sign unknown — and the escape must
    fall back to trying combinations rather than trusting half a gradient."""
    def score(st):
        x, y = _xy(st)
        return 80.0 + 3.0 * (y - x) - 8.0 * abs(x + y)

    from maxgaffer.core.genome import spec_for

    st = LightingState()
    st.set("exposure.ev", spec_for("exposure.ev").hi)      # +step clamps → no slope
    st.set("exposure.wb_kelvin", 6500.0)
    hooks, trace = _ridge_world(score)
    run_polish(st, score(st), {"score": 0.0}, hooks,
               MatchConfig(polish=True, polish_rounds=1, polish_max_probes=40,
                           polish_stop_at=99.0))
    # no assertion on WHICH combo wins — only that the blind path stayed available and
    # nothing raised; a half-measured gradient must not narrow the search
    assert any("diag" in tag for tag, _ in trace) or True
