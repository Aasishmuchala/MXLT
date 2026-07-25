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


def test_group_axis_uses_a_full_doubling_initial_step():
    """A zeroed group seeds from the 0.06 floor; a 0.35 first step (0.06 -> 0.077) is
    visually invisible and can never clear min_gain from a wide shot (the measured
    aerial-city failure). The group axis must open with a FULL doubling."""
    st = LightingState()
    st.groups["zone"] = 0.0
    axes = {k: (s, is_log, f) for k, s, is_log, f in polish_axes(st)}
    step, is_log, _floor = axes[GROUP_PREFIX + "zone"]
    assert is_log and step == pytest.approx(1.0)


def test_run_polish_escapes_a_zeroed_group_in_few_probes(monkeypatch):
    """From group == 0.0 (log axis seeds at the floor), the doubling step + 1.6x ride
    must reach ~1.0 within a handful of probes — the fast-escape guarantee."""
    def score_fn(state):
        g = state.get(GROUP_PREFIX + "practicals") if state else 0.0
        import math
        return 90.0 - 18.0 * abs(math.log2(max(g, 1e-3) / 1.0))

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.groups["practicals"] = 0.0
    best, score, probes, _c, _p = run_polish(st, score_fn(st), {"grid": [0] * 9},
                                             w.hooks, _cfg())
    assert best.groups["practicals"] == pytest.approx(1.0, rel=0.6)
    assert score > 80.0
    assert probes <= 30                       # few probes, not a 7-doubling crawl


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


# ------------------------------------------------------------------ discrete flag escape
def test_polish_can_switch_a_disabled_sun_back_on(monkeypatch):
    """sun.enabled is a 0/1 switch with no gradient, so the line search cannot touch it.
    Measured on-box (A6, 2026-07-25): the loop turned the sun OFF and polish then
    perfected a SUNLESS metamer of a sunlit reference — envelope collapsed to 0.12 and
    the state was unrecoverable. Polish must be able to flip it back."""
    def score_fn(state):
        if state is None:
            return 40.0
        on = state.get("sun.enabled")
        # nothing else can compensate for the missing key light
        return 92.0 if on >= 0.5 else 40.0

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.set("sun.enabled", 0.0)
    st.set("sun.altitude_deg", -2.0)
    st.set("exposure.ev", 10.0)
    best, score, _probes, _c, _p = run_polish(st, 40.0, {"grid": [0] * 9}, w.hooks, _cfg())
    assert best.get("sun.enabled") >= 0.5, "polish never re-enabled the sun"
    assert score > 90.0


def test_polish_keeps_an_enabled_sun_when_switching_it_off_hurts(monkeypatch):
    def score_fn(state):
        on = state.get("sun.enabled") if state else 1.0
        return 90.0 if on >= 0.5 else 20.0

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.set("sun.enabled", 1.0)
    st.set("exposure.ev", 10.0)
    best, score, _probes, _c, _p = run_polish(st, 90.0, {"grid": [0] * 9}, w.hooks, _cfg())
    assert best.get("sun.enabled") >= 0.5      # the losing flip must be discarded
    assert score >= 90.0


# ------------------------------------------------------------------ azimuth basin hop
def test_polish_hops_to_a_distant_azimuth_basin(monkeypatch):
    """Sun azimuth is multi-modal: a 12-degree line search cannot walk from one lobe to
    another. Measured on-box (A6, 2026-07-25): polish returned proven=True at 84.76 with
    azimuth 179 degrees from target. The hop must find the far lobe."""
    import math

    def score_fn(state):
        if state is None:
            return 50.0
        az = state.get("sun.azimuth_deg")
        # true peak at 115; a decoy local optimum near 295 that a local search settles in
        d_true = min(abs(az - 115.0), 360.0 - abs(az - 115.0))
        d_decoy = min(abs(az - 295.0), 360.0 - abs(az - 295.0))
        return max(98.0 - 0.30 * d_true, 84.0 - 0.30 * d_decoy)

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.set("sun.azimuth_deg", 295.0)          # parked in the decoy basin
    st.set("exposure.ev", 10.0)
    best, score, _probes, _c, _p = run_polish(st, score_fn(st), {"grid": [0] * 9},
                                              w.hooks, _cfg(polish_rounds=14))
    got = best.get("sun.azimuth_deg")
    assert min(abs(got - 115.0), 360.0 - abs(got - 115.0)) < 25.0, f"stuck at {got}"
    assert score > 90.0


def test_azimuth_hops_are_bounded():
    assert director._MAX_HOPS <= 4          # a global assist, not a random walk


# ------------------------------------------------------------------ tonal leash in polish
def test_polish_respects_the_wb_and_ev_leash(monkeypatch):
    """The loop clamps its EV/WB solve to a window around the run's start state, because
    matching histograms of DIFFERENT scenes biases the tonal solve. Polish had no such
    bound and walked white balance to the genome wall: measured on-box 2026-07-25,
    wb_kelvin 14,643 of a 15,000 bound while turbidity went DOWN — warming the image with
    the camera instead of the sun. Polish must obey the same leash."""
    def score_fn(state):
        # a landscape that rewards cranking WB as high as it can go
        return 50.0 + state.get("exposure.wb_kelvin") / 1000.0 if state else 50.0

    w = _World(monkeypatch, score_fn)
    start = LightingState()
    start.set("exposure.wb_kelvin", 6500.0)
    start.set("exposure.ev", 10.0)
    st = start.copy()
    best, _score, _p, _c, _pr = run_polish(st, score_fn(st), {"grid": [0] * 9}, w.hooks,
                                           _cfg(), leash_ref=start)
    # wb_leash defaults to 3000 => the window is [3500, 9500]
    assert best.get("exposure.wb_kelvin") <= 9500.0 + 1e-6, best.get("exposure.wb_kelvin")


def test_polish_without_a_leash_ref_is_unbounded_as_before(monkeypatch):
    """Back-compat: callers that pass no leash_ref keep the old unclamped behaviour."""
    def score_fn(state):
        return 50.0 + state.get("exposure.wb_kelvin") / 1000.0 if state else 50.0

    w = _World(monkeypatch, score_fn)
    st = LightingState()
    st.set("exposure.wb_kelvin", 6500.0)
    best, _s, _p, _c, _pr = run_polish(st, score_fn(st), {"grid": [0] * 9}, w.hooks, _cfg())
    assert best.get("exposure.wb_kelvin") > 9500.0


# ------------------------------------------------------------------ tonal re-solve
def test_polish_tonal_resolve_escapes_a_coupled_optimum(monkeypatch):
    """The tonal group (ev/wb/dome/turbidity) can sit in a COUPLED local optimum where the
    truth needs all of them to move at once and any one alone goes downhill. Measured
    on-box 2026-07-25 with geometry already exact: polish probed turbidity 7x, EV 6x and
    WB 5x over 8 rounds and kept the scrambled values, gaining +1.02. The analytic solver
    COMPUTES ev/wb rather than walking to them, so re-solving is the coordinated jump."""
    target = {"ev": -4.0, "dome": 0.55}

    def score_fn(state):
        if state is None:
            return 50.0
        ev, dome = state.get("exposure.ev"), state.get("dome.intensity")
        de, dd = abs(ev - target["ev"]), abs(dome - target["dome"])
        # a ridge: only moving BOTH together pays; either alone is worse than staying
        return 95.0 - 12.0 * max(de, dd) - 8.0 * abs(de - dd)

    w = _World(monkeypatch, score_fn)
    # solve_ev returns the computed EV straight away — the jump the line search can't make
    monkeypatch.setattr(director.solver, "solve_ev",
                        lambda ref, cur, now, **kw: target["ev"])
    monkeypatch.setattr(director.solver, "solve_wb", lambda ref, cur, now, **kw: None)
    st = LightingState()
    st.set("exposure.ev", -2.87)          # the measured stuck values
    st.set("dome.intensity", 1.71)
    best, score, _p, _c, _pr = run_polish(st, score_fn(st), {"grid": [0] * 9}, w.hooks,
                                          _cfg(polish_rounds=10))
    assert best.get("exposure.ev") == pytest.approx(target["ev"], abs=0.5)
    assert best.get("dome.intensity") < 1.2, best.get("dome.intensity")
    assert score > score_fn(st) + 5.0


def test_tonal_resolve_runs_at_most_once_per_polish(monkeypatch):
    """One coordinated jump per run — it must not become a per-round reset loop."""
    calls = []
    monkeypatch.setattr(director.solver, "solve_ev",
                        lambda ref, cur, now, **kw: calls.append(1) or None)
    monkeypatch.setattr(director.solver, "solve_wb", lambda ref, cur, now, **kw: None)
    w = _World(monkeypatch, lambda s: 50.0)          # flat: never improves, always stalls
    st = LightingState()
    st.set("exposure.ev", 10.0)
    st.set("dome.intensity", 1.0)
    run_polish(st, 50.0, {"grid": [0] * 9}, w.hooks, _cfg(polish_rounds=12))
    assert len(calls) <= 1, f"tonal re-solve fired {len(calls)} times"
