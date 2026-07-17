"""Deep-match 99 mode — annealing, tightening deadbands, the polish finisher, ceiling honesty."""

import json

from maxgaffer.core import solver
from maxgaffer.core.director import (Hooks, MatchConfig, POLISH_PARAMS, _anneal, run_match,
                                     run_polish)
from maxgaffer.core.genome import LightingState, limit_step

from tests.test_stress_round2 import REF, near


def state_full(az=100.0):
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": az, "sun.altitude_deg": 30.0,
                 "sun.intensity": 1.0, "sun.size": 2.0, "sun.turbidity": 3.0,
                 "dome.enabled": 1, "dome.rotation_deg": 0.0, "dome.intensity": 1.0,
                 "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    return st


# ------------------------------------------------------------------ annealing
def test_annealed_step_limits():
    assert _anneal(None) == 1.0 and _anneal(50) == 1.0
    assert _anneal(75) == 0.5 and _anneal(90) == 0.25
    # azimuth step 60 → at 0.25 scale a 60° request moves only 15°
    assert limit_step("sun.azimuth_deg", 0.0, 60.0, scale=0.25) == 15.0
    assert limit_step("sun.azimuth_deg", 0.0, 60.0) == 60.0
    # log-scale: intensity 1→8 (3 stops) at 0.25 scale limits to 0.25 stops
    assert abs(limit_step("sun.intensity", 1.0, 8.0, scale=0.25) - 2 ** 0.25) < 1e-9


def test_tightened_deadbands_catch_fine_errors():
    # 0.1-stop error: exploration deadband swallows it, the tightened one corrects it
    ref, cur = dict(REF), near(REF["log_key"] / (2 ** 0.1))
    assert solver.solve_ev(ref, cur, 12.0) is None
    tightened = solver.solve_ev(ref, cur, 12.0, tighten=0.25)
    assert tightened is not None and abs(tightened - 11.9) < 0.01
    # WB: 0.8 b* error passes at deadband 1.5, caught at 0.375
    warm = dict(REF); warm["lab_mean"] = [55.0, 2.0, 12.8]
    assert solver.solve_wb(warm, REF, 6500.0) is None
    assert solver.solve_wb(warm, REF, 6500.0, tighten=0.25) is not None


# ------------------------------------------------------------------ polish pass
class PolishWorld:
    """Score = 100 - |azimuth-210|*0.3 - |size-4|*2 (peak at az 210, size 4)."""

    def __init__(self):
        self.current = None
        self.renders = 0

    def hooks(self):
        return Hooks(apply=lambda st: setattr(self, "current", st.copy()),
                     render=lambda tag: (setattr(self, "renders", self.renders + 1)
                                         or f"/tmp/{tag}.png"),
                     stats=lambda p: {"__probe__": True},
                     llm_deltas=lambda ctx: "", log=lambda m: None)

    def score_of(self, st):
        az_err = abs((st.get("sun.azimuth_deg") - 210 + 180) % 360 - 180)
        return 100.0 - az_err * 0.3 - abs(st.get("sun.size") - 4.0) * 2.0


def test_polish_climbs_to_local_optimum_and_proves_ceiling(monkeypatch):
    from maxgaffer.core import critic

    world = PolishWorld()
    monkeypatch.setattr(critic, "score", lambda ref, cur, w=None: critic.Verdict(
        score=world.score_of(world.current), components={}))
    start = state_full(az=195.0)   # 15° off, size 2 (2 units off) → score 91.5
    start.set("sun.size", 2.0)
    st, sc, probes, converged, proven = run_polish(
        start, world.score_of(start), {"any": "ref"}, world.hooks(),
        MatchConfig(polish_rounds=12, polish_stop_at=200.0,
                    polish_max_probes=250))
    assert sc > 99.0                                   # climbed to the peak neighborhood
    assert abs(st.get("sun.azimuth_deg") - 210) <= 3.0
    assert abs(st.get("sun.size") - 4.0) < 1.3
    assert converged is True
    assert probes == world.renders and probes > 10


def test_polish_respects_locks_and_stop_score(monkeypatch):
    from maxgaffer.core import critic

    world = PolishWorld()
    monkeypatch.setattr(critic, "score", lambda ref, cur, w=None: critic.Verdict(
        score=world.score_of(world.current), components={}))
    start = state_full(az=195.0)
    st, sc, probes, converged, _proven = run_polish(
        start, world.score_of(start), {}, world.hooks(),
        MatchConfig(polish_rounds=8, polish_stop_at=200.0),
        locks={"sun.azimuth_deg"})
    assert st.get("sun.azimuth_deg") == 195.0          # locked never moves
    # stop score short-circuits without claiming a converged ceiling — the world must be
    # able to CROSS the stop score for this to trigger (else exhaustion is correct)
    world2 = PolishWorld()
    monkeypatch.setattr(critic, "score", lambda ref, cur, w=None: critic.Verdict(
        score=world2.score_of(world2.current), components={}))
    start2 = state_full(az=209.0)
    start2.set("sun.size", 4.0)               # at the size optimum; az 1° off → 99.7 reachable
    st2, sc2, _, conv2, proven2 = run_polish(
        start2, 99.0, {}, world2.hooks(),
        MatchConfig(polish_rounds=8, polish_stop_at=99.2))
    assert sc2 >= 99.2
    assert conv2 is False and proven2 is False


def test_run_match_wires_polish_and_reports_gain():
    stats_seq = [near(0.10), near(0.12)]
    polish_stats = near(0.19)                          # every polish probe lands closer

    def stats(path):
        return polish_stats if "polish" in path else (
            stats_seq.pop(0) if stats_seq else near(0.12))

    hooks = Hooks(apply=lambda st: None, render=lambda t: f"/tmp/{t}.png", stats=stats,
                  llm_deltas=lambda ctx: json.dumps(
                      {"assessment": "", "changes": [], "stop": False}),
                  log=lambda m: None)
    res = run_match(state_full(), REF, {}, hooks,
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99,
                                polish=True, polish_rounds=1, polish_stop_at=101))
    assert res.polish_probes > 0
    assert res.polish_gain > 0
    assert res.best_score > 90


def test_polish_params_valid_and_floored():
    from maxgaffer.core.genome import spec_for

    for key, init, is_log, floor in POLISH_PARAMS:
        assert spec_for(key) is not None
        assert init > floor > 0
        assert isinstance(is_log, bool)


def test_polish_closes_a_post_sweep_gap():
    """The Phase-C requirement: a 30° azimuth error (one sweep bucket) must be closable —
    the fixed-nudge design couldn't; the adaptive line search must."""
    from maxgaffer.core import critic

    world = PolishWorld()
    real = critic.score
    critic.score = lambda ref, cur, w=None: critic.Verdict(
        score=world.score_of(world.current), components={})
    try:
        start = state_full(az=180.0)          # sweep landed a bucket away from 210
        start.set("sun.size", 4.0)
        st, sc, probes, _conv, _proven = run_polish(
            start, world.score_of(start), {}, world.hooks(),
            MatchConfig(polish_rounds=6, polish_stop_at=99.8, polish_max_probes=60))
        assert abs(st.get("sun.azimuth_deg") - 210.0) <= 3.0
        assert sc >= 99.0
        assert probes <= 60
    finally:
        critic.score = real
