"""Fixer-cluster-F regressions — director.py hardening:

  * keep-best survives a dead hook (apply/render/stats/gateway) — the best state is
    re-applied with the audit trail intact before the error surfaces;
  * a failed sun sweep hands the scene back at the entry azimuth, not the last probe;
  * polish: zeroed log axes are explorable again, the climb polls should_cancel, it is
    skipped after a render failure, and its landed state is not applied twice;
  * leash accounting counts genome-bound saturation, so the SPEC §2 albedo diagnosis
    fires when the solver wants more than the rig can express;
  * the loop's final landing is a no-op when the scene already wears the best state
    (a phantom apply = a phantom Ctrl+Z step).

Pure python, fake hooks — no Max, no network, no images.
"""

import json

import pytest

import maxgaffer.core.director as director
from maxgaffer.core import critic
from maxgaffer.core.director import Hooks, MatchConfig, run_match, run_polish, run_sun_sweep
from maxgaffer.core.genome import LightingState

REF = {"log_key": 0.20, "lab_mean": [55.0, 2.0, 12.0], "lab_std": [20, 4, 6],
       "p": {"5": 0.03, "25": 0.2, "50": 0.45, "75": 0.7, "95": 0.92},
       "contrast": 0.89,
       "lum_hist": [0.0] * 10 + [0.5, 0.5] + [0.0] * 20,
       "hue_hist": [0.6, 0.4] + [0.0] * 10}


def near(log_key):
    s = dict(REF)
    s["log_key"] = log_key
    return s


def start_state(ev=12.0, dome=None):
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.altitude_deg": 30.0,
                 "sun.intensity": 1.0, "sun.size": 2.0, "sun.turbidity": 3.0,
                 "exposure.ev": ev, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    if dome is not None:
        st.set("dome.enabled", 1)
        st.set("dome.rotation_deg", 0.0)
        st.set("dome.intensity", dome)
    return st


class World:
    """Fake hooks: renders 'succeed', stats come from a queue/function, everything the
    loop applies is recorded. Optionally makes stats raise on chosen frame tags."""

    def __init__(self, stats_seq=None, llm_replies=None):
        self.applied = []
        self.renders = []
        self.logs = []
        self.stats_seq = list(stats_seq or [])
        self.llm_replies = list(llm_replies or [])
        self.fail_stats_at = ()       # tag substrings whose stats call raises
        self.stats_fn = None

    def hooks(self):
        def apply(st):
            self.applied.append(st.copy())

        def render(tag):
            self.renders.append(tag)
            return f"/tmp/{tag}.png"

        def stats(path):
            if any(s in path for s in self.fail_stats_at):
                raise RuntimeError("corrupt png")
            if self.stats_fn:
                return self.stats_fn(path)
            if self.stats_seq:
                return self.stats_seq.pop(0)
            return dict(REF)

        def llm(ctx):
            if self.llm_replies:
                return self.llm_replies.pop(0)
            return json.dumps({"assessment": "", "changes": [], "stop": False})

        return Hooks(apply=apply, render=render, stats=stats, llm_deltas=llm,
                     log=self.logs.append)


# ------------------------------------------------------------------ keep-best guard
def test_stats_hook_exception_restores_best_state_and_reraises():
    # iter0 lands the best (perfect stats); the LLM explores altitude 40; iter1's stats
    # hook dies on the corrupt frame — the scene must return to the best (altitude 30)
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "sun.altitude_deg", "value": 40.0, "why": "explore"}], "stop": False})
    world = World(stats_seq=[dict(REF)], llm_replies=[reply])
    world.fail_stats_at = ("iter01",)
    with pytest.raises(RuntimeError, match="corrupt png"):
        run_match(start_state(), REF, {}, world.hooks(),
                  MatchConfig(max_iterations=4, target_score=101, stall_patience=99))
    assert world.applied[-2].get("sun.altitude_deg") != 30.0   # exploratory was live
    assert world.applied[-1].diff(start_state()) == {}         # best re-applied
    assert any("aborted" in m and "re-applied" in m for m in world.logs)
    assert any("iter 0: score" in m for m in world.logs)       # audit trail intact


def test_hook_exception_before_any_score_restores_start_state():
    world = World()
    hooks = world.hooks()
    hooks.render = lambda tag: (_ for _ in ()).throw(RuntimeError("vray died"))
    with pytest.raises(RuntimeError, match="vray died"):
        run_match(start_state(), REF, {}, hooks, MatchConfig(max_iterations=3))
    assert len(world.applied) == 2                             # iter0 apply + restore
    assert world.applied[-1].diff(start_state()) == {}


def test_llm_gateway_error_restores_best_and_keeps_trail():
    # a gateway failure is NOT a ParseError — it must abort with the best state landed
    world = World(stats_seq=[near(0.05)])
    hooks = world.hooks()
    hooks.llm_deltas = lambda ctx: (_ for _ in ()).throw(RuntimeError("gateway down"))
    with pytest.raises(RuntimeError, match="gateway down"):
        run_match(start_state(), REF, {}, hooks,
                  MatchConfig(max_iterations=3, target_score=101, stall_patience=99))
    assert world.applied[-1].diff(start_state()) == {}
    assert any("aborted" in m for m in world.logs)
    assert any("iter 0: score" in m for m in world.logs)


def test_restore_failure_logs_and_surfaces_original_error():
    world = World(stats_seq=[near(0.05), near(0.05)])
    calls = {"n": 0}

    def apply(st):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("apply exploded")
        world.applied.append(st.copy())

    hooks = world.hooks()
    hooks.apply = apply
    with pytest.raises(RuntimeError, match="apply exploded"):
        run_match(start_state(), REF, {}, hooks,
                  MatchConfig(max_iterations=3, target_score=101, stall_patience=99))
    assert any("use Restore" in m for m in world.logs)


# ------------------------------------------------------------------ sun sweep restore
def test_failed_sweep_restores_entry_azimuth():
    world = World()
    hooks = world.hooks()
    hooks.render = lambda tag: None                        # every probe render fails
    az, hint, why = run_sun_sweep(start_state(), [0.0, 90.0, 180.0], hooks,
                                  llm_pick=lambda p, a: "{}")
    assert az is None and "not enough" in why
    assert round(world.applied[-1].get("sun.azimuth_deg")) == 100   # entry restored
    assert any("restored" in m for m in world.logs)

    world2 = World()                                       # renders fine, reply garbage
    az2, _, why2 = run_sun_sweep(start_state(), [0.0, 90.0], world2.hooks(),
                                 llm_pick=lambda p, a: "definitely not json")
    assert az2 is None and "unusable" in why2
    assert round(world2.applied[-1].get("sun.azimuth_deg")) == 100


def test_sweep_hook_exception_restores_entry_azimuth():
    ref = dict(REF)
    ref["grid"] = [0.5] * 9                                # makes the sweep read stats
    world = World()
    world.stats_fn = lambda p: (_ for _ in ()).throw(RuntimeError("stats died"))
    with pytest.raises(RuntimeError, match="stats died"):
        run_sun_sweep(start_state(), [0.0, 90.0], world.hooks(),
                      llm_pick=lambda p, a: "{}", ref_stats=ref)
    assert round(world.applied[-1].get("sun.azimuth_deg")) == 100


def test_sweep_cancel_before_first_probe_applies_nothing():
    world = World()
    hooks = world.hooks()
    hooks.should_cancel = lambda: True
    az, _, why = run_sun_sweep(start_state(), [0.0, 90.0], hooks,
                               llm_pick=lambda p, a: "{}")
    assert az is None and why == "cancelled"
    assert world.applied == []          # scene never touched → nothing to restore


# ------------------------------------------------------------------ polish fixes
def test_polish_turns_a_zeroed_dome_back_up(monkeypatch):
    # dome.intensity == 0.0 is a dead axis for multiplicative probes (0 * 2**x == 0);
    # a dome-favoring landscape must be able to climb it off the floor again
    current = {}

    def apply(st):
        current["st"] = st.copy()

    monkeypatch.setattr(critic, "score", lambda ref, cur, w=None: critic.Verdict(
        score=50.0 + current["st"].get("dome.intensity") * 10.0, components={}))
    hooks = Hooks(apply=apply, render=lambda t: "/tmp/p.png",
                  stats=lambda p: {"x": 1}, llm_deltas=lambda c: "", log=lambda m: None)
    st = start_state(dome=0.0)
    out, sc, probes, _, _ = run_polish(st, 50.0, {"any": "ref"}, hooks,
                                       MatchConfig(polish_rounds=2, polish_stop_at=200.0,
                                                   polish_max_probes=60))
    assert out.get("dome.intensity") > 0.0
    assert sc > 50.0


def test_polish_climb_honors_cancel_mid_climb(monkeypatch):
    # an ever-improving EV axis would climb to the genome bound (~7 probes); cancel
    # after the 3rd render must cut the climb within one probe, not after the param
    current = {}
    renders = {"n": 0}

    def apply(st):
        current["st"] = st.copy()

    def render(tag):
        renders["n"] += 1
        return "/tmp/p.png"

    monkeypatch.setattr(critic, "score", lambda ref, cur, w=None: critic.Verdict(
        score=50.0 + current["st"].get("exposure.ev") * 2.0, components={}))
    hooks = Hooks(apply=apply, render=render, stats=lambda p: {"x": 1},
                  llm_deltas=lambda c: "", log=lambda m: None,
                  should_cancel=lambda: renders["n"] >= 3)
    out, sc, probes, converged, _ = run_polish(
        start_state(), 60.0, {"r": 1}, hooks,
        MatchConfig(polish_rounds=8, polish_stop_at=1e9, polish_max_probes=500))
    assert renders["n"] <= 4                               # cancel cut the climb at once
    assert converged is False
    assert current["st"].diff(out) == {}                   # exit landed the best state


def test_polish_skipped_when_loop_stopped_on_render_failure():
    # iter0 scores; iter1's render dies → render_failed → polish must not burn renders
    # against a dead renderer (each attempt is a potential V-Ray error dialog)
    world = World(stats_seq=[near(0.1)])
    calls = {"n": 0}

    def render(tag):
        calls["n"] += 1
        if calls["n"] > 1:
            return None
        world.renders.append(tag)
        return f"/tmp/{tag}.png"

    hooks = world.hooks()
    hooks.render = render
    res = run_match(start_state(), REF, {}, hooks,
                    MatchConfig(max_iterations=5, target_score=101, stall_patience=99,
                                polish=True, polish_stop_at=101))
    assert res.stop_reason == "render_failed"
    assert res.polish_probes == 0 and res.polish_gain == 0.0
    assert not any("polish" in t for t in world.renders)
    assert not any(m.startswith("polish:") for m in world.logs)


def test_polish_result_is_not_applied_twice(monkeypatch):
    # every run_polish return path already lands `best` — run_match must not apply the
    # identical state again (one undo record per apply; the duplicate is a no-op Ctrl+Z)
    landed = start_state()
    landed.set("sun.altitude_deg", 33.0)
    world = World(stats_seq=[near(0.1)])
    marks = []

    def fake_polish(state, score, ref, hooks, cfg, locks, **kw):
        hooks.apply(landed)                                # polish's contract: land it
        marks.append(len(world.applied))
        return landed, score + 1.0, 3, False, False

    monkeypatch.setattr(director, "run_polish", fake_polish)
    res = run_match(start_state(), REF, {}, world.hooks(),
                    MatchConfig(max_iterations=1, target_score=101,
                                polish=True, polish_stop_at=101))
    assert len(world.applied) == marks[0]                  # zero applies after polish
    assert world.applied[-1].diff(landed) == {}
    assert res.polish_gain == 1.0


def test_polish_hook_exception_restores_loop_best(monkeypatch):
    rogue = start_state()
    rogue.set("sun.altitude_deg", 60.0)

    def fake_polish(state, score, ref, hooks, cfg, locks, **kw):
        hooks.apply(rogue)                                 # exploratory probe live…
        raise RuntimeError("render hook died")

    monkeypatch.setattr(director, "run_polish", fake_polish)
    world = World(stats_seq=[near(0.1)])
    with pytest.raises(RuntimeError, match="render hook died"):
        run_match(start_state(), REF, {}, world.hooks(),
                  MatchConfig(max_iterations=1, target_score=101,
                              polish=True, polish_stop_at=101))
    assert world.applied[-1].diff(start_state()) == {}     # loop-best re-applied
    assert any("polish aborted" in m for m in world.logs)


# ------------------------------------------------------------------ leash accounting
def test_leash_counts_genome_bound_saturation_as_hit():
    # start EV 19 (genome hi 20) with the render 2.5+ stops BRIGHTER than the ref: the
    # solver wants EV 21.5 but solve_ev pre-clamps to 20 — the leash window [15, 23]
    # never clips, yet this is the albedo trap and the SPEC §2 diagnosis must fire
    bright = near(0.20 * 2 ** 2.5)
    world = World(stats_seq=[bright] * 4)
    res = run_match(start_state(ev=19.0), REF, {}, world.hooks(),
                    MatchConfig(max_iterations=4, target_score=101, stall_patience=99,
                                ev_leash=4.0))
    assert any("genome bound" in m for m in world.logs)
    assert any("albedo" in m for m in world.logs)          # the diagnosis fired
    evs = [r.analytic_changes["exposure.ev"] for r in res.iterations
           if "exposure.ev" in r.analytic_changes]
    assert evs and max(evs) <= 20.0 + 1e-9                 # genome bound still respected


# ------------------------------------------------------------------ no-op final landing
def test_final_apply_skipped_when_scene_already_wears_best():
    # one improving iteration reaches the target → the scene IS the best state; a
    # trailing apply would be a phantom undo step
    world = World(stats_seq=[dict(REF)])
    res = run_match(start_state(), REF, {}, world.hooks(),
                    MatchConfig(max_iterations=1, target_score=50.0))
    assert res.stop_reason == "target_reached"
    assert len(world.applied) == 1                           # iter0's apply, nothing more
    assert world.applied[-1].diff(res.best_state) == {}


def test_final_apply_still_lands_best_after_a_worse_exploration():
    # iter0 best (100), iter1 explores and scores worse → the loop must still re-apply
    # the iter-0 best: no-op detection must NEVER skip a real landing
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "sun.altitude_deg", "value": 45.0, "why": "explore"}], "stop": False})
    world = World(stats_seq=[dict(REF), near(0.02)], llm_replies=[reply])
    res = run_match(start_state(), REF, {}, world.hooks(),
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99))
    assert len(world.applied) == 3                           # iter0 · iter1 · landing
    assert world.applied[1].get("sun.altitude_deg") != 30.0  # exploration was live
    assert world.applied[-1].diff(res.best_state) == {}
    assert res.best_state.get("sun.altitude_deg") == 30.0    # best = iter0 state
