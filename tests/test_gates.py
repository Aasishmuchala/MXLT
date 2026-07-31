"""I5 — escalate uncertainty, do not guess through it.

On 2026-07-30 ANALYZE disagreed with itself about the sun's bearing by ±76° across
samples, logged a warning, dropped its own trust to 25%, and then spent about NINETY
MINUTES probing 44 directions anyway. The artist was sitting right there and could have
answered in five seconds by looking at the photograph.

Every headless default here is TODAY'S BEHAVIOUR, with two deliberate exceptions: the
basin floor (where today's behaviour is provably worthless — it ranked black frames) and
the frozen gate (where continuing is provably worthless — the picture cannot change).
"""

import os

import pytest

import honesty_harness as H
from maxgaffer.core import ask as askmod
from maxgaffer.core import consensus, plate, scenarios
from maxgaffer.core.errors import MatchCancelled, PreflightBlocked
from maxgaffer.maxbridge import controller as ctl


def _q(**over):
    kw = dict(key="cost", headline="a decision", detail="the measured facts",
              options=(("go", "Go on"), ("stop", "Stop")), default="go",
              facts={"minutes": 3.0})
    kw.update(over)
    return askmod.Question(**kw)


# ------------------------------------------------------------------ the mechanism
def test_the_question_and_its_facts_are_logged_before_anyone_is_asked(tmp_path,
                                                                     monkeypatch):
    """A question the transcript does not record is the silent-degradation failure again,
    just with better manners. It has to land even when nobody is there to answer."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    logs = []
    c._escalate(_q(), logs.append)
    assert logs[0].startswith("? a decision")
    assert any("the measured facts" in ln for ln in logs)


def test_the_answer_is_logged_with_who_gave_it(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="ask")
    c.ask = lambda q: "stop"
    logs = []
    assert c._escalate(_q(), logs.append) == "stop"
    assert any("(artist)" in ln for ln in logs)

    c.cfg.uncertainty_policy = "assume"
    logs = []
    assert c._escalate(_q(), logs.append) == "go"
    assert any("(policy)" in ln for ln in logs)


def test_headless_assume_returns_the_default_and_says_it_did_not_ask(tmp_path,
                                                                    monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    c.ask = lambda q: pytest.fail("assume must not reach the dialog")
    logs = []
    assert c._escalate(_q(), logs.append) == "go"
    assert any("policy: assume" in ln for ln in logs)


def test_headless_abort_raises(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="abort")
    with pytest.raises(PreflightBlocked):
        c._escalate(_q(), lambda _m: None)


def test_a_raising_ask_degrades_to_the_default_and_says_so(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="ask")
    c.ask = lambda q: (_ for _ in ()).throw(RuntimeError("no dock"))
    logs = []
    assert c._escalate(_q(), logs.append) == "go"
    assert any("could not be put to you" in ln for ln in logs)


def test_an_unknown_answer_degrades_to_the_default_and_says_so(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="ask")
    c.ask = lambda q: "banana"
    logs = []
    assert c._escalate(_q(), logs.append) == "go"
    assert any("not one of this question's answers" in ln for ln in logs)


def test_an_unknown_policy_reads_as_ask(tmp_path, monkeypatch):
    """The same defensive default probe_backend already uses — a typo must degrade to
    today's behaviour, never to a new failure surface."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="whatever")
    c.ask = lambda q: "stop"
    logs = []
    assert c._escalate(_q(), logs.append) == "stop"
    assert any("not a known value" in ln for ln in logs)


def test_every_decision_survives_into_run_json(tmp_path, monkeypatch):
    import glob
    import json

    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume",
                cost_ask_minutes=0.1)
    H.stub_renders(c, monkeypatch, seconds=30.0)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    c.run_match("CamA", lambda _m: None, multi_start=True, do_sweep=False)
    paths = glob.glob(str(tmp_path / "sessions" / "**" / "run.json"), recursive=True)
    with open(paths[-1], encoding="utf-8") as f:
        data = json.load(f)
    assert any(d["key"] == "cost" for d in data["decisions"])
    assert all("who" in d and "answer" in d for d in data["decisions"])


# ------------------------------------------------------------------ absent evidence
def test_default_semantics_carry_zero_agreement():
    """DEFAULT_SEMANTICS is the gateway-down fallback and its -60° bearing is an
    INVENTION. Without these keys the controller read the invention at FULL trust and gave
    it a quarter of the match objective, then printed "sun mid @ bearing -60°" in the same
    sentence shape it uses for a real measurement."""
    assert scenarios.DEFAULT_SEMANTICS["sun_bearing_agreement"] == 0.0
    assert scenarios.DEFAULT_SEMANTICS["sun_bearing_spread_deg"] is None


def test_n_equals_one_consolidation_stamps_zero_agreement():
    """analyze_samples: 1 is a documented config value. `spread` is undefined for N=1, so
    stamping a COMPUTED number here would launder absence into measurement — the honest
    answer is "no agreement evidence", which the consumer can then say out loud."""
    out = consensus.consolidate_analyses([{"time_of_day": "morning",
                                           "sun_bearing_deg": 40.0}])
    assert out["sun_bearing_agreement"] == 0.0
    assert out["sun_bearing_spread_deg"] is None


def test_a_missing_agreement_key_reads_as_zero_trust(tmp_path, monkeypatch):
    """The `.get(..., 1.0)` regression, in the controller."""
    import inspect

    src = inspect.getsource(ctl.Controller.run_match)
    assert 'semantics.get("sun_bearing_agreement", 1.0)' not in src
    assert 'semantics.get("sun_bearing_agreement", 0.0)' in src


def test_a_failed_write_ev_does_not_switch_software_exposure_on(tmp_path, monkeypatch):
    """write_ev returns False whenever set_prop finds no matching spelling, the host kind
    is 'none', or the legacy ISO path fails. p2 then renders at ev0, moved ≈ 0.0, and the
    session was flagged display-stage-only with the confident, WRONG sentence "+2 EV moved
    the render only 0.00 stops"."""
    c = H.build(tmp_path, monkeypatch, real_exposure_host=True)
    c._exposure_host_checked = False
    c.cfg.software_exposure = False

    class _Host:
        kind = "vray_physical"

        def __init__(self, cam):
            pass

        def read_ev(self):
            return 10.0

        def write_ev(self, v):
            return False                    # refused

    import maxgaffer.maxbridge.exposure as exmod

    monkeypatch.setattr(exmod, "ExposureHost", _Host)
    # a REAL plate at the size asked for: the canary blocks on a missing file or a wrong
    # size, and this test is about the EV write, not about either of those
    def _render(cam, out, w, h):
        from maxgaffer.core import png_min

        png_min.write_png_rgb(out, [[(90, 90, 90)] * w for _ in range(h)])
        return out

    monkeypatch.setattr(ctl.rd, "render_frame", _render)
    logs = []
    c._verify_exposure_host(object(), str(tmp_path), logs.append)
    assert c.cfg.software_exposure is False
    assert any("REFUSED" in ln for ln in logs)


# ------------------------------------------------------------------ gate 1: direction
def _contested(**over):
    sem = dict(H.SEMANTICS)
    sem["sun_bearing_agreement"] = 0.0
    sem["sun_bearing_spread_deg"] = 76.0
    sem.update(over)
    return sem


def _bearing_run(tmp_path, monkeypatch, answer, sem=None, **cfg):
    cfg.setdefault("uncertainty_policy", "ask")
    cfg.setdefault("cost_ask_minutes", 0.0)
    c = H.build(tmp_path, monkeypatch, **cfg)
    monkeypatch.setattr(c, "analyze_reference",
                        lambda name: dict(sem if sem is not None else _contested()))
    H.stub_renders(c, monkeypatch)
    asked = []
    c.ask = lambda q: asked.append(q) or answer
    solved = []
    monkeypatch.setattr(ctl.sunsolve, "solve_sun_angles",
                        lambda *a, **k: solved.append(1) or None)
    logs = []
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    return c, asked, solved, logs


def test_the_direction_gate_fires_at_a_spread_of_76(tmp_path, monkeypatch):
    _c, asked, solved, _logs = _bearing_run(tmp_path, monkeypatch, "left")
    assert [q.key for q in asked] == ["sun_bearing"]
    assert asked[0].facts["spread_deg"] == 76.0


def test_the_direction_gate_does_not_fire_at_a_spread_of_20(tmp_path, monkeypatch):
    """0.34 agreement is a circular spread above 40° — the samples cannot agree on a
    QUADRANT. 20° of scatter still constrains one, so the grid is cheaper than a dialog."""
    sem = _contested(sun_bearing_agreement=0.67, sun_bearing_spread_deg=20.0)
    _c, asked, solved, _logs = _bearing_run(tmp_path, monkeypatch, "left", sem=sem)
    assert asked == []
    assert solved == [1], "the solve must still run when the reading constrains something"


def test_a_quadrant_answer_skips_the_whole_grid_solve(tmp_path, monkeypatch):
    _c, asked, solved, logs = _bearing_run(tmp_path, monkeypatch, "left")
    assert solved == [], "44 probes must not fire after the artist answered"
    assert any("SKIPPED" in ln for ln in logs)


def test_a_quadrant_answer_does_not_lock_the_axis(tmp_path, monkeypatch):
    """An artist answering "over my left shoulder" has given a QUADRANT, not an angle.
    Locking outright would forfeit ±45° of refinement the render can still do, so the
    answer becomes a start value plus slack and the loop may still move it."""
    c, _asked, _solved, logs = _bearing_run(tmp_path, monkeypatch, "left")
    assert "sun.azimuth_deg" not in c.session.cameras["CamA"].locks
    assert any("45" in ln and "held to" in ln for ln in logs)


def test_answering_stop_stops_before_the_grid_is_spent(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="ask", cost_ask_minutes=0.0)
    monkeypatch.setattr(c, "analyze_reference", lambda name: _contested())
    H.stub_renders(c, monkeypatch)
    c.ask = lambda q: "stop"
    monkeypatch.setattr(ctl.sunsolve, "solve_sun_angles",
                        lambda *a, **k: pytest.fail("the artist said stop"))
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    with pytest.raises(MatchCancelled):
        c.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)


def test_the_headless_default_is_todays_behaviour(tmp_path, monkeypatch):
    """"solve" — so no existing API caller changes shape."""
    _c, asked, solved, logs = _bearing_run(tmp_path, monkeypatch, "",
                                           uncertainty_policy="assume")
    assert solved == [1]
    assert any("Solve it on the grid anyway" in ln for ln in logs)


# ------------------------------------------------------------------ gate 3: basin floor
def _basin_run(tmp_path, monkeypatch, score, answer="stop", **cfg):
    cfg.setdefault("uncertainty_policy", "ask")
    cfg.setdefault("cost_ask_minutes", 0.0)
    c = H.build(tmp_path, monkeypatch, **cfg)
    H.stub_renders(c, monkeypatch)
    import types as _t

    monkeypatch.setattr(ctl.critic, "score",
                        lambda ref, cur, w: _t.SimpleNamespace(score=score,
                                                               components={}))
    monkeypatch.setattr(ctl, "blend_transfer", lambda v, st, h, cfg_: v)
    asked = []
    c.ask = lambda q: asked.append(q) or answer
    H.stub_director(monkeypatch, lambda *a, **k: H.done(score=score))
    logs = []
    return c, asked, logs


def test_the_basin_floor_fires_at_twelve(tmp_path, monkeypatch):
    """THE gate that would have stopped 2026-07-30 within about seven renders. The black
    frames scored 12.0, 10.9, 8.7 and 2.7 and the picker announced a "best basin" from
    them, then ranked black for hours."""
    c, asked, logs = _basin_run(tmp_path, monkeypatch, 12.0)
    with pytest.raises(PreflightBlocked):
        c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert [q.key for q in asked] == ["basin_floor"]
    assert asked[0].facts["best_score"] == pytest.approx(12.0)


@pytest.mark.parametrize("score", [63.2, 77.21, 77.6, 80.35, 84.95, 91.47, 94.75, 96.6])
def test_the_basin_floor_never_fires_on_a_score_this_repo_has_measured(
        tmp_path, monkeypatch, score):
    """The false-positive guard, calibrated on this repo's own recorded numbers. 45 is
    below every legitimate basin ever measured here and above every black-frame number."""
    c, asked, logs = _basin_run(tmp_path, monkeypatch, score)
    c.ask = lambda q: pytest.fail(f"{score} is a legitimate basin")
    c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)


def test_continuing_from_a_bad_basin_is_recorded_as_a_degradation(tmp_path, monkeypatch):
    c, asked, logs = _basin_run(tmp_path, monkeypatch, 12.0, answer="continue")
    result = c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert any("on your say-so" in d for d in result.degradations)


def test_a_six_way_tie_is_reported_as_not_a_choice(tmp_path, monkeypatch):
    c, asked, logs = _basin_run(tmp_path, monkeypatch, 80.0)
    c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert any("leads by only" in ln for ln in logs)


def test_a_basin_that_produced_no_plate_is_named_not_silently_dropped(tmp_path,
                                                                     monkeypatch):
    """A silent `continue` turned a 6-candidate board into a 1-candidate board."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    H.stub_renders(c, monkeypatch)
    monkeypatch.setattr(c, "stats_for", lambda p: None)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    logs = []
    c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert any("NOT in the comparison" in ln for ln in logs)


# ------------------------------------------------------------------ gate 4: frozen
def test_six_frozen_plates_escalate_and_the_default_skips_the_sun_stages(
        tmp_path, monkeypatch):
    """Spending 38 more renders that are provably the same picture is not a defensible
    default for anyone, so the headless answer is "skip", not "continue"."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    H.stub_renders(c, monkeypatch, stats=dict(H.LIT))     # every plate identical
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for i in range(8):
            start.set("sun.azimuth_deg", 30.0 * i)
            hooks.apply(start)
            hooks.render(f"sunsolve_a{i:03d}")
        return H.done()

    H.stub_director(monkeypatch, fake)
    logs = []
    result = c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert c._skip_sun_stages is True
    assert any(d["key"] == "frozen_plates" for d in result.decisions)
    assert any("PIXEL-IDENTICAL" in ln for ln in logs)


def test_skipping_the_sun_stages_ends_the_stage_ALREADY_RUNNING(tmp_path, monkeypatch):
    """2026-07-31. The gate fires from INSIDE render_hook — at the 6th frozen plate, i.e.
    coarse probe 7 of 44 — but _skip_sun_stages was read only at the two stage ENTRANCES.
    So an artist who answered "skip the sun stages" (and the headless default) still paid
    for the ~37 remaining probes of the solve already in flight: on TULA, 37 minutes spent
    after they agreed to stop spending them, and a final report that then said the sun
    stages had been SKIPPED.

    Driven through the REAL sunsolve grid, not a stand-in for it — the stage that would
    not stop is the thing under test."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    H.stub_renders(c, monkeypatch, stats=dict(H.LIT))     # every plate identical
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=True)
    grid = [out for out, _w, _h in c.h.renders
            if os.path.basename(out).startswith(("sunsolve_a", "sunsolve_fine",
                                                 "sweep0", "sweep1", "sweep2", "sweep3"))]
    assert c._skip_sun_stages is True
    assert len(grid) <= plate.FROZEN_ESCALATE_RUN + 3, (
        f"{len(grid)} direction probes rendered after the gate said stop: {grid}")
    assert any("SKIPPED from here" in ln for ln in logs)


def test_the_tone_align_plate_is_not_collateral_of_the_sun_skip(tmp_path, monkeypatch):
    """sunsolve_tonealign shares the prefix and is NOT a sun stage — it sets the artist's
    exposure via solve_ev and must still render. Same for sweep_basin_*, the multi-start
    picker."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    c._skip_sun_stages = True
    hook = _render_hook_of(c, monkeypatch)
    assert hook("sunsolve_tonealign") is not None
    assert hook("sweep_basin_warm") is not None
    assert hook("sunsolve_a090") is None
    assert hook("sweep045") is None


def _render_hook_of(c, monkeypatch):
    """Reach run_match's render_hook without running a match: capture it from the director
    call, then hand it back for direct interrogation."""
    grabbed = {}

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        grabbed["render"] = hooks.render
        return H.done()

    H.stub_director(monkeypatch, fake)
    H.quiet_sun(monkeypatch)
    skip = c._skip_sun_stages
    c.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)
    c._skip_sun_stages = skip          # run_match resets it; the caller's setting is the test
    c._skip_sun_said = True            # do not re-announce inside the assertions below
    return grabbed["render"]


def test_three_frozen_plates_report_once_not_once_per_probe(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    H.stub_renders(c, monkeypatch, stats=dict(H.LIT))
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for i in range(4):
            start.set("sun.azimuth_deg", 30.0 * i)
            hooks.apply(start)
            hooks.render(f"sunsolve_a{i:03d}")
        return H.done()

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    # the ⚠ line is the diagnosis at the moment it happens; the final report replays the
    # same text from the degradation ledger, which is the point of the ledger
    frozen = [ln for ln in logs if "PIXEL-IDENTICAL" in ln and ln.startswith("⚠")]
    assert len(frozen) == 1
    assert "sun.azimuth_deg" in frozen[0]


def test_the_frozen_run_is_state_gated_so_determinism_is_not_a_symptom(tmp_path,
                                                                      monkeypatch):
    """V-Ray renders are deterministic in the state — director.py records that the same
    state re-rendered scores 100.0 against itself — so re-rendering ONE state must never
    look like a stuck sun."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="assume")
    H.stub_renders(c, monkeypatch, stats=dict(H.LIT))
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        for i in range(8):
            hooks.apply(start)               # the SAME state, eight times
            hooks.render(f"iter{i:02d}")
        return H.done()

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert not any("PIXEL-IDENTICAL" in ln for ln in logs)
    assert c._skip_sun_stages is False


# ------------------------------------------------------------------ batch
def test_match_all_answers_its_own_questions_and_never_blocks(tmp_path, monkeypatch):
    """A 45-camera overnight queue must not stop on the first dialog and block until
    morning — and the artist's interactive policy is theirs, restored in a finally."""
    c = H.build(tmp_path, monkeypatch, uncertainty_policy="ask", cost_ask_minutes=0.01)
    H.stub_renders(c, monkeypatch, seconds=30.0)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    for name in ("CamB", "CamC"):
        c.session.entry(name).reference = c.h.ref
    c.ask = lambda q: pytest.fail("a batch must never wait for a human")
    logs = []
    results = c.match_all(logs.append, do_sweep=False)
    assert set(results) == {"CamA", "CamB", "CamC"}
    assert c.cfg.uncertainty_policy == "ask", "the artist's policy must be restored"
    assert any("answered from their defaults" in ln for ln in logs)
