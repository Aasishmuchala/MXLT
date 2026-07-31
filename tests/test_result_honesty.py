"""I6 — prove the work. Every fallback announced; the report states what IMPROVED.

The 2026-07-30 delivery was a cool, sunless dusk courtyard for a warm golden-hour
reference, and no stage of the pipeline ever said "this does not look like the reference".
Every number the reality check needs was already computed by that run — and not one of
them was compared to a threshold. That, and not the metric, was the gap.

The most important test in this file is the false-positive guard: a wrong "this does not
look like your reference" on a good match is a credibility hit worse than silence, so the
score floor is calibrated on this repo's own recorded numbers and enforced by a
parameterized list of them.
"""

import glob
import json

import pytest

import honesty_harness as H
from maxgaffer.maxbridge import controller as ctl


class _R:
    """The three fields _reality_check reads, and nothing else."""

    def __init__(self, score=88.0, colour=0.9, highlight=0.9, transfer=None):
        self.best_score = score
        self.best_components = {"color": colour, "key": 0.9, "envelope": 0.9}
        self.highlight = highlight
        self.transfer = {"score": transfer} if transfer is not None else None
        self.unlike_reference = False
        self.unlike_reasons = []


REF_WITH_SUN = {"hot_frac": 0.04}
REF_WITHOUT_SUN = {"hot_frac": 0.0}


def _fake_critic(monkeypatch, score=35.1, colour=0.10, highlight=0.05):
    """A cool, sunless dusk courtyard scored against a warm golden-hour reference."""
    import types as _t

    monkeypatch.setattr(ctl.critic, "score",
                        lambda ref, cur, w: _t.SimpleNamespace(
                            score=score, components={"color": colour, "key": 0.9}))
    monkeypatch.setattr(ctl.metrics, "highlight_similarity",
                        lambda ref, cur: highlight)


def _check(ctrl, r, ref=REF_WITH_SUN):
    logs = []
    ctrl._reality_check(r, ref, logs.append)
    return logs


# ------------------------------------------------------------------ F1: the verdict
def test_a_cool_dusk_against_a_golden_reference_says_it_does_not_look_like_it(
        tmp_path, monkeypatch):
    """Worked through by hand on the 2026-07-30 delivery: colour ~0.10 (LAB Δb ≈ 26,
    blue-grey against golden), highlight ~0.05 (the reference is full of sun patches and
    the render has none), weighted score ≈ 35. The critic was already capable of
    screaming; nothing was listening."""
    c = H.build(tmp_path, monkeypatch)
    r = _R(score=35.1, colour=0.10, highlight=0.05, transfer=41.0)
    logs = _check(c, r)
    assert r.unlike_reference is True
    assert len(r.unlike_reasons) == 3
    joined = " ".join(logs)
    assert "THIS DOES NOT LOOK LIKE YOUR REFERENCE" in joined
    assert "colour" in joined and "sun-patch agreement" in joined
    assert "41/100" in joined
    assert "Do not deliver this frame" in joined


@pytest.mark.parametrize("score", [63.2, 77.21, 77.6, 80.35, 84.95, 91.47, 94.75, 96.6])
def test_a_score_this_repo_has_actually_recorded_is_never_flagged_on_score_alone(
        tmp_path, monkeypatch, score):
    """THE false-positive guard, and the reason the absolute term is the WEAKEST of the
    three at 45.0 rather than the 60.0 a cost audit proposed. This repo's record has
    legitimate basins at 77.6 and 80.35 and structurally-wrong-but-plausible finished
    matches at 63.2 and 77.21; 45 is below every one of them."""
    c = H.build(tmp_path, monkeypatch)
    r = _R(score=score, colour=0.8, highlight=0.8)
    _check(c, r)
    assert r.unlike_reference is False, score


@pytest.mark.parametrize("score", [2.7, 8.7, 10.9, 12.0])
def test_the_black_frame_scores_of_2026_07_30_are_flagged(tmp_path, monkeypatch, score):
    """The four numbers the tool actually reported for 100% black plates, and ranked."""
    c = H.build(tmp_path, monkeypatch)
    r = _R(score=score, colour=0.8, highlight=0.8)
    _check(c, r)
    assert r.unlike_reference is True


def test_colour_alone_is_enough(tmp_path, monkeypatch):
    """Weakest-link, matching the critic's own aggregation philosophy: a warm/cool
    inversion is a failed match however good the histogram is."""
    c = H.build(tmp_path, monkeypatch)
    r = _R(score=88.0, colour=0.10, highlight=0.9)
    _check(c, r)
    assert r.unlike_reference is True
    assert len(r.unlike_reasons) == 1


def test_the_highlight_term_is_silent_when_the_reference_has_no_sun_to_miss(
        tmp_path, monkeypatch):
    """sunsolve already skips the solve entirely at hot_frac 0. An overcast reference has
    no sun patch to be missing, so a render with none agrees with it."""
    c = H.build(tmp_path, monkeypatch)
    r = _R(score=88.0, colour=0.8, highlight=0.0)
    _check(c, r, ref=REF_WITHOUT_SUN)
    assert r.unlike_reference is False


def test_a_clean_match_says_nothing_at_all(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch)
    assert _check(c, _R()) == []


def test_the_verdict_survives_into_run_json(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        r = H.done(score=35.1)
        r.best_render = str(tmp_path / "best.png")
        return r

    H.stub_director(monkeypatch, fake)
    monkeypatch.setattr(c, "stats_for", lambda p: dict(H.LIT))
    # the final full re-render RE-SCORES and overwrites best_score/components/highlight —
    # that is the whole point of it, so the bad numbers have to come from the critic
    _fake_critic(monkeypatch)
    result = c.run_match("CamA", lambda _m: None, multi_start=False, do_sweep=False)
    assert result.unlike_reference is True
    paths = glob.glob(str(tmp_path / "sessions" / "**" / "run.json"), recursive=True)
    with open(paths[-1], encoding="utf-8") as f:
        data = json.load(f)
    assert data["unlike_reference"] is True
    assert data["unlike_reasons"]


def test_the_batch_summary_says_failed_not_a_bare_score(tmp_path, monkeypatch):
    """"SCENE 04 shot02: 35.1" reads as a result. It was not one."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        r = H.done(score=35.1)
        r.best_render = str(tmp_path / "best.png")
        return r

    H.stub_director(monkeypatch, fake)
    monkeypatch.setattr(c, "stats_for", lambda p: dict(H.LIT))
    # the final full re-render RE-SCORES and overwrites best_score/components/highlight —
    # that is the whole point of it, so the bad numbers have to come from the critic
    _fake_critic(monkeypatch)
    # …and with the critic reading 35, the BASIN FLOOR would stop this run before the
    # summary ever existed — which is gate 3 doing its job and test_gates.py's subject.
    # Here the board is empty so the basin stage does not run, and the question under test
    # is only what the batch summary SAYS about a finished-but-wrong match.
    monkeypatch.setattr(ctl.scen, "build_scenarios", lambda *a, **k: [])
    results = c.match_all(lambda _m: None, do_sweep=False)
    assert "FAILED" in results["CamA"]
    assert "does not look like the reference" in results["CamA"]


# ------------------------------------------------------------------ F2: the ledger
def test_every_degradation_reaches_the_final_report(tmp_path, monkeypatch):
    """They already log at the moment they happen and then scroll away under 190 THUMB
    lines. The ledger is what makes "every fallback announced" survive to the verdict."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        c._degrade(hooks.log, "software exposure needed Pillow — frames scored un-exposed")
        c._degrade(hooks.log, "vantage backend demoted to V-Ray at probe 12")
        return H.done()

    H.stub_director(monkeypatch, fake)
    logs = []
    result = c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert len(result.degradations) == 2
    replayed = [ln for ln in logs if ln.strip().startswith("DEGRADED")]
    assert len(replayed) == 2
    assert any("Pillow" in ln for ln in replayed)


def test_a_clean_run_says_it_was_clean(tmp_path, monkeypatch):
    """A report that only speaks up when something broke cannot be trusted when it is
    silent — that is the whole 2026-07-30 lesson, applied to the report itself."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert any("no degradations" in ln for ln in logs)


def test_polish_gain_is_reported_with_its_probe_cost(tmp_path, monkeypatch):
    """polish_gain is the ONLY measured-improvement number in the pipeline and it was
    reported without the 120 probes it bought."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch, seconds=60.0)
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        hooks.render("iter00")
        r = H.done(score=71.8)
        r.polish_gain = 0.34
        r.polish_probes = 118
        return r

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False,
                )
    assert any("points per probe" in ln for ln in logs)
    assert any("already at its basin's ceiling" in ln for ln in logs)


def test_the_measured_gain_is_stated_start_to_finish(tmp_path, monkeypatch):
    from maxgaffer.core.director import IterationRecord

    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        r = H.done(score=71.8)
        r.iterations = [IterationRecord(index=0, state={}, score=62.4),
                        IterationRecord(index=1, state={}, score=71.8)]
        return r

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert any("62.4" in ln and "71.8" in ln and "+9.4" in ln for ln in logs)


# ------------------------------------------------------------------ F2: the else-less
@pytest.mark.parametrize("break_it,expected", [
    ("no_renders", "no-render mode"),
    ("no_reference", "no reference bound"),
    ("bad_reference", "could not be decoded"),
    ("no_plate", "no usable plate"),
    ("no_stats", "could not be measured"),
])
def test_plan_effect_not_measured_names_the_reason(tmp_path, monkeypatch, break_it,
                                                   expected):
    """probe_score returns None six different ways and execute_plan had no `else` on
    either branch, so the most destructive operation in the plugin could complete with no
    effect key and no explanation."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    monkeypatch.setattr(ctl.ex, "execute_plan",
                        lambda ops, cam: {"changes": [], "created": [], "warnings": []})
    if break_it == "no_renders":
        c.cfg.no_renders = True
    elif break_it == "no_reference":
        c.session.entry("CamA").reference = ""
    elif break_it == "bad_reference":
        monkeypatch.setattr(c, "ref_stats", lambda p: None)
    elif break_it == "no_plate":
        monkeypatch.setattr(c, "_render_raw", lambda *a, **k: None)
    elif break_it == "no_stats":
        monkeypatch.setattr(c, "stats_for", lambda p: None)
    logs = []
    report = c.execute_plan([], "CamA", logs.append)
    assert "effect" not in report
    assert any("plan effect: NOT measured" in ln and expected in ln for ln in logs), logs


def test_a_measured_plan_still_reports_its_effect(tmp_path, monkeypatch):
    import types as _t

    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    monkeypatch.setattr(ctl.ex, "execute_plan",
                        lambda ops, cam: {"changes": [], "created": [], "warnings": []})
    scores = iter([60.0, 72.0])
    monkeypatch.setattr(ctl.critic, "score",
                        lambda ref, cur, w: _t.SimpleNamespace(score=next(scores),
                                                               components={}))
    logs = []
    report = c.execute_plan([], "CamA", logs.append)
    assert report["effect"] == {"before": 60.0, "after": 72.0}


def test_a_run_where_nothing_was_measured_says_so_and_does_not_record(tmp_path,
                                                                     monkeypatch):
    """Repeated stats failures produce a FULL match at best_score None under stop_reason
    "max_iterations", which was NOT in the no-measurement set — so the controller recorded
    the match, applied the final state, and handed the artist "best score n/a" with
    nothing saying no frame in the run had ever been measured."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)
    c.session.record_match("CamA", H.make_state(ev=9.0), 88.0)
    recorded = []
    monkeypatch.setattr(c, "_record_match", lambda *a, **k: recorded.append(1))

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        r = H.done(score=None)
        r.stop_reason = "max_iterations"
        return r

    H.stub_director(monkeypatch, fake)
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert recorded == []
    assert any("NO FRAME IN THIS RUN WAS EVER MEASURED" in ln for ln in logs)


def test_the_sweep_reason_is_logged_rather_than_discarded(tmp_path, monkeypatch):
    """run_sun_sweep produces four distinct failure strings and every one was dropped at
    the call site — the sweep's own "metric-only" basis among them."""
    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)
    monkeypatch.setattr(ctl, "run_sun_sweep",
                        lambda *a, **k: (None, "na", "the gateway was down — "
                                                    "metric-only"))
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    logs = []
    result = c.run_match("CamA", logs.append, multi_start=False, do_sweep=True)
    assert any("metric-only" in ln for ln in logs)
    assert any("returned NO direction" in d for d in result.degradations)


def test_config_warnings_reach_the_transcript_not_only_the_listener(tmp_path,
                                                                   monkeypatch):
    """config._warn print()s to Max's Listener, which is not where the artist reads —
    and 29bbae6 added the "config could not be read" warning precisely because that
    sentence was the difference between three lost hours and a two-second diagnosis."""
    from maxgaffer.maxbridge import config as cfgmod

    c = H.build(tmp_path, monkeypatch)
    H.stub_renders(c, monkeypatch)
    H.quiet_sun(monkeypatch)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    cfgmod._warn("config.json could not be read — using DEFAULTS")
    logs = []
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    assert any("using DEFAULTS" in ln for ln in logs)
