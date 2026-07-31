"""I2 — never spend the artist's time without saying what it will cost FIRST.

On 2026-07-30 a single 240×135 probe took about 60 seconds on TULA, a standard profile
budgets up to 133 probes by the number the artist was shown and 190 by the number it
actually fires, and nobody was told any of this before the run started. The artist lost a
day to a bill they were never quoted.

Timing here is always injected. A test that sleeps is a test that lies about being fast.
"""

import pytest

import honesty_harness as H
from maxgaffer.core import profiles
from maxgaffer.maxbridge import controller as ctl


def _profile(name="standard"):
    return profiles.resolve_profile(name, loop_width=480, loop_height=270,
                                    max_iterations=5, sweep_count=8, target_score=95.0)


# ------------------------------------------------------------------ ONE budget
def test_standard_worst_case_is_190_not_133():
    """The number printed to the artist was max_iterations + sweep + polish = 133. It
    omitted the multi-start basin (6), the two tone-align passes, the sun solve (44), the
    two exposure-host plates, the two plan probes and the final full re-render. On TULA
    the missing 57 renders are about an hour that nobody had agreed to."""
    stages = profiles.planned_renders(_profile(), plan_ran=True, verify_exposure=True)
    assert sum(s.count for s in stages) == 190
    assert _profile().worst_case_renders == 133      # the old number, kept and deprecated


def test_planned_renders_and_the_progress_bar_cannot_disagree():
    """TWO hand-maintained copies of one budget is how they came to differ by 50. The
    progress bar's _stage_budget is now derived from planned_renders."""
    import inspect

    src = inspect.getsource(ctl.Controller.run_match)
    assert "_budget_of = {s.key: s.count for s in self._cost_stages}" in src
    assert "hard_cap = sum(s.count for s in self._cost_stages)" in src


@pytest.mark.parametrize("name", ["fast", "standard", "hero"])
def test_every_profile_prices_and_the_stages_are_ordered(name):
    stages = profiles.planned_renders(_profile(name))
    assert stages, name
    assert [s.key for s in stages] == sorted(
        [s.key for s in stages],
        key=lambda k: ["evcheck", "plan", "basin", "tone", "sunsolve", "sweep", "iter",
                       "polish", "final"].index(k))
    for s in stages:
        assert s.count > 0 and s.width > 0 and s.height > 0


def test_most_of_a_match_is_one_resolution_so_one_probe_prices_it():
    """The useful fact: polish_size == sweep size at default settings, so 180 of a
    standard profile's 188 renders are the same 240×135. One timed probe prices 96% of
    the match and no resolution model is needed for the profiles that hurt."""
    stages = profiles.planned_renders(_profile())
    total = sum(s.count for s in stages)
    at_sweep = sum(s.count for s in stages
                   if (s.width, s.height) == (_profile().sweep_width,
                                              _profile().sweep_height))
    assert at_sweep / total > 0.95


# ------------------------------------------------------------------ measurement
def _cold(c, w=160, h=90, seconds=120.0):
    """Burn the session's COLD sample. The first V-Ray plate of a Controller carries
    scene translation and BVH build, so it is recorded as warmup and never as a steady
    sample at its size; a test that wants to talk about steady samples has to get past
    it first, exactly as a real run does."""
    c._observe_render(w, h, seconds, "vray", None)
    assert c._render_times == {}, "the cold sample must not enter the timing table"


def test_vantage_samples_never_price_a_vray_plan(tmp_path, monkeypatch):
    """A ~50 ms window grab multiplied by 181 V-Ray renders reads as "9 seconds" for a
    three-hour run. Same rule the black guard already lives by: gate on the backend that
    actually answered."""
    c = H.build(tmp_path, monkeypatch)
    c._cost_stages = profiles.planned_renders(_profile())
    _cold(c, 240, 135, 90.0)
    c._observe_render(240, 135, 0.05, "vantage", None)
    assert c.cost_estimate() is None, "a grab must not price a render plan"
    c._observe_render(240, 135, 60.0, "vray", None)
    est = c.cost_estimate()
    assert est["seconds_each"] == pytest.approx(60.0)


def test_the_cold_sample_is_added_once_and_never_multiplied(tmp_path, monkeypatch):
    """The first plate of a session carries one-time scene translation, BVH build and the
    light-cache prepass — on 18M triangles with 460 lights that is minutes. Multiplying it
    by 181 is how you get a wildly wrong estimate."""
    c = H.build(tmp_path, monkeypatch)
    c._cost_stages = profiles.planned_renders(_profile())
    c._observe_render(240, 135, 203.0, "vray", None)     # 143 s cold + 60 s steady
    c._observe_render(240, 135, 60.0, "vray", None)
    est = c.cost_estimate()
    assert est["warmup_seconds"] == pytest.approx(143.0)
    # ≈ 205 min priced off the 60 s steady sample, NOT 188 × 203 s ≈ 636 min
    assert 195.0 < est["minutes"] < 215.0


def test_one_sample_refuses_to_extrapolate_silently(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch)
    _cold(c, 64, 36, 9.0)
    c._observe_render(64, 36, 1.0, "vray", None)
    seconds, how = c._price(1920 * 1080)                 # 900× the sampled pixel count
    assert "EXTRAPOLATED" in how and "lower bound" in how
    assert seconds == pytest.approx(4.0)                 # capped at 4×, not 900×


def test_two_sizes_fit_an_affine_model_and_say_so(tmp_path, monkeypatch):
    """Render time on a heavy V-Ray scene is affine, not proportional — the fixed term
    (translation, BVH) dominates at small sizes, so a pure ratio under-prices badly."""
    c = H.build(tmp_path, monkeypatch)
    _cold(c)
    c._observe_render(160, 90, 50.0, "vray", None)       # 14400 px
    c._observe_render(480, 270, 54.0, "vray", None)      # 129600 px
    seconds, how = c._price(240 * 135)                   # 32400 px
    assert "affine" in how
    assert 50.0 < seconds < 54.0


def test_the_cold_canary_never_anchors_the_fit(tmp_path, monkeypatch):
    """The 2026-07-31 zero-second bill, with 2026-07-30's own numbers.

    The canary is the ONLY sample at 160×90 and it is by construction the cold one, so
    filing it as a steady sample made _price fit a line through (14400 px, 180 s) and
    (32400 px, 60 s). That slope is NEGATIVE — more pixels, less time — and every larger
    size clamped to max(0.0, …): the seven full-size loop renders and the delivered final,
    the most expensive frames in the run, priced at 0.0 s and announced as a measured fit.
    """
    c = H.build(tmp_path, monkeypatch)
    c._cost_stages = profiles.planned_renders(_profile())
    c._observe_render(160, 90, 180.0, "vray", None)      # the canary: cold, 3 minutes
    c._observe_render(240, 135, 60.0, "vray", None)      # the first basin probe: warm
    for w, h in ((480, 270), (1920, 1080)):
        seconds, how = c._price(w * h)
        assert seconds > 0.0, f"{w}×{h} priced at zero seconds ({how})"
    est = c.cost_estimate()
    # 188 renders at 60 s is ~3 hours; the broken model quoted 182 minutes of which the
    # 480×270 stages contributed nothing at all
    assert est["minutes"] > 180.0
    assert est["basis"] != "measured at this size"


def test_a_negative_slope_is_refused_rather_than_clamped(tmp_path, monkeypatch):
    """Directly: two samples where the bigger frame was faster cannot be a fit."""
    c = H.build(tmp_path, monkeypatch)
    _cold(c)
    c._observe_render(160, 90, 120.0, "vray", None)      # 14400 px
    c._observe_render(480, 270, 60.0, "vray", None)      # 129600 px, FASTER
    seconds, how = c._price(1920 * 1080)
    assert "affine" not in how
    assert seconds > 0.0


def test_the_headline_basis_is_the_weakest_stage_not_the_first(tmp_path, monkeypatch):
    """A plan whose most expensive stage was EXTRAPOLATED must not be announced as
    "measured" because its two cheapest exposure plates happened to be."""
    c = H.build(tmp_path, monkeypatch)
    c._cost_stages = profiles.planned_renders(_profile(), verify_exposure=True)
    _cold(c)
    c._observe_render(160, 90, 30.0, "vray", None)       # the evcheck stage's own size
    est = c.cost_estimate()
    assert est["stages"][0]["label"]                     # the first stage IS priced exactly
    assert c._price(160 * 90)[1] == "measured at this size"
    assert est["basis"] == c._price(480 * 270)[1] != "measured at this size"


# ------------------------------------------------------------------ the report
def _run_to_first_probe(c, monkeypatch, seconds=60.0):
    logs = []
    H.stub_renders(c, monkeypatch, seconds=seconds)

    def fake(start, ref, sem, hooks, cfg, locks, **kw):
        hooks.render("iter00")
        return H.done()

    H.stub_director(monkeypatch, fake)
    c.run_match("CamA", logs.append, multi_start=False, do_sweep=False)
    return logs


def test_the_estimate_is_reported_before_the_sun_solve(tmp_path, monkeypatch):
    """"Before committing" has to mean before the 44-probe solve and the 120-probe
    polish, not at iter00 — which on TULA is reached an hour and sixty renders into the
    run it is supposed to be pricing."""
    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=0.0)
    logs = []
    H.stub_renders(c, monkeypatch, seconds=60.0)
    at = {}
    H.stub_director(monkeypatch, lambda *a, **k: H.done())

    def spy(*a, **k):
        # HOW MUCH TRANSCRIPT EXISTED when the 44-probe solve began. The previous version
        # of this test asserted `cost_at >= 0` and `order.index("sunsolve") >= 0` —
        # neither of which can be false — and so passed with the estimate reported after
        # the solve, which is the entire thing it is named for.
        at["sunsolve"] = len(logs)
        return None

    monkeypatch.setattr(ctl.sunsolve, "solve_sun_angles", spy)
    c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert "sunsolve" in at, "the solve has to have been reached for this to mean anything"
    cost_at = next(i for i, ln in enumerate(logs) if ln.startswith("cost:"))
    assert cost_at < at["sunsolve"], (
        "the bill must be in the transcript BEFORE the 44-probe solve starts spending:\n"
        + "\n".join(logs[:at["sunsolve"] + 1]))
    assert "renders planned" in logs[cost_at]


def test_the_report_states_the_cancel_bound_in_the_same_breath(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=0.0)
    logs = _run_to_first_probe(c, monkeypatch, seconds=60.0)
    assert any(ln.startswith("cost:") and "renders planned" in ln for ln in logs)
    assert any("cancel will take up to" in ln.lower() for ln in logs)
    assert any("cannot be interrupted" in ln for ln in logs)


def test_the_cost_gate_fires_at_the_configured_minutes(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=10.0,
                uncertainty_policy="ask")
    asked = []
    c.ask = lambda q: asked.append(q) or "stop"
    H.stub_renders(c, monkeypatch, seconds=60.0)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    with pytest.raises(Exception):
        c.run_match("CamA", lambda _m: None, multi_start=True, do_sweep=False)
    assert asked and asked[0].key == "cost"
    assert asked[0].facts["renders"] > 100


def test_a_cheap_scene_is_never_interrupted_by_the_cost_gate(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=20.0, uncertainty_policy="ask")
    c.ask = lambda q: pytest.fail("a 30-second run must not stop to ask")
    H.stub_renders(c, monkeypatch, seconds=0.1)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    c.run_match("CamA", lambda _m: None, multi_start=True, do_sweep=False)


def test_the_gate_headless_assume_proceeds_and_logs_the_answer(tmp_path, monkeypatch):
    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=1.0,
                uncertainty_policy="assume")
    logs = []
    H.stub_renders(c, monkeypatch, seconds=60.0)
    H.stub_director(monkeypatch, lambda *a, **k: H.done())
    c.run_match("CamA", logs.append, multi_start=True, do_sweep=False)
    assert any("policy: assume" in ln for ln in logs)
    assert any("→ Proceed" in ln for ln in logs)


def test_the_estimate_survives_into_run_json(tmp_path, monkeypatch):
    import glob
    import json

    c = H.build(tmp_path, monkeypatch, cost_ask_minutes=0.0)
    _run_to_first_probe(c, monkeypatch, seconds=12.0)
    paths = glob.glob(str(tmp_path / "sessions" / "**" / "run.json"), recursive=True)
    assert paths
    with open(paths[-1], encoding="utf-8") as f:
        data = json.load(f)
    assert data["cost"]["renders"] > 0
    assert data["cost"]["seconds_each"] > 0


def test_human_minutes_reads_like_a_decision_not_a_measurement():
    assert ctl._human_minutes(192.4) == "3 h 12 min"
    assert ctl._human_minutes(45.0) == "45 min"
    assert ctl._human_minutes(0.5) == "30 s"
