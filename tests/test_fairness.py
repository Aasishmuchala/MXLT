"""Off-box tests for ``core.fairness.assess`` — the read-only reference-vs-scene estimate.

Pure Python, no Max, no renders. These tests pin the FROZEN B1–B8 contract:

  * the exact B3 key set on every path (including the degenerate "unknown" path);
  * ``verdict``/``constrainable``/``albedo_risk`` bounded and correctly banded;
  * NUMERIC IDENTITY with the sources fairness borrows from — the director's log2 key-ratio
    (with the ``max(1e-5, …)`` floor, director.py:253-255) for ``predicted_ev_gap`` and the
    solver's highlight-b* gap (with the ``lab_mean_hi_full`` same-scene switch and
    ``WB_KELVIN_PER_B`` slope, solver.py:107-129) for ``predicted_wb_gap``;
  * the same-scene EXEMPTION zeroing ``albedo_risk`` AND gating ``predicts_leash_trip``;
  * the AUTHORITATIVE-SIGNAL contract — ``leash_hits >= 2`` forces "unfair", ``content_gap``
    forces at least "marginal", and the verdict is monotone: fairness can only read
    as-bad-or-worse than the director/critic signals it is handed, never softer;
  * the honest "unreconstructable" flags — the single-view back-hemisphere caveat is present
    for BOTH single- and multi-reference (Route A never consumes extra views in the solve);
  * ``constrainable`` NOT raised by ``n_references``; ``remedy`` only on "unfair".

The numeric-identity assertions import ``solver.WB_KELVIN_PER_B`` / ``solver.SYMMETRIC_CLIP_FRAC``
and cross-check against ``solver.solve_wb`` directly, so they track the live sources rather than
a copied constant.
"""

import math

import pytest

from maxgaffer.core import fairness, solver
from maxgaffer.core.director import MatchConfig

# The FROZEN B3 return shape — exactly these keys, no more, no fewer, on every path.
B3_KEYS = {
    "verdict", "constrainable", "albedo_risk", "predicted_ev_gap", "predicted_wb_gap",
    "same_scene", "coverage", "n_references", "unreconstructable", "predicts_leash_trip",
    "remedy", "notes",
}

# Verdict severity ordering for monotonicity checks ("unknown" is a separate, off-ladder state).
_SEVERITY = {"fair": 0, "marginal": 1, "unfair": 2}


def make_stats(log_key=0.18, b_hi=5.0, b_hi_full=None, b_mean=None,
               hi_clip=0.0, a=0.0, L=50.0):
    """A minimal but well-formed stats dict carrying exactly the fields ``assess`` reads.

    ``b_hi`` is the highlight b* (``lab_mean_hi``), ``b_hi_full`` the symmetric-inclusive
    highlight b* (``lab_mean_hi_full``, used in the same-scene regime), ``b_mean`` the
    full-frame fallback (``lab_mean``). ``hi_clip`` drives the same-scene predicate.
    """
    if b_hi_full is None:
        b_hi_full = b_hi
    if b_mean is None:
        b_mean = b_hi
    return {
        "log_key": log_key,
        "hi_clip_frac": hi_clip,
        "lab_mean": [L, a, b_mean],
        "lab_mean_hi": [L, a, b_hi],
        "lab_mean_hi_full": [L, a, b_hi_full],
    }


def same_scene_pair(b_full_ref=5.0, b_full_cur=5.0, **kw):
    """A pair of stats that BOTH clip >= SYMMETRIC_CLIP_FRAC of the highlight quartile —
    the solver's same-scene regime (a relight / hero shot)."""
    clip = solver.SYMMETRIC_CLIP_FRAC + 0.05
    ref = make_stats(hi_clip=clip, b_hi_full=b_full_ref, **kw)
    cur = make_stats(hi_clip=clip, b_hi_full=b_full_cur, **kw)
    return ref, cur


# ============================================================ shape / never-raise / never-None

def test_returns_exact_b3_key_set():
    result = fairness.assess(make_stats(), make_stats())
    assert set(result) == B3_KEYS


@pytest.mark.parametrize("ref,cur", [
    (None, None),
    (None, make_stats()),
    (make_stats(), None),
    ({}, {}),                                  # dicts but no log_key
    ("junk", 12),                              # not even dicts
    ({"log_key": 0.2}, {"log_key": 0.2}),      # log_key present but no chroma at all
    ({"log_key": None}, make_stats()),         # present-but-None key reads as ABSENT
    ({"log_key": "nan"}, make_stats()),        # mistyped key must not raise
])
def test_missing_or_degenerate_stats_are_unknown_never_raise(ref, cur):
    result = fairness.assess(ref, cur)
    assert set(result) == B3_KEYS                      # fully shaped even on the failure path
    assert result["verdict"] == "unknown"
    assert result["remedy"] == ""
    # numeric fields still carry floats (return type never goes None on any path)
    assert isinstance(result["predicted_ev_gap"], float)
    assert isinstance(result["predicted_wb_gap"], float)
    assert isinstance(result["notes"], str) and result["notes"]


def test_unknown_keeps_the_single_view_caveat():
    # the back-hemisphere caveat is a property of the SOLVE METHOD, not of the stats, so it
    # must survive even when the stats are unusable
    result = fairness.assess(None, None)
    assert fairness.BACK_HEMI_CAVEAT in result["unreconstructable"]


def test_never_returns_none_across_wild_inputs():
    for ref in (None, {}, make_stats(log_key=0.0), make_stats(log_key=-3.0)):
        for cur in (None, {}, make_stats(log_key=0.0), make_stats(b_hi=999.0)):
            assert fairness.assess(ref, cur) is not None


# ================================================================ predicted_ev_gap == director

@pytest.mark.parametrize("ref_key,cur_key", [
    (0.18, 0.18),        # identical → 0
    (0.36, 0.18),        # +1 stop
    (0.18, 0.72),        # -2 stops
    (0.16, 0.01),        # exactly +4 stops (ratio 16 → log2 == 4.0)
    (0.9, 0.02),
])
def test_predicted_ev_gap_equals_log2_key_ratio(ref_key, cur_key):
    result = fairness.assess(make_stats(log_key=ref_key), make_stats(log_key=cur_key))
    # verbatim director.py:253-255
    expected = abs(math.log2(max(1e-5, ref_key) / max(1e-5, cur_key)))
    assert result["predicted_ev_gap"] == pytest.approx(expected, abs=1e-12)
    assert result["predicted_ev_gap"] >= 0.0


@pytest.mark.parametrize("ref_key,cur_key", [
    (0.0, 0.18),         # zero key on a side → the 1e-5 floor engages, no div/log blow-up
    (0.18, 0.0),
    (0.0, 0.0),
    (-0.5, 0.18),        # negative key floors to 1e-5 exactly like the director
])
def test_predicted_ev_gap_uses_the_max_1e5_guard(ref_key, cur_key):
    result = fairness.assess(make_stats(log_key=ref_key), make_stats(log_key=cur_key))
    expected = abs(math.log2(max(1e-5, ref_key) / max(1e-5, cur_key)))
    assert math.isfinite(result["predicted_ev_gap"])
    assert result["predicted_ev_gap"] == pytest.approx(expected, abs=1e-12)


def test_missing_log_key_is_unknown_not_a_zero_gap():
    # a MISSING key must abstain (unknown), never masquerade as a 0-stop gap
    result = fairness.assess({"lab_mean": [50, 0, 5]}, make_stats())
    assert result["verdict"] == "unknown"


# ================================================================= predicted_wb_gap == solver

def test_wb_gap_uses_lab_mean_hi_when_not_same_scene():
    # hi_clip below threshold → cross-scene default → the CLIP-EXCLUSIVE highlight mean
    ref = make_stats(b_hi=8.0, b_hi_full=99.0, b_mean=-99.0, hi_clip=0.0)
    cur = make_stats(b_hi=2.0, b_hi_full=99.0, b_mean=-99.0, hi_clip=0.0)
    result = fairness.assess(ref, cur)
    assert result["same_scene"] is False
    expected = abs(8.0 - 2.0) * solver.WB_KELVIN_PER_B
    assert result["predicted_wb_gap"] == pytest.approx(expected, abs=1e-9)


def test_wb_gap_uses_lab_mean_hi_full_when_same_scene():
    # BOTH sides clip the highlight quartile → the solver switches to the SYMMETRIC-inclusive
    # highlight mean; the draft's fixed lab_mean_hi was wrong precisely here
    ref = make_stats(b_hi=-99.0, b_hi_full=9.0, b_mean=-99.0, hi_clip=0.30)
    cur = make_stats(b_hi=-99.0, b_hi_full=1.0, b_mean=-99.0, hi_clip=0.30)
    result = fairness.assess(ref, cur)
    assert result["same_scene"] is True
    expected = abs(9.0 - 1.0) * solver.WB_KELVIN_PER_B
    assert result["predicted_wb_gap"] == pytest.approx(expected, abs=1e-9)


def test_wb_gap_falls_back_to_lab_mean_when_highlight_absent_on_a_side():
    # highlight mean missing on one side → hi_both False → BOTH sides fall back to lab_mean
    ref = {"log_key": 0.18, "hi_clip_frac": 0.0, "lab_mean": [50, 0, 7.0],
           "lab_mean_hi": [50, 0, 8.0]}
    cur = {"log_key": 0.18, "hi_clip_frac": 0.0, "lab_mean": [50, 0, 1.0]}   # no lab_mean_hi
    result = fairness.assess(ref, cur)
    expected = abs(7.0 - 1.0) * solver.WB_KELVIN_PER_B      # from lab_mean, NOT lab_mean_hi
    assert result["predicted_wb_gap"] == pytest.approx(expected, abs=1e-9)


def test_wb_gap_unknown_when_all_chroma_absent_on_a_side():
    ref = {"log_key": 0.18, "hi_clip_frac": 0.0}           # no lab_mean, no highlight
    result = fairness.assess(ref, make_stats())
    assert result["verdict"] == "unknown"


def test_predicted_wb_gap_numeric_identity_with_solver_solve_wb():
    # Non-same-scene, above the WB deadband, below WB_MAX_STEP and inside the genome kelvin
    # range → solve_wb's returned kelvin delta is exactly abs(b_ref - b_cur) * kelvin_per_b,
    # which is precisely fairness.predicted_wb_gap. This ties the two to the SAME source math.
    ref = make_stats(b_hi=8.0, hi_clip=0.0)
    cur = make_stats(b_hi=2.0, hi_clip=0.0)
    new_kelvin = solver.solve_wb(ref, cur, current_kelvin=6500.0)
    assert new_kelvin is not None
    solver_gap = abs(new_kelvin - 6500.0)                  # == |db| * WB_KELVIN_PER_B, unclamped
    result = fairness.assess(ref, cur)
    assert result["predicted_wb_gap"] == pytest.approx(solver_gap, abs=1e-9)


# =============================================================== same-scene exemption (B4)

def test_same_scene_zeroes_albedo_risk_even_with_large_gaps():
    # huge EV + WB divergence, but both frames clip → same-scene → albedo risk is WAIVED
    ref, cur = same_scene_pair(b_full_ref=40.0, b_full_cur=-40.0,
                               log_key=0.9)
    cur["log_key"] = 0.02
    result = fairness.assess(ref, cur)
    assert result["same_scene"] is True
    assert result["albedo_risk"] == 0.0


def test_same_scene_gates_predicts_leash_trip():
    ref, cur = same_scene_pair(b_full_ref=60.0, b_full_cur=-60.0, log_key=0.9)
    cur["log_key"] = 0.01                                   # gaps well past both leashes
    result = fairness.assess(ref, cur)
    assert result["predicted_ev_gap"] >= MatchConfig().ev_leash
    assert result["predicts_leash_trip"] is False          # exemption gates the prediction
    assert result["verdict"] == "fair"                     # no overriding signal → fair


def test_same_scene_offframe_and_albedo_caveats_are_dropped():
    ref, cur = same_scene_pair(b_full_ref=40.0, b_full_cur=-40.0)
    result = fairness.assess(ref, cur)
    assert fairness.BACK_HEMI_CAVEAT in result["unreconstructable"]   # single-view caveat stays
    # ...but the direction/albedo-family caveats are honestly gated OFF in the same-scene regime
    assert not any("off-frame" in c for c in result["unreconstructable"])
    assert not any("albedo/material family" in c for c in result["unreconstructable"])


# ============================================================ predicts_leash_trip alignment

def test_leash_trip_when_ev_gap_at_or_above_ev_leash():
    ev_leash = MatchConfig().ev_leash
    # log_key ratio 16 → exactly 4.0 stops == ev_leash (the >= boundary must trip)
    ref = make_stats(log_key=0.16, b_hi=5.0, hi_clip=0.0)
    cur = make_stats(log_key=0.01, b_hi=5.0, hi_clip=0.0)
    result = fairness.assess(ref, cur)
    assert result["predicted_ev_gap"] == pytest.approx(ev_leash, abs=1e-9)
    assert result["predicts_leash_trip"] is True
    assert result["verdict"] == "unfair"


def test_leash_trip_when_wb_gap_at_or_above_wb_leash():
    wb_leash = MatchConfig().wb_leash
    # b* gap chosen so |db| * 90 == wb_leash exactly
    db = wb_leash / solver.WB_KELVIN_PER_B
    ref = make_stats(log_key=0.18, b_hi=db, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=0.0, hi_clip=0.0)
    result = fairness.assess(ref, cur)
    assert result["predicted_wb_gap"] == pytest.approx(wb_leash, abs=1e-6)
    assert result["predicts_leash_trip"] is True
    assert result["verdict"] == "unfair"


def test_no_leash_trip_when_both_gaps_below_leashes():
    ref = make_stats(log_key=0.20, b_hi=6.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)   # ~0.15 stop, 90 K gaps
    result = fairness.assess(ref, cur)
    assert result["predicted_ev_gap"] < MatchConfig().ev_leash
    assert result["predicted_wb_gap"] < MatchConfig().wb_leash
    assert result["predicts_leash_trip"] is False


# ================================================= authoritative-signal contract (B4 / B8)

def test_leash_hits_ge_2_forces_unfair_even_when_stats_look_fair():
    # tiny gaps AND same-scene (which would otherwise exempt everything) — the director's
    # leash_hits>=2 is authoritative and must still force "unfair"
    ref, cur = same_scene_pair(b_full_ref=5.0, b_full_cur=5.0, log_key=0.18)
    result = fairness.assess(ref, cur, leash_hits=2)
    assert result["verdict"] == "unfair"
    assert result["remedy"] == fairness.REMEDY_LOCK


def test_leash_hits_1_does_not_force_unfair():
    ref = make_stats(log_key=0.19, b_hi=5.2, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    result = fairness.assess(ref, cur, leash_hits=1)
    assert result["verdict"] == "fair"


def test_content_gap_true_forces_at_least_marginal():
    # small gaps → base would be well under 0.33 → without the floor this reads "fair"
    ref = make_stats(log_key=0.19, b_hi=5.1, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    fair_no_gap = fairness.assess(ref, cur)
    assert fair_no_gap["verdict"] == "fair"                # baseline is genuinely fair
    with_gap = fairness.assess(ref, cur, content_gap=True)
    assert _SEVERITY[with_gap["verdict"]] >= _SEVERITY["marginal"]
    assert with_gap["albedo_risk"] >= 0.50                 # deterministic content-gap floor


def test_content_gap_via_components_second_critic_clause():
    # light axes strong (>= 0.78) but tonal shape weak (< 0.62) → content, not lighting
    comps = {"key": 0.85, "color": 0.85, "direction": 0.85,
             "envelope": 0.50, "histogram": 0.55}
    ref = make_stats(log_key=0.19, b_hi=5.1, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    result = fairness.assess(ref, cur, components=comps)
    assert _SEVERITY[result["verdict"]] >= _SEVERITY["marginal"]


def test_content_gap_via_ceiling_clause_needs_a_score_signal():
    # ceiling clause: ceiling_proven AND a derived-low score (mean component < 0.95)
    low = {"key": 0.90, "color": 0.90, "direction": 0.90,
           "envelope": 0.90, "histogram": 0.90}                # mean 0.90 < 0.95
    ref = make_stats(log_key=0.19, b_hi=5.1, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    fired = fairness.assess(ref, cur, components=low, ceiling_proven=True)
    assert _SEVERITY[fired["verdict"]] >= _SEVERITY["marginal"]
    # ...but with NO components the ceiling clause degrades to False (no score signal to see)
    degraded = fairness.assess(ref, cur, ceiling_proven=True)
    assert degraded["verdict"] == "fair"


def test_content_gap_still_marginal_in_same_scene_regime():
    # same-scene zeroes albedo_risk, but an authoritative content_gap still floors verdict
    ref, cur = same_scene_pair(b_full_ref=5.0, b_full_cur=5.0)
    result = fairness.assess(ref, cur, content_gap=True)
    assert result["albedo_risk"] == 0.0
    assert _SEVERITY[result["verdict"]] >= _SEVERITY["marginal"]


# --------------------------------------------- monotonicity: never LESS severe than the signal

@pytest.mark.parametrize("ref,cur", [
    (make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0),
     make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)),          # fair-looking
    same_scene_pair(),                                          # same-scene exempt
    (make_stats(log_key=0.9, b_hi=40.0, hi_clip=0.0),
     make_stats(log_key=0.02, b_hi=-40.0, hi_clip=0.0)),        # already unfair
])
def test_leash_hits_ge_2_never_reads_softer_than_director(ref, cur):
    result = fairness.assess(ref, cur, leash_hits=3)
    assert result["verdict"] == "unfair"                        # can only be as-bad-or-worse


@pytest.mark.parametrize("ref,cur", [
    (make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0),
     make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)),
    same_scene_pair(),
    (make_stats(log_key=0.30, b_hi=8.0, hi_clip=0.0),
     make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)),
])
def test_content_gap_never_reads_softer_than_critic(ref, cur):
    result = fairness.assess(ref, cur, content_gap=True)
    assert _SEVERITY[result["verdict"]] >= _SEVERITY["marginal"]


def test_signalled_verdict_is_monotone_vs_unsignalled():
    # feeding the authoritative signals can only hold or RAISE severity, never lower it
    ref = make_stats(log_key=0.30, b_hi=9.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    bare = fairness.assess(ref, cur)
    signalled = fairness.assess(ref, cur, leash_hits=2, content_gap=True)
    assert _SEVERITY[signalled["verdict"]] >= _SEVERITY[bare["verdict"]]


# ================================================= single-view "unreconstructable" honesty (B5)

@pytest.mark.parametrize("n_refs", [1, 2, 5, 12])
def test_back_hemisphere_caveat_present_for_every_ref_count(n_refs):
    result = fairness.assess(make_stats(), make_stats(), n_references=n_refs)
    assert fairness.BACK_HEMI_CAVEAT in result["unreconstructable"]
    assert "back-hemisphere" in fairness.BACK_HEMI_CAVEAT


def test_offframe_direction_caveat_only_when_not_same_scene():
    not_same = fairness.assess(make_stats(hi_clip=0.0), make_stats(hi_clip=0.0))
    assert any("off-frame key-source direction" in c for c in not_same["unreconstructable"])
    same = fairness.assess(*same_scene_pair())
    assert not any("off-frame key-source direction" in c for c in same["unreconstructable"])


def test_albedo_family_caveat_appears_when_risk_high_and_cross_scene():
    ref = make_stats(log_key=0.9, b_hi=40.0, hi_clip=0.0)
    cur = make_stats(log_key=0.02, b_hi=-40.0, hi_clip=0.0)
    result = fairness.assess(ref, cur)
    assert result["albedo_risk"] >= 0.5
    assert any("albedo/material family" in c for c in result["unreconstructable"])


# ================================================= constrainable NOT raised by n_references (B4)

def test_constrainable_is_independent_of_n_references():
    ref = make_stats(log_key=0.30, b_hi=8.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    one = fairness.assess(ref, cur, n_references=1)
    many = fairness.assess(ref, cur, n_references=9)
    assert one["constrainable"] == many["constrainable"]       # extra views do NOT tighten
    assert one["n_references"] == 1 and many["n_references"] == 9   # ...but ARE reported


def test_n_references_is_coerced_to_int():
    result = fairness.assess(make_stats(), make_stats(), n_references="4")
    assert result["n_references"] == 4
    junk = fairness.assess(make_stats(), make_stats(), n_references=None)
    assert junk["n_references"] == 1                           # falls back to the default 1


# ============================================================ coverage alignment with critic (B4)

def test_coverage_scales_constrainable_and_is_reported_clamped():
    ref = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)      # albedo_risk ~ 0
    half = fairness.assess(ref, cur, coverage=0.5)
    assert half["coverage"] == pytest.approx(0.5)
    assert half["constrainable"] == pytest.approx((1.0 - half["albedo_risk"]) * 0.5, abs=1e-9)


@pytest.mark.parametrize("cov,expected_report", [
    (None, 1.0),        # unobserved coverage → reported full
    (1.5, 1.0),         # clamped high
    (-0.2, 0.0),        # clamped low
    (0.0, 0.0),         # reported 0.0 ...
])
def test_coverage_reported_field_is_clamped(cov, expected_report):
    result = fairness.assess(make_stats(), make_stats(), coverage=cov)
    assert result["coverage"] == pytest.approx(expected_report)


def test_zero_coverage_does_not_zero_constrainable():
    # per the B4 `clamp01(coverage or 1.0)` factor: coverage 0.0 reports 0.0 but acts as full,
    # so a fair reference is not spuriously reported unconstrainable
    ref = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    result = fairness.assess(ref, cur, coverage=0.0)
    assert result["coverage"] == 0.0
    assert result["constrainable"] == pytest.approx(1.0 - result["albedo_risk"], abs=1e-9)


# ============================================================================ remedy scoping (B6)

def test_remedy_only_present_on_unfair():
    unfair = fairness.assess(make_stats(log_key=0.9, b_hi=40.0, hi_clip=0.0),
                             make_stats(log_key=0.02, b_hi=-40.0, hi_clip=0.0))
    assert unfair["verdict"] == "unfair"
    assert unfair["remedy"] == fairness.REMEDY_LOCK

    marginal = fairness.assess(make_stats(log_key=0.19, b_hi=5.1, hi_clip=0.0),
                               make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0),
                               content_gap=True)
    assert marginal["verdict"] == "marginal"
    assert marginal["remedy"] == ""

    fair = fairness.assess(make_stats(), make_stats())
    assert fair["verdict"] == "fair"
    assert fair["remedy"] == ""


def test_remedy_empty_on_unknown():
    assert fairness.assess(None, None)["remedy"] == ""


# ============================================================================ bounds & banding

@pytest.mark.parametrize("ref,cur", [
    (make_stats(log_key=0.18, b_hi=5.0), make_stats(log_key=0.18, b_hi=5.0)),
    (make_stats(log_key=0.9, b_hi=90.0), make_stats(log_key=0.001, b_hi=-90.0)),
    (make_stats(log_key=0.0), make_stats(log_key=0.0)),
    same_scene_pair(b_full_ref=50.0, b_full_cur=-50.0),
    (make_stats(log_key=1e-9, b_hi=1e9), make_stats(log_key=1e9, b_hi=-1e9)),
])
def test_scores_are_bounded_and_gaps_nonnegative(ref, cur):
    result = fairness.assess(ref, cur, coverage=1.3, n_references=3)
    assert 0.0 <= result["constrainable"] <= 1.0
    assert 0.0 <= result["albedo_risk"] <= 1.0
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["predicted_ev_gap"] >= 0.0
    assert result["predicted_wb_gap"] >= 0.0
    assert result["verdict"] in ("fair", "marginal", "unfair", "unknown")
    assert isinstance(result["same_scene"], bool)
    assert isinstance(result["predicts_leash_trip"], bool)


def test_identical_reference_and_scene_reads_fair_and_highly_constrainable():
    # same albedo family, same exposure → the reference IS a fair yardstick
    s = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    result = fairness.assess(dict(s), dict(s))
    assert result["verdict"] == "fair"
    assert result["albedo_risk"] == pytest.approx(0.0, abs=1e-9)
    assert result["constrainable"] == pytest.approx(1.0, abs=1e-9)


def test_wildly_different_albedo_family_reads_unfair():
    # white-room reference (bright key, warm highlight) vs dark-wood scene (crushed, cool)
    ref = make_stats(log_key=0.85, b_hi=35.0, hi_clip=0.0)
    cur = make_stats(log_key=0.03, b_hi=-30.0, hi_clip=0.0)
    result = fairness.assess(ref, cur)
    assert result["verdict"] == "unfair"
    assert result["albedo_risk"] >= 0.66
    assert result["remedy"] == fairness.REMEDY_LOCK


# ============================================================ roles / notes / signature defaults

def test_roles_are_accepted_but_do_not_change_the_assessment():
    ref = make_stats(log_key=0.30, b_hi=8.0, hi_clip=0.0)
    cur = make_stats(log_key=0.18, b_hi=5.0, hi_clip=0.0)
    without = fairness.assess(ref, cur, n_references=3)
    with_roles = fairness.assess(ref, cur, n_references=3,
                                 roles=["primary", "angle_1", "angle_2"])
    assert with_roles["verdict"] == without["verdict"]
    assert with_roles["constrainable"] == without["constrainable"]


def test_notes_signal_the_guarantee_scope():
    # both authoritative signals present → notes claim alignment with the director/critic
    aligned = fairness.assess(make_stats(), make_stats(), leash_hits=0, content_gap=False)
    assert "director" in aligned["notes"] and "content-gap" in aligned["notes"]
    # neither present → notes narrow the guarantee to "the numbers fairness can see"
    predictive = fairness.assess(make_stats(), make_stats())
    assert "predictive" in predictive["notes"]


def test_signature_defaults_match_the_live_leashes_and_slopes():
    import inspect

    defaults = {p.name: p.default
                for p in inspect.signature(fairness.assess).parameters.values()
                if p.default is not inspect.Parameter.empty}
    assert defaults["ev_leash"] == MatchConfig().ev_leash
    assert defaults["wb_leash"] == MatchConfig().wb_leash
    assert defaults["wb_kelvin_per_b"] == solver.WB_KELVIN_PER_B
    assert defaults["same_scene_clip_frac"] == solver.SYMMETRIC_CLIP_FRAC
