"""Adversarial sweep over everything shipped 2026-07-26, written against the two failure
shapes this codebase has actually produced under pressure:

  1. CONFIDENTLY WRONG — a measurement or solve returning a clean, high-confidence answer
     for a question the data cannot support (the withdrawn hardness gate; the 0.841-vs-
     0.250 noise frame; the sweep defended at full trust while 195 degrees out).
  2. SCORING BETTER BY GIVING UP — a candidate improving its number by deleting the
     criteria it was failing (the sun-off exploit: 83.33 for surrender vs 79.09 for
     trying and missing).

Every test states the property it guards and why it matters. Invariants over magic
numbers throughout.
"""

import math

import pytest

from maxgaffer.core import cct, critic, metrics, patchgeom, solver, sunsolve
from maxgaffer.core.genome import LightingState


# ---------------------------------------------------------------- shared builders
def stats(key=0.2, p5=0.05, p50=0.45, p95=0.92, lab=(55.0, 2.0, 12.0),
          hue=(0.6, 0.4), grid=None, hot=0.03, hot_grid=None):
    lum = [0.0] * 32
    lum[10], lum[11] = 0.5, 0.5
    hue_hist = list(hue) + [0.0] * (12 - len(hue))
    g = grid if grid is not None else [0.1, 0.2, -0.1, 0.0, 0.05, -0.05, 0.1, -0.2, -0.1]
    hg = hot_grid if hot_grid is not None else [1.0 / 25] * 25
    return {"log_key": key, "p": {"5": p5, "25": 0.2, "50": p50, "75": 0.7, "95": p95},
            "contrast": p95 - p5, "lab_mean": list(lab), "lab_std": [20, 4, 6],
            "lum_hist": lum, "hue_hist": hue_hist, "grid": list(g),
            "hot_frac": hot, "hot_grid": list(hg)}


def rig(**kw):
    st = LightingState()
    base = {"sun.enabled": 1.0, "sun.azimuth_deg": 0.0, "sun.altitude_deg": 30.0,
            "exposure.ev": 0.0}
    base.update(kw)
    for k, v in base.items():
        st.set(k, v)
    return st


# ================================================================ the ceiling
def test_identity_scores_exactly_100_for_many_varied_frames():
    """The whole search rests on this: rendering the same state twice must score EXACTLY
    100.0, because any identity gap becomes a noise floor the optimizer chases forever
    (measured on-box: V-Ray is deterministic, so the true ceiling IS 100.0)."""
    frames = [stats(),
              stats(key=0.02, p5=0.0, p95=0.3, hot=0.0, hot_grid=[0.0] * 25),
              stats(key=0.9, p5=0.4, p95=1.0, lab=(90.0, 0.1, 0.1), hue=(1.0,)),
              stats(hot=0.4, hot_grid=[0.04] * 25),
              stats(grid=[0.0] * 9)]
    for f in frames:
        assert critic.score(f, f).score == 100.0


def test_junk_frames_score_badly_against_a_real_reference():
    ref = stats()
    black = stats(key=0.0001, p5=0.0, p50=0.0, p95=0.02, lab=(2.0, 0.0, 0.0),
                  hue=(0.0,), hot=0.0, hot_grid=[0.0] * 25, grid=[0.0] * 9)
    white = stats(key=0.98, p5=0.97, p50=0.99, p95=1.0, lab=(99.0, 0.0, 0.0),
                  hue=(0.0,), hot=0.0, hot_grid=[0.0] * 25, grid=[0.0] * 9)
    assert critic.score(ref, black).score < 30.0
    assert critic.score(ref, white).score < 30.0


# ================================================== giving up must never pay
def test_fewer_measurable_components_never_raises_the_score():
    """The sun-off exploit, generalised: strip components from the candidate one at a time
    and the score must never go UP relative to the fully-measured honest miss."""
    ref = stats()
    honest = stats(key=0.15, p5=0.08, lab=(50.0, 1.0, 9.0))
    base = critic.score(ref, honest).score
    for missing in ("hot_frac", "grid", "hue_hist", "lum_hist"):
        crippled = {k: v for k, v in honest.items() if k != missing}
        if missing == "hot_frac":
            crippled.pop("hot_grid", None)
        got = critic.score(ref, crippled).score
        assert got <= base + 1e-6, (
            f"dropping {missing} RAISED the score {base:.2f} -> {got:.2f}")


def test_a_candidate_with_no_patches_scores_zero_highlight_not_absent():
    ref, bare = stats(hot=0.05), stats(hot=0.0, hot_grid=[0.0] * 25)
    v = critic.score(ref, bare)
    assert v.components.get("highlight") == 0.0, "absence must be a zero, never a skip"


def test_weakest_link_cannot_be_escaped_by_shrinking_the_component_set():
    """The penalty gates on len(comps) >= 3; a candidate must not dodge it by presenting
    only two components — with fewer measurables its score is capped by renormalisation
    anyway, and must still sit at or below the fully-measured equivalent."""
    ref = stats()
    two_only = {"log_key": 0.2, "lab_mean": [55.0, 2.0, 12.0], "lab_std": [20, 4, 6]}
    v = critic.score(ref, two_only)
    assert v.score <= critic.score(ref, stats()).score


# ================================================================ highlight metric
def test_highlight_similarity_survives_adversarial_grids():
    good = {"hot_frac": 0.05, "hot_grid": [0.04] * 25}
    for evil in ({"hot_frac": 0.05, "hot_grid": [0.04] * 24},          # wrong length
                 {"hot_frac": 0.05, "hot_grid": [-1.0] * 25},          # negative mass
                 {"hot_frac": float("nan"), "hot_grid": [0.04] * 25},  # NaN fraction
                 {"hot_frac": 0.05, "hot_grid": [float("nan")] * 25}):
        got = metrics.highlight_similarity(good, evil)
        assert got is None or (isinstance(got, float) and 0.0 <= got <= 1.0
                               and got == got), evil


def test_highlight_asymmetry_is_the_contract():
    lit = {"hot_frac": 0.05, "hot_grid": [0.04] * 25}
    dark = {"hot_frac": 0.0, "hot_grid": [0.0] * 25}
    assert metrics.highlight_similarity(lit, dark) == 0.0     # gave up on the sun
    assert metrics.highlight_similarity(dark, dark) == 1.0    # both overcast: agreement
    assert metrics.highlight_similarity(lit, lit) == 1.0


# ================================================================ ambient solver
def test_solve_ambient_refuses_corrupt_percentiles():
    good = stats()
    for evil in (stats(p5=0.5, p95=0.5),                    # zero dynamic range
                 stats(p5=0.9, p95=0.1),                    # inverted
                 stats(p5=0.0, p95=0.0),                    # crushed to black
                 {"p": {"5": float("nan"), "95": 0.9}},     # NaN
                 {"p": "nonsense"}, {}):
        out = solver.solve_ambient(good, evil, current_dome=1.0, current_sun=1.0)
        for v in out.values():
            assert isinstance(v, float) and v == v and v > 0, (evil, out)


def test_solve_ambient_octave_cap_binds_and_zero_hosts_are_skipped():
    ref, cur = stats(p5=0.02, p95=0.99), stats(p5=0.60, p95=0.99)
    out = solver.solve_ambient(ref, cur, current_dome=1.0, current_sun=1.0)
    if "dome.intensity" in out:
        assert out["dome.intensity"] >= 2.0 ** -solver.AMBIENT_MAX_OCTAVES - 1e-9
    # a rig with no dome host must never be proposed one; a zero current must not divide
    assert "dome.intensity" not in solver.solve_ambient(ref, cur, None, 1.0)
    out0 = solver.solve_ambient(ref, cur, current_dome=0.0, current_sun=0.0)
    for v in out0.values():
        assert v == v and abs(v) != float("inf")


# ================================================================ sun solve
def _world(peak_az, sharp=True, partial_at=(), flat=False):
    applied = {"az": 0.0, "alt": 30.0}
    n = {"probes": 0}

    def apply(st):
        applied["az"], applied["alt"] = st.get("sun.azimuth_deg"), st.get("sun.altitude_deg")

    def render(tag):
        return f"/tmp/{tag}.png"

    def stats_fn(path):
        n["probes"] += 1
        if n["probes"] in partial_at:
            return {"lum_hist": [0.03] * 32}            # a dict MISSING the patch map
        if flat:
            return {"hot_frac": 0.05, "hot_grid": [0.04] * 25}
        d = abs((applied["az"] - peak_az + 180.0) % 360.0 - 180.0)
        frac = 0.17 * max(0.02, 1.0 - (d / 180.0) ** (1 if sharp else 0.15))
        grid = [0.0] * 25
        grid[12] = 1.0
        return {"hot_frac": frac, "hot_grid": grid}

    return apply, render, stats_fn


REF = {"hot_frac": 0.17, "hot_grid": [0.0] * 12 + [1.0] + [0.0] * 12}


def test_a_flat_landscape_is_never_reported_as_a_confident_answer():
    """The sweep's 195-degree failure, forbidden at the solve level: when every direction
    lights the scene alike, the winner is an artefact and confidence must say so."""
    apply, render, stats_fn = _world(0.0, flat=True)
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats_fn)
    assert out is not None and out["confidence"] < 0.5


def test_partial_stats_tables_cannot_crown_a_probe_by_default():
    """A probe whose rivals went unmeasured must not win on a partial table — the exact
    bug class run_sun_sweep had to fix. Some probes return stats missing the patch map."""
    apply, render, stats_fn = _world(105.0, partial_at=tuple(range(2, 30, 3)))
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats_fn)
    if out is not None:
        err = abs((out["azimuth_deg"] - 105.0 + 180.0) % 360.0 - 180.0)
        assert err <= 45.0 or out["confidence"] < 0.5, (
            "wrong answer at high confidence off a partial table")


def test_zero_and_one_probe_budgets_degrade_cleanly():
    apply, render, stats_fn = _world(105.0)
    assert sunsolve.solve_sun_angles(rig(), REF, apply, render, stats_fn,
                                     max_probes=0) is None
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats_fn, max_probes=1)
    assert out is None or 0.0 <= out["confidence"] <= 1.0


def test_cancellation_mid_grid_never_raises():
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 7

    apply, render, stats_fn = _world(105.0)
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats_fn,
                                    should_cancel=cancel)
    assert out is None or 0.0 <= out["confidence"] <= 1.0


def test_nan_in_the_reference_map_is_refused_not_propagated():
    apply, render, stats_fn = _world(105.0)
    bad_ref = {"hot_frac": float("nan"), "hot_grid": [0.04] * 25}
    out = sunsolve.solve_sun_angles(rig(), bad_ref, apply, render, stats_fn)
    if out is not None:
        assert out["confidence"] == out["confidence"], "NaN confidence escaped"
        assert out["azimuth_deg"] == out["azimuth_deg"]


# ================================================================ cct
def test_cct_refuses_off_locus_light_rather_than_naming_it():
    """Ungated, McCamy reports a magenta light at ~29000 K and pure blue at ~1667 K — a
    blue light labelled candlelight. The gate is the module's whole value."""
    for bad in ([0.9, 0.1, 0.9],       # magenta
                [0.0, 0.0, 1.0],       # blue primary
                [0.1, 0.9, 0.1]):      # deep green
        st = {"illum": bad, "illum_sog": bad, "illum_edge": bad}
        got = cct.illuminant_kelvin(st)
        assert got is None or abs(got.get("duv", 0.0)) <= 0.05, bad


def test_cct_degenerate_vectors_return_none():
    for bad in ([0, 0, 0], [1e-15] * 3, [float("nan")] * 3, [1.0], "junk", None):
        st = {"illum": bad, "illum_sog": bad, "illum_edge": bad}
        assert cct.illuminant_kelvin(st) is None, bad


def test_estimator_disagreement_lowers_confidence():
    agree = {"illum": [0.60, 0.55, 0.45], "illum_sog": [0.60, 0.55, 0.45],
             "illum_edge": [0.60, 0.55, 0.45]}
    disagree = {"illum": [0.60, 0.55, 0.45], "illum_sog": [0.60, 0.55, 0.45],
                "illum_edge": [0.45, 0.55, 0.62]}
    a, d = cct.illuminant_kelvin(agree), cct.illuminant_kelvin(disagree)
    if a is not None and d is not None:
        assert d["confidence"] <= a["confidence"] + 1e-9


# ================================================================ patchgeom
def test_bearing_hint_confidence_never_reaches_the_solve_threshold():
    """A free geometric guess must never outrank a rendered measurement: the controller
    accepts a solve at confidence >= 0.5, so every hint must sit strictly below it."""
    grid_left, grid_right = [0.0] * 25, [0.0] * 25
    grid_left[10] = 1.0
    grid_right[14] = 1.0
    ref = {"hot_frac": 0.05, "hot_grid": grid_right}
    cur = {"hot_frac": 0.05, "hot_grid": grid_left}
    for az in (0.0, 45.0, 170.0, 200.0, 350.0):
        hint = patchgeom.bearing_hint(ref, cur, az)
        if hint is not None:
            assert hint["confidence"] < 0.5, (az, hint)


def test_bearing_agreement_is_none_not_zero_on_missing_maps():
    """Unlike the SCORED highlight metric, agreement is diagnostic: a missing map is
    ignorance, and fabricating 0.0 would read as 'they point opposite ways' — a claim the
    evidence never made."""
    lit = {"hot_frac": 0.05, "hot_grid": [0.04] * 25}
    assert patchgeom.bearing_agreement(lit, {"hot_frac": 0.0, "hot_grid": [0.0] * 25}) is None
    assert patchgeom.bearing_agreement(None, lit) is None


# ================================================================ weight tables
def test_every_preference_profile_is_normalised_and_carries_highlight():
    for name, w in critic.PREFERENCE_PROFILES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, name
        assert w.get("highlight", 0.0) > 0.0, name
        assert set(w) == set(critic.DEFAULT_WEIGHTS), (
            f"{name} drifts from the component set — renormalisation would silently skew")


def test_a_raising_hook_skips_the_probe_and_never_kills_the_solve():
    """Found by escalating this very suite: apply hits pymxs (a deleted node raises),
    render hits V-Ray, stats hits a decoder — and one such failure on ONE frame of the
    44-probe grid propagated out of solve_sun_angles and killed the entire match. The
    basin picker already lives by "one bad candidate is a skip, not a dead match"; the
    solve, being newest, did not."""
    n = {"i": 0}

    def apply_raises(_s):
        n["i"] += 1
        if n["i"] % 5 == 0:
            raise RuntimeError("deleted scene node")

    def stats_raises(_p):
        if n["i"] % 7 == 0:
            raise RuntimeError("stats engine died")
        return {"hot_frac": 0.1, "hot_grid": [0.04] * 25}

    out = sunsolve.solve_sun_angles(rig(), REF, apply_raises,
                                    lambda t: "/tmp/x.png", stats_raises)
    assert out is None or 0.0 <= out["confidence"] <= 1.0

    def apply_always_raises(_s):
        raise RuntimeError("scene gone entirely")

    assert sunsolve.solve_sun_angles(rig(), REF, apply_always_raises,
                                     lambda t: "/tmp/x.png",
                                     lambda p: None) is None
