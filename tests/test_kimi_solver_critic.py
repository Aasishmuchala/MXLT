"""Cluster C regressions — solver/critic/rules/feedback hardening against the sidecar
trust boundary (unvalidated stats + human-editable semantics/notes). Pure core, off-Max."""

import math

from maxgaffer.core import critic, feedback, rules, solver
from maxgaffer.core.genome import LightingState

NAN = float("nan")


def rig_state():
    st = LightingState()
    for k, v in {
        "sun.enabled": 1, "sun.azimuth_deg": 0.0, "sun.altitude_deg": 45.0,
        "sun.intensity": 1.0, "sun.size": 1.0, "sun.turbidity": 3.0,
        "dome.enabled": 1, "dome.rotation_deg": 0.0, "dome.intensity": 1.0,
        "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0,
    }.items():
        st.set(k, v)
    st.groups["practicals"] = 1.0
    return st


def stats(**over):
    base = {
        "log_key": 0.18,
        "lab_mean": [50.0, 0.0, 5.0],
        "p": {"5": 0.02, "95": 0.9},
        "lum_hist": [0.0] * 10 + [0.6] + [0.0] * 9 + [0.4] + [0.0] * 12,
        "hue_hist": [0.0, 0.7] + [0.0] * 5 + [0.3] + [0.0] * 4,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------- solver
def test_solve_ev_noops_on_malformed_log_key():
    good = stats()
    for bad in ({"log_key": None}, {"log_key": "junk"}, {"log_key": [0.2]}):
        assert solver.solve_ev(bad, good, 12.0) is None
        assert solver.solve_ev(good, bad, 12.0) is None


def test_solve_ev_noops_on_nan_instead_of_max_step():
    assert solver.solve_ev({"log_key": NAN}, stats(log_key=0.05), 12.0) is None
    assert solver.solve_ev(stats(log_key=0.05), {"log_key": NAN}, 12.0) is None
    assert solver.solve_ev({"log_key": math.inf}, stats(), 12.0) is None


def test_solve_wb_noops_on_malformed_lab():
    good = stats()
    for bad in ({"lab_mean": None}, {"lab_mean": [0, 0]}, {"lab_mean": "warm"},
                {"lab_mean": [50, 0, NAN]}):
        assert solver.solve_wb(bad, good, 6500.0) is None
        assert solver.solve_wb(good, bad, 6500.0) is None
    assert solver.solve_wb({}, {}, 6500.0) is None


def test_solve_wb_highlight_choice_is_validity_based_not_presence_based():
    # ref carries a null lab_mean_hi, cur a valid one: must NOT compare ref's full mean
    # against cur's highlight mean — falls back to full-vs-full instead
    ref = {"lab_mean_hi": None, "lab_mean": [50.0, 0.0, 20.0]}
    cur = {"lab_mean_hi": [80.0, 0.0, 3.0], "lab_mean": [50.0, 0.0, 2.0]}
    full_only = solver.solve_wb({"lab_mean": [50.0, 0.0, 20.0]},
                                {"lab_mean": [50.0, 0.0, 2.0]}, 6500.0)
    assert solver.solve_wb(ref, cur, 6500.0) == full_only == 8000.0
    # both highlight means valid → highlight path still preferred
    hi = solver.solve_wb({"lab_mean": [50, 0, 20], "lab_mean_hi": [80, 0, 4.0]},
                         {"lab_mean": [50, 0, 2], "lab_mean_hi": [80, 0, 3.5]}, 6500.0)
    assert hi is None  # highlights agree within the deadband


def test_analytic_pass_survives_malformed_stats():
    st = rig_state()
    bad = {"log_key": None, "lab_mean": None}
    assert solver.analytic_pass(st, bad, stats()) == {}
    assert solver.analytic_pass(st, stats(), bad) == {}
    # well-formed stats still produce the analytic move
    changes = solver.analytic_pass(st, stats(log_key=0.4), stats(log_key=0.1))
    assert "exposure.ev" in changes


# --------------------------------------------------------------------- critic
def test_critic_weights_mistyped_fall_back_to_defaults():
    ref, cur = stats(), stats(log_key=0.1, lab_mean=[40.0, 5.0, -10.0])
    default = critic.score(ref, cur).score
    for junk in ("junk", [0.2] * 6, 42, {"key": "lots"}, {"key": -20.0},
                 {"key": NAN}, {"key": None}):
        v = critic.score(ref, cur, junk)
        assert v.score == default
        assert 0.0 <= v.score <= 100.0


def test_critic_weights_partial_junk_keeps_valid_overrides():
    ref, cur = stats(), stats(lab_mean=[50.0, 0.0, 25.0])  # only color differs
    mixed = critic.score(ref, cur, {"color": 0.9, "key": "junk"}).score
    assert mixed == critic.score(ref, cur, {"color": 0.9}).score
    assert mixed < critic.score(ref, cur).score  # the valid override took effect


def test_critic_survives_mistyped_stats_fields():
    good = stats()
    for bad in ({"log_key": None}, {"log_key": "junk"}, {"log_key": NAN},
                {"lab_mean": None}, {"lab_mean": [0, 0]}, {"lab_mean": [0, 0, NAN]},
                {"p": {"5": None, "95": "x"}}, {"p": None},
                {"hue_hist": None}, {"lum_hist": None}, {"lum_hist": ["x", 1]},
                {"grid": "junk", "grid5": [1, "a"]}):
        for a, b in ((bad, good), (good, bad), (bad, bad)):
            v = critic.score(a, b)
            assert 0.0 <= v.score <= 100.0


def test_critic_achromatic_pair_renormalizes_hue_out():
    # black reference vs white render: both hue histograms empty = no hue information,
    # so hue must not award free credit (direction component mirrors this guard)
    black = stats(log_key=1e-5, lab_mean=[0.0, 0.0, 0.0], p={"5": 0.0, "95": 0.0},
                  lum_hist=[1.0] + [0.0] * 31, hue_hist=[0.0] * 12)
    white = stats(log_key=0.95, lab_mean=[100.0, 0.0, 0.0], p={"5": 0.99, "95": 1.0},
                  lum_hist=[0.0] * 31 + [1.0], hue_hist=[0.0] * 12)
    v = critic.score(black, white)
    assert "hue" not in v.components
    assert v.score < 1.0  # every remaining component sees the maximal mismatch


def test_critic_hue_one_sided_achromatic_is_signal_not_skipped():
    # a grey render of a COLORFUL reference is a real mismatch: the hue component
    # joins at 0 (cosine vs zeros) and pulls the score down — skipping it would
    # inflate exactly the renders that lost the reference's color
    v = critic.score(stats(), stats(hue_hist=[0.0] * 12))
    assert v.components.get("hue") == 0.0
    # two ACHROMATIC images carry no hue information — skipped and renormalized
    both = critic.score(stats(hue_hist=[0.0] * 12), stats(hue_hist=[0.0] * 12))
    assert "hue" not in both.components
    # identical chromatic stats keep the hue component at full marks
    same = critic.score(stats(), stats())
    assert same.components["hue"] == 1.0 and same.score == 100.0


# --------------------------------------------------------------------- rules
def test_initial_state_coerces_hand_edited_semantics():
    sem = {"scene_type": "exterior", "time_of_day": "afternoon", "sky": "clear",
           "sun_active": True, "sun_bearing_deg": "high", "sun_altitude_band": "mid",
           "light_quality": "hard", "wb_kelvin_estimate": None, "practicals_on": True,
           "atmosphere": "none", "contrast_character": "normal",
           "key_notes": "", "confidence": 0.9}
    st, _why = rules.initial_state(sem, rig_state(), camera_yaw_deg=90.0)
    assert st.get("sun.azimuth_deg") == 90.0        # junk bearing → 0.0 offset
    assert st.get("exposure.wb_kelvin") == 6500.0   # null estimate → craft default


def test_initial_state_coerces_nan_and_numeric_strings():
    sem = {"time_of_day": "afternoon", "sky": "clear", "sun_active": True,
           "sun_bearing_deg": NAN, "sun_altitude_band": "mid",
           "wb_kelvin_estimate": "5200"}
    st, _why = rules.initial_state(sem, rig_state(), camera_yaw_deg=45.0)
    assert math.isfinite(st.get("sun.azimuth_deg"))
    assert st.get("sun.azimuth_deg") == 45.0        # NaN bearing → 0.0, never NaN state
    assert st.get("exposure.wb_kelvin") == 5200.0   # numeric strings still coerce


# --------------------------------------------------------------------- feedback
def test_negated_note_produces_no_nudge():
    keys, groups = rig_state().keys(), ["practicals"]
    assert feedback.nudges_from_note("it's not too bright", keys, groups) == {}
    assert feedback.nudges_from_note("not blown at all", keys, groups) == {}
    assert feedback.nudges_from_note("don't make it too dark", keys, groups) == {}


def test_negation_guard_is_clause_local():
    st = rig_state()
    d = feedback.nudges_from_note("not bad, still too bright",
                                  st.keys(), list(st.groups))
    assert d == {"exposure.ev": 0.7}                # negator in the other clause only
    d2 = feedback.nudges_from_note("exposure is too much and it's too warm",
                                   st.keys(), list(st.groups))
    assert d2["exposure.ev"] == 0.7 and d2["exposure.wb_kelvin"] == -800.0


def test_zeroed_group_nudge_bootstraps_off_zero():
    st = rig_state()
    st.groups["practicals"] = 0.0                   # daylight first guess killed them
    d = feedback.nudges_from_note("practicals too dim", st.keys(), list(st.groups))
    assert d == {"group.practicals": 0.5}
    _new, applied = feedback.apply_note_deltas(st, d)
    assert applied["group.practicals"] > 0.0        # no longer a silent no-op
    assert abs(applied["group.practicals"] - 0.25 * 2 ** 0.5) < 1e-9
    # non-zero groups keep the pure multiplicative path
    _new2, applied2 = feedback.apply_note_deltas(rig_state(), d)
    assert abs(applied2["group.practicals"] - 2 ** 0.5) < 1e-9
