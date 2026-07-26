"""Separating the two lights. The search kept trading ambient against exposure because
nothing measured them apart; the histogram's two ends do."""

from maxgaffer.core import solver
from maxgaffer.core.genome import LightingState


def stats(p5, p95, key=0.2):
    return {"log_key": key, "p": {"5": p5, "25": 0.3, "50": 0.5, "75": 0.7, "95": p95},
            "lab_mean": [55.0, 2.0, 12.0], "lab_std": [20, 4, 6],
            "lum_hist": [0.03] * 32, "hue_hist": [0.5, 0.5] + [0.0] * 10,
            "contrast": p95 - p5}


def test_lifted_shadows_pull_the_ambient_down_not_the_exposure():
    """Measured on-box 2026-07-26: a match with the sun's DIRECTION right to 2.5 degrees
    still recovered dome 1.66 against a 0.55 target, with exposure compensating and the
    shadows coming back lifted (p5 0.183 against the reference's 0.164). Nothing was
    solving the two lights apart."""
    ref, cur = stats(0.164, 0.958), stats(0.183, 0.971)
    out = solver.solve_ambient(ref, cur, current_dome=1.66, current_sun=1.0)
    assert "dome.intensity" in out
    assert out["dome.intensity"] < 1.66, out
    # crushed shadows pull the other way
    out2 = solver.solve_ambient(stats(0.30, 0.958), cur, current_dome=1.0, current_sun=1.0)
    assert out2["dome.intensity"] > 1.0


def test_the_sun_is_measured_above_the_ambient_floor_not_from_raw_p95():
    """Comparing raw p95 would credit the sun for ambient the dome is already supplying.
    Here both frames have the same highlight level but different floors, so the SUN's own
    contribution differs and only the decomposed solve sees it."""
    ref, cur = stats(0.10, 0.90), stats(0.40, 0.90)
    out = solver.solve_ambient(ref, cur, current_dome=1.0, current_sun=1.0)
    assert out["sun.intensity"] > 1.0, (
        "the reference's sun does more work above a darker floor")
    assert out["dome.intensity"] < 1.0


def test_a_match_that_already_agrees_is_left_alone():
    same = stats(0.164, 0.958)
    assert solver.solve_ambient(same, same, 1.0, 1.0) == {}


def test_it_degrades_on_junk_and_respects_capability_and_locks():
    good = stats(0.164, 0.958)
    assert solver.solve_ambient({}, good, 1.0, 1.0) == {}
    assert solver.solve_ambient(good, {"p": "nonsense"}, 1.0, 1.0) == {}
    assert solver.solve_ambient(stats(0.0, 0.9), stats(0.0, 0.9), 1.0, 1.0) == {}
    # a rig with no dome host must never be proposed one
    assert "dome.intensity" not in solver.solve_ambient(
        good, stats(0.30, 0.958), current_dome=None, current_sun=1.0)


def test_the_correction_is_capped_so_one_bad_frame_cannot_swing_the_rig():
    ref, cur = stats(0.02, 0.99), stats(0.60, 0.99)      # absurdly far apart
    out = solver.solve_ambient(ref, cur, current_dome=1.0, current_sun=1.0)
    assert out["dome.intensity"] >= 1.0 * 2.0 ** -solver.AMBIENT_MAX_OCTAVES - 1e-9


def test_the_analytic_pass_carries_it_and_honours_locks():
    st = LightingState()
    for k, v in {"exposure.ev": 10.0, "dome.intensity": 1.66,
                 "sun.intensity": 1.0}.items():
        st.set(k, v)
    ref, cur = stats(0.164, 0.958), stats(0.183, 0.971)
    assert "dome.intensity" in solver.analytic_pass(st, ref, cur)
    assert "dome.intensity" not in solver.analytic_pass(
        st, ref, cur, locks={"dome.intensity"})
