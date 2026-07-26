"""Global sun-angle solve — the answer to four runs of one scene returning 2.8, 171, 168
and 78 degrees of azimuth error. Local search cannot climb a gradient that is not there;
now that patch agreement discriminates direction, the angle is solved, not searched."""

from maxgaffer.core import sunsolve
from maxgaffer.core.genome import LightingState


def rig(**kw):
    st = LightingState()
    base = {"sun.enabled": 1.0, "sun.azimuth_deg": 0.0, "sun.altitude_deg": 30.0,
            "exposure.ev": 0.0}
    base.update(kw)
    for k, v in base.items():
        st.set(k, v)
    return st


def world(true_az=105.0, true_alt=10.0, flat=False):
    """A scene whose patch map depends on the sun angle, so the solve has something real to
    find. `flat` models the honest hard case: every direction lights the room alike."""
    applied = {"az": 0.0, "alt": 30.0}
    rendered = []

    def apply(state):
        applied["az"] = state.get("sun.azimuth_deg")
        applied["alt"] = state.get("sun.altitude_deg")

    def render(tag):
        rendered.append((tag, applied["az"], applied["alt"]))
        return f"/tmp/{tag}.png"

    def stats(path):
        if flat:
            return {"hot_frac": 0.05, "hot_grid": [0.04] * 25}
        d_az = abs((applied["az"] - true_az + 180.0) % 360.0 - 180.0)
        d_alt = abs(applied["alt"] - true_alt)
        # patch AREA falls off as the sun swings away, and its placement shifts with it —
        # both of which highlight_similarity reads
        frac = 0.17 * max(0.02, 1.0 - d_az / 180.0) * max(0.3, 1.0 - d_alt / 90.0)
        grid = [0.0] * 25
        cell = int(min(4, max(0, round(2 + 2 * (d_az / 180.0)))))
        grid[10 + cell] = 1.0
        return {"hot_frac": frac, "hot_grid": grid}

    return apply, render, stats, rendered


REF = {"hot_frac": 0.17, "hot_grid": [0.0] * 12 + [1.0] + [0.0] * 12}


def test_the_solve_finds_the_sun_globally_not_locally():
    """The whole point: it looks everywhere before it refines, so a wrong basin cannot trap
    it. The rig starts at azimuth 0 with the true sun at 105."""
    apply, render, stats, rendered = world(true_az=105.0)
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats)
    assert out is not None
    err = abs((out["azimuth_deg"] - 105.0 + 180.0) % 360.0 - 180.0)
    assert err <= 15.0, (out["azimuth_deg"], err)
    assert out["probes"] == len(rendered)


def test_it_reports_when_every_direction_looks_the_same():
    """A flat landscape is a FINDING. Reporting the arbitrary winner as a measurement is
    how the sweep put the sun 168 degrees out and then had its answer defended."""
    apply, render, stats, _ = world(flat=True)
    logs = []
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats, log=logs.append)
    assert out is not None
    assert out["confidence"] < 0.5, out
    assert any("about equally" in m for m in logs)


def test_a_reference_with_no_directional_light_is_not_ground_over():
    """Overcast, night, a flat interior — there is nothing to solve toward, and burning
    fifty renders to discover that is worse than saying so."""
    apply, render, stats, rendered = world()
    logs = []
    out = sunsolve.solve_sun_angles(rig(), {"hot_frac": 0.0, "hot_grid": [0.0] * 25},
                                    apply, render, stats, log=logs.append)
    assert out is None
    assert rendered == [], "it must not render anything it cannot learn from"
    assert any("no directional light" in m for m in logs)


def test_a_locked_azimuth_is_never_solved_over():
    apply, render, stats, rendered = world()
    assert sunsolve.solve_sun_angles(rig(), REF, apply, render, stats,
                                     locks={"sun.azimuth_deg"}) is None
    assert rendered == []
    # ...and a rig with no sun axis at all is simply not this stage's problem
    dome_only = LightingState()
    dome_only.set("dome.intensity", 1.0)
    assert sunsolve.solve_sun_angles(dome_only, REF, apply, render, stats) is None


def test_the_probe_budget_is_finite_and_respected():
    apply, render, stats, rendered = world()
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats, max_probes=9)
    assert out is not None
    assert out["probes"] <= 9 and len(rendered) <= 9


def test_a_dead_renderer_or_a_cancel_degrades_instead_of_raising():
    apply, render, stats, _ = world()
    assert sunsolve.solve_sun_angles(rig(), REF, apply, lambda t: None, stats) is None
    assert sunsolve.solve_sun_angles(rig(), REF, apply, render, lambda p: None) is None
    assert sunsolve.solve_sun_angles(rig(), REF, apply, render, stats,
                                     should_cancel=lambda: True) is None


def test_altitude_is_solved_too_when_the_rig_has_it():
    """Sun HEIGHT was consistently wrong as well (44 degrees against a 10-degree target in
    one measured run), and it is the second half of the same two-parameter question."""
    apply, render, stats, _ = world(true_az=105.0, true_alt=10.0)
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats)
    assert out["altitude_deg"] is not None
    assert out["altitude_deg"] < 30.0, "a low sun must not be reported as a high one"


def test_the_controller_solves_the_sun_before_it_sweeps_for_it():
    """Order matters: the solve is deterministic, dense and scored on the measure that
    actually discriminates direction, while the sweep is 12 samples judged partly by an
    LLM. When the solve is decisive the sweep is skipped entirely — it would cost twelve
    more renders to re-answer a question already answered better."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    solve_at = src.index("sunsolve.solve_sun_angles(")
    sweep_at = src.index("run_sun_sweep(")
    assert solve_at < sweep_at, "the weaker instrument must not go first"
    between = src[solve_at:sweep_at]
    assert "do_sweep = False" in between, "a decisive solve must skip the sweep"
    # and a NON-decisive solve must fall through to the sweep rather than be trusted
    assert 'solved.get("confidence", 0.0) >= 0.5' in between


def test_a_weak_solve_does_not_get_defended_at_full_strength():
    """Same rule the sweep had to learn: an answer is defended only as far as it was
    earned. The transfer weight scales with the solve's own margin."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    seg = src[src.index("sunsolve.solve_sun_angles("):]
    assert 'TRANSFER_WEIGHT * max(' in seg[:2200]
    assert 'sem_live["sun_bearing_agreement"] = solved["confidence"]' in seg[:2200]


def test_the_solve_renders_at_sweep_size_not_loop_size():
    """It probes up to 56 directions and is comparing WHERE the light falls, not judging
    tone. Measured: because its tags start with "sunsolve" rather than "sweep" it was
    rendering at full loop resolution, turning a two-minute stage into most of a
    fifty-minute match."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    hook = src[src.index("def render_hook("):src.index("def apply_hook(")]
    assert 'tag.startswith(("sweep", "sunsolve"))' in hook
    assert "profile.sweep_width" in hook


def test_a_reference_from_another_building_is_ranked_on_side_not_layout():
    """Patch PLACEMENT only carries direction when the reference shares the scene's
    geometry — the patches land on the same walls. Against a photograph of a different
    building those positions encode THAT building, so matching them is worse than
    uninformative: measured on-box 2026-07-26, a real dawn plate onto the CG room settled at
    azimuth 348 with the transfer reading calling the bearing 169 degrees out.

    The tell needs no knowledge of the reference: if the BEST achievable agreement is low,
    the scene cannot reproduce that layout at any sun angle."""
    applied = {"az": 0.0}
    # the reference's light lands HIGH on a right-hand wall — that building's geometry
    ref_grid = [0.0] * 25
    ref_grid[4] = ref_grid[9] = 0.5
    ref = {"hot_frac": 0.03, "hot_grid": ref_grid}

    def apply(state):
        applied["az"] = state.get("sun.azimuth_deg")

    def stats(path):
        # OUR room puts its patches low on the floor — no cell is ever shared, so placement
        # agreement is 0 at every angle. The SIDE still tracks the sun, and that is the one
        # thing that survives a change of building.
        col = 0 if applied["az"] < 180 else 4
        grid = [0.0] * 25
        grid[15 + col] = grid[20 + col] = 0.5
        return {"hot_frac": 0.028, "hot_grid": grid}

    out = sunsolve.solve_sun_angles(rig(), ref, apply, lambda t: f"/tmp/{t}.png", stats,
                                    log=lambda m: None)
    assert out is not None
    assert out["cross_domain"] is True, out["score"]
    # it must pick the half-turn whose light is on the reference's side...
    assert out["azimuth_deg"] >= 180.0, out
    # ...and never claim a confident answer from a coarser question on weaker evidence
    assert out["confidence"] <= 0.45


def test_a_same_scene_solve_is_untouched_by_the_fallback():
    """The fallback must not fire when placement genuinely works — row 4 solved to 2.5
    degrees on it."""
    apply, render, stats, _ = world(true_az=105.0)
    out = sunsolve.solve_sun_angles(rig(), REF, apply, render, stats)
    assert out["cross_domain"] is False
    assert out["confidence"] > 0.45
