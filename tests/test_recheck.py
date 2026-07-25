"""Full-codebase recheck regressions — each test encodes a defect found by the review."""

import json

from maxgaffer.core.director import Hooks, MatchConfig, run_match
from maxgaffer.core.genome import LightingState
from maxgaffer.core.solver import analytic_pass

REF = {"log_key": 0.20, "lab_mean": [55.0, 2.0, 12.0], "lab_std": [20, 4, 6],
       "p": {"5": 0.03, "25": 0.2, "50": 0.45, "75": 0.7, "95": 0.92},
       "contrast": 0.89,
       "lum_hist": [0.0] * 10 + [0.5, 0.5] + [0.0] * 20,
       "hue_hist": [0.6, 0.4] + [0.0] * 10}


def dark(ref):
    s = dict(ref)
    s["log_key"] = 0.02
    return s


# ------------------------------------------------- solver capability gate (exposure-less rig)
def test_solver_never_proposes_params_the_rig_lacks():
    st = LightingState()
    st.set("sun.azimuth_deg", 100.0)           # a rig with no exposure host at all
    assert analytic_pass(st, REF, dark(REF)) == {}
    st.set("exposure.ev", 12.0)                # EV host only — WB still absent
    changes = analytic_pass(st, REF, dark(REF))
    assert "exposure.ev" in changes
    assert "exposure.wb_kelvin" not in changes


def test_exposure_less_rig_runs_clean_no_phantom_keys_no_false_leash():
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.altitude_deg": 30.0,
                 "sun.intensity": 1.0}.items():
        st.set(k, v)
    logs = []
    hooks = Hooks(apply=lambda s: None, render=lambda t: f"/tmp/{t}.png",
                  stats=lambda p: dark(REF),
                  llm_deltas=lambda ctx: json.dumps(
                      {"assessment": "", "changes": [], "stop": False}),
                  log=logs.append)
    res = run_match(st, REF, {}, hooks, MatchConfig(max_iterations=3, stall_patience=99))
    assert "exposure.ev" not in res.best_state.values          # no phantom key created
    assert all(not r.analytic_changes for r in res.iterations)  # solver stayed silent
    assert not any("leash" in m or "albedo" in m for m in logs)  # no false diagnosis


# ------------------------------------------------- slump-revert must re-measure, not re-tweak
def test_revert_iteration_reasons_from_the_restored_states_own_stats():
    """After a slump-revert the loop must never solve or prompt from the ABANDONED frame's
    stats (the original recheck: it was applying stale-evidence changes to the restored
    state). It used to buy that safety by skipping the iteration and re-rendering.

    Renders are deterministic in the lighting state (on-box 2026-07-25: a state re-rendered
    scores 100.0 against itself), and the restored state was already measured when it
    became best — so the loop now ADOPTS those cached stats. Same guarantee, no wasted
    render: measured on-box, 5 of 9 iterations in a live match were being spent
    re-measuring states the loop already had."""
    good, bad = dict(REF), dark(REF)
    # iter0 great (best), iter1+2 slump (revert fires on iter2)
    stats_seq = [good, bad, bad, good]
    reply = json.dumps({"assessment": "", "changes": [
        {"param": "sun.intensity", "value": 1.4, "why": "ratio"}], "stop": False})
    replies = [reply] * 4
    rendered = []

    def llm(ctx):
        return replies.pop(0)

    hooks = Hooks(apply=lambda s: None,
                  render=lambda t: rendered.append(t) or f"/tmp/{t}.png",
                  stats=lambda p: stats_seq.pop(0) if stats_seq else dict(REF),
                  llm_deltas=llm, log=lambda m: None)
    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.intensity": 1.0,
                 "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    res = run_match(st, REF, {}, hooks,
                    MatchConfig(max_iterations=4, target_score=101, stall_patience=99,
                                slump_tolerance=1.0))
    reverted = [r for r in res.iterations if r.reverted_to_best]
    assert reverted, "test setup should trigger a revert"
    rev = reverted[0]
    # the revert iteration reports the RESTORED state's score and frame — never the
    # abandoned one it had just measured
    assert rev.score == res.best_score
    assert rev.render_path == res.best_render
    # and it did NOT spend an extra render to learn what was already known
    assert len(rendered) == len(res.iterations)
    assert any(r.index > rev.index for r in res.iterations)


# ------------------------------------------------- vantage output detection (frame suffixes)
def test_vantage_output_written_accepts_frame_suffixes(tmp_path):
    from maxgaffer.maxbridge.vantage import _output_written

    exact = tmp_path / "Cam01.png"
    assert not _output_written(str(exact))
    (tmp_path / "Cam01.0000.png").write_bytes(b"x" * 10)   # Vantage-style frame suffix
    assert _output_written(str(exact))
    exact.write_bytes(b"")                                  # empty exact file ≠ success
    assert _output_written(str(exact))                      # (suffix file still counts)
    (tmp_path / "Cam01.0000.png").unlink()
    assert not _output_written(str(exact))                  # empty-only → not written


# ------------------------------------------------- bridge query functions degrade off-Max
def test_bridge_queries_never_raise_off_max():
    from maxgaffer.maxbridge import scene as sc

    assert sc.scene_path() == ""
    assert sc.list_cameras() == []
    assert sc.get_camera("X") is None
    assert sc.set_active_camera("X") is False


def test_controller_session_and_prune_work_off_max(tmp_path):
    from maxgaffer.maxbridge.controller import Controller

    ctrl = Controller()
    assert ctrl.session is not None            # unsaved-scene in-memory session
    assert ctrl.cameras() == []                # graceful, no raise
    assert ctrl.save_session() is False        # nothing to persist → honest False


# ------------------------------------------------- LLM repeat guard (#8)
def test_llm_reproposing_a_measured_state_is_dropped():
    """The prompt already carries parameter and score history and tells the model to hold
    a ping-ponged value, and it re-proposed an identical failing move anyway. Measured
    on-box 2026-07-25 (A3): 9 iterations, three visits to the same ~77.6 state and two
    identical ev=-4.0 proposals three apart — five of nine renders re-measured something
    already known. A rebuilt state we have already scored must be refused."""
    from maxgaffer.core.director import _state_fingerprint

    st = LightingState()
    st.set("sun.azimuth_deg", 100.0)
    st.set("sun.intensity", 1.0)

    # the model keeps proposing the SAME move back to azimuth 100 (a no-op repeat)
    same = json.dumps({"assessment": "", "changes": [
        {"param": "sun.azimuth_deg", "value": 100.0, "why": "again"}], "stop": False})
    seen_ctx = []

    def llm(ctx):
        seen_ctx.append(ctx["iteration"])
        return same

    hooks = Hooks(apply=lambda s: None, render=lambda t: f"/tmp/{t}.png",
                  stats=lambda p: dict(REF), llm_deltas=llm, log=lambda m: None)
    res = run_match(st, REF, {}, hooks,
                    MatchConfig(max_iterations=4, target_score=101, stall_patience=99,
                                analytic=False))
    # every iteration after the first must have had its repeat proposal dropped
    for rec in res.iterations[1:]:
        assert rec.llm_accepted == {}, f"iter {rec.index} accepted a repeat"
    # and the fingerprint helper is stable for equal states
    other = st.copy()
    assert _state_fingerprint(other) == _state_fingerprint(st)


# ------------------------------------------------- basin choice uses the loop's objective
def test_basin_probe_is_judged_on_the_same_objective_as_the_loop():
    """The basin picker hands the loop its starting state and polish is a basin-finisher, so
    that one comparison caps the whole run. It was scoring candidates on PIXELS while the
    loop scored them on pixels blended with the reference's lighting reading. Measured
    on-box 2026-07-25: it picked a SUNLESS basin for a sunlit golden-hour reference, which
    then suppressed the sun lock (that lock deliberately leaves a genuinely sunless basin
    free), and the match finished at 80.35 with the sun switched off. Two judges, one
    handing work to the other, must apply the same rule."""
    import inspect

    from maxgaffer.core.director import TRANSFER_WEIGHT
    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller._pick_start_basin)
    assert "blend_transfer(" in src, "basin probe fell back to a pixels-only comparison"
    # and the weight is shared, not re-typed — two literals drift apart
    assert "TRANSFER_WEIGHT" in src
    assert TRANSFER_WEIGHT == 0.25


def test_sun_lock_is_installed_no_matter_how_the_start_state_was_chosen():
    """The lock that stops the loop abolishing a sunlit reference's key light used to live
    INSIDE the multi-start branch, which is skipped whenever a start_override is supplied.
    `refine` always supplies one — so a refine round was the single path that could hand the
    loop a sun-off start with nothing pinning the sun back on. The guard belongs to the
    STATE, not to how the state was picked."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.run_match)
    lock_at = src.index('locks = set(locks) | {"sun.enabled"}')
    branch_at = src.index("if multi_start and start_override is None")
    # the lock must not be nested under the multi-start guard: compare indentation
    lock_indent = len(src[:lock_at].rsplit("\n", 1)[-1])
    branch_indent = len(src[:branch_at].rsplit("\n", 1)[-1])
    assert lock_indent <= branch_indent + 4, (
        "sun lock is nested inside the multi-start branch again — refine skips it")


def test_refine_picks_its_branch_on_the_same_objective_as_the_loop():
    """refine's winning branch becomes run_match's start_override, so this comparison
    decides the whole round. On pixels alone it can crown a structurally wrong branch that
    the loop then scores worse than the one it rejected."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    src = inspect.getsource(Controller.refine)
    assert "blend_transfer(" in src, "refine's branch probe is pixels-only again"
    assert "TRANSFER_WEIGHT" in src


def test_the_reported_score_is_pixel_similarity_not_the_search_objective():
    """The objective blends pixel similarity with agreement to the reference's lighting
    reading, and after the sweep it aims at the SWEEP's own answer — so the search scores
    itself as agreeing with a target it picked. Measured on-box 2026-07-25: a match whose
    sun ended 171 degrees from the reference reported 86.27 while its own best plate scored
    77.21 against that reference. Steering on the blend is right; reporting it is not."""
    import inspect

    from maxgaffer.core.director import MatchResult
    from maxgaffer.maxbridge.controller import Controller

    assert hasattr(MatchResult(best_state=None, best_score=None, best_render=None,
                               stop_reason="x"), "objective_score")
    src = inspect.getsource(Controller.run_match)
    assert "result.objective_score = result.best_score" in src
    assert "result.best_score = verdict.score" in src, (
        "the headline number must be the plain pixel similarity the artist can verify")
