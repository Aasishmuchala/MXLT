"""Full-codebase recheck regressions — each test encodes a defect found by the review."""

import json

import pytest

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
    # the VALUE is tuned (0.25 -> 0.10 once the critic could see direction for itself);
    # what this locks is that both judges read it from the same constant
    assert 0.0 < TRANSFER_WEIGHT <= 0.30


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


# ------------------------------------------------- the critic cannot see a sun patch
def test_highlight_similarity_sees_what_the_direction_component_cannot():
    """Measured on-box 2026-07-25 on a golden-hour interior: the reference carried sun
    patches in 15 of 25 cells at 17.3% of the frame, the match carried hot pixels in 3 —
    all of them the blown window itself, no floor patch, no wall patch, no directional
    light anywhere. Unmistakable side by side. The critic's direction component scored the
    pair 0.92, because averaging a grid cell is exactly what erases a small very bright
    patch. This reads 0.56."""
    from maxgaffer.core.metrics import highlight_similarity

    def stats(frac, grid):
        return {"hot_frac": frac, "hot_grid": grid}

    patches = [0.0] * 25
    for c in (5, 6, 10, 11, 15, 16):        # sun raking across the lower-left floor
        patches[c] = 1.0 / 6
    window = [0.0] * 25
    for c in (3, 8, 13):                     # only the blown window, no cast light
        window[c] = 1.0 / 3

    assert highlight_similarity(stats(0.17, patches), stats(0.17, patches)) == 1.0
    wrong = highlight_similarity(stats(0.17, patches), stats(0.058, window))
    assert wrong < 0.7, wrong


def test_a_rig_with_no_sun_patch_scores_zero_not_unmeasurable():
    """Same rule as the transfer metric: the candidate must not be able to delete a
    criterion by giving up on it. No directional light against a reference full of it is a
    total miss, not an absent measurement."""
    from maxgaffer.core.metrics import highlight_similarity

    lit = {"hot_frac": 0.17, "hot_grid": [0.04] * 25}
    dark = {"hot_frac": 0.0, "hot_grid": [0.0] * 25}
    assert highlight_similarity(lit, dark) == 0.0
    assert highlight_similarity(dark, dark) == 1.0      # both overcast: they agree
    assert highlight_similarity({}, lit) is None        # stats predating the map


def test_the_match_reports_sun_patch_agreement_beside_the_score():
    import inspect

    from maxgaffer.core.director import MatchResult
    from maxgaffer.maxbridge.controller import Controller

    assert hasattr(MatchResult(best_state=None, best_score=None, best_render=None,
                               stop_reason="x"), "highlight")
    assert "highlight_similarity(ref_stats, honest)" in inspect.getsource(
        Controller.run_match)


# ------------------------------------------------- best_render must BE the best state
def test_polish_updates_best_render_not_just_best_state():
    """run_match set result.best_state and best_score from polish but left best_render
    pointing at the LOOP's plate — a render of a state the match discarded. Measured on-box
    2026-07-25: a run whose polish gained 55.67 saved a blown frame (81% of pixels clipped,
    scoring 26.5 against its reference) as the 'best render' of a state that actually
    renders at 91.2. Another saved a 77.2 plate for a state rendering 55.5. It is wrong in
    both directions, the artist is handed the wrong picture, and anything that scores
    best_render scores the wrong image."""
    # loop iterations measure poorly, polish probes measure perfectly — so polish is
    # guaranteed to climb away from the loop's best and land on a different state
    def stats_for(path):
        return dict(REF) if "polish" in path else dark(REF)

    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.intensity": 1.0,
                 "exposure.ev": 12.0, "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)
    rendered = []

    def render(tag):
        path = f"/tmp/{tag}.png"
        rendered.append(path)
        return path

    hooks = Hooks(apply=lambda s: None, render=render, stats=stats_for,
                  llm_deltas=lambda ctx: json.dumps(
                      {"assessment": "", "changes": [], "stop": False}),
                  log=lambda m: None)
    res = run_match(st, REF, {}, hooks,
                    MatchConfig(max_iterations=2, target_score=101, stall_patience=99,
                                polish=True, polish_rounds=2, polish_max_probes=12,
                                analytic=False))
    if res.polish_probes:
        assert res.best_render is not None
        # whatever polish landed on, the saved plate must come from polish — never be left
        # behind on a loop iteration once polish has moved the state
        assert "polish" in res.best_render, res.best_render


def test_the_critic_now_punishes_a_tonally_perfect_but_sunless_match():
    """The whole point of the highlight component. A match that nails exposure, histogram,
    colour and hue while carrying NONE of the reference's directional light used to score
    100 — every component the critic had was satisfied. Measured on-box 2026-07-25, that is
    not hypothetical: the failing golden-hour run scored 0.922 on direction with its sun
    171 degrees out and no sun patch anywhere in the room."""
    from maxgaffer.core import critic

    perfect = {"key": 1.0, "envelope": 1.0, "histogram": 1.0, "color": 1.0,
               "hue": 1.0, "direction": 1.0}
    w = critic.DEFAULT_WEIGHTS

    def agg(comps):
        tw = sum(w[k] for k in comps if k in w) or 1.0
        tot = sum(w[k] * v for k, v in comps.items() if k in w) / tw
        if len(comps) >= 3:
            tot -= 0.35 * max(0.0, tot - min(comps.values()))
        return 100.0 * tot

    assert agg(dict(perfect, highlight=1.0)) == pytest.approx(100.0)
    assert agg(dict(perfect, highlight=0.554)) < 85.0    # the measured failure
    assert agg(dict(perfect, highlight=0.0)) < 60.0      # no directional light at all


def test_highlight_carries_real_weight_in_every_taste_profile():
    """Including 'direction' — that profile means "I care where the light is", so its
    emphasis belongs on the component that can actually tell. The old direction profile
    weighted the grid cosine at 0.33, which on a golden-hour interior meant weighting a
    number that read 0.92 for a 171-degree miss and 0.917 for a 13.5-degree one."""
    from maxgaffer.core import critic

    for name, w in critic.PREFERENCE_PROFILES.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, (name, sum(w.values()))
        assert w.get("highlight", 0) > 0, name
    assert critic.PREFERENCE_PROFILES["direction"]["highlight"] > \
        critic.PREFERENCE_PROFILES["direction"]["direction"]


def test_reverting_to_the_champion_takes_its_plate_with_it():
    """polish may adopt a restart up to 8 points WORSE as a bet, with the champion fallback
    making the gamble free. _record_best only fires when `best` improves, so reverting to
    the champion left best_render on the state the run had just abandoned. Measured on-box
    2026-07-26: the saved plate scored 88.83 against the reference while a re-render of the
    state actually applied scored 56.20 — a 32-point lie about which picture the artist
    got, in a run whose sun was 168 degrees out."""
    import inspect

    from maxgaffer.core.director import run_polish

    src = inspect.getsource(run_polish)
    assert "champion_render" in src
    finish = src[src.index("def _finish("):src.index("def _finish(") + 2000]
    assert "hooks._polish_best_render = final_render" in finish, (
        "the plate must follow whichever state _finish actually lands on")
    # _record_best pins BOTH channels, so a revert must restore both — restoring only the
    # plate left the component breakdown describing the abandoned descent, and that
    # breakdown drives scorecard's weakest/likely-gap and fairness.assess
    assert "champion_components" in src
    assert "hooks._polish_best_components = final_components" in finish


def test_the_transfer_blend_can_no_longer_outvote_the_pixels():
    """transfer_weight existed because pixels were blind to sun direction. The highlight
    component ended that blindness (it scored two states 0.912 and 0.546 on exactly that
    question), and the workaround then turned costly: on-box the sweep put the sun 168
    degrees wrong, the transfer term defended it, and that overrode a 32-point pixel gap."""
    from maxgaffer.core.director import TRANSFER_WEIGHT

    assert TRANSFER_WEIGHT <= 0.12, "the reading must not outvote what the render shows"
    assert TRANSFER_WEIGHT > 0.0, "some weight still breaks a genuine tie"


def test_a_globally_brightened_frame_cannot_fake_sun_patches():
    """The metamer this codebase keeps meeting: crank the dome and open the exposure, and
    the frame fills with absolutely-bright pixels that satisfy a plain threshold. Measured
    on-box 2026-07-26 — a match with dome 2.29 against a 0.55 target and white balance
    10400 K against 4800 K carried MORE absolutely-bright pixels than the reference (0.195
    vs 0.173) and scored 0.891 on absolute-threshold agreement, with its sun 78 degrees
    out. A patch is bright RELATIVE TO ITS SURROUNDINGS; on local contrast the same pair
    reads 0.514, because the reference has 4.1% of frame in genuine patches and the match
    1.4%."""
    from maxgaffer.core import metrics

    assert metrics.HOT_LOCAL_LIFT > 0, "a highlight must clear its own neighbourhood"
    assert metrics.HOT_THRESHOLD < 0.5, (
        "with a local-contrast test the absolute floor only rejects shadow detail")

    # a flat bright field has no patches at any brightness; a field with a bright blob does
    import tempfile

    from maxgaffer.core.png_min import write_png_rgb

    def frame(blob):
        rows = [[(200, 200, 200) if (blob and 20 <= x < 32 and 20 <= y < 32)
                 else (120, 120, 120) for x in range(64)] for y in range(64)]
        fh = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fh.close()
        write_png_rgb(fh.name, rows)
        return metrics.compute_stats(fh.name)

    flat, patched = frame(False), frame(True)
    assert flat["hot_frac"] == 0.0, "an evenly lit frame has no directional patch"
    assert patched["hot_frac"] > 0.0
    assert metrics.highlight_similarity(patched, flat) == 0.0


def test_every_polish_probe_renders_to_its_own_file():
    """The probe tag was f"polish{round}_{axis}" — a pure function of round and axis — but a
    climb probes one axis repeatedly, and render_frame DELETES its target before writing.
    Each probe therefore destroyed the last one's plate, and a climb only stops when a probe
    fails, so the recorded winning plate was reliably overwritten by the rejected state one
    step past the optimum (typically 1.6 stops out once the stride accelerates). Correct
    path, wrong bytes. sun.intensity and dome.intensity also both reduce to "intensity",
    colliding two axes on one filename in the same round.

    No existing test could catch it: every fake is render=lambda t: f"/tmp/{t}.png", which
    models a path as a content-stable token nothing ever writes to. This one asserts
    uniqueness directly."""
    seen_paths = []

    st = LightingState()
    for k, v in {"sun.enabled": 1, "sun.azimuth_deg": 100.0, "sun.intensity": 1.0,
                 "dome.intensity": 1.0, "exposure.ev": 12.0,
                 "exposure.wb_kelvin": 6500.0}.items():
        st.set(k, v)

    def render(tag):
        path = f"/tmp/{tag}.png"
        seen_paths.append(path)
        return path

    # imperfect stats, so the score stays under polish_stop_at and polish actually probes
    hooks = Hooks(apply=lambda s: None, render=render, stats=lambda p: dark(REF),
                  llm_deltas=lambda ctx: json.dumps(
                      {"assessment": "", "changes": [], "stop": False}),
                  log=lambda m: None)
    run_match(st, REF, {}, hooks,
              MatchConfig(max_iterations=2, target_score=101, stall_patience=99,
                          polish=True, polish_rounds=3, polish_max_probes=40,
                          analytic=False))
    polish_paths = [p for p in seen_paths if "polish" in p]
    assert polish_paths, "test setup should have run polish probes"
    assert len(polish_paths) == len(set(polish_paths)), (
        "two polish probes shared a filename — the second destroys the first's plate")


def test_the_diagonal_escape_adopts_the_state_it_actually_measured():
    """`measure` applies the tonal leash IN PLACE, so the object it renders may be clamped.
    The ridge escape discarded that candidate and rebuilt the move from scratch, bypassing
    the leash, then adopted the rebuilt one. exposure.ev or exposure.wb_kelvin appear in six
    of the seven polish pairs, so this was the hot path: a ridden diagonal could adopt an EV
    two stops away from the render its score and plate came from, and it silently defeated
    the leash the block is supposed to respect."""
    import inspect

    from maxgaffer.core.director import run_polish

    src = inspect.getsource(run_polish)
    diag = src[src.index("def _diag_probe("):src.index("if not escaped:")]
    assert "return sc, cand" in diag, "the probe must hand back what it measured"
    # and no caller may reconstruct the move itself
    after = diag[diag.index("for ka, kb in pairs:"):]
    assert "cand.set(ka," not in after, "a caller rebuilt the diagonal instead of adopting it"
    assert "sc, cand = got" in after and "sc2, cand2 = got2" in after


def test_the_match_reports_progress_from_the_single_render_seam():
    """A match runs for minutes and the log alone cannot distinguish a working run from a
    hung one. Every unit of work already passes through render_hook and the tag names the
    stage, so progress is counted there rather than threaded separately through the solver,
    the sweep, the loop and polish — one seam, no plumbing, and it cannot drift out of sync
    with what is actually being rendered."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    sig = inspect.signature(Controller.run_match)
    assert "on_progress" in sig.parameters
    assert sig.parameters["on_progress"].default is None, "progress must be opt-in"

    src = inspect.getsource(Controller.run_match)
    hook_at = src.index("def render_hook(")
    assert "_tick(tag)" in src[hook_at:hook_at + 900], "the seam does not count its work"
    # every stage the artist waits through must be nameable, or the label lies
    for stage in ("basin", "sunsolve", "sweep", "polish"):
        assert f'"{stage}"' in src
    # a stage that overruns its budget must not push the bar past 100
    assert "min(_seen[key], budget)" in src
    assert "min(100.0," in src
    # and a broken readout must never take a match down with it
    tick_at = src.index("def _tick(")
    assert "except Exception" in src[tick_at:tick_at + 1200]


def test_start_fresh_restores_the_scene_before_it_forgets_anything():
    """A reset has two halves and both are needed or "fresh" is a lie: put the artist's
    light back, THEN forget. Order matters — pre_match is the only record of the light
    before MaxGaffer touched it, so a camera whose restore FAILS must keep its entry rather
    than have that snapshot dropped along with everything else."""
    import inspect

    from maxgaffer.maxbridge.controller import Controller

    assert hasattr(Controller, "start_fresh")
    src = inspect.getsource(Controller.start_fresh)
    restore_at = src.index("restore_pre_match")
    clear_at = src.index("self.session.cameras = keep")
    assert restore_at < clear_at, "it forgets before it restores"
    assert "summary[\"failed\"]" in src and "keep = {" in src, (
        "a failed restore must keep its snapshot for a second attempt")
    # baselines are the artist's own authored multipliers, not ours to discard
    assert "self._baselines = dict(self.session.baselines)" in src
    # one bad camera cannot strand the others
    assert "except Exception" in src[:clear_at]


def test_the_reset_button_asks_first_and_says_what_it_will_do():
    """RESET sits beside the camera re-scan, which is cheap and idempotent — this one throws
    work away, so it must not be confusable with it, and the confirm has to name the SCENE
    half too. That is the part an artist would be upset to discover afterwards."""
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")     # the box's Max python has Qt; CI may not

    from maxgaffer.ui import dock as dockmod

    src = inspect.getsource(dockmod.MaxGafferDock._on_reset)
    assert "QMessageBox" in src and "setInformativeText" in src
    assert "restore each camera's light" in src, "the confirm hides the scene half"
    assert "exposure control MaxGaffer created" in src
    assert "StandardButtons" in src and "Cancel" in src
    assert "setDefaultButton" in src and "QMessageBox.Cancel" in src, (
        "the safe choice must be the default")
    assert "if self._busy" in src, "a reset mid-match would race the worker"
    # and the panels are emptied, or a stale reading outlives the session that produced it
    assert "_clear_after_reset" in src


def test_progress_survives_a_phase_that_renders_nothing():
    """The stall an artist actually hits is BEFORE the first render: three ANALYZE image
    calls plus a plan call, all network, all at near-zero CPU. A meter driven only by
    renders shows a frozen bar there — which is precisely the moment "am I stuck?" gets
    asked. A clock that keeps counting is the difference between working and hung."""
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    from maxgaffer.ui import dock as dockmod

    for name in ("_on_heartbeat", "_progress_stage", "_clock"):
        assert hasattr(dockmod.MaxGafferDock, name), name
    beat = inspect.getsource(dockmod.MaxGafferDock._on_heartbeat)
    assert "self._busy" in beat, "the heartbeat must stop when the run does"
    start = inspect.getsource(dockmod.MaxGafferDock._start_match)
    # the phases that render nothing are NAMED, so the label is never stale during a wait
    assert '_progress_begin("reading the reference")' in start
    assert '_progress_stage("planning")' in start
    assert '_progress_stage("matching")' in start
    # and the timer is stopped on the way out, or it ticks forever over an idle dock
    assert "self._beat.stop()" in inspect.getsource(dockmod.MaxGafferDock._progress_end)


def test_the_reload_script_purges_modules_and_the_cached_dock():
    """show_dock REUSES an existing panel and Python caches modules for the process
    lifetime, so editing the plugin and calling launch() again returns the OLD code in the
    OLD widget — indistinguishable from the change not working."""
    src = open("scripts/reload_dock.py", encoding="utf-8").read()
    assert "_dock_wrapper" in src and "_dock_instance" in src, "a reused panel is not reloaded"
    assert "sys.modules.pop" in src
    assert "reverse=True" in src, "submodules must be dropped before their packages"
    assert "import maxgaffer.bootstrap" in src


def test_cancel_can_release_a_run_that_will_not_respond_to_it():
    """Cancel can only ever be a REQUEST — it sets a flag the running code has to notice,
    and between checks sit long uninterruptible stretches (a gateway round trip, a V-Ray
    frame). That is fine when the run is healthy and useless when it is wedged somewhere
    that never looks at the flag again, which locks the dock with no way out but reloading.
    Measured on-box 2026-07-26: a match sat 70 minutes at 0.3% CPU having rendered nothing,
    with MATCH greyed and no dialog to answer.

    The second press hands the controls back. It must NOT claim to have killed anything —
    a released UI that silently leaves work running is its own kind of lie."""
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    from maxgaffer.ui import dock as dockmod

    src = inspect.getsource(dockmod.MaxGafferDock._cancel_match)
    assert "if not self._cancel:" in src, "the first press must still be a polite request"
    assert "press ✕ again" in src, "the escape hatch has to be discoverable"
    assert "_force_release" in src
    rel = inspect.getsource(dockmod.MaxGafferDock._force_release)
    assert "self._busy = False" in rel and "setEnabled(True)" in rel
    assert "does NOT stop" in src, "it must not imply the work was killed"


def test_a_long_run_always_shows_its_own_log():
    """THE bug behind "it is just stuck half the time". The transcript widget is created
    hidden and was only ever revealed by clicking "Transcript ▾" — so a running match
    displayed nothing at all: no log, no meter, no clock. That is indistinguishable from a
    hang, and the plugin was logging every step of its work the whole time.

    Anything that takes minutes reveals its transcript when it starts. The toggle stays,
    for hiding it afterwards."""
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    from maxgaffer.ui import dock as dockmod

    begin = inspect.getsource(dockmod.MaxGafferDock._progress_begin)
    assert "self._show_log()" in begin, "a run that shows nothing looks hung"
    show = inspect.getsource(dockmod.MaxGafferDock._show_log)
    assert "self.log.setVisible(True)" in show
    assert 'setText("Transcript ▴")' in show, "the toggle label must not contradict it"
    # an error must reveal the transcript too — one written into a hidden or freshly
    # cleared panel is an error nobody sees, which is how this went undiagnosed
    match_src = inspect.getsource(dockmod.MaxGafferDock._start_match)
    assert match_src.count("self._show_log()") >= 2


def test_progress_never_starts_for_a_run_that_was_refused():
    """_progress_begin resets the clock and reveals the panel, so it must sit AFTER the
    busy guard and every early return. Called first, a click while busy would restart the
    timer for a run that never began — the readout would then be describing nothing."""
    import re

    src = open("maxgaffer/ui/dock.py", encoding="utf-8").read()
    for fn in ("_start_match", "_start_match_all", "_start_refine"):
        body = re.search(r"\n    def %s\(self\):\n(.*?)(?=\n    def )" % fn, src, re.S)
        assert body, fn
        guard = body.group(1).find("if self._busy")
        begin = body.group(1).find("_progress_begin")
        assert guard != -1, f"{fn} lost its busy guard"
        assert begin == -1 or guard < begin, f"{fn} starts progress before it commits"


def test_no_handler_can_strand_the_dock_between_disabling_and_its_finally():
    """The permanent-stuck bug. _start_match took the busy flag, disabled the buttons and
    cleared the transcript OUTSIDE its try — eight statements, any of which could raise (a
    missing action, a config attribute, a Qt call). When one did, the finally never ran and
    the dock was locked for good: buttons dead, transcript cleared and EMPTY, no meter, no
    message. Indistinguishable from a hang, and undiagnosable, because the widget that
    would have shown the error had just been emptied.

    The rule: from the moment a handler takes self._busy, every statement until the
    try/finally that releases it must be inside that try."""
    import re

    src = open("maxgaffer/ui/dock.py", encoding="utf-8").read()
    offenders = []
    for match in re.finditer(r"\n    def (_[a-z_]+)\(self.*?\):\n(.*?)(?=\n    def )",
                             src, re.S):
        fn, body = match.group(1), match.group(2)
        at = body.find("self._busy = True")
        if at == -1:
            continue
        before = body[:at].rstrip()
        after = body[at + len("self._busy = True"):]
        # safe either way: the flag is taken inside a try, or a try opens immediately after
        # it with nothing in between that could raise
        if before.endswith("try:"):
            continue
        gap = [ln.strip() for ln in after.splitlines()
               if ln.strip() and not ln.strip().startswith("#")]
        if gap and gap[0] == "try:":
            continue
        offenders.append(fn)
    assert not offenders, (
        "these handlers can leave the dock permanently disabled if a statement raises "
        "before their finally: " + ", ".join(offenders))


def test_an_option_checkbox_can_never_take_a_match_down():
    """The exception that was locking the dock, finally visible once errors reached the
    transcript: "Internal C++ object (PySide6.QtGui.QAction) already deleted". PySide6
    raises that when a widget's Qt side is destroyed while Python still holds the shell,
    and the match handler read one of those actions between taking the busy flag and the
    try that releases it.

    Two fixes, and both matter. The Options QMenu is now held on the dock like lock_menu
    always was — it was the odd one out, kept only in a local. And every option read goes
    through _opt, which falls back to the default: an option CHECKBOX must never be able to
    stop a match, because not running at all is strictly worse than running with a default.
    """
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    from maxgaffer.ui import dock as dockmod

    class _Dead:
        def isChecked(self):
            raise RuntimeError("Internal C++ object (PySide6.QtGui.QAction) already deleted.")

    assert dockmod.MaxGafferDock._opt(_Dead(), True) is True
    assert dockmod.MaxGafferDock._opt(_Dead(), False) is False
    assert dockmod.MaxGafferDock._opt(None, True) is True          # attribute gone entirely

    src = inspect.getsource(dockmod)
    assert "self.opts_menu = QtWidgets.QMenu(self)" in src, "the menu is a local again"
    # no raw .isChecked() on an option action may survive anywhere
    for name in ("act_sweep", "act_autoexec", "act_draft", "act_popup", "act_live"):
        assert f"self.{name}.isChecked()" not in src, f"{name} is read unguarded"


def test_show_dock_closes_an_orphan_panel_before_making_another():
    """The module globals are the only handle on the previous panel, and a reload that
    replaces this module loses them — leaving a live dock nobody tracks. Measured on-box:
    the transcript logged the same failure TWICE, from two instances answering one click."""
    import inspect
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtWidgets")

    from maxgaffer.ui import dock as dockmod

    src = inspect.getsource(dockmod.show_dock)
    assert 'findChildren(QtWidgets.QDockWidget, "MaxGafferDock")' in src
    assert "old.close()" in src


def test_polish_renders_cheap_but_the_answer_is_verified_full_size():
    """Polish is ~82% of a match: 120 renders of "nudge one parameter, did that help". That
    is a comparison, not a verdict, and the cheaper render answers it just as well —
    measured on-box 2026-07-26 across eight states spanning exposure, dome, azimuth,
    turbidity and altitude, half-resolution scored within 0.63 of full and ranked all eight
    IDENTICALLY, at half the render time.

    Quarter also preserved the ranking but drifted up to 1.33 points, which is too coarse:
    polish accepts a move on a 0.03 gain, so a shift that size would have it chasing
    measurement error instead of light. Half is the tier the evidence supports.

    The saving must not be paid for in honesty, so the state polish lands on is re-rendered
    at FULL size before it is scored or shown — one render against 120 saved."""
    import inspect

    from maxgaffer.core.profiles import resolve_profile
    from maxgaffer.maxbridge.controller import Controller

    p = resolve_profile("standard", loop_width=400, loop_height=600,
                        max_iterations=10, sweep_count=8, target_score=95.0)
    assert p.polish_size == (200, 300)
    assert p.polish_size[0] < p.loop_width and p.polish_size[1] < p.loop_height

    src = inspect.getsource(Controller.run_match)
    hook = src[src.index("def render_hook("):src.index("def apply_hook(")]
    assert 'tag.startswith("polish")' in hook and "profile.polish_size" in hook
    # ...and the delivered plate is re-rendered full size before anyone sees a number
    assert "final_full.png" in src
    verify_at = src.index("final_full.png")
    score_at = src.index("honest = self.stats_for(")
    assert verify_at < score_at, "the reported score would come from a cheap frame"
