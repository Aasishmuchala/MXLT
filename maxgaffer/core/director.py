"""The match loop — a sans-IO state machine. Hooks do the touching; this does the thinking.

Iteration shape (one render per iteration — renders are the expensive resource):
    apply(state) → render → stats → score → [converged? stop]
    → analytic solve (EV/WB, deterministic) + LLM deltas (geometry/mood, bounded)
    → merge into next state → repeat

Reliability guards, in the MaxDirector tradition:
  * keep-best — the best-scoring state is tracked and ALWAYS re-applied at the end, so an
    exploratory move that made things worse can never be the final answer;
  * revert-on-slump — two consecutive scores meaningfully below best snap the loop back to
    the best state before asking the LLM again (one exploratory move is allowed, a slide
    is not);
  * every LLM proposal passes genome validation (unknown → dropped, locked → refused,
    bounds → clamped, per-iteration step → limited);
  * metrics missing (no stats engine / no scores) degrades to LLM-visual-only mode with the
    analytic solver off — the loop still runs, the log says so loudly.

All hooks are injected, so the whole loop is unit-tested off-Max with fakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

from . import critic, solver
from .genome import GROUP_PREFIX, LightingState, apply_changes, spec_for, state_table
from .parse import ParseError


@dataclass
class MatchConfig:
    max_iterations: int = 5
    target_score: float = 82.0
    stall_delta: float = 1.5      # min improvement over best to count as progress
    stall_patience: int = 2       # iterations without progress before stopping
    slump_tolerance: float = 3.0  # how far below best counts as a slump
    analytic: bool = True         # run the EV/WB histogram solver each iteration
    max_changes: int = 4
    weights: Optional[Dict[str, float]] = None
    # analytic LEASH — total movement from the run's start state. The solver matches
    # histograms of DIFFERENT scenes, so scene-vs-reference albedo mismatch (white room
    # matched to a walnut library) biases it systematically; the leash bounds the damage
    # and hitting it is reported as a diagnosis, not silently absorbed.
    ev_leash: float = 4.0
    wb_leash: float = 3000.0
    # when the solver had to move EV by more than this in one iteration, the render the
    # LLM just saw was badly mis-exposed — its absolute-brightness judgments (intensities,
    # group levels) are contaminated and get dropped for that iteration
    contaminated_ev_step: float = 1.5
    # DEEP-MATCH finisher: after the loop, an LLM-free adaptive coordinate line search —
    # climb while a rendered nudge improves the score, halve the step when it doesn't,
    # converged when every step bottoms out. That exhaustion IS a provable local optimum:
    # the scene's ceiling for this reference.
    polish: bool = False
    polish_rounds: int = 10
    polish_min_gain: float = 0.03
    polish_stop_at: float = 99.5
    # converged = a strict no-improve round with all steps floored, OR two consecutive
    # rounds each gaining < round_eps — on smooth landscapes every round finds crumbs
    # forever, and "within 2ε of the optimum along every probed axis" IS the ceiling
    polish_round_eps: float = 0.2
    # a full convergence PROOF costs ~2 probes × 9 params × 5 step levels + the climbs;
    # stop_at usually exits far earlier — the cap is the overnight safety rail
    polish_max_probes: int = 120
    #: How much of the objective is the SEMANTIC transfer reading rather than pixel
    #: statistics. 0 keeps the historical pixels-only behaviour. Measured need: with
    #: pixels alone the search settled on uniformly-decent but structurally WRONG rigs
    #: (sun 64 degrees out) because no pixel component punishes it enough.
    transfer_weight: float = 0.0


def _anneal(best_score: Optional[float]) -> float:
    """Step/deadband scale from convergence: explore big, finish small."""
    if best_score is None or best_score < 70.0:
        return 1.0
    if best_score < 85.0:
        return 0.5
    return 0.25


def _bound_pinned(key: str, wanted: float) -> bool:
    """True when a solver target arrives sitting exactly ON the genome bound.

    solve_ev/solve_wb pre-clamp to the genome, so the leash-window delta below is blind
    to it: when the window is wider than the genome, "the solver wants more than the rig
    can express" never registers as a leash hit — yet it is the SAME albedo-mismatch
    signal the leash exists to report (SPEC §2 analytic leash)."""
    spec = spec_for(key)
    return spec is not None and (wanted <= spec.lo or wanted >= spec.hi)


# adaptive coordinate line-search table: (key, initial_step, is_log2_step, fine_floor).
# EV and WB are axes here too — measured descent, so no analytic-ownership conflict —
# because freezing them while geometry moves invites COMPENSATION DRIFT: live sim showed
# altitude climbing AWAY from its target to fake the exposure key at a stale EV.
POLISH_PARAMS = (
    ("exposure.ev", 0.4, False, 0.05),
    ("exposure.wb_kelvin", 400.0, False, 50.0),
    ("sun.azimuth_deg", 12.0, False, 1.0),
    ("sun.altitude_deg", 6.0, False, 0.75),
    ("sun.size", 0.5, True, 0.08),
    ("sun.intensity", 0.35, True, 0.06),
    ("sun.turbidity", 1.0, False, 0.2),
    ("dome.intensity", 0.35, True, 0.06),
    ("dome.rotation_deg", 12.0, False, 1.5),
)

# the compensation couples — probed diagonally when single-axis search stalls (ridge
# escape): exposure↔WB fake each other, and both fake low-sun warmth/direction.
# ev↔dome is the dome-rig metamer valley: dome × 2^-EV holds the image nearly constant,
# so single-axis probes see a ridge and the dome pins at its start value while EV
# compensates (measured on-box 2026-07-24: three consecutive hero self-matches recovered
# dome.intensity == the scramble value exactly, EV off by the canceling stops).
_POLISH_PAIRS = (
    ("exposure.ev", "exposure.wb_kelvin"),
    ("exposure.wb_kelvin", "sun.altitude_deg"),
    ("exposure.ev", "sun.altitude_deg"),
    ("sun.azimuth_deg", "sun.altitude_deg"),
    ("exposure.ev", "dome.intensity"),
    # measured couples from the 2026-07-24 archetype matrix: overall brightness can be
    # faked by the sun as well as the dome, and haze warmth trades against camera WB
    ("exposure.ev", "sun.intensity"),
    ("sun.turbidity", "exposure.wb_kelvin"),
)


#: Bounded azimuth basin hops per polish run. Each hop costs a few probes and re-opens
#: the step sizes, so this is a global-search assist, not a licence to wander.
_MAX_HOPS = 3

#: How far BELOW the incumbent a coordinated tonal restart may land and still be adopted.
#: A jump into the right region routinely lands a few points under a finely-tuned wrong
#: answer and only wins after it descends; the champion fallback makes the gamble free.
_RESTART_TOLERANCE = 8.0


#: Share of the match objective given to the reference's own lighting reading, over the
#: pixel critic. The basin picker and the loop MUST use the same number: the basin picker
#: hands the loop its starting state, and if the two judge by different rules the picker
#: can hand over a state the loop immediately scores worse than what it rejected.
TRANSFER_WEIGHT = 0.25


def blend_transfer(pixel_score: float, state: LightingState, hooks: "Hooks",
                   cfg: "MatchConfig") -> float:
    """Fold the SEMANTIC transfer reading into the pixel score.

    Pixel statistics cannot pin sun direction on an interior: measured on-box 2026-07-25,
    a 64-degree azimuth error still scored 90.92 because the 3x3 luminance grid barely
    moves when a sun patch stays roughly in frame, and no other component notices at all.
    ANALYZE, meanwhile, reads the sun's bearing straight off the reference — a DIRECT
    measurement of where the light is, independent of the render. Weighting it into the
    objective is what stops the search settling on a uniformly-decent but structurally
    wrong rig. Absent hook or zero weight → the historical pixels-only score."""
    w = float(getattr(cfg, "transfer_weight", 0.0) or 0.0)
    if w <= 0.0 or hooks.transfer is None:
        return pixel_score
    try:
        agreement = hooks.transfer(state)
    except Exception:  # noqa: BLE001 — a diagnostic must never break scoring
        return pixel_score
    if agreement is None:
        return pixel_score
    return (1.0 - w) * pixel_score + w * (100.0 * max(0.0, min(1.0, float(agreement))))


def _state_fingerprint(state: LightingState) -> tuple:
    """A state's identity for repeat detection, rounded so float noise does not make two
    identical rigs look different."""
    vals = tuple(sorted((k, round(float(v), 4)) for k, v in state.values.items()))
    grps = tuple(sorted((k, round(float(v), 4)) for k, v in state.groups.items()))
    return (vals, grps)


def _has_axis(state: LightingState, key: str) -> bool:
    """Group-aware membership: ``group.<name>`` lives in state.groups, not state.values
    (state.get/set already route the prefix — only membership tests need this)."""
    if key.startswith(GROUP_PREFIX):
        return key[len(GROUP_PREFIX):] in state.groups
    return key in state.values


def polish_axes(state: LightingState) -> tuple:
    """The static POLISH_PARAMS plus this run's DYNAMIC axes: one log2 dimmer axis per
    light group present on the state, and the fog distance when the rig carries an
    atmosphere. Measured need (2026-07-24 archetype matrix): group recovery rode on the
    LLM alone — an aerial city scene never re-lit its 4 zones — and the fog solve landed
    in the right haze BUCKET (250 m vs a 20 m target) with no way to fine-tune. Both are
    ordinary measurable axes; polish just needs to know they exist for this rig."""
    axes = list(POLISH_PARAMS)
    for g in sorted(state.groups):
        # initial step = a FULL doubling: a zeroed group seeds from the 0.06 floor and a
        # 0.35-step probe (0.06 -> 0.077) is visually invisible, so an off zone could
        # never clear min_gain from a wide shot (measured: the aerial city's 4 zones
        # stayed at 0 while the street-level city climbed fine). With 1.0 the ride
        # reaches unity in ~3 accelerated probes; the fine floor still finishes small.
        axes.append((GROUP_PREFIX + g, 1.0, True, 0.06))
    if "atmosphere.distance_m" in state.values:
        axes.append(("atmosphere.distance_m", 0.5, True, 0.08))
    return tuple(axes)


@dataclass
class Hooks:
    """The loop's only contact with the world. ``apply``/``render`` run on Max's main thread
    (the UI layer guarantees that); llm/stats may be slow and are wrapped by the caller."""
    apply: Callable[[LightingState], None]
    render: Callable[[str], Optional[str]]            # tag → image path (None = failed)
    stats: Callable[[str], Optional[Dict]]            # image path → stats dict
    llm_deltas: Callable[[Dict], str]                 # context → raw reply text
    #: state → 0..1 lighting-TRANSFER agreement with the reference's ANALYZE reading.
    #: Pixel statistics cannot pin sun direction on an interior — a 64-degree swing barely
    #: moves the 3x3 luminance grid (measured 2026-07-25) — but ANALYZE reads the bearing
    #: straight off the reference, which is a DIRECT measurement of where the light is.
    transfer: Optional[Callable[[LightingState], Optional[float]]] = None
    log: Callable[[str], None] = lambda msg: None
    should_cancel: Callable[[], bool] = lambda: False


@dataclass
class IterationRecord:
    index: int
    state: Dict
    render_path: Optional[str] = None
    score: Optional[float] = None
    components: Dict[str, float] = field(default_factory=dict)
    analytic_changes: Dict[str, float] = field(default_factory=dict)
    llm_accepted: Dict[str, float] = field(default_factory=dict)
    llm_rejected: List[str] = field(default_factory=list)
    assessment: str = ""
    reverted_to_best: bool = False


@dataclass
class MatchResult:
    best_state: LightingState
    best_score: Optional[float]
    best_render: Optional[str]
    stop_reason: str
    iterations: List[IterationRecord] = field(default_factory=list)
    transfer: Optional[Dict] = None     # lighting-TRANSFER reading (see core.transfer)
    objective_score: Optional[float] = None   # what the SEARCH steered on: the pixel score
                                        # blended with agreement to the reference's lighting
                                        # reading. Kept for diagnosis, never the headline —
                                        # after the sweep the objective aims at the sweep's
                                        # own answer, so part of that agreement is the
                                        # search congratulating itself. best_score is the
                                        # plain pixel similarity the artist can verify.
    polish_gain: float = 0.0            # score added by the coordinate-descent finisher
    polish_probes: int = 0
    ceiling_converged: bool = False     # polish ended in a converged condition (either kind)
    ceiling_proven: bool = False        # STRONG claim: every step at its fine floor and a
                                        # full round improved nothing — a proven local
                                        # optimum. converged without proven = plateau
                                        # (two low-gain rounds; finer steps untested)
    best_components: Dict[str, float] = field(default_factory=dict)
    scorecard: Dict = field(default_factory=dict)

    def to_summary(self) -> Dict:
        """JSON-safe run record — the controller writes it to the run dir as run.json.
        This is the calibration trail: which critic scores humans later accept is the
        data that turns the proxy metric into a promise."""
        return {
            "best_score": self.best_score,
            "stop_reason": self.stop_reason,
            "best_render": self.best_render,
            "polish_gain": self.polish_gain,
            "polish_probes": self.polish_probes,
            "ceiling_converged": self.ceiling_converged,
            "ceiling_proven": self.ceiling_proven,
            "best_components": self.best_components,
            "scorecard": self.scorecard,
            "best_state": self.best_state.to_dict(),
            "iterations": [
                {"index": r.index, "score": r.score, "render": r.render_path,
                 "analytic": r.analytic_changes, "llm": r.llm_accepted,
                 "rejected": r.llm_rejected, "assessment": r.assessment,
                 "reverted": r.reverted_to_best}
                for r in self.iterations],
        }


def run_match(
    start_state: LightingState,
    ref_stats: Optional[Dict],
    semantics: Dict,
    hooks: Hooks,
    cfg: Optional[MatchConfig] = None,
    locks: Optional[Set[str]] = None,
    rig_notes: str = "",
    director_note: str = "",
) -> MatchResult:
    cfg = cfg or MatchConfig()
    locks = locks or set()
    metrics_ok = ref_stats is not None
    if not metrics_ok:
        hooks.log("⚠ metrics unavailable — LLM-visual mode (no analytic solve, no scores)")

    state = start_state.copy()
    best_state = state.copy()
    best_score: Optional[float] = None
    best_render: Optional[str] = None
    best_components: Dict[str, float] = {}
    best_stats: Optional[Dict] = None      # stats of best_state, reused on revert
    measured_states: Set[tuple] = set()    # every rig already rendered — repeat guard
    live: Optional[LightingState] = None   # what the scene is actually wearing right now
    records: List[IterationRecord] = []
    score_history: List[Tuple[int, float]] = []
    slump_count = 0
    stall_count = 0
    stop_reason = "max_iterations"
    leash_ev_lo = start_state.get("exposure.ev", 10.0) - cfg.ev_leash
    leash_ev_hi = start_state.get("exposure.ev", 10.0) + cfg.ev_leash
    leash_wb_lo = start_state.get("exposure.wb_kelvin", 6500.0) - cfg.wb_leash
    leash_wb_hi = start_state.get("exposure.wb_kelvin", 6500.0) + cfg.wb_leash
    leash_hits = 0
    budget = f"budget: ≤{cfg.max_iterations} loop renders"
    if cfg.polish:
        budget += f" + ≤{cfg.polish_max_probes} polish probes"
    hooks.log(budget)

    # keep-best is a GUARANTEE, not the happy path: a hook dying mid-iteration (apply,
    # render, stats, gateway) must never strand the scene on an exploratory state —
    # land the best known state with the audit trail complete, then surface the error.
    try:
        import time as _time

        for i in range(cfg.max_iterations):
            if hooks.should_cancel():
                stop_reason = "cancelled"
                break
            rec = IterationRecord(index=i, state=state.to_dict())
            hooks.apply(state)
            live = state
            _t0 = _time.time()
            path = hooks.render(f"iter{i:02d}")
            if i == 0 and path is not None:
                _dt = _time.time() - _t0
                worst = cfg.max_iterations + (cfg.polish_max_probes if cfg.polish else 0)
                hooks.log(f"~{_dt:.1f}s/render — worst case ≈ "
                          f"{_dt * worst / 60.0:.0f} min for this run")
            rec.render_path = path
            if path is None:
                hooks.log(f"iter {i}: render failed — stopping")
                stop_reason = "render_failed"
                records.append(rec)
                break

            measured_states.add(_state_fingerprint(state))
            cur_stats = hooks.stats(path) if metrics_ok else None
            # MEASURED mis-exposure of the frame the LLM is about to judge — drives the
            # contamination guard directly (the capped/annealed applied delta understates it)
            misexposure = 0.0
            if cur_stats is not None and ref_stats is not None:
                import math as _math

                misexposure = abs(_math.log2(
                    max(1e-5, float(ref_stats.get("log_key", 0.0)))
                    / max(1e-5, float(cur_stats.get("log_key", 0.0)))))
            if cur_stats is not None and ref_stats is not None:
                verdict = critic.score(ref_stats, cur_stats, cfg.weights)
                verdict.score = blend_transfer(verdict.score, state, hooks, cfg)
                rec.score, rec.components = verdict.score, verdict.components
                score_history.append((i, verdict.score))
                hooks.log(f"iter {i}: score {verdict.summary()}")

                improved = best_score is None or verdict.score > best_score + 1e-9
                if improved:
                    if best_score is not None and verdict.score < best_score + cfg.stall_delta:
                        stall_count += 1
                    else:
                        stall_count = 0
                    best_score, best_state, best_render = verdict.score, state.copy(), path
                    best_components = dict(verdict.components)
                    best_stats = dict(cur_stats)   # measurements OF the best state — the
                    # revert path below restores this exact state, so its stats are already
                    # in hand and re-rendering it would reproduce them frame-for-frame
                    slump_count = 0
                else:
                    if verdict.score < (best_score or 0) - cfg.slump_tolerance:
                        # slump iterations count toward slump-revert ONLY — feeding them
                        # into the stall counter too let one marginal gain plus one slump
                        # end a run as "stalled" before the 2-strike revert ever engaged
                        slump_count += 1
                        if slump_count >= 2:
                            hooks.log(f"iter {i}: slumping — reverting to best "
                                      f"({best_score:.1f})")
                            state = best_state.copy()
                            rec.reverted_to_best = True
                            slump_count = 0
                    else:
                        stall_count += 1
                        slump_count = 0

                if verdict.score >= cfg.target_score:
                    stop_reason = "target_reached"
                    records.append(rec)
                    break
                if stall_count >= cfg.stall_patience and i >= 1:
                    stop_reason = "stalled"
                    records.append(rec)
                    break
            else:
                # unscored iteration (LLM-visual mode, or one flaky stats read): while no
                # score exists to rank by, LATEST is best — a once-only assignment here
                # left the UI showing iteration 0's *before* frame as the final result
                if best_score is None or best_render is None:
                    best_state, best_render = state.copy(), path

            if rec.reverted_to_best:
                # The stats in hand describe the state we just ABANDONED, so the next
                # iteration must not reason from them. The restored state's OWN
                # measurements are already known, though — they were taken when it became
                # best, and a render is deterministic in the state (on-box 2026-07-25: the
                # same state re-rendered scores 100.0 against itself). So adopt the cached
                # measurements instead of burning an iteration re-rendering a known frame.
                # Measured waste this removes: 5 of 9 iterations in a live A3 match were
                # spent re-measuring states the loop already had.
                if best_stats is not None and best_render is not None:
                    cur_stats, path = dict(best_stats), best_render
                    rec.render_path = path
                    rec.score = best_score
                    # misexposure was measured from the ABANDONED frame above; recompute it
                    # against the restored state's stats or the contamination guard would
                    # judge this iteration on the wrong frame
                    if ref_stats is not None:
                        import math as _math

                        misexposure = abs(_math.log2(
                            max(1e-5, float(ref_stats.get("log_key", 0.0)))
                            / max(1e-5, float(cur_stats.get("log_key", 0.0)))))
                    hooks.log(f"iter {i}: reverted to best ({best_score:.1f}) — reusing its "
                              "measurements (no re-render)")
                else:
                    hooks.log(f"iter {i}: reverted — re-measuring before further changes")
                    records.append(rec)
                    continue

            if i == cfg.max_iterations - 1:  # last render measured; no point proposing more
                records.append(rec)
                break

            # ---- analytic solve (deterministic, before/independent of the LLM)
            # annealed: exploration-sized steps and deadbands shrink as the score climbs
            anneal = _anneal(best_score)
            analytic: Dict[str, float] = {}
            if cfg.analytic and cur_stats is not None and ref_stats is not None:
                analytic = solver.analytic_pass(state, ref_stats, cur_stats, locks,
                                                tighten=anneal)
                if "exposure.ev" in analytic:
                    wanted = analytic["exposure.ev"]
                    leashed = min(leash_ev_hi, max(leash_ev_lo, wanted))
                    if abs(leashed - wanted) > 1e-6:
                        leash_hits += 1
                        hooks.log(f"iter {i}: EV solve hit its leash "
                                  f"({leashed:+.1f} vs wanted {wanted:+.1f})")
                    elif _bound_pinned("exposure.ev", wanted):
                        leash_hits += 1
                        hooks.log(f"iter {i}: EV solve pinned at the genome bound "
                                  f"({wanted:+.1f}) — wants more than the rig can express")
                    analytic["exposure.ev"] = leashed
                elif ("exposure.ev" in state.values
                        and "exposure.ev" not in locks
                        and misexposure >= 0.5
                        and _bound_pinned("exposure.ev", state.get("exposure.ev"))):
                    # the solver ABSTAINED (pinned solve = a no-op re-write, emitted as
                    # None) but EV is sitting ON the genome bound with the frame still
                    # measurably mis-exposed — the same albedo signal, sustained; count
                    # it or the 2-strike diagnosis below never fires after iteration 0
                    leash_hits += 1
                    hooks.log(f"iter {i}: EV stays pinned at the genome bound "
                              f"({state.get('exposure.ev'):+.1f}) — render still "
                              f"{misexposure:.1f} stops off the reference")
                if "exposure.wb_kelvin" in analytic:
                    wanted = analytic["exposure.wb_kelvin"]
                    leashed = min(leash_wb_hi, max(leash_wb_lo, wanted))
                    if abs(leashed - wanted) > 1e-6:
                        leash_hits += 1
                        hooks.log(f"iter {i}: WB solve hit its leash ({leashed:.0f}K)")
                    elif _bound_pinned("exposure.wb_kelvin", wanted):
                        leash_hits += 1
                        hooks.log(f"iter {i}: WB solve pinned at the genome bound "
                                  f"({wanted:.0f}K) — wants more than the rig can express")
                    analytic["exposure.wb_kelvin"] = leashed
                if analytic:
                    state, accepted, _ = apply_changes(state, analytic, locks, limit=False)
                    rec.analytic_changes = accepted
                    hooks.log("iter %d: analytic %s" % (
                        i, ", ".join(f"{k}={v:.2f}" for k, v in accepted.items())))

            # ---- LLM deltas
            if hooks.should_cancel():
                stop_reason = "cancelled"
                records.append(rec)
                break
            # per-param trajectory — the model sees its own oscillation (live sim showed
            # altitude ping-ponging 6→-1→6 when each iteration judged in isolation)
            param_history: Dict[str, List[float]] = {}
            for r in records:
                for k, v in list(r.analytic_changes.items()) + list(r.llm_accepted.items()):
                    param_history.setdefault(k, []).append(round(v, 2))
            history_txt = "\n".join(
                f"  {k}: {' → '.join(str(x) for x in vs[-5:])}"
                for k, vs in sorted(param_history.items()) if len(vs) >= 2)
            ctx = {
                "iteration": i,
                "max_iterations": cfg.max_iterations,
                "state_table": state_table(state, locks),
                "semantics": semantics,
                "score_history": score_history,
                "analytic_applied": rec.analytic_changes,
                "param_history": history_txt,
                "render_path": path,
                "rig_notes": rig_notes,
                "director_note": director_note,
                "max_changes": cfg.max_changes,
            }
            try:
                from .parse import validate_deltas

                proposal = validate_deltas(hooks.llm_deltas(ctx), cfg.max_changes)
                # REPEAT GUARD. The prompt already carries the parameter and score
                # history, and the model is told to hold a ping-ponged value — but it
                # re-proposed an identical failing move anyway. Measured on-box
                # 2026-07-25 (A3): nine iterations produced three visits to the SAME
                # ~77.6 state and two identical ev=-4.0 proposals three iterations
                # apart, so five of nine renders re-measured something already known.
                # Drop deltas that would rebuild a state we have already scored; the
                # loop then spends the iteration somewhere new instead.
                if proposal.get("changes"):
                    would_be = state.copy()
                    for param, value in dict(proposal["changes"]).items():
                        try:
                            would_be.set(str(param), float(value))
                        except (TypeError, ValueError):
                            pass
                    if _state_fingerprint(would_be) in measured_states:
                        hooks.log(f"iter {i}: LLM re-proposed an already-measured state "
                                  "— dropped, keeping the analytic step")
                        proposal = dict(proposal)
                        proposal["changes"] = {}
                        proposal["reasons"] = {}
            except ParseError as e:
                hooks.log(f"iter {i}: LLM reply unusable ({e}) — keeping analytic-only step")
                proposal = {"assessment": "", "changes": {}, "reasons": {}, "stop": False}
            rec.assessment = proposal["assessment"]
            if proposal["assessment"]:
                hooks.log(f"iter {i}: gaffer: {proposal['assessment']}")
            # structural, not just prompted: while the analytic solver is running, ANALYTIC
            # params are the solver's alone — live fire showed the model overriding a perfect
            # EV solve and costing two iterations of re-correction (sim_match, 2026-07-16)
            if cfg.analytic and metrics_ok:
                from .genome import spec_for

                for k in [k for k in proposal["changes"]
                          if (spec_for(k) is not None and spec_for(k).analytic)]:
                    proposal["changes"].pop(k, None)
                    rec.llm_rejected.append(f"{k}: analytic — the solver owns it")
                    hooks.log(f"iter {i}: refused {k} (analytic — solver owns it)")
            # contaminated-iteration guard: the frame the LLM critiqued was MEASURABLY
            # mis-exposed — its absolute-brightness judgments (intensities, groups) are
            # contamination regardless of how much of the error the solver corrected
            if misexposure >= cfg.contaminated_ev_step:
                dropped = [k for k in proposal["changes"]
                           if k.endswith(".intensity") or k.startswith("group.")]
                for k in dropped:
                    proposal["changes"].pop(k, None)
                    rec.llm_rejected.append(
                        f"{k}: dropped — render was {misexposure:.1f} stops mis-exposed, "
                        "brightness judgment contaminated")
                if dropped:
                    hooks.log(f"iter {i}: dropped {len(dropped)} intensity change(s) — "
                              "the model judged a mis-exposed frame")
            from .genome import rig_keys

            state, accepted, rejected = apply_changes(state, proposal["changes"], locks,
                                                      limit=True, step_scale=anneal,
                                                      known=rig_keys(state))
            rec.llm_accepted = accepted
            rec.llm_rejected.extend(rejected)   # extend — the contamination guard logged here too
            for r in rejected:
                hooks.log(f"iter {i}: rejected {r}")
            for k, v in accepted.items():
                hooks.log(f"iter {i}: Δ {k} → {v:.2f}  ({proposal['reasons'].get(k, '')})")
            records.append(rec)
            if proposal["stop"] and not accepted and not rec.analytic_changes:
                stop_reason = "llm_satisfied"
                break
    except Exception:
        # keep-best survives a crash: a hook/critic raise must never leave the
        # last exploratory state standing in the scene (documented guarantee)
        try:
            hooks.apply(best_state)
        except Exception as restore_err:
            hooks.log(f"⚠ match aborted AND re-applying the best state failed "
                      f"({restore_err}) — use Restore to recover the pre-match rig")
        else:
            note = f" (best {best_score:.1f})" if best_score is not None else ""
            hooks.log(f"⚠ match aborted by an error — best-known state re-applied{note}; "
                      "the audit trail above is intact")
        raise

    if leash_hits >= 2:
        hooks.log("⚠ the exposure/WB solver kept hitting its leash — the reference and "
                  "this scene likely disagree in albedo (e.g. white room vs dark wood). "
                  "Consider locking exposure.ev / exposure.wb_kelvin and setting them by eye.")

    # ---- always land on the best known state
    if best_score is None:
        # unscored LLM-visual mode (metrics off): land on the latest applied state
        best_state = state
    # ...but only touch the scene when it is NOT already wearing that state — a no-op
    # re-apply is a phantom Ctrl+Z step (one undo record per apply, SPEC trust model).
    # live None = cancelled/failed before the FIRST apply: the scene was never touched
    # and must stay exactly as the run found it (matches are explorations).
    if live is not None and live.to_dict() != best_state.to_dict():
        hooks.apply(best_state)
    # else: cancelled/failed before the FIRST successful render — leave the scene exactly
    # as the run found it (matches are explorations, never commitments) and return
    # best_score=None so the controller keeps the camera's previously accepted state
    result = MatchResult(
        best_state=best_state,
        best_score=best_score,
        best_render=best_render,
        stop_reason=stop_reason,
        iterations=records,
        best_components=best_components,
    )

    # ---- DEEP-MATCH finisher: squeeze to the scene's ceiling, then prove it
    # (never after a render failure — polish would burn renders against a dead renderer)
    if (cfg.polish and best_score is not None and ref_stats is not None
            and best_score < cfg.polish_stop_at
            and stop_reason not in ("cancelled", "render_failed")):
        try:
            p_state, p_score, probes, converged, proven = run_polish(
                best_state, best_score, ref_stats, hooks, cfg, locks,
                leash_ref=start_state)
        except Exception:
            # polish probes are exploratory too — a dead hook mid-climb must leave the
            # loop's best state live, not the last probe
            try:
                hooks.apply(best_state)
            except Exception:
                pass
            hooks.log("⚠ polish aborted by an error — loop-best state re-applied")
            raise
        result.polish_gain = round(p_score - best_score, 2)
        result.polish_probes = probes
        result.ceiling_converged = converged
        result.ceiling_proven = proven
        result.best_state, result.best_score = p_state, p_score
        if getattr(hooks, "_polish_best_components", None):
            result.best_components = dict(hooks._polish_best_components)
        # NO trailing apply here — every run_polish return path already landed `best`;
        # re-applying the identical state is a no-op undo step (SPEC: one undo per apply)
        if converged and p_score < cfg.polish_stop_at:
            if proven:
                hooks.log(f"ceiling: no fine move improves {p_score:.1f} — that score IS "
                          "this scene's optimum for this reference (content gap, not "
                          "lighting)")
            else:
                hooks.log(f"polish: plateau at {p_score:.1f} — two consecutive low-gain "
                          "rounds (finer steps untested; a plateau, not a proven ceiling)")
    return result


def run_polish(
    state: LightingState,
    score_now: float,
    ref_stats: Dict,
    hooks: Hooks,
    cfg: MatchConfig,
    locks: Optional[Set[str]] = None,
    leash_ref: Optional[LightingState] = None,
) -> Tuple[LightingState, float, int, bool, bool]:
    """LLM-free ADAPTIVE coordinate line search. Per parameter: nudge, keep climbing in a
    direction while each rendered probe measurably improves the score; when neither
    direction improves, that parameter's step halves next round.
    → (best_state, best_score, probes_rendered, converged, proven): ``proven`` is the
    STRONG claim (every step at its fine floor + a full no-improve round = provable
    local optimum); ``converged`` without ``proven`` is the diminishing-returns
    plateau exit (two consecutive low-gain rounds; finer steps untested)."""
    locks = locks or set()
    best = state.copy()
    best_score = score_now
    hooks._polish_best_components = {}
    probes = 0
    axes = polish_axes(state)               # static params + this rig's groups/fog
    # dynamic ridge couples: EV can fake ANY dimmer group (measured 2026-07-24 on the
    # 400-lamp city: one zone ran away to 9.4x compensating exposure while, from an
    # aerial camera, zeroed zones could never clear min_gain single-axis — the diagonal
    # fixes both directions of the same valley)
    pairs = _POLISH_PAIRS + tuple(("exposure.ev", GROUP_PREFIX + g)
                                  for g in sorted(state.groups))
    steps = {k: s for k, s, _log, _floor in axes}
    # fail-memo: (step, score) at last failure per param — while neither has changed,
    # re-probing would render the exact same comparison again
    dead: Dict[str, Tuple[float, float]] = {}
    hooks.log(f"polish: adaptive line search from {best_score:.2f} "
              f"(≤{cfg.polish_rounds} rounds · ≤{cfg.polish_max_probes} probes)")

    def _memo_note() -> str:
        return (f" · {memo_hits} repeat states served from the probe memo (renders saved)"
                if memo_hits else "")

    # EXACT probe memo. A render is a deterministic function of the lighting state
    # (measured on-box 2026-07-25: the same state rendered twice scores a perfect 100.0
    # against itself at every resolution — V-Ray reproduces the frame bit-for-bit), so a
    # state already measured NEVER needs re-rendering. The line search revisits states
    # constantly: it climbs a direction, overshoots, and steps back onto a point it just
    # measured; the diagonal ridge-escape re-probes the same corners. Serving those from
    # the memo costs nothing and buys real probes — on a heavy scene each saved probe is
    # a whole frame.
    seen: Dict[tuple, Optional[float]] = {}
    memo_components: Dict[tuple, Dict[str, float]] = {}
    stats_of: Dict[tuple, Dict] = {}
    memo_hits = 0

    def _memo_key(st: LightingState) -> tuple:
        vals = tuple(sorted((k, round(float(v), 6)) for k, v in st.values.items()))
        grps = tuple(sorted((k, round(float(v), 6)) for k, v in st.groups.items()))
        return (vals, grps)

    def measure(cand: LightingState, tag: str) -> Optional[float]:
        nonlocal probes, memo_hits
        _leash(cand)                    # every probe respects the loop's tonal leash
        key = _memo_key(cand)
        if key in seen:
            memo_hits += 1
            hooks.apply(cand)          # the scene must still BE the state we report on
            cached = seen[key]
            if cached is not None:
                hooks._last_polish_components = dict(
                    memo_components.get(key, getattr(hooks, "_last_polish_components", {})))
            return cached
        if probes >= cfg.polish_max_probes:
            return None
        hooks.apply(cand)
        path = hooks.render(tag)
        if path is None:
            return None
        st = hooks.stats(path)
        if st is None:
            return None
        probes += 1
        verdict = critic.score(ref_stats, st, cfg.weights)
        verdict.score = blend_transfer(verdict.score, cand, hooks, cfg)
        hooks._last_polish_components = dict(verdict.components)
        seen[key] = verdict.score
        memo_components[key] = dict(verdict.components)
        stats_of[key] = st            # the frame's measurements, for the tonal re-solve
        return verdict.score

    seen[_memo_key(state)] = score_now      # the seed state is already measured
    hops_used = 0                           # azimuth basin hops taken (bounded)
    resolved_tonally = False                # the coordinated tonal jump, once
    # A RESTART needs somewhere safe to fall back to. The tonal jump is allowed to land
    # slightly WORSE than the incumbent and then descend — a coordinated move into the
    # right region rarely beats a well-tuned wrong one on its first raw probe (measured
    # 2026-07-25: the jump fired, missed the gain gate by a hair, and was discarded, so
    # the run finished on the scrambled tonal values again). champion holds the best state
    # ever seen so a failed restart can never cost the run anything.
    champion, champion_score = best.copy(), best_score

    def _finish(conv: bool, proven: bool):
        """Land on the better of the working point and the pre-restart champion."""
        final, final_score = best, best_score
        if champion_score > final_score + 1e-9:
            final, final_score = champion, champion_score
            hooks.log(f"polish: restart did not pay — returning the earlier best "
                      f"{final_score:.2f}")
        hooks.apply(final)
        return final, final_score, probes, conv, proven

    # The ANALYTIC LEASH also binds polish. The loop clamps its EV/WB solve to a window
    # around the run's start state, because matching histograms of DIFFERENT scenes biases
    # the tonal solve systematically — but polish had no such bound and walked white
    # balance to the genome wall. Measured 2026-07-25 on the realistic golden-hour room:
    # once the sun was pinned on, azimuth converged correctly (315°→123° against a 105°
    # target) yet the score stalled at 91.47 with exposure.wb_kelvin at 14,643 of a 15,000
    # bound and turbidity DOWN at 2.5 — the search making the image warm with the CAMERA
    # instead of the SUN. Bounding polish the same way as the loop forces the physically
    # correct route (turbidity/altitude carry the warmth).
    leash_bounds: Dict[str, Tuple[float, float]] = {}
    if leash_ref is not None:
        for key, span in (("exposure.ev", cfg.ev_leash),
                          ("exposure.wb_kelvin", cfg.wb_leash)):
            if key in state.values and key in leash_ref.values and span and span > 0:
                anchor = leash_ref.get(key)
                leash_bounds[key] = (anchor - span, anchor + span)

    def _leash(cand: LightingState) -> LightingState:
        """Clamp the leashed tonal axes back into the loop's window (in place)."""
        for key, (lo, hi) in leash_bounds.items():
            v = cand.get(key)
            if v < lo:
                cand.set(key, lo)
            elif v > hi:
                cand.set(key, hi)
        return cand

    try:
        low_gain_rounds = 0
        for rnd in range(cfg.polish_rounds):
            improved_any = False
            round_start = best_score
            for key, _init, is_log, floor in axes:
                if hooks.should_cancel() or best_score >= cfg.polish_stop_at \
                        or probes >= cfg.polish_max_probes:
                    return _finish(False, False)
                if key in locks or not _has_axis(best, key):
                    continue
                step = steps[key]
                if dead.get(key) == (step, best_score):
                    continue    # same step, same landscape — the answer hasn't changed
                param_moved = False
                for direction in (1.0, -1.0):
                    climbing = True
                    stride = step        # accelerating line search: consecutive keeps
                    while climbing and not hooks.should_cancel() \
                            and probes < cfg.polish_max_probes \
                            and best_score < cfg.polish_stop_at:
                        cand = best.copy()
                        v = cand.get(key)
                        # a log-scale probe MULTIPLIES: v == 0 stays 0 forever (a dome the
                        # LLM drove to 0 could never be turned back up). Seed the probe from
                        # the fine floor so a zeroed log axis is explorable again.
                        base = floor if is_log and v <= 0.0 else v
                        cand.set(key, base * (2.0 ** (direction * stride)) if is_log
                                 else v + direction * stride)
                        if abs(cand.get(key) - v) < 1e-6:
                            break            # clamped to a bound — nowhere to go
                        sc = measure(cand, f"polish{rnd}_{key.split('.')[-1]}")
                        if sc is not None and sc > best_score + cfg.polish_min_gain:
                            hooks.log(f"polish: {key} {v:.2f}→{cand.get(key):.2f} · "
                                      f"{best_score:.2f}→{sc:.2f} ✓")
                            best, best_score = cand, sc
                            hooks._polish_best_components = dict(
                                getattr(hooks, "_last_polish_components", {}))
                            improved_any = True
                            param_moved = True
                            stride *= 1.6          # keep riding the slope, faster
                        else:
                            climbing = False
                    if param_moved:
                        break                # THIS param rode uphill; its mirror is downhill
                if not param_moved:
                    dead[key] = (step, best_score)
            low_gain_rounds = (low_gain_rounds + 1
                               if best_score - round_start < cfg.polish_round_eps else 0)
            if not improved_any:
                # COMPENSATION-RIDGE escape: tonal and geometry axes can ratchet each
                # other AWAY from the target (altitude↑ fakes warmth, WB↑ cancels it —
                # the v0.9.5+ stats made this a live trap: polish stalled at 97.8 on the
                # ridge while the 99 summit sat one diagonal away). Single-axis moves
                # can't cross a rotated valley floor, so before halving, probe the
                # coupled pairs diagonally with the CURRENT steps (Powell-style) — one
                # bounded pass per stall, still under the probe budget.
                escaped = False
                is_log = {k: l for k, _s, l, _f in axes}

                def _diag_probe(ka: str, kb: str, sa: float, sb: float,
                                mult: float) -> Optional[float]:
                    cand = best.copy()
                    va, vb = cand.get(ka), cand.get(kb)
                    da, db = sa * steps[ka] * mult, sb * steps[kb] * mult
                    cand.set(ka, va * (2.0 ** da) if is_log[ka] else va + da)
                    cand.set(kb, vb * (2.0 ** db) if is_log[kb] else vb + db)
                    if abs(cand.get(ka) - va) < 1e-6 and abs(cand.get(kb) - vb) < 1e-6:
                        return None             # both clamped — no move at all
                    sc = measure(cand, f"polish{rnd}_diag")
                    if sc is not None and sc > best_score + cfg.polish_min_gain:
                        return sc
                    return None

                for ka, kb in pairs:
                    if escaped or hooks.should_cancel() \
                            or probes >= cfg.polish_max_probes:
                        break
                    if ka in locks or kb in locks or not _has_axis(best, ka) \
                            or not _has_axis(best, kb):
                        continue
                    for sa in (1.0, -1.0):
                        if escaped:
                            break
                        for sb in (1.0, -1.0):
                            sc = _diag_probe(ka, kb, sa, sb, 1.0)
                            if sc is None:
                                continue
                            hooks.log(f"polish: {ka}&{kb} diagonal · "
                                      f"{best_score:.2f}→{sc:.2f} ✓ (ridge escape)")
                            va, vb = best.get(ka), best.get(kb)
                            cand = best.copy()
                            da, db = sa * steps[ka], sb * steps[kb]
                            cand.set(ka, va * (2.0 ** da) if is_log[ka] else va + da)
                            cand.set(kb, vb * (2.0 ** db) if is_log[kb] else vb + db)
                            best, best_score = cand, sc
                            hooks._polish_best_components = dict(
                                getattr(hooks, "_last_polish_components", {}))
                            improved_any = escaped = True
                            # ride the valley: accelerate along the winning diagonal
                            # exactly like the single-axis climb does (stride ×1.6)
                            mult = 1.6
                            while best_score < cfg.polish_stop_at \
                                    and probes < cfg.polish_max_probes:
                                sc2 = _diag_probe(ka, kb, sa, sb, mult)
                                if sc2 is None:
                                    break
                                hooks.log(f"polish: {ka}&{kb} diagonal ×{mult:.1f} · "
                                          f"{best_score:.2f}→{sc2:.2f} ✓")
                                va, vb = best.get(ka), best.get(kb)
                                cand = best.copy()
                                da, db = sa * steps[ka] * mult, sb * steps[kb] * mult
                                cand.set(ka, va * (2.0 ** da) if is_log[ka] else va + da)
                                cand.set(kb, vb * (2.0 ** db) if is_log[kb] else vb + db)
                                best, best_score = cand, sc2
                                hooks._polish_best_components = dict(
                                    getattr(hooks, "_last_polish_components", {}))
                                mult *= 1.6
                            break
                if not escaped:
                    # DISCRETE escape: the on/off flags are not line-search axes (a 0..1
                    # switch has no gradient), so once the loop turns the sun OFF nothing
                    # can turn it back on — polish then perfects a SUNLESS metamer of a
                    # sunlit reference. Measured on-box 2026-07-25 (A6): recovered
                    # sun.enabled=0, altitude -2°, WB pinned at 15000, envelope collapsed
                    # to 0.12 against a golden-hour reference. Flipping a flag is one
                    # probe; if the flip does not pay, it is reverted immediately.
                    for flag in ("sun.enabled", "dome.enabled"):
                        if escaped or flag in locks or flag not in best.values \
                                or hooks.should_cancel() \
                                or probes >= cfg.polish_max_probes:
                            continue
                        cand = best.copy()
                        cand.set(flag, 0.0 if best.get(flag) >= 0.5 else 1.0)
                        if abs(cand.get(flag) - best.get(flag)) < 1e-9:
                            continue
                        sc = measure(cand, f"polish{rnd}_flip_{flag.split('.')[0]}")
                        if sc is not None and sc > best_score + cfg.polish_min_gain:
                            hooks.log(f"polish: {flag} {best.get(flag):.0f}→"
                                      f"{cand.get(flag):.0f} · {best_score:.2f}→{sc:.2f} ✓ "
                                      "(discrete escape)")
                            best, best_score = cand, sc
                            hooks._polish_best_components = dict(
                                getattr(hooks, "_last_polish_components", {}))
                            improved_any = escaped = True

                if not escaped:
                    # TONAL RE-SOLVE. The tonal group (ev / wb / dome / turbidity) can sit
                    # in a COUPLED local optimum where reaching the truth needs all four to
                    # move at once — dome down, turbidity up, WB down, EV down — while any
                    # single one alone goes downhill first. Measured on-box 2026-07-25 once
                    # geometry was exact (azimuth 105.0 against a 105 target): polish probed
                    # turbidity 7x, EV 6x and WB 5x across 8 rounds and kept the SCRAMBLED
                    # values, gaining +1.02, because no one-axis move pays.
                    #
                    # The analytic solver does not walk — it COMPUTES ev/wb straight from
                    # the histogram and highlight chromaticity. Re-solving them together
                    # against the current frame is exactly the coordinated jump the line
                    # search cannot make, and pairing it with a dome/turbidity step makes it
                    # the 4-way move the landscape demands.
                    best_stats_here = stats_of.get(_memo_key(best))
                    if best_stats_here is None and not resolved_tonally \
                            and probes < cfg.polish_max_probes:
                        # The seed state arrives with a SCORE but no STATS (the caller
                        # measured it), and it is pre-registered in the memo — so measure()
                        # would serve the cached score and still leave us without a frame
                        # to re-solve from. Drop the memo entry so this one goes through a
                        # real render; a run that never improves past its seed would
                        # otherwise never be able to fire the jump at all.
                        seen.pop(_memo_key(best), None)
                        measure(best.copy(), f"polish{rnd}_tonal_seed")
                        best_stats_here = stats_of.get(_memo_key(best))
                    if (best_stats_here is not None and not resolved_tonally
                            and "exposure.ev" in best.values
                            and probes < cfg.polish_max_probes):
                        ev_now = best.get("exposure.ev")
                        wb_now = best.get("exposure.wb_kelvin", 6500.0)
                        ev_new = solver.solve_ev(ref_stats, best_stats_here, ev_now)
                        wb_new = solver.solve_wb(ref_stats, best_stats_here, wb_now)
                        # pair the computed tonal jump with the ambient/haze partners that
                        # have to move with it, so the whole group travels together
                        jump_best, jump_score = None, None
                        for dome_mult, turb_step in ((1.0, 0.0), (0.5, 1.0), (0.35, 2.0)):
                            if hooks.should_cancel() \
                                    or probes >= cfg.polish_max_probes:
                                break
                            cand = best.copy()
                            if ev_new is not None and "exposure.ev" not in locks:
                                cand.set("exposure.ev", ev_new)
                            if wb_new is not None and "exposure.wb_kelvin" not in locks:
                                cand.set("exposure.wb_kelvin", wb_new)
                            if dome_mult != 1.0 and "dome.intensity" in cand.values \
                                    and "dome.intensity" not in locks:
                                cand.set("dome.intensity",
                                         cand.get("dome.intensity") * dome_mult)
                            if turb_step and "sun.turbidity" in cand.values \
                                    and "sun.turbidity" not in locks:
                                cand.set("sun.turbidity",
                                         cand.get("sun.turbidity") + turb_step)
                            sc = measure(cand, f"polish{rnd}_tonal{int(dome_mult * 100)}")
                            if sc is not None and (jump_score is None or sc > jump_score):
                                jump_best, jump_score = cand, sc
                        resolved_tonally = True   # one coordinated jump per polish run

                        # RESTART, not a probe. A coordinated jump lands in the right
                        # REGION; it rarely out-scores a finely-tuned wrong answer on the
                        # raw landing. Judging it by the usual gain gate threw it away
                        # (measured 2026-07-25: fired once, missed, run finished on the
                        # scrambled tonal values). So adopt it whenever it lands within
                        # _RESTART_TOLERANCE of the incumbent, re-open every step, and let
                        # it DESCEND — champion holds the old peak, so if the descent never
                        # pays the run still returns the better state.
                        if jump_best is not None \
                                and jump_score > best_score - _RESTART_TOLERANCE:
                            if best_score > champion_score:
                                champion, champion_score = best.copy(), best_score
                            hooks.log(
                                f"polish: tonal restart (ev→{jump_best.get('exposure.ev'):.2f} "
                                f"wb→{jump_best.get('exposure.wb_kelvin'):.0f}) · "
                                f"{best_score:.2f}→{jump_score:.2f}, descending from here")
                            best, best_score = jump_best, jump_score
                            hooks._polish_best_components = dict(
                                getattr(hooks, "_last_polish_components", {}))
                            improved_any = escaped = True
                            for k, s0, _l, _f in axes:   # fresh descent from the jump
                                steps[k] = s0

                if not escaped:
                    # AZIMUTH BASIN HOP. Sun azimuth is the one genuinely MULTI-MODAL
                    # axis: a rig lit from 294° and one lit from 115° are different
                    # basins, and a 12°-step line search can never walk between them
                    # downhill. Measured on-box 2026-07-25 (A6): polish returned
                    # proven=True — a real local optimum — at 84.76 with azimuth 179° from
                    # target and direction the weakest component at 0.74. So when the
                    # local search is otherwise exhausted, sample the OTHER lobes: jump a
                    # long way round the compass, keep a jump only if it measurably wins,
                    # and re-open the step sizes so the new basin gets a real descent.
                    if "sun.azimuth_deg" in best.values \
                            and "sun.azimuth_deg" not in locks and hops_used < _MAX_HOPS:
                        base_az = best.get("sun.azimuth_deg")
                        for offset in (180.0, 90.0, -90.0, 135.0, -135.0):
                            if hooks.should_cancel() or probes >= cfg.polish_max_probes:
                                break
                            cand = best.copy()
                            cand.set("sun.azimuth_deg", (base_az + offset) % 360.0)
                            sc = measure(cand, f"polish{rnd}_hop{int(offset)}")
                            if sc is not None and sc > best_score + cfg.polish_min_gain:
                                hooks.log(
                                    f"polish: azimuth basin hop {base_az:.0f}°→"
                                    f"{cand.get('sun.azimuth_deg'):.0f}° · "
                                    f"{best_score:.2f}→{sc:.2f} ✓")
                                best, best_score = cand, sc
                                hooks._polish_best_components = dict(
                                    getattr(hooks, "_last_polish_components", {}))
                                improved_any = escaped = True
                                hops_used += 1
                                # a new basin deserves a fresh descent, not the fine
                                # steps the previous basin had annealed down to
                                for k, s0, _l, _f in axes:
                                    steps[k] = s0
                                break

                if escaped:
                    low_gain_rounds = 0
                    continue
                all_floored = all(steps[k] <= floor + 1e-9
                                  for k, _s, _l, floor in axes)
                if all_floored:
                    hooks.log(f"polish: converged at {best_score:.2f}{_memo_note()}")
                    return _finish(True, True)          # proven local optimum
                # A no-improve round at COARSE steps means the step size is wrong, not
                # that the climb is over — refine and keep going. Exiting here was
                # measured to abandon ~9 points of real headroom (2026-07-24: three
                # archetypes quit at 91 with converged=True while their self-match
                # optimum was 100 by construction).
                for k, _s, _l, floor in axes:
                    steps[k] = max(floor, steps[k] / 2.0)
            elif low_gain_rounds >= 2:
                if all(steps[k] <= floor + 1e-9 for k, _s, _l, floor in axes):
                    hooks.log(f"polish: plateau at {best_score:.2f}{_memo_note()}")
                    return _finish(True, False)
                for k, _s, _l, floor in axes:   # still coarse — refine before concluding
                    steps[k] = max(floor, steps[k] / 2.0)
                low_gain_rounds = 0
        hooks.log(f"polish: budget spent at {best_score:.2f}{_memo_note()}")
        return _finish(False, False)
    except Exception:
        # polish is exploratory too — a dead hook mid-climb must leave the best state
        # live, not the last probe (the caller re-applies and logs as well; this is the
        # in-function guarantee so even direct callers keep it)
        try:
            hooks.apply(best)
        except Exception:
            pass
        raise


#: Contrast margin at which the sweep's winner is considered decisively ahead. The same
#: 0.15 already gates the metric overriding the LLM's pick, so a winner clearing it is one
#: the code elsewhere already trusts enough to overrule a judgement on.
_SWEEP_DECISIVE_MARGIN = 0.15


def _report_sweep(out: Optional[Dict], contrast: List[float], idx: Optional[int],
                  basis: str) -> None:
    """Record HOW SURE the sweep is, not just what it picked.

    The winner's lead over the runner-up is the whole signal. Measured on-box 2026-07-25:
    on one interior the sweep landed ~195 degrees from the true sun, and because the caller
    was told only the answer and not the margin, it trusted that answer completely and the
    match defended a sun on the wrong side of the building. A near-tie between probes and a
    decisive win must not be reported the same way."""
    if out is None:
        return
    out["basis"] = basis
    out["contrast"] = list(contrast)
    if idx is None or len(contrast) < 2:
        # no measurable table: the pick rests on the LLM's comparison alone. That is real
        # evidence, but unverified — never full confidence.
        out["margin"] = None
        out["confidence"] = 0.5
        return
    rest = [c for i, c in enumerate(contrast) if i != idx]
    margin = contrast[idx] - max(rest)
    out["margin"] = round(margin, 4)
    out["confidence"] = round(max(0.0, min(1.0, margin / _SWEEP_DECISIVE_MARGIN)), 3)


def run_sun_sweep(
    state: LightingState,
    azimuths: List[float],
    hooks: Hooks,
    llm_pick: Callable[[List[str], List[float]], str],
    ref_stats: Optional[Dict] = None,
    out: Optional[Dict] = None,
) -> Tuple[Optional[float], str, str]:
    """Grid-solve the sun direction: render one low-res frame per azimuth, let the LLM do
    multiple-choice (estimation is hard, comparison is easy) — CROSS-CHECKED by the
    deterministic direction metric (3×3 luminance-grid cosine vs the reference) when
    ``ref_stats`` carries a grid. A clear metric winner overrides an LLM pick it beats by
    a margin: two independent judges beat one on the system's weakest call.
    Returns (azimuth | None, altitude_hint, why)."""
    from .metrics import cosine
    from .parse import validate_sweep

    paths: List[str] = []
    kept: List[float] = []
    dir_scores: List[Optional[float]] = []
    probe_grids: List[List[float]] = []
    ref_grid = (ref_stats or {}).get("grid")
    # probes move the sun BEFORE their render — a FAILED sweep (too few renders, an
    # unusable reply, a dead hook) must hand back the scene it was given, not the
    # last probed azimuth (the loop's iter0 apply only masked this on the happy path)
    entry = state.copy()
    touched = False                 # no probe applied yet → nothing to restore
    completed = False
    try:
        for az in azimuths:
            if hooks.should_cancel():
                return None, "na", "cancelled"
            probe = state.copy()
            probe.set("sun.azimuth_deg", az)
            hooks.apply(probe)
            touched = True
            path = hooks.render(f"sweep{az:03.0f}")
            if path:
                paths.append(path)
                kept.append(az)
                score = None
                if ref_grid and any(abs(v) > 1e-6 for v in ref_grid):
                    st = hooks.stats(path)
                    if st and st.get("grid"):
                        score = (cosine(ref_grid, st["grid"]) + 1.0) / 2.0
                        probe_grids.append(list(st["grid"]))
                dir_scores.append(score)
            else:
                hooks.log(f"sweep: render failed at azimuth {az:.0f}° — skipping")
        if len(paths) < 2:
            return None, "na", "not enough sweep renders"
        # CONTRASTIVE metric table — computed BEFORE the LLM call so a dead gateway can
        # still solve direction. Cross-check only when EVERY probe was measurable (a
        # partial table could crown a probe merely because its rivals went unmeasured).
        # All probes share the scene's dominant pattern (sky gradient), which swamps the
        # sun's contribution — live fire showed a SUNLESS probe scoring 0.97 raw
        # similarity — so subtract the probes' mean grid: only what varies WITH sun
        # direction is compared. Negligible residue disables the metric entirely.
        measured_all = (all(s is not None for s in dir_scores)
                        and len(probe_grids) == len(kept))
        metric_idx: Optional[int] = None
        contrast: List[float] = []
        if measured_all:
            mean_grid = [sum(g[i] for g in probe_grids) / len(probe_grids)
                         for i in range(9)]
            ref_d = [ref_grid[i] - mean_grid[i] for i in range(9)]
            if sum(abs(v) for v in ref_d) > 0.01:
                for g in probe_grids:
                    d = [g[i] - mean_grid[i] for i in range(9)]
                    contrast.append((cosine(ref_d, d) + 1.0) / 2.0)
                metric_idx = max(range(len(contrast)), key=lambda i: contrast[i])
        try:
            picked = validate_sweep(llm_pick(paths, kept), len(paths))
        except ParseError as e:
            if metric_idx is not None:        # gateway down ≠ direction unsolved
                az = kept[metric_idx]
                hooks.log(f"sweep: LLM pick unusable ({e}) — metric-only winner "
                          f"{az:.0f}° (contrast {contrast[metric_idx]:.2f})")
                completed = True
                _report_sweep(out, contrast, metric_idx, "metric-only")
                return az, "na", "metric-only pick (LLM unavailable)"
            return None, "na", f"sweep reply unusable: {e}"
        idx = picked["best_index"]
        if metric_idx is not None:
            if metric_idx != idx and contrast[metric_idx] - contrast[idx] > 0.15:
                hooks.log(f"sweep: direction metric overrides — {kept[metric_idx]:.0f}° "
                          f"(contrast {contrast[metric_idx]:.2f}) beats the pick of "
                          f"{kept[idx]:.0f}° ({contrast[idx]:.2f})")
                idx = metric_idx
        elif measured_all:
            hooks.log("sweep: direction residue too small to cross-check — LLM pick stands")
        az = kept[idx]
        _report_sweep(out, contrast, idx, "llm+metric" if contrast else "llm-only")
        conf = (out or {}).get("confidence")
        hooks.log(f"sweep: azimuth {az:.0f}° — {picked['why']}"
                  + (f" (confidence {conf:.0%})" if conf is not None else ""))
        completed = True
        return az, picked["altitude_hint"], picked["why"]
    finally:
        if not completed and touched:
            try:
                hooks.apply(entry)
            except Exception:
                pass            # the real failure is already surfacing — never mask it
            else:
                hooks.log("sweep: failed — sun azimuth restored to "
                          f"{entry.get('sun.azimuth_deg'):.0f}°")
