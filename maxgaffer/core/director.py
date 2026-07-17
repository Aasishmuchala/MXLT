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
from .genome import LightingState, apply_changes, state_table
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


def _anneal(best_score: Optional[float]) -> float:
    """Step/deadband scale from convergence: explore big, finish small."""
    if best_score is None or best_score < 70.0:
        return 1.0
    if best_score < 85.0:
        return 0.5
    return 0.25


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


@dataclass
class Hooks:
    """The loop's only contact with the world. ``apply``/``render`` run on Max's main thread
    (the UI layer guarantees that); llm/stats may be slow and are wrapped by the caller."""
    apply: Callable[[LightingState], None]
    render: Callable[[str], Optional[str]]            # tag → image path (None = failed)
    stats: Callable[[str], Optional[Dict]]            # image path → stats dict
    llm_deltas: Callable[[Dict], str]                 # context → raw reply text
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
    polish_gain: float = 0.0            # score added by the coordinate-descent finisher
    polish_probes: int = 0
    ceiling_converged: bool = False     # polish exhausted: no fine move improves — this
                                        # score IS the scene's ceiling for this reference

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

    for i in range(cfg.max_iterations):
        if hooks.should_cancel():
            stop_reason = "cancelled"
            break
        rec = IterationRecord(index=i, state=state.to_dict())
        hooks.apply(state)
        path = hooks.render(f"iter{i:02d}")
        rec.render_path = path
        if path is None:
            hooks.log(f"iter {i}: render failed — stopping")
            stop_reason = "render_failed"
            records.append(rec)
            break

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
                slump_count = 0
            else:
                stall_count += 1
                if verdict.score < (best_score or 0) - cfg.slump_tolerance:
                    slump_count += 1
                    if slump_count >= 2:
                        hooks.log(f"iter {i}: slumping — reverting to best "
                                  f"({best_score:.1f})")
                        state = best_state.copy()
                        rec.reverted_to_best = True
                        slump_count = 0
                else:
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
            if best_render is None:
                best_state, best_render = state.copy(), path

        if rec.reverted_to_best:
            # the stats in hand describe the state we just ABANDONED — solving or asking
            # the LLM from them would tweak the restored state on stale evidence. Re-render
            # first; next iteration reasons from coherent measurements.
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
                leashed = min(leash_ev_hi, max(leash_ev_lo, analytic["exposure.ev"]))
                if abs(leashed - analytic["exposure.ev"]) > 1e-6:
                    leash_hits += 1
                    hooks.log(f"iter {i}: EV solve hit its leash "
                              f"({leashed:+.1f} vs wanted {analytic['exposure.ev']:+.1f})")
                analytic["exposure.ev"] = leashed
            if "exposure.wb_kelvin" in analytic:
                leashed = min(leash_wb_hi, max(leash_wb_lo, analytic["exposure.wb_kelvin"]))
                if abs(leashed - analytic["exposure.wb_kelvin"]) > 1e-6:
                    leash_hits += 1
                    hooks.log(f"iter {i}: WB solve hit its leash ({leashed:.0f}K)")
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
        state, accepted, rejected = apply_changes(state, proposal["changes"], locks,
                                                  limit=True, step_scale=anneal)
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

    if leash_hits >= 2:
        hooks.log("⚠ the exposure/WB solver kept hitting its leash — the reference and "
                  "this scene likely disagree in albedo (e.g. white room vs dark wood). "
                  "Consider locking exposure.ev / exposure.wb_kelvin and setting them by eye.")

    # ---- always land on the best known state
    if best_score is not None:
        hooks.apply(best_state)
    else:
        best_state = state
        hooks.apply(best_state)
    result = MatchResult(
        best_state=best_state,
        best_score=best_score,
        best_render=best_render,
        stop_reason=stop_reason,
        iterations=records,
    )

    # ---- DEEP-MATCH finisher: squeeze to the scene's ceiling, then prove it
    if (cfg.polish and best_score is not None and ref_stats is not None
            and best_score < cfg.polish_stop_at and stop_reason != "cancelled"):
        p_state, p_score, probes, converged = run_polish(
            best_state, best_score, ref_stats, hooks, cfg, locks)
        result.polish_gain = round(p_score - best_score, 2)
        result.polish_probes = probes
        result.ceiling_converged = converged
        result.best_state, result.best_score = p_state, p_score
        hooks.apply(p_state)
        if converged and p_score < cfg.polish_stop_at:
            hooks.log(f"ceiling: no fine move improves {p_score:.1f} — that score IS this "
                      "scene's optimum for this reference (content gap, not lighting)")
    return result


def run_polish(
    state: LightingState,
    score_now: float,
    ref_stats: Dict,
    hooks: Hooks,
    cfg: MatchConfig,
    locks: Optional[Set[str]] = None,
) -> Tuple[LightingState, float, int, bool]:
    """LLM-free ADAPTIVE coordinate line search. Per parameter: nudge, keep climbing in a
    direction while each rendered probe measurably improves the score; when neither
    direction improves, that parameter's step halves next round. Converged when every
    unlocked parameter's step is at its fine floor and a full round changed nothing — a
    provable local optimum. → (best_state, best_score, probes_rendered, converged)."""
    locks = locks or set()
    best = state.copy()
    best_score = score_now
    probes = 0
    steps = {k: s for k, s, _log, _floor in POLISH_PARAMS}
    # fail-memo: (step, score) at last failure per param — while neither has changed,
    # re-probing would render the exact same comparison again
    dead: Dict[str, Tuple[float, float]] = {}
    hooks.log(f"polish: adaptive line search from {best_score:.2f} "
              f"(≤{cfg.polish_rounds} rounds · ≤{cfg.polish_max_probes} probes)")

    def measure(cand: LightingState, tag: str) -> Optional[float]:
        nonlocal probes
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
        return critic.score(ref_stats, st, cfg.weights).score

    low_gain_rounds = 0
    for rnd in range(cfg.polish_rounds):
        improved_any = False
        round_start = best_score
        for key, _init, is_log, floor in POLISH_PARAMS:
            if hooks.should_cancel() or best_score >= cfg.polish_stop_at \
                    or probes >= cfg.polish_max_probes:
                hooks.apply(best)
                return best, best_score, probes, False
            if key in locks or key not in best.values:
                continue
            step = steps[key]
            if dead.get(key) == (step, best_score):
                continue    # same step, same landscape — the answer hasn't changed
            param_moved = False
            for direction in (1.0, -1.0):
                climbing = True
                stride = step        # accelerating line search: consecutive keeps
                while climbing and probes < cfg.polish_max_probes \
                        and best_score < cfg.polish_stop_at:
                    cand = best.copy()
                    v = cand.get(key)
                    cand.set(key, v * (2.0 ** (direction * stride)) if is_log
                             else v + direction * stride)
                    if abs(cand.get(key) - v) < 1e-6:
                        break            # clamped to a bound — nowhere to go
                    sc = measure(cand, f"polish{rnd}_{key.split('.')[-1]}")
                    if sc is not None and sc > best_score + cfg.polish_min_gain:
                        hooks.log(f"polish: {key} {v:.2f}→{cand.get(key):.2f} · "
                                  f"{best_score:.2f}→{sc:.2f} ✓")
                        best, best_score = cand, sc
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
        if low_gain_rounds >= 2:
            hooks.apply(best)
            return best, best_score, probes, True   # diminishing returns = ceiling
        if not improved_any:
            all_floored = all(steps[k] <= floor + 1e-9
                              for k, _s, _l, floor in POLISH_PARAMS)
            if all_floored:
                hooks.apply(best)
                return best, best_score, probes, True   # proven local optimum
            for k, _s, _l, floor in POLISH_PARAMS:
                steps[k] = max(floor, steps[k] / 2.0)
    hooks.apply(best)
    return best, best_score, probes, False


def run_sun_sweep(
    state: LightingState,
    azimuths: List[float],
    hooks: Hooks,
    llm_pick: Callable[[List[str], List[float]], str],
    ref_stats: Optional[Dict] = None,
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
    for az in azimuths:
        if hooks.should_cancel():
            return None, "na", "cancelled"
        probe = state.copy()
        probe.set("sun.azimuth_deg", az)
        hooks.apply(probe)
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
    # CONTRASTIVE metric table — computed BEFORE the LLM call so a dead gateway can still
    # solve direction. Cross-check only when EVERY probe was measurable (a partial table
    # could crown a probe merely because its rivals went unmeasured). All probes share the
    # scene's dominant pattern (sky gradient), which swamps the sun's contribution — live
    # fire showed a SUNLESS probe scoring 0.97 raw similarity — so subtract the probes'
    # mean grid: only what varies WITH sun direction is compared. Negligible residue
    # disables the metric entirely.
    measured_all = all(s is not None for s in dir_scores) and len(probe_grids) == len(kept)
    metric_idx: Optional[int] = None
    contrast: List[float] = []
    if measured_all:
        mean_grid = [sum(g[i] for g in probe_grids) / len(probe_grids) for i in range(9)]
        ref_d = [ref_grid[i] - mean_grid[i] for i in range(9)]
        if sum(abs(v) for v in ref_d) > 0.01:
            for g in probe_grids:
                d = [g[i] - mean_grid[i] for i in range(9)]
                contrast.append((cosine(ref_d, d) + 1.0) / 2.0)
            metric_idx = max(range(len(contrast)), key=lambda i: contrast[i])
    try:
        picked = validate_sweep(llm_pick(paths, kept), len(paths))
    except ParseError as e:
        if metric_idx is not None:            # gateway down ≠ direction unsolved
            az = kept[metric_idx]
            hooks.log(f"sweep: LLM pick unusable ({e}) — metric-only winner {az:.0f}° "
                      f"(contrast {contrast[metric_idx]:.2f})")
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
    hooks.log(f"sweep: azimuth {az:.0f}° — {picked['why']}")
    return az, picked["altitude_hint"], picked["why"]
