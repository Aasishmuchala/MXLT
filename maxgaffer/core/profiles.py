"""Bounded render profiles for interactive and hero lighting matches.

Profiles are pure data so the UI, public API, and Max bridge all describe the same
budget. Scored loop renders keep one resolution; only the directional sweep is reduced,
which avoids resolution-dependent score drift while still saving expensive probes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MatchProfile:
    name: str
    label: str
    loop_width: int
    loop_height: int
    sweep_width: int
    sweep_height: int
    max_iterations: int
    sweep_count: int
    target_score: float
    polish: bool
    polish_rounds: int
    polish_max_probes: int
    #: iterations without progress before the loop gives up. Measured 2026-07-25: with the
    #: MatchConfig default of 2, hero runs ended after 3 of their 10 iterations on a single
    #: dip -- 70% of the loop budget went unused, and the loop is where GEOMETRY is solved
    #: (polish only refines the basin it is handed). A deep profile needs a longer leash.
    stall_patience: int = 2

    @property
    def polish_size(self):
        """Resolution for POLISH probes — half the loop's, and measured, not assumed.

        Polish is ~82% of a match: 120 renders of "nudge one parameter, did that help".
        It never needs a pixel-accurate frame, it needs a correct ORDERING, and ordering
        survives the cheaper render. Measured on-box 2026-07-26 across eight states spanning
        exposure, dome, azimuth, turbidity and altitude: half-resolution scores landed within
        0.63 of full and ranked all eight IDENTICALLY, for half the render time.

        Quarter resolution also preserved the ranking but drifted up to 1.33 points, which is
        too coarse to be safe here — polish accepts a move on a 0.03 gain, so a shift that
        size would let it chase measurement error instead of light. Half is the tier the
        evidence supports.

        That first measurement compared GROSS states (scores 16 to 79) and so proved the
        easy claim. What polish actually needs is the harder one: that a single nudge of one
        parameter moves the score the same WAY at both sizes. Measured separately, one nudge
        per polish axis in both directions, 13 in total: every one agreed on direction.
        Median disagreement on the delta itself was 0.19 and the worst 1.20, and the two
        nudges that fell under polish's own 0.03 gate were rejected at both sizes, so the
        decision matched there too.

        Known limit, stated because it is not covered by either measurement: nudges taken
        near CONVERGENCE, where the true delta is a few hundredths, are not represented in
        that set. The direction was right in every case tested, but the magnitude error
        means polish can stop at a marginally different point than a full-size run would.
        The score reported to the artist never comes from these frames — the final state is
        re-rendered full size — so the cost of being wrong here is a slightly different
        answer, never a wrong number about it.
        """
        return (max(64, self.loop_width // 2), max(36, self.loop_height // 2))

    @property
    def worst_case_renders(self) -> int:
        """DEPRECATED — kept because tests and the public API assert on it.

        It counts iterations + sweep + polish and NOTHING else, which on a standard
        profile is 133 of the 190 renders a match can actually fire: it omits the
        multi-start basin (≤6), the two tone-align passes, the sun solve (up to 44), the
        two exposure-host plates, the two plan probes and the final full re-render. The
        artist was quoted 133 and charged 190 — roughly an hour of extra TULA renders that
        nobody had agreed to. ``planned_renders`` is the one budget; this stays only so
        old callers keep working, and it is derived from the new function so the two can
        never drift apart again.
        """
        return sum(s.count for s in planned_renders(
            self, do_sweep=True, has_sun=False, multi_start=False,
            exposure_free=False, plan_ran=False, verify_exposure=False,
            final_render=False))


@dataclass(frozen=True)
class Stage:
    """One budgeted block of renders, at ONE resolution.

    ``count`` is the WORST case for that stage — what the artist could be charged, which
    is the only honest number to quote before committing. The resolution rides along
    because pricing needs it: on a heavy scene 181 of a standard profile's 190 renders are
    the same 240×135 size, so one timed probe at that size prices 96% of the match.
    """
    key: str
    label: str
    count: int
    width: int
    height: int


def planned_renders(profile: "MatchProfile", *, do_sweep: bool = True,
                    has_sun: bool = True, multi_start: bool = True,
                    n_scenarios: int = 6, exposure_free: bool = True,
                    plan_ran: bool = False, verify_exposure: bool = True,
                    final_render: bool = True,
                    ref_has_patches: bool = True) -> List[Stage]:
    """EVERY render a match can fire, as ordered stages. ONE budget, one source of truth.

    Written 2026-07-31. There were two hand-maintained copies of this number and they
    disagreed: ``controller.run_match`` computed ``hard_cap = max_iterations + sweep +
    polish`` = 133 and printed it to the artist, while its own progress bar's
    ``_stage_budget`` a few lines later knew about the basin (6) and the sun solve (44)
    and totalled 183 — and neither counted the tone-align passes, the exposure-host
    plates, the plan probes or the final full re-render. The true worst case on a standard
    profile is 190. Being short by 57 renders on TULA is being short by an hour.

    Same failure shape as the draft-cap bug 29bbae6 fixed: two sources of truth for one
    setting. Both callers now derive from this.
    """
    from . import sunsolve                    # local: keeps the module import-cheap

    sweep_size = (profile.sweep_width, profile.sweep_height)
    loop_size = (profile.loop_width, profile.loop_height)
    stages: List[Stage] = []
    if verify_exposure:
        # _verify_exposure_host fires TWO 160×90 rd.render_frame calls with no
        # announcement at all — on TULA that is ~2 minutes of silence right after
        # "run dir: …". They are renders; they are counted.
        stages.append(Stage("evcheck", "exposure-host check", 2, 160, 90))
    if plan_ran:
        stages.append(Stage("plan", "plan effect", 2, *loop_size))
    if multi_start:
        stages.append(Stage("basin", "multi-start", max(1, int(n_scenarios)),
                            *sweep_size))
    if has_sun and exposure_free:
        stages.append(Stage("tone", "tone-align", 2, *sweep_size))
    if has_sun and ref_has_patches:
        coarse = int(getattr(sunsolve, "COARSE_AZIMUTHS", 12)) * len(
            sunsolve.COARSE_ALTITUDES)
        fine = len(sunsolve.FINE_AZIMUTH_OFFSETS) + len(sunsolve.FINE_ALTITUDE_OFFSETS)
        stages.append(Stage("sunsolve", "sun solve", coarse + fine, *sweep_size))
    if do_sweep:
        stages.append(Stage("sweep", "sun sweep", max(0, int(profile.sweep_count)),
                            *sweep_size))
    stages.append(Stage("iter", "match loop", max(1, int(profile.max_iterations)),
                        *loop_size))
    if profile.polish and profile.polish_max_probes:
        stages.append(Stage("polish", "polish", int(profile.polish_max_probes),
                            *profile.polish_size))
    if final_render:
        stages.append(Stage("final", "final re-render", 1, *loop_size))
    return [s for s in stages if s.count > 0]


def _scaled(width: int, height: int, scale: float, floor_w: int = 160):
    width = max(1, int(width))
    height = max(1, int(height))
    out_w = min(width, max(min(width, floor_w), int(round(width * scale))))
    out_h = max(1, int(round(height * out_w / width)))
    return out_w, out_h


def resolve_profile(name: str, *, loop_width: int, loop_height: int,
                    max_iterations: int, sweep_count: int,
                    target_score: float) -> MatchProfile:
    """Resolve user settings into a finite Fast, Standard, or Hero budget."""
    key = str(name or "standard").strip().lower()
    if key in ("deep", "quality", "final"):
        key = "hero"
    if key not in ("fast", "standard", "hero"):
        raise ValueError(f"unknown match profile {name!r}; use fast, standard, or hero")

    width = max(64, int(loop_width))
    height = max(36, int(loop_height))
    iterations = max(1, int(max_iterations))
    sweeps = max(0, int(sweep_count))
    target = float(target_score)

    if key == "fast":
        fw, fh = _scaled(width, height, 2.0 / 3.0, floor_w=192)
        sw, sh = _scaled(fw, fh, 0.5, floor_w=128)
        return MatchProfile("fast", "Fast", fw, fh, sw, sh,
                            min(iterations, 3), min(sweeps, 4), min(target, 78.0),
                            False, 0, 0, stall_patience=2)
    if key == "hero":
        sw, sh = _scaled(width, height, 0.5, floor_w=192)
        # Polish budget 24 rounds / 500 probes. Every raise so far was measured, not
        # guessed: 6/48 -> 10/160 when the cap was hit with gains still coming, and now
        # -> 24/500 because on-box hero runs still ended ceiling_converged=False at 240
        # probes while climbing. The budget is affordable because a probe on an
        # already-measured state no longer renders (run_polish's exact memo — renders are
        # deterministic in the state), and because polish STOPS when it proves a local
        # optimum: the deterministic sim converges at 185 probes from a bad basin, so this
        # is headroom for harder real landscapes, not a licence to grind.
        # Sweep 12 directions, not 6. Six is 60-degree resolution, and the sweep's answer
        # is the ONE decision that sets the basin everything downstream refines inside:
        # measured on-box 2026-07-25 with locks applied, the true sun sat at 105 degrees
        # and a 6-way sweep could only offer 60 or 120. It picked 60 and the match ended 45
        # degrees out. Twelve small half-res renders against a 500-probe polish budget is
        # not where this profile's time goes. Raised the way iterations already are — hero
        # lifts a too-low setting rather than capping a generous one.
        return MatchProfile("hero", "Hero", width, height, sw, sh,
                            max(iterations, 8), max(sweeps, 12), 99.0,
                            True, 24, 500, stall_patience=5)

    sw, sh = _scaled(width, height, 0.5, floor_w=192)
    # Standard used to have NO polish at all, so the default match handed back whatever
    # the loop happened to land on. A modest finisher (8 rounds / 120 probes) costs a few
    # minutes and is the difference between a rough guess and a usable match; Hero keeps
    # the deep 24/500 budget and Fast keeps none.
    return MatchProfile("standard", "Standard", width, height, sw, sh,
                        iterations, sweeps, target, True, 8, 120, stall_patience=3)
