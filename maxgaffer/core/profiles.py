"""Bounded render profiles for interactive and hero lighting matches.

Profiles are pure data so the UI, public API, and Max bridge all describe the same
budget. Scored loop renders keep one resolution; only the directional sweep is reduced,
which avoids resolution-dependent score drift while still saving expensive probes.
"""

from __future__ import annotations

from dataclasses import dataclass


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
        """
        return (max(64, self.loop_width // 2), max(36, self.loop_height // 2))

    @property
    def worst_case_renders(self) -> int:
        return self.max_iterations + self.sweep_count + self.polish_max_probes


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
