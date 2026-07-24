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
                            False, 0, 0)
    if key == "hero":
        sw, sh = _scaled(width, height, 0.5, floor_w=192)
        # polish budget 10 rounds / 160 probes: the old 6/48 cap was measured EXHAUSTED
        # with gains still coming (2026-07-24 on-box hero runs: polish_gain +43..+49 and
        # ceiling_converged False at the cap, three runs in a row) — and the axis list
        # is now dynamic (groups + fog), so the budget must cover more parameters too.
        return MatchProfile("hero", "Hero", width, height, sw, sh,
                            max(iterations, 8), min(sweeps, 6), 99.0,
                            True, 10, 160)

    sw, sh = _scaled(width, height, 0.5, floor_w=192)
    return MatchProfile("standard", "Standard", width, height, sw, sh,
                        iterations, sweeps, target, False, 0, 0)
