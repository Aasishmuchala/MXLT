"""Is this frame worth scoring at all? — the validator every render passes through.

Written 2026-07-31. On 2026-07-30 a TULA run scored 100% black V-Ray frames as 12.0,
10.9, 8.7 and 2.7 and picked a "best basin" from them, for hours. The tool could not tell
it was looking at nothing. That is the invariant this module exists to enforce: NEVER
SCORE AN UNVALIDATED IMAGE.

Four tests, in cost order — the first two cost no pixels at all:

  1. WRONG SIZE  — ``png_min.read_png_size`` reads IHDR and nothing else.
  2. BLACK       — every sampled pixel exactly zero (``metrics.is_black``'s semantics).
  3. NEAR BLACK  — dark AND FLAT. The discriminator is structure, not level (see below).
  4. FROZEN      — pixel-identical to the previous plate while the applied state CHANGED.

Pure, stdlib-only, PIL-free, and it takes a stats dict rather than a path: the caller has
already paid for ``compute_stats`` and this must not compute it a second time.

WHY NEAR-BLACK IS NOT JUST "DARK"
---------------------------------
The requirement is that this must never fire on a legitimately dark render — the
``practicals_dusk`` and ``night_drive`` archetypes are dark on purpose and an artist
matching a moonlit exterior must not be locked out of their own tool. Level alone cannot
separate them from a dead renderer, because both are dark. STRUCTURE can: a real dark
render carries practicals, a sky gradient and V-Ray's own sampling noise, so p95 − p5 sits
comfortably above one 8-bit quantisation step even when p95 is 6/255. A frame produced by
a renderer that returned nothing is dark AND flat AND carries no hot pixel anywhere.
Contrast is already computed by ``metrics.compute_stats`` ("contrast": p95 − p5), so this
costs nothing.

``metrics.is_black`` deliberately keeps its exact-zero semantics — it is the abort
condition 29bbae6 shipped and the sentence the artist reads names distributed rendering.
Near-black is a SEPARATE, differently-worded reason so the transcript never conflates
"the renderer returned nothing" with "the renderer returned almost nothing".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from . import metrics

#: NEAR BLACK. p95 at or under 2/255 AND the whole 5th-95th percentile span inside one
#: 8-bit step AND no hot pixel anywhere. Measured against this repo's own archetypes:
#: the dusk/night states in core/scenarios.py render with practicals and a sky gradient,
#: which is several quantisation steps of span; a plate a dead renderer handed back is
#: flat to the bit. Both terms are required — either alone false-positives.
NEAR_BLACK_P95 = 2.0 / 255.0
NEAR_BLACK_CONTRAST = 1.0 / 255.0

#: Consecutive pixel-identical plates (with the state CHANGING between them) before the
#: run says so. Not 1: a dusk scene with the sun below the horizon legitimately renders
#: identical frames across several coarse azimuths, and a wall of warnings is the same
#: unreadable transcript this whole exercise exists to fix. Three in a row while the
#: azimuth moved 90° is not a coincidence — it is a sun that is not reaching the frame.
FROZEN_WARN_RUN = 3
#: …and this many is no longer a diagnosis, it is a decision for the artist (gate 4).
FROZEN_ESCALATE_RUN = 6


@dataclass(frozen=True)
class PlateVerdict:
    ok: bool
    #: "" when ok; else "black" | "near_black" | "frozen" | "wrong_size"
    reason: str = ""
    #: the artist-facing sentence, house voice — never a bare code
    detail: str = ""
    #: this plate's fingerprint, for the NEXT frozen test. Always populated (even on a
    #: reject) so the caller can carry it forward without a second code path.
    signature: Tuple = ()
    #: how many consecutive frozen plates this one makes, counting itself
    frozen_run: int = 0
    #: True only on the transition INTO a reportable frozen run — so the caller logs once
    frozen_report: bool = False


def signature(stats: Optional[Dict]) -> Tuple:
    """A fingerprint of the PICTURE, drawn from stats the critic already paid for.

    Deliberately not a hash of the file bytes: two renders of the same state can differ in
    PNG metadata (timestamps, chunk ordering) while being the same picture, and hashing
    bytes would then miss every real freeze. ``log_key`` is the frame's overall level,
    ``grid_log`` is where the light lives, ``hot_frac`` is how much directional light there
    is — any real change to the lighting moves at least one of them, and none of them costs
    anything extra to read.
    """
    if not isinstance(stats, dict):
        return ()
    try:
        key = round(float(stats.get("log_key") or 0.0), 6)
        grid = tuple(round(float(v), 5) for v in (stats.get("grid_log") or ()))
        hot = round(float(stats.get("hot_frac") or 0.0), 5)
    except (TypeError, ValueError):
        return ()
    return (key, grid, hot)


def _near_black(stats: Dict) -> bool:
    try:
        p95 = float((stats.get("p") or {}).get("95", 1.0))
        contrast = float(stats.get("contrast", 1.0))
        hot = float(stats.get("hot_frac") or 0.0)
    except (TypeError, ValueError):     # a partial/hand-made stats dict is not evidence
        return False
    return p95 <= NEAR_BLACK_P95 and contrast <= NEAR_BLACK_CONTRAST and hot <= 0.0


def validate(
    stats: Optional[Dict],
    *,
    want: Optional[Tuple[int, int]] = None,
    got: Optional[Tuple[int, int]] = None,
    prev_sig: Tuple = (),
    state_changed: bool = False,
    frozen_run: int = 0,
    changed_axes: Sequence[str] = (),
) -> PlateVerdict:
    """Judge one rendered plate. → a ``PlateVerdict``; never raises.

    ``want`` is the size the caller ASKED for and ``got`` what ``read_png_size`` found —
    ``got is None`` means "could not verify" (a non-PNG, a fixture that wrote bytes) and
    is never a rejection. ``state_changed`` gates the frozen test: V-Ray renders are
    deterministic in the state (director.py records "the same state re-rendered scores
    100.0 against itself"), so two identical plates are only suspicious when the applied
    state moved between them. ``frozen_run`` is the caller's running count.
    """
    sig = signature(stats)
    if stats is None:
        # an unmeasurable plate is not evidence either way — the caller already treats a
        # None stats as "skip", and saying "ok" here would be a lie about a non-measurement
        return PlateVerdict(ok=False, reason="unmeasured",
                            detail="the plate could not be measured — no stats engine "
                                   "could read it, so it was not scored",
                            signature=sig)

    if want and got and (int(got[0]), int(got[1])) != (int(want[0]), int(want[1])):
        # render_frame has THREE layered size spellings and the third mutates
        # rt.renderWidth/Height globally (render.py:57-73). A build where all three fall
        # through saves a perfectly valid frame at SCENE resolution; compute_stats
        # downsamples to 256 px so the numbers look normal. Nothing checked this before.
        return PlateVerdict(
            ok=False, reason="wrong_size",
            detail=(f"the renderer returned a {got[0]}×{got[1]} frame when "
                    f"{want[0]}×{want[1]} was asked for — the render size did not take "
                    f"on this build, so this plate is not comparable with the others"),
            signature=sig)

    if metrics.is_black(stats):
        return PlateVerdict(
            ok=False, reason="black",
            detail="the plate is 100% BLACK — every sampled pixel is exactly zero",
            signature=sig)

    if _near_black(stats):
        return PlateVerdict(
            ok=False, reason="near_black",
            detail=("the plate is near-black AND FLAT (p95 "
                    f"{float((stats.get('p') or {}).get('95', 0.0)) * 255.0:.1f}/255, "
                    f"span {float(stats.get('contrast', 0.0)) * 255.0:.1f}/255, no hot "
                    "pixel anywhere) — a legitimately dark render still carries "
                    "practicals, a sky gradient and sampler noise; this carries nothing"),
            signature=sig)

    if state_changed and sig and prev_sig and sig == prev_sig:
        run = int(frozen_run) + 1
        moved = ", ".join(str(a) for a in changed_axes) or "the applied state"
        report = run in (FROZEN_WARN_RUN, FROZEN_ESCALATE_RUN)
        # NOT a rejection. A frozen plate is a real measurement of a real render — it is
        # the SEARCH that is broken, not the picture. Rejecting it would abort a run that
        # a dusk scene can legitimately produce; reporting it names the cause.
        return PlateVerdict(
            ok=True, reason="frozen", signature=sig, frozen_run=run,
            frozen_report=report,
            detail=(f"{run} probes in a row rendered PIXEL-IDENTICAL while {moved} "
                    "changed. The light MaxGaffer is steering is not reaching this "
                    "frame. Likeliest: the sun it is driving is switched OFF, or it is "
                    "an untargeted VRaySun, which aims by node rotation so azimuth "
                    "writes do not re-aim it. Every remaining probe will rank the same "
                    "picture."))

    return PlateVerdict(ok=True, signature=sig, frozen_run=0)
