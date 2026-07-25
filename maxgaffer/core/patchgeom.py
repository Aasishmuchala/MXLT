"""Sun DIRECTION from where the patches landed — the free half of the direction problem.

The question this module answers: *which side of the frame is the light coming from, and
does the render agree with the reference?*

Sun direction is this plugin's weakest link. ANALYZE reads it off the reference with an LLM
and is measurably unreliable — four runs over ONE golden-hour reference returned bearings of
45.0, -52.5, 77.6 and 64.9 degrees, and on A2 the sign came back inverted, which put the sun
on the wrong side of the room. `sunsolve` answers the same question properly, but pays ~50
renders to do it.

Meanwhile part of the answer is already sitting in stats that every probe computes.
`metrics.compute_stats` reports `hot_grid`: WHERE in the frame the locally-contrasty bright
patches are. A patch's POSITION is geometric evidence about direction — light travels away
from the sun, so the shaft entering a window lands displaced across the frame, opposite to
the sun's lateral direction. Nothing was reading that.

So what this module produces is a PRIOR and a CROSS-CHECK, never a measurement: zero
renders, zero IO, pure stdlib. It is worth having because it is free, and because it catches
the failure that actually happened — a bearing whose SIGN is wrong puts the light on the
other side of the frame, and that is the one thing a patch map states plainly.

What it does NOT know, said up front, because the sign is the only thing defended here:

  * nothing about the room. Window position, room depth, camera field of view and sun
    altitude all set HOW FAR a patch slides per degree of rotation, and none of them are
    visible in a 5x5 map. Magnitudes here are capped, deliberately conservative nudges.
  * the reference is usually a photograph of a DIFFERENT building, so its patch position
    encodes ITS aperture, not ours. What the two frames genuinely share is the direction
    light travels across the frame; what they do not share is where the window sits, which
    adds noise. A reference room windowed on the opposite side to ours is a real,
    unfixable failure mode for this evidence.
  * exteriors invert the rule — there the bright patches are sun-FACING facades, sitting on
    the SAME side as the sun instead of displaced away from it — so `bearing_hint` declines
    them rather than guessing which population won.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

#: `hot_grid` is metrics.compute_stats' 5x5 patch map, row-major with row 0 at the TOP of
#: frame (it fills `hot_cells[row * 5 + col]`), normalised to sum to 1. Mass therefore lands
#: on cell centres, at (col + 0.5) / 5 across and (row + 0.5) / 5 down.
GRID = 5

#: The widest lateral disagreement the map can express: mass in the leftmost column against
#: mass in the rightmost puts two centroids 0.8 of a frame apart, because a cell centre can
#: never sit closer to an edge than half a cell.
MAX_CENTROID_SEPARATION = (GRID - 1) / float(GRID)

#: Below this share of frame the map is a handful of speculars rather than a shaft, and its
#: centroid moves with noise. Real directional light is whole percents of the frame:
#: measured on-box 2026-07-25, references carried 17.3% and 4.1% of frame in genuine patches
#: while the failing matches carried 5.8% and 1.4%. 0.2% at the stats working resolution
#: (max_dim 256) is roughly an 11x11 pixel blob — the floor, not a typical case.
MIN_HOT_FRAC = 0.002
#: ...and at this much the patch is unmistakably a shaft, so presence stops capping
#: confidence. Same measured scale: the honest references sat well above it.
CONFIDENT_HOT_FRAC = 0.02

#: Lateral spread (see `patch_spread`) above which an x-centroid is the middle of a smear
#: rather than the position of anything. Mass spread evenly across all five columns scores
#: 0.283, so this refuses a full-width smear while still accepting the two-blob
#: window-plus-floor-patch pattern a real shaft produces (half the mass at x=0.3 and half at
#: x=0.7 scores 0.20).
MAX_LATERAL_SPREAD = 0.26

#: Centroid differences under half a cell are below the map's own resolution. Reporting a
#: rotation for them would defend precision the 5x5 grid does not have — the same mistake
#: transfer.score fixed by widening its tolerance to the sweep's sampling step.
LATERAL_DEADBAND = 0.10

#: The largest rotation this hint will ever ask for. Deliberately smaller than the geometry
#: could imply: the loop iterates, so an under-shooting nudge with the right sign converges,
#: while an over-shooting one oscillates and can shove the sun through the +-90 band where
#: this model's sign flips. It also sits just above sunsolve's 30-degree coarse step, so a
#: hint can name a different coarse cell without leaping across the sky.
MAX_HINT_DEG = 40.0

#: Within this many degrees of pure side-on (bearing +-90) the patch's lateral offset is at
#: its extremum: the derivative passes through zero, bearings either side of it produce the
#: same lateral position, and the map genuinely cannot say which way to turn.
AMBIGUOUS_HALF_WIDTH_DEG = 15.0

#: Confidence ceiling. This is a prior from an uncalibrated model read across two different
#: rooms; it must never present itself as a measurement. Kept strictly under the 0.5 at
#: which the controller accepts sunsolve's RENDERED answer as decisive, so a free hint can
#: never outrank a paid solve.
MAX_CONFIDENCE = 0.45

#: Scene types whose bright patches are light CAST across the room, which is the transport
#: geometry `bearing_hint` models. Anything else (an exterior, an unset field) is declined.
TRANSPORT_SCENES = ("interior", "interior_with_view")


# --------------------------------------------------------------------------- validation
def _finite(value) -> Optional[float]:
    """A real number, or None. Junk — strings, None, NaN, inf — never reaches the geometry,
    so every entry point degrades instead of raising."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _weights(stats: Optional[Dict]) -> Optional[List[float]]:
    """Validated `hot_grid`: exactly 25 finite non-negative weights that sum above zero.

    Guards BEFORE any arithmetic, exactly as metrics._illum_vec does, so a degenerate map
    can never leak a fabricated position downstream. A negative entry is corruption rather
    than a signal — the grid is a normalised distribution — so the whole map is refused
    instead of being repaired into something that looks measured."""
    if not isinstance(stats, dict):
        return None
    grid = stats.get("hot_grid")
    if not isinstance(grid, (list, tuple)) or len(grid) != GRID * GRID:
        return None                     # legacy stats predate the map, or this is junk
    out: List[float] = []
    for cell in grid:
        weight = _finite(cell)
        if weight is None or weight < 0.0:
            return None
        out.append(weight)
    return out if sum(out) > 1e-9 else None      # all zero: no patches to have a position


def _centroid(weights: List[float]) -> Tuple[float, float]:
    total = sum(weights)
    cx = sum(w * ((i % GRID) + 0.5) / GRID for i, w in enumerate(weights)) / total
    cy = sum(w * ((i // GRID) + 0.5) / GRID for i, w in enumerate(weights)) / total
    return cx, cy


# --------------------------------------------------------------------------- description
def patch_centroid(stats: Optional[Dict]) -> Optional[Tuple[float, float]]:
    """Where the sun patches sit, as a centre of mass in normalised frame coordinates.

    → (x, y), x 0..1 left-to-right and y 0..1 TOP-to-bottom (image order, which is the order
    `hot_grid` itself is stored in — the y axis points DOWN and getting that backwards
    silently mirrors every floor patch into the ceiling). None when there are no patches, so
    there is no centre to report.

    Quantised by construction: mass lands on cell centres, so both coordinates only ever
    fall in 0.1..0.9 and any difference under 0.2 is inside a single cell."""
    weights = _weights(stats)
    if weights is None:
        return None
    cx, cy = _centroid(weights)
    return round(cx, 4), round(cy, 4)


def patch_spread(stats: Optional[Dict], axis: str = "radial") -> Optional[float]:
    """How concentrated the patches are: weight-RMS distance from the centroid, in frames.

    One shaft of light scores low, a scatter of small speculars scores high — which is the
    difference between evidence and noise for everything below. `axis` selects "radial"
    (both), "x" (lateral only) or "y".

    Landmarks, all exact for 5x5 cell centres. Radial: 0.0 a single cell, 0.10 two adjacent
    cells, 0.40 mass spread evenly over the whole frame, 0.566 the maximum (split between
    opposite corners). Lateral: 0.283 spread evenly across the width, 0.40 split between the
    outer columns.

    The axes are separable on purpose. A window and the floor patch it throws are two blobs
    in the SAME column — perfectly localised as a DIRECTION, but a large radial spread.
    Judging that pattern "scattered" would refuse the clearest evidence this module ever
    gets, so direction work reads the x axis alone."""
    weights = _weights(stats)
    if weights is None or axis not in ("radial", "x", "y"):
        return None
    cx, cy = _centroid(weights)
    total = sum(weights)
    var = 0.0
    for i, w in enumerate(weights):
        dx = ((i % GRID) + 0.5) / GRID - cx
        dy = ((i // GRID) + 0.5) / GRID - cy
        if axis == "x":
            var += w * dx * dx
        elif axis == "y":
            var += w * dy * dy
        else:
            var += w * (dx * dx + dy * dy)
    return round(math.sqrt(var / total), 4)


def _lateral_evidence(stats: Optional[Dict]) -> Optional[Tuple[float, float, float]]:
    """(lateral centroid, lateral spread, hot fraction) when this frame's patch map can be
    used as DIRECTION EVIDENCE — else None.

    Stricter than `patch_centroid` deliberately. The centroid describes whatever map it is
    handed; an inference has to first believe the map means something, so this also demands
    that `hot_frac` exists and clears MIN_HOT_FRAC. sunsolve gates on the same key for the
    same reason: stats that predate the patch map, or that carry a few stray specular
    pixels, are not a shaft of light."""
    weights = _weights(stats)
    if weights is None:
        return None
    frac = _finite(stats.get("hot_frac"))       # _weights already proved stats is a dict
    if frac is None or frac < MIN_HOT_FRAC:
        return None
    cx, _cy = _centroid(weights)
    total = sum(weights)
    var = sum(w * (((i % GRID) + 0.5) / GRID - cx) ** 2
              for i, w in enumerate(weights)) / total
    return cx, math.sqrt(var), frac


# --------------------------------------------------------------------------- inference
def bearing_agreement(ref_stats: Optional[Dict],
                      cur_stats: Optional[Dict]) -> Optional[float]:
    """Do the reference and the render put their light on the SAME SIDE of frame? → 0..1.

    The cheap, model-free half of the direction question: no geometry, no assumption about
    interiors, just the two lateral centroids compared. 1.0 is the same column; 0.0 is the
    widest disagreement a 5x5 map can express (outermost column against outermost,
    MAX_CENTROID_SEPARATION apart) — which is exactly what a sign-flipped bearing looks
    like, the error ANALYZE produced on A2.

    None — not 0.0 — when either side has no usable map. `metrics.highlight_similarity`
    returns 0.0 there and is right to: it is SCORED, and a candidate must not be able to
    delete a criterion by switching its sun off. This is not scored, so nothing can be
    gained by admitting ignorance, and a fabricated 0.0 would read as "they point opposite
    ways" — a claim the evidence never made."""
    ref = _lateral_evidence(ref_stats)
    cur = _lateral_evidence(cur_stats)
    if ref is None or cur is None:
        return None
    return round(max(0.0, 1.0 - abs(ref[0] - cur[0]) / MAX_CENTROID_SEPARATION), 4)


def bearing_hint(ref_stats: Optional[Dict], cur_stats: Optional[Dict],
                 current_azimuth_deg: float, camera_yaw_deg: float = 0.0,
                 scene_type: str = "interior") -> Optional[Dict]:
    """Which way to rotate the sun so the render's patches move toward the reference's.

    → ``{"delta_deg", "confidence", "bearing_deg", "lateral_shift", "why"}``, or None when
    the evidence is too weak to sign an answer.

    SIGN CONVENTION. ``delta_deg`` is ADDED to ``sun.azimuth_deg``; because the
    camera-relative bearing is ``azimuth - camera_yaw`` (transfer.score, domeseed), the
    identical delta applies to ``sun_bearing_deg``. POSITIVE rotates the sun toward
    camera-RIGHT — the +90 end of the ANALYZE convention in prompts.py.

    THE MODEL, in full, because it is small enough to state and you should trust exactly
    this much of it. Light travels away from the sun, so in a room lit through an aperture
    the patch is displaced from that aperture along the propagation direction, whose lateral
    component in camera space is ``-sin(bearing)``: a sun camera-RIGHT throws its shaft
    camera-LEFT. Differentiate and the patch's lateral motion per degree of rotation is
    ``-cos(bearing)``. So with the sun anywhere in FRONT of the camera, moving the patch
    RIGHT means rotating the sun LEFT; with the sun BEHIND the camera the sign inverts,
    which is why the current azimuth is required rather than optional. That is the whole
    physical content: a sign, monotone across each half-turn.

    What is NOT modelled and therefore not claimed: window position, room depth, field of
    view and sun altitude all set how far the patch slides per degree, and none of them are
    visible here. The magnitude is a capped conservative nudge (MAX_HINT_DEG) — use it to
    MOVE, not to arrive. This is a prior for, or a cross-check on, `sunsolve`, which
    measures the same quantity properly for the price of ~50 renders.

    Declines (None) rather than guessing when:
      * either frame has no usable patch map (see `_lateral_evidence`);
      * the patches are too laterally scattered to localise (MAX_LATERAL_SPREAD);
      * the sun is within AMBIGUOUS_HALF_WIDTH_DEG of side-on, where the lateral offset is
        stationary and the bearings either side of it look identical;
      * ``scene_type`` is not an interior — on an exterior the bright patches are sun-FACING
        facades, which sit on the SAME side as the sun rather than displaced away from it,
        so the rule inverts, and mixed in with sunlit glass and sky there is nothing here
        that says which population won.

    A ZERO delta with a live confidence is a result, not a refusal: it says the two frames
    already put their light on the same side, to the resolution a 5x5 map has."""
    ref = _lateral_evidence(ref_stats)
    cur = _lateral_evidence(cur_stats)
    if ref is None or cur is None:
        return None
    if str(scene_type or "").strip().lower() not in TRANSPORT_SCENES:
        return None
    azimuth = _finite(current_azimuth_deg)
    yaw = _finite(camera_yaw_deg)
    if azimuth is None or yaw is None:
        return None
    if max(ref[1], cur[1]) > MAX_LATERAL_SPREAD:
        return None                     # the centroid is the middle of a smear, not a place
    bearing = (azimuth - yaw + 180.0) % 360.0 - 180.0
    sensitivity = math.cos(math.radians(bearing))       # -d(lateral offset)/d(bearing)
    if abs(abs(bearing) - 90.0) < AMBIGUOUS_HALF_WIDTH_DEG:
        return None                     # side-on: the offset is at its extremum, no sign
    shift = ref[0] - cur[0]             # positive: the render's light must move RIGHT
    # every factor is 0..1 and the WORST one wins, the same posture consensus.py takes with
    # its headline agreement — a confident sub-measure must not paper over a weak one
    factors = [min(1.0, min(ref[2], cur[2]) / CONFIDENT_HOT_FRAC),
               1.0 - max(ref[1], cur[1]) / MAX_LATERAL_SPREAD,
               abs(sensitivity)]
    if abs(shift) < LATERAL_DEADBAND:
        delta = 0.0
        why = ("both frames already put their light on the same side, within half a cell "
               "of the patch map")
    else:
        # confidence rises with the SIZE of the disagreement over the whole expressible
        # range, not just past the deadband: a patch that has crossed the frame is a claim
        # about which side of the room the light enters, and that claim survives the fact
        # that the reference's window is somewhere else. A one-cell nudge does not — it is
        # the same order as the noise two different rooms put into this comparison.
        factors.append(min(1.0, abs(shift) / MAX_CENTROID_SEPARATION))
        # opposite to the shift because a patch moves against the sun, and inverted again
        # when the sun is behind the camera, where the lateral motion per degree changed
        # sign along with cos(bearing)
        delta = (-MAX_HINT_DEG * min(1.0, abs(shift) / MAX_CENTROID_SEPARATION)
                 * math.copysign(1.0, shift) * math.copysign(1.0, sensitivity))
        why = ("the reference's light sits %.2f of a frame to the %s of the render's; at "
               "bearing %+.0f° that asks for %+.0f°"
               % (abs(shift), "right" if shift > 0 else "left", bearing, delta))
    return {"delta_deg": round(delta, 1),
            "confidence": round(MAX_CONFIDENCE * max(0.0, min(factors)), 3),
            "bearing_deg": round(bearing, 1),
            "lateral_shift": round(shift, 4),
            "why": why}
