"""Shadow-edge sharpness and haze density — MEASURED off the reference, not guessed at.

The two questions this module answers: *how hard is the light that cast these shadows*, and
*how much air is between the camera and the scene?*

Both are read today by asking a vision model for `light_quality` ("hard" | "soft" | "mixed")
and `atmosphere` ("none" | "light_haze" | "heavy_haze" | "fog"), and both drive real
controls — `sun.size` through transfer.HARDNESS_SIZE and `atmosphere.distance_m` through
transfer.ATMOSPHERE_DISTANCE_M. Measured on-box 2026-07-26 on the cross-domain run, the
hardness reading collapsed to 0.0 agreement: the model is being asked to eyeball a physical
quantity that is sitting in the pixels.

It IS in the pixels, and neither measurement needs a network:

  HARDNESS is the width of the penumbra. A point-like source (the sun, 0.53° across) puts a
  lit/shadow transition inside a pixel or two; a broad source (overcast sky, a big window)
  smears it over tens of pixels. So hardness is the reciprocal of the transition WIDTH at the
  boundaries between lit and shadowed regions — and width, in turn, is the level difference
  across the boundary divided by the peak gradient at it, which is the standard slope
  estimate of edge width and is invariant to how bright the scene is.

  HAZE lifts the darkest part of every patch. The dark-channel prior (He, Sun & Tang 2009):
  in a haze-free outdoor patch, at least one pixel is very dark in at least one colour
  channel — a shadow, a dark surface, a saturated colour. Airlight adds a floor to every
  channel at once, so the minimum-over-window-and-channel rises with the density of the air.

Pure stdlib, sans-IO, O(n) in pixels. Both entry points degrade to None and never raise:
they run beside the loop's own stats, which can never fail.

SCALE IS PART OF THE MEASUREMENT. An edge width in pixels only means something at a fixed
working resolution, so both functions load through `metrics._load_pixels` at WORK_MAX_DIM,
the same 256 `compute_stats` uses. Handing in pre-loaded pixels from anywhere else is
supported and is the caller's promise that they came from that loader at that size.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import metrics

#: The one resolution these numbers mean anything at — `metrics.compute_stats`' own default.
#: A penumbra 3% of the frame wide is 8 px here and 60 px at 2K; report either as "8 pixels"
#: and the label flips. Every threshold below is quoted in pixels AT THIS SIZE.
WORK_MAX_DIM = 256


# --------------------------------------------------------------------------- input
def _frame(source) -> Optional[Tuple[Sequence[Tuple[int, int, int]], int, int]]:
    """(pixels, w, h) from a path or from an already-loaded triple — or None.

    Accepting both is not politeness: `compute_stats` has already paid for the decode on
    every polish probe, and making a caller decode the same PNG twice to add two more
    numbers is how cheap measurements stop being cheap."""
    if isinstance(source, str):
        return metrics._load_pixels(source, max_dim=WORK_MAX_DIM)
    if isinstance(source, (tuple, list)) and len(source) == 3:
        pixels, w, h = source
        try:
            w, h = int(w), int(h)
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0 or not pixels or len(pixels) < w * h:
            return None
        return pixels, w, h
    return None


# ======================================================================= edge hardness
#: A central difference over ±1 px measures a perfect step as a slope of Δ/2, so it cannot
#: report a width under 2 px however sharp the edge really is. That floor IS hardness 1.0 —
#: claiming to resolve finer would be claiming precision the operator does not have.
EDGE_OPERATOR_FLOOR_PX = 2.0

#: ...and the width at which an edge is unambiguously a broad-source penumbra: hardness 0.0.
#: 16 px of 256 is 6% of the frame. Physically the split is wide, not marginal: sun geometry
#: puts a penumbra at ~0.93% of the occluder-to-receiver distance (a body 2 m above its own
#: shadow → 1.9 cm, well under one pixel in any architectural frame), while an overcast sky
#: or a 1 m window at the same 2 m gives a penumbra of the order of a metre — tens of pixels.
#: Nothing physical lives at 16 px except a source of intermediate size, which is "mixed".
WIDTH_SOFT_PX = 16.0

#: Between the two the map is in OCTAVES of width, matching how transfer.py and
#: highlight_similarity already score multiplicative quantities. 3 octaves span 2→16 px.
_WIDTH_OCTAVES = math.log(WIDTH_SOFT_PX / EDGE_OPERATOR_FLOOR_PX, 2.0)

#: Label bands, CALIBRATED against generated ground truth. One scene was rendered five times
#: in 3ds Max with nothing varying but V-Ray's sun size, giving a monotonic truth series
#: rather than a handful of anecdotes (2026-07-26):
#:      sun.size  1 → 0.984      sun.size 16 → 0.152
#:      sun.size  3 → 0.776      sun.size 30 → 0.148
#:      sun.size  8 → 0.356
#: Sizes 1 and 3 are unambiguously hard, 16 and 30 unambiguously soft, and 8 is genuinely in
#: between. The thresholds are the midpoints of the two gaps that separate those groups —
#: (0.776 + 0.356)/2 and (0.356 + 0.152)/2 — so each band is centred on measured truth
#: rather than on a guess about pixel widths.
HARD_LABEL = 0.57
SOFT_LABEL = 0.25

#: Where the plateaus are sampled, either side of the boundary, along the gradient normal.
#: The NEAR pair sets the level difference Δ; the FAR pair only has to agree that the
#: difference is still there. PLATEAU_NEAR_PX is deliberately tied to WIDTH_SOFT_PX / 2: a
#: transition wider than the near probes cannot be fully spanned, so its measured Δ shrinks
#: in exact proportion and the width estimate SATURATES at WIDTH_SOFT_PX instead of lying.
PLATEAU_NEAR_PX = 8
PLATEAU_FAR_PX = 16
PLATEAU_RADIUS = 2                       # 5×5 patch means, so one noisy pixel cannot speak

#: Boundary samples are only taken this far inside the frame — the far probe must fit.
EDGE_MARGIN_PX = PLATEAU_FAR_PX + PLATEAU_RADIUS

#: A gradient below one 8-bit code per pixel is quantisation, not an edge.
MIN_ABS_GRAD = 1.0 / 255.0

#: Cost gate, not a filter: only the top fifth of pixels by gradient are worth testing.
#: Kept generous ON PURPOSE. A tight gradient threshold would admit hard edges and reject
#: soft ones by construction — the exact bias this module exists to avoid — so the gate is
#: loose and the REJECTION is done by the sustain/ratio/chroma tests below, none of which
#: care how steep the edge is.
GRAD_KEEP_FRACTION = 0.20
_GRAD_BINS = 256

#: The lit side must be at least this much brighter than the shaded side, in display-referred
#: sRGB. 1.35 there is about a 2:1 ratio in linear light — the shallowest thing that is
#: honestly a shadow rather than shading. Filled interior shadows sit near it; a sunlit/shade
#: boundary is 3:1 to 8:1 and clears it easily.
MIN_STEP_RATIO = 1.35
#: ...and an absolute floor, so a 2:1 ratio between two nearly-black patches is not a
#: "boundary" measured on eight code values of noise.
MIN_STEP = 0.02

#: THE TEXTURE TEST. The level difference measured across ±PLATEAU_FAR_PX must keep this
#: much of the difference measured across ±PLATEAU_NEAR_PX, with the same sign. Texture is
#: zero-mean at small scale — averaging over a wider baseline collapses it toward nothing.
#: A real lit/shadow boundary is a DC level change between two regions, so it survives
#: averaging at every scale, and a soft ramp survives it too (its far step is if anything
#: LARGER), which is why the test is a sustain check and not a flatness check: demanding
#: flat plateaus would have thrown away precisely the soft edges.
SUSTAIN_FRAC = 0.6

#: THE MATERIAL TEST — a shared SHADOW RATIO, which is an exact invariant rather than a
#: tolerance on colours. Across a true shadow boundary the surface is the same and only the
#: illumination changes:
#:      shadow = albedo · a                (ambient alone)
#:      lit    = albedo · (a + s)          (ambient plus sun)
#: so the per-channel ratio is (a + s)/a = 1 + s/a and THE ALBEDO CANCELS EXACTLY. A material
#: boundary is albedo₁/albedo₂, which is arbitrary; foliage against sky is arbitrary. Every
#: genuine shadow edge in one image therefore agrees on one direction, and nothing else does.
#:
#: The quantity clustered is `ratio − 1`, NOT the ratio. This matters and is not a detail.
#: Surfaces sit in different amounts of ambient occlusion, so the ambient at one shadow edge
#: is k times the ambient at another. Then ratio = 1 + s/(k·a), which is a DIFFERENT vector
#: for every k — the ratios of genuine shadow edges do not share a direction, they lie on a
#: ray anchored at (1,1,1). Subtract the 1 and that ray becomes a direction: (ratio − 1) =
#: s/(k·a), whose direction is s/a for every k. So `ratio − 1` is invariant to how deep the
#: shade is, and the raw ratio is not. Measured both ways on the plates (2026-07-26) the
#: difference is decisive and is recorded in `edge_hardness`.
#:
#: What survives is the SUN-TO-AMBIENT SPECTRAL RATIO, reported as `ratio_rgb`.
#: This replaces a normalised-chromaticity tolerance that was measured to be unfixable: the
#: bands overlapped (a wood/plaster edge at 0.258 was LESS chromatic than a genuine 3200 K
#: shadow edge at 0.301) and foliage against pale sky sat at 0.095, a quarter of any usable
#: tolerance. No threshold on colour distance could separate those; a consensus on the ratio
#: does not have to, because it asks a different question — not "are these two colours
#: similar" but "do all these boundaries agree on one illuminant change".

#: Half-angle of the inlier cone, in degrees, about the consensus direction. Real scenes are
#: not perfectly bi-illuminant — bounce off a coloured wall genuinely changes the local
#: ambient SPECTRUM, which moves the direction — so this is loose enough to hold one scene's
#: shadows together and tight enough that an arbitrary albedo ratio has to be lucky to land
#: inside it. A cone of this half-angle covers ~1.4% of the sphere of unit directions.
RATIO_INLIER_DEG = 12.0

#: How many ratio clusters to enumerate before giving up on finding a sun population. The
#: search stops early the moment a red-weighted one appears — clusters come out largest
#: first, so the first warm one IS the largest warm one — so this bound is only ever paid in
#: full by a frame that has no sun in it, which is exactly the frame that should pay for
#: being sure.
MAX_CLUSTERS = 6

#: THE SHARE OF THE FRAME IS NOT EVIDENCE, and this is deliberately not a gate.
#:
#: An earlier version required the chosen cluster to hold a minimum share of candidates. Two
#: measurements killed it. First, share does not separate signal from noise: eight random
#: albedo patchworks with no shadow anywhere in them reached shares of 0.41, 0.45, 0.50 and
#: 0.73 (2026-07-26) — that last one HIGHER than any of the three real plates, which sit at
#: 0.52, 0.52 and 0.68. Second, and decisively, the share of the RIGHT answer is small: on
#: the golden-hour interior the actual sun-shadow cluster is 10% of candidates while 53% is
#: ambient occlusion. A floor on share would have rejected the one correct population in the
#: frame for being outnumbered by the wrong one. What gates instead is an ABSOLUTE sample
#: count, because a hundred boundaries are a hundred boundaries however much else is going on.

#: Hypotheses tested in the consensus. Every sample's own direction is a candidate axis, but
#: scoring all of them is O(k²) and k runs to ~1500 on a detailed plate. An evenly-strided
#: subset is enough: missing the true cluster entirely requires every one of these to be an
#: outlier, which at even a 20% inlier fraction is (0.8)^96 ≈ 4e-10. Strided rather than
#: random so the answer is reproducible — the loop must not return a different rig twice.
CONSENSUS_HYPOTHESES = 96

#: Sun light is ADDITIVE, so the lit side cannot be darker than the shadow side in ANY
#: channel. A material edge routinely is (red brick against grey mortar is brighter in red
#: and darker in blue). Allowed a small negative slack for 8-bit noise, as a fraction of the
#: strongest channel's lift.
RATIO_NEG_SLACK = 0.06

#: Linear-light floor for the shadow side. Ratios taken against a near-black window are
#: numerically meaningless — the denominator is quantisation.
MIN_SHADOW_LINEAR = 0.002

#: WHICH CLUSTER IS THE SUN'S. A sun-vs-ambient ratio is warm by construction — the key is a
#: 2000–6500 K sun and the fill is cooler skylight — so s/a is red-weighted in every daylight
#: scene, while an albedo ratio has no reason to be. The threshold is r ≥ b in the normalised
#: direction, i.e. a margin of exactly 1.0, and it is a physical dividing line rather than a
#: fitted one. Pushing the repo's own `colortemp.kelvin_to_rgb` across the whole plausible
#: daylight range (2026-07-26) puts every sun/sky pairing above it, with the LEAST warm case
#: — a 6500 K sun against a 7000 K sky, which is daylight barely distinguishable from its own
#: fill — at r/b = 1.07, and the warmest (2000 K against 20000 K) at 27.4. Measured material
#: pairs sit far below: grey/brick 0.10, plaster/wood 0.19, sky/foliage 0.62. Nothing here is
#: tuned to an image; 1.0 is simply where "the ratio is warm" stops being true.
#:
#: KNOWN FALSE NEGATIVE, and it is a real one: an interior whose ambient is warm BOUNCE
#: rather than blue sky inverts this. A 4500 K sun against 3000 K bounce gives r/b = 0.59, so
#: its sun cluster reads cool and is declined. Measured across that family the ratio runs
#: 0.39–0.89 — always below the line. Such a frame returns None (no sun found) rather than a
#: wrong hardness, which is the safe direction but is still a miss.
SUN_WARMTH_MIN = 1.0

#: Warmth margin at which the choice is unambiguous, for confidence. r/b of 1.5 is a clearly
#: warm key against a cool fill; 1.07 is the edge of the daylight range and must be held
#: loosely, because at that point sun and sky are nearly the same colour and the two
#: populations are not really separable.
CONFIDENT_WARMTH = 1.5

#: sRGB→linear as a 256-entry table. The decode is a pow() per channel per pixel otherwise,
#: and there are only 256 possible 8-bit inputs, so the whole transfer function is a lookup.
#: Scaled to integers so the window sums stay exact, like the summed-area tables.
_LIN_SCALE = 65535
_LIN_LUT = tuple(
    int(round((v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4) * _LIN_SCALE))
    for v in (i / 255.0 for i in range(256))
)

#: Fewer boundary samples than this and the median is an anecdote → None.
MIN_SAMPLES = 40
#: ...and this many is a settled median. Lowered from 400 when share stopped being a
#: confidence term: the quantity that matters is how many boundaries the CHOSEN cluster
#: holds, not how much of the frame it covers, and a hundred-odd boundary samples pin a
#: median perfectly well. Adjacent samples along one edge are not independent, so this is
#: counting evidence generously — hence a floor set well above the 40 that merely permits.
CONFIDENT_SAMPLES = 150

#: Both populations at least this large ⇒ the frame is BIMODAL: it carries a hard population
#: and a soft one rather than one typical boundary width. Reported as a diagnostic, and it
#: no longer overrides the label.
#:
#: It used to. The override was removed when the ladder arrived, on the ladder's evidence:
#: the calibrated, monotonic quantity is the MEDIAN, and the override let a population count
#: contradict it — the golden-hour reference measures a median of 0.947, as hard as anything
#: in the truth series, and was being labelled "mixed" because its boundaries split 54/46.
#: Not one of the five ladder rungs triggers the override, so it was never exercised by
#: ground truth while it was overruling the one quantity that was.
MIXED_MINORITY = 0.30


def _sat3(pixels, w: int, h: int) -> Tuple[List[int], List[int], List[int]]:
    """Summed-area tables for R, G and B — the same technique metrics' sun-patch map uses,
    for the same reason: every window mean below is then O(1), so the whole pass stays O(n)
    however wide the probe baseline gets. Integer sums, so they are exact and fast.

    Three channels rather than one luminance table because the plateau probes need the
    window's COLOUR as well as its brightness — that is the material test — and luminance
    is a fixed combination of the three, so one set of tables serves both."""
    stride = w + 1
    size = stride * (h + 1)
    sr = [0] * size
    sg = [0] * size
    sb = [0] * size
    for y in range(h):
        run_r = run_g = run_b = 0
        base = y * w
        above = y * stride
        here = (y + 1) * stride
        for x in range(w):
            r, g, b = pixels[base + x]
            run_r += r
            run_g += g
            run_b += b
            i = x + 1
            sr[here + i] = sr[above + i] + run_r
            sg[here + i] = sg[above + i] + run_g
            sb[here + i] = sb[above + i] + run_b
    return sr, sg, sb


def _window(sats, w: int, h: int, cx: int, cy: int,
            radius: int = PLATEAU_RADIUS) -> Optional[Tuple[float, float, float]]:
    """(mean luminance 0..1, r̄, b̄) over the square window at (cx, cy) — or None off-frame.

    Chromaticity is a ratio of SUMS, not a mean of per-pixel ratios: it is the patch's own
    colour, and it does not let one near-black pixel with a wild ratio speak for the patch."""
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    stride = w + 1
    br, tr_, bl, tl = (y1 * stride + x1, y0 * stride + x1,
                       y1 * stride + x0, y0 * stride + x0)
    sr, sg, sb = sats
    tot_r = sr[br] - sr[tr_] - sr[bl] + sr[tl]
    tot_g = sg[br] - sg[tr_] - sg[bl] + sg[tl]
    tot_b = sb[br] - sb[tr_] - sb[bl] + sb[tl]
    area = (x1 - x0) * (y1 - y0)
    lum = (0.2126 * tot_r + 0.7152 * tot_g + 0.0722 * tot_b) / (area * 255.0)
    total = tot_r + tot_g + tot_b
    if total <= 0:
        return lum, 1.0 / 3.0, 1.0 / 3.0     # a black patch has no chromaticity to compare
    return lum, tot_r / total, tot_b / total


def _linear_window(pixels, w: int, h: int, cx: int, cy: int,
                   radius: int = PLATEAU_RADIUS) -> Optional[Tuple[float, float, float]]:
    """Mean LINEAR-light R, G, B over the window at (cx, cy) — or None off-frame.

    Summed directly rather than from a summed-area table on purpose. Only accepted
    candidates ever need a linear reading, and there are a few hundred of those against
    65k pixels, so paying per candidate (25 lookups) beats building three more tables over
    the whole frame. A ratio of DISPLAY-referred values is not a ratio of light — the sRGB
    curve has to come off first or the invariant below is being computed on the wrong
    numbers entirely."""
    x0, x1 = cx - radius, cx + radius + 1
    y0, y1 = cy - radius, cy + radius + 1
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    tr = tg = tb = 0
    for yy in range(y0, y1):
        base = yy * w
        for xx in range(x0, x1):
            r, g, b = pixels[base + xx]
            tr += _LIN_LUT[r]
            tg += _LIN_LUT[g]
            tb += _LIN_LUT[b]
    area = (x1 - x0) * (y1 - y0) * float(_LIN_SCALE)
    return tr / area, tg / area, tb / area


def _shadow_direction(lit, shade) -> Optional[Tuple[float, float, float]]:
    """Unit direction of (lit / shade − 1) in linear RGB — the sun-to-ambient spectral
    ratio s/a — or None when this pair cannot be a shadow boundary.

    Declines when any channel of the shadow side is below MIN_SHADOW_LINEAR (the ratio would
    be quantisation noise) and when the lit side is meaningfully DARKER in some channel,
    which additive sunlight cannot do and an albedo change routinely does."""
    lift = []
    for i in range(3):
        if shade[i] < MIN_SHADOW_LINEAR:
            return None
        lift.append(lit[i] / shade[i] - 1.0)
    peak = max(lift)
    if peak <= 0.0:
        return None                          # the "lit" side is not brighter in any channel
    if min(lift) < -RATIO_NEG_SLACK * peak:
        return None                          # darker somewhere: an albedo change, not sun
    clamped = [v if v > 0.0 else 0.0 for v in lift]      # tiny negatives are noise
    norm = math.sqrt(sum(v * v for v in clamped))
    if norm <= 0.0:
        return None
    return clamped[0] / norm, clamped[1] / norm, clamped[2] / norm


def _is_warm(axis) -> bool:
    """Is this ratio direction a SUN-over-ambient one? See SUN_WARMTH_MIN."""
    return axis[0] >= SUN_WARMTH_MIN * axis[2]


def _one_cluster(directions, members, tol_cos):
    """Largest cone of ``tol_cos`` over the given member indices → (axis, [indices], mean
    cosine) or None.

    Strided-hypothesis vote: every sample's own direction is a candidate axis, each scored by
    how many samples fall inside a cone about it, and the most-populated wins. A spherical
    MEAN would be pulled off the cluster by the outliers this exists to reject — on the dawn
    plate the outliers are the majority — so the estimator has to be a vote, not an average.
    Two refinement passes re-centre the winner on the mean of its own inliers, recovering the
    precision a discrete hypothesis set gives up."""
    if not members:
        return None
    stride = max(1, len(members) // CONSENSUS_HYPOTHESES)
    best_axis, best_count = None, -1
    for hi in members[::stride]:
        ax, ay, az = directions[hi]
        count = 0
        for j in members:
            dx, dy, dz = directions[j]
            if dx * ax + dy * ay + dz * az >= tol_cos:
                count += 1
        if count > best_count:
            best_axis, best_count = directions[hi], count
    if best_axis is None:
        return None
    for _ in range(2):
        ax, ay, az = best_axis
        sx = sy = sz = 0.0
        for j in members:
            dx, dy, dz = directions[j]
            if dx * ax + dy * ay + dz * az >= tol_cos:
                sx += dx
                sy += dy
                sz += dz
        norm = math.sqrt(sx * sx + sy * sy + sz * sz)
        if norm <= 0.0:
            break
        best_axis = (sx / norm, sy / norm, sz / norm)
    ax, ay, az = best_axis
    kept, dot_sum = [], 0.0
    for j in members:
        dx, dy, dz = directions[j]
        dot = dx * ax + dy * ay + dz * az
        if dot >= tol_cos:
            kept.append(j)
            dot_sum += dot
    if not kept:
        return None
    return best_axis, kept, dot_sum / len(kept)


def _sun_cluster(directions: List[Tuple[float, float, float]],
                 tol_deg: float = RATIO_INLIER_DEG):
    """The frame's SUN-shadow population. → (axis, [indices], mean cosine, rank) or None.

    Clusters are peeled off largest-first and the first RED-WEIGHTED one wins — not the
    largest one. That distinction is the whole point of this function and it was learned the
    hard way: the invariant is symmetric. A shadow edge's ratio is albedo-independent, but a
    material edge's ratio is illumination-independent — the same two surfaces meeting in sun
    and in shade give the same albedo ratio both times — so a material pair that REPEATS
    across a frame forms a cluster every bit as tight as the shadows do, and taking the
    majority just takes whichever population is bigger. Measured 2026-07-26, the majority was
    never the sun's: 53% ambient occlusion against a 10% sun cluster on the golden-hour
    interior, and on the dawn landscape a 52% cluster that was 88% tree canopy.

    Warmth breaks the tie on physics rather than on size (see SUN_WARMTH_MIN). ``rank`` is
    which cluster it turned out to be — 0 means the sun also happened to be the biggest thing
    in frame, which on real plates it usually is not.

    None when no cluster is warm, which is the right answer for a scene lit only by a dome:
    there is no sun whose size a width could be evidence about."""
    k = len(directions)
    if k == 0:
        return None
    tol_cos = math.cos(math.radians(tol_deg))
    remaining = list(range(k))
    for rank in range(MAX_CLUSTERS):
        if len(remaining) < MIN_SAMPLES:
            return None                  # nothing left that could support a median
        found = _one_cluster(directions, remaining, tol_cos)
        if found is None:
            return None
        axis, members, mean_cos = found
        if _is_warm(axis):
            return axis, members, mean_cos, rank
        drop = set(members)
        remaining = [j for j in remaining if j not in drop]
    return None


def _hardness_from_width(width_px: float) -> float:
    """Transition width in pixels → 0..1, in octaves between the operator floor and the
    broad-source width. Clamped at both ends, so a razor step and a 40 px wrap both report
    the extreme they actually are rather than running off the scale."""
    w = max(EDGE_OPERATOR_FLOOR_PX, min(WIDTH_SOFT_PX, width_px))
    octaves = math.log(w / EDGE_OPERATOR_FLOOR_PX, 2.0)
    return max(0.0, min(1.0, 1.0 - octaves / _WIDTH_OCTAVES))


def edge_hardness(source) -> Optional[Dict]:
    """How sharp are this frame's lit/shadow boundaries? → 0..1, or None.

    → ``{"hardness", "label", "confidence", "samples", "width_px", "hard_frac",
    "soft_frac"}``. 1.0 is a razor-sharp boundary (a point-like sun); 0.0 is a boundary that
    takes 6% of the frame to happen (overcast, or a very large source). ``label`` is the
    "hard" | "mixed" | "soft" vocabulary `transfer.HARDNESS_SIZE` scores against.

    ``source`` is a path, or an already-loaded ``(pixels, w, h)`` triple from
    ``metrics._load_pixels`` at WORK_MAX_DIM. None when the frame is too small, undecodable,
    or carries fewer than MIN_SAMPLES boundary samples — a frame with no shadow boundary in
    it has no hardness, and inventing 0.5 for it would be worse than the guess this replaces.

    HOW A SHADOW BOUNDARY IS TOLD FROM ORDINARY TEXTURE, and how well that works, because it
    is the whole load-bearing claim here. A candidate is a local gradient maximum along its
    own gradient direction, and it must then pass three independent tests:

      * SUSTAIN. The level difference across ±8 px must persist across ±16 px, same sign,
        at SUSTAIN_FRAC of its size. Texture — foliage, carpet, brick, noise, a bookshelf —
        is zero-mean at small scale, so widening the baseline collapses its difference. This
        is the test that does the real work, and it is the one that cannot be fooled by
        making the texture higher-contrast.
      * STEP. The bright side must be MIN_STEP_RATIO brighter than the dark side, i.e. a
        real lighting ratio rather than a shading wobble on one surface.
      * SHARED RATIO. The linear-light lit/shadow ratio, minus one, must point the same way
        as the rest of the frame's — see CHROMA's replacement above. The albedo cancels out
        of a true shadow ratio, so every genuine shadow edge in one image agrees on one
        direction and a material edge cannot. Samples outside the consensus cone are
        discarded and the hardness is measured on the inliers alone.

    WHAT THAT STILL DOES NOT CATCH, measured rather than assumed. The shared-ratio test does
    what it claims — it separates ILLUMINATION boundaries from MATERIAL boundaries — but
    hardness needs a different separation that it does not provide: sun-PENUMBRA boundaries
    from CONTACT and ambient-occlusion boundaries. A contact shadow is geometrically sharp
    under a source of any size, because the penumbra closes to nothing where the occluder
    meets the surface, and an ambient-occlusion boundary is a genuine illumination change
    that passes the invariant honestly. Clustering the three session plates (2026-07-26,
    12° cone) shows this directly:

        dome-only archetype, TRUE soft, no sun at all
            cluster 0  68% of candidates  BLUE-weighted axis  median hardness 0.93
            cluster 1  15%                BLUE                              1.00
            cluster 2   8%                BLUE                              0.38
        golden-hour interior, TRUE hard sun
            cluster 0  53%  BLUE  0.67   <- ambient occlusion, the majority
            cluster 2  10%  RED   0.95   <- the actual sun-shadow population
        real dawn landscape
            cluster 0  52%  BLUE  0.98   <- 88% of it in the canopy half of the frame
            cluster 1  24%  RED   1.00

    So the largest coherent population is usually NOT the sun's. The consensus picks the
    biggest cluster, and in all three plates that is ambient occlusion. What the axis does
    reveal is whether a sun is present at all: s/a is red-weighted in any daylight scene
    (a 2000–5500 K key against a 10000 K+ fill), so a blue-weighted consensus means there is
    no sun population to have measured — which is why that case caps confidence.

    The failure is ONE-SIDED and that is the useful part: nothing here makes a genuinely
    hard-lit frame read soft. A "soft" reading is therefore strong evidence.

    WHAT ``confidence`` MEANS, exactly, because it does not mean what it looks like. It is
    *how well determined this width measurement is, GIVEN that the warm cluster really is
    the sun's* — support, cluster tightness, coherence with the frame's other warm evidence,
    and how closely the chosen boundaries agree on a width. It is NOT the probability that
    the warm cluster is the sun. Those are different questions and only the first one is
    answerable from a single frame. On the rendered sun-size ladder, where every rung is a
    correct reading, it sits in a narrow 0.41–0.58 band, which is the honest report: all five
    are usable and none is certain. It does not rank a good reading above a bad one — see the
    tension below — so treat a high value as "this number is stable", never as "this number
    is right".

    THE UNRESOLVED TENSION IN ``confidence``, stated because it is load-bearing. Cluster
    SHARE is deliberately not a confidence term, because the share of the RIGHT answer is
    small — the golden-hour plate's genuine sun cluster is 10% of candidates, outnumbered
    five to one by ambient occlusion — and scoring on share reported the one correct
    population in the frame at low confidence for being outnumbered. Removing it fixed that
    and cost something real: measured over twelve random albedo patchworks with no shadow in
    them at all (2026-07-26), five were refused outright but the worst of the rest scored
    0.841, ABOVE the 0.358 this gives the golden-hour sun. Random albedo pairs skew warm
    often enough to form a coherent warm cluster by chance, and within a single frame that
    is not distinguishable from a sun population. Share-in suppresses noise but buries the
    right answer; share-out lets the right answer score but lets noise score too. This
    implementation chose share-out. Until that is resolved, a high confidence from this
    function is not by itself grounds to overrule a reading from elsewhere — check
    ``cluster_rank``, ``inlier_frac`` and ``warmth_ratio``, which are reported for exactly
    this purpose.

    Diagonal boundaries are measured on a ramp, where the central-difference magnitude is
    exact at any orientation; only true steps (already clamped at the operator floor) carry
    an orientation bias. Boundaries within EDGE_MARGIN_PX of the frame border are not
    sampled at all, because their far probe would fall outside.

    ``width_px`` is the RAW median width and is a diagnostic, not a clamped one: a sharpened
    or aliased plate overshoots at its edges, which makes the local slope steeper than the
    plateau difference implies and reports widths under EDGE_OPERATOR_FLOOR_PX. That is not
    an error — it is the operator saying "sharper than I can resolve" — and ``hardness``
    clamps it to 1.0."""
    try:
        return _edge_hardness(source)
    except Exception:                        # the loop's stats can never fail — see metrics
        return None


def _edge_hardness(source) -> Optional[Dict]:
    loaded = _frame(source)
    if not loaded:
        return None
    pixels, w, h = loaded
    if w < 2 * EDGE_MARGIN_PX + 3 or h < 2 * EDGE_MARGIN_PX + 3:
        return None                          # no interior left to hold a sample
    n = w * h
    lum = [0.0] * n
    for i in range(n):
        r, g, b = pixels[i]
        lum[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    # ---- gradient magnitude everywhere, then a HISTOGRAM percentile for the cost gate.
    # A histogram keeps this O(n) where a sort would be O(n log n) on 65k values, and
    # compute_stats already builds its luminance histogram the same way.
    mag = [0.0] * n
    gx_all = [0.0] * n
    gy_all = [0.0] * n
    hist = [0] * _GRAD_BINS
    interior = 0
    for y in range(1, h - 1):
        base = y * w
        for x in range(1, w - 1):
            i = base + x
            gx = (lum[i + 1] - lum[i - 1]) * 0.5
            gy = (lum[i + w] - lum[i - w]) * 0.5
            m = math.sqrt(gx * gx + gy * gy)
            mag[i] = m
            gx_all[i] = gx
            gy_all[i] = gy
            hist[min(_GRAD_BINS - 1, int(m * _GRAD_BINS))] += 1
            interior += 1
    if not interior:
        return None
    want = max(1, int(GRAD_KEEP_FRACTION * interior))
    seen = 0
    cutoff_bin = 0
    for b_i in range(_GRAD_BINS - 1, -1, -1):
        seen += hist[b_i]
        if seen >= want:
            cutoff_bin = b_i
            break
    cutoff = max(MIN_ABS_GRAD, cutoff_bin / float(_GRAD_BINS))

    sats = _sat3(pixels, w, h)
    hardnesses: List[float] = []
    widths: List[float] = []
    directions: List[Tuple[float, float, float]] = []
    lo_y, hi_y = EDGE_MARGIN_PX, h - EDGE_MARGIN_PX
    lo_x, hi_x = EDGE_MARGIN_PX, w - EDGE_MARGIN_PX
    for y in range(lo_y, hi_y):
        base = y * w
        for x in range(lo_x, hi_x):
            i = base + x
            m = mag[i]
            if m < cutoff:
                continue
            gx, gy = gx_all[i], gy_all[i]
            # unit normal, rounded to the pixel grid — the direction the profile is read in
            ux, uy = gx / m, gy / m
            sx, sy = int(round(ux)), int(round(uy))
            if sx == 0 and sy == 0:
                continue
            # Non-maximum suppression ALONG the normal: without it every pixel on the flank
            # of one soft edge becomes its own "boundary" and the soft edge outvotes the
            # hard one purely by being WIDER — a bias straight into the answer.
            # The trailing comparison is `>=`, not `>`, which breaks ties: a linear ramp has
            # no strict maximum anywhere, so a plain NMS keeps all of it and re-introduces
            # exactly the width-proportional inflation this is here to remove. With the
            # tie-break a run of equal gradient yields one sample, like a step does.
            if mag[i + sy * w + sx] > m or mag[i - sy * w - sx] >= m:
                continue
            near_dx, near_dy = int(round(ux * PLATEAU_NEAR_PX)), int(round(uy * PLATEAU_NEAR_PX))
            far_dx, far_dy = int(round(ux * PLATEAU_FAR_PX)), int(round(uy * PLATEAU_FAR_PX))
            a = _window(sats, w, h, x + near_dx, y + near_dy)
            b = _window(sats, w, h, x - near_dx, y - near_dy)
            fa = _window(sats, w, h, x + far_dx, y + far_dy)
            fb = _window(sats, w, h, x - far_dx, y - far_dy)
            if a is None or b is None or fa is None or fb is None:
                continue
            near_step = a[0] - b[0]
            far_step = fa[0] - fb[0]
            if abs(near_step) < MIN_STEP:
                continue
            # SUSTAIN — the difference must still be there on a wider baseline
            if far_step * near_step <= 0.0 or abs(far_step) < SUSTAIN_FRAC * abs(near_step):
                continue
            hi_lum, lo_lum = (a[0], b[0]) if near_step > 0 else (b[0], a[0])
            if lo_lum <= 1e-6 or hi_lum / lo_lum < MIN_STEP_RATIO:
                continue
            # MATERIAL — the shadow ratio, in LINEAR light, for the consensus below. The
            # brighter plateau is the lit side by construction of near_step's sign.
            if near_step > 0:
                lit_xy, shade_xy = (x + near_dx, y + near_dy), (x - near_dx, y - near_dy)
            else:
                lit_xy, shade_xy = (x - near_dx, y - near_dy), (x + near_dx, y + near_dy)
            lit_lin = _linear_window(pixels, w, h, lit_xy[0], lit_xy[1])
            shade_lin = _linear_window(pixels, w, h, shade_xy[0], shade_xy[1])
            if lit_lin is None or shade_lin is None:
                continue
            direction = _shadow_direction(lit_lin, shade_lin)
            if direction is None:
                continue
            width = abs(near_step) / m       # Δ / slope: the edge's own width, in pixels
            widths.append(width)
            hardnesses.append(_hardness_from_width(width))
            directions.append(direction)

    # ---- CONSENSUS. Keep only the boundaries that agree on one illuminant change; measure
    # hardness on those alone. This is the step that decides whether the frame contains
    # shadow edges at all, so it runs before the sample-count gate.
    candidates = len(hardnesses)
    if candidates < MIN_SAMPLES:
        return None
    chosen = _sun_cluster(directions)
    if chosen is None:
        # No warm cluster anywhere ⇒ no sun-shadow population in this frame. A width
        # measured on ambient occlusion is not evidence about the size of a sun, so the
        # answer is absence rather than a number with a caveat attached.
        return None
    axis, members, mean_cos, rank = chosen
    inlier_frac = len(members) / float(candidates)
    # how much OTHER warm evidence disagreed with the chosen cone — the sun population
    # should be one cluster, not several pointing different ways
    rival_warm = sum(1 for j, d in enumerate(directions)
                     if _is_warm(d) and j not in set(members))
    hardnesses = [hardnesses[j] for j in members]
    widths = [widths[j] for j in members]
    samples = len(hardnesses)
    if samples < MIN_SAMPLES:
        return None

    hardnesses.sort()
    median = hardnesses[samples // 2]
    hard_frac = sum(1 for v in hardnesses if v >= HARD_LABEL) / samples
    soft_frac = sum(1 for v in hardnesses if v <= SOFT_LABEL) / samples
    # the label rides on the MEDIAN alone — the quantity the ladder showed monotonic. See
    # MIXED_MINORITY for why a population count no longer gets to overrule it.
    bimodal = hard_frac >= MIXED_MINORITY and soft_frac >= MIXED_MINORITY
    if median >= HARD_LABEL:
        label = "hard"
    elif median <= SOFT_LABEL:
        label = "soft"
    else:
        label = "mixed"
    # CONFIDENCE = how much do I believe this width is the SUN's penumbra. Share of frame is
    # deliberately absent from it. The share of the right answer is small — the golden-hour
    # interior's sun cluster is 10% of candidates — so scoring on share would report the one
    # correct population in the frame at low confidence for being outnumbered by ambient
    # occlusion, which is the mistake this whole selection exists to undo. What is left is:
    #   support    enough boundaries to settle a median, counted ABSOLUTELY
    #   warmth     how decisively the axis is a sun-over-ambient ratio rather than a coin toss
    #   tightness  how concentrated the cluster is inside its cone
    #   coherent   whether the other warm evidence joined this cluster or contradicted it
    #   agreement  how closely the chosen boundaries agree on a WIDTH — the direct statement
    #              of whether this median is well determined
    #
    # DISTANCE FROM A LABEL BOUNDARY IS NOT ONE OF THEM, and used to be. It measured the
    # wrong thing: a frame whose median lands mid-scale has an ambiguous LABEL but a
    # perfectly well determined NUMBER, and the number is what drives sun.size. Measured on
    # the ladder, that term reported the sun.size 8 rung — a correct mid reading, dead centre
    # of a monotonic series — at confidence 0.047, purely for landing near a threshold.
    # Label ambiguity is now reported separately as `label_margin`, where a caller who cares
    # about the category rather than the quantity can see it.
    support = min(1.0, samples / float(CONFIDENT_SAMPLES))
    warmth_ratio = axis[0] / axis[2] if axis[2] > 1e-9 else float("inf")
    warmth = min(1.0, max(0.0, (warmth_ratio - SUN_WARMTH_MIN)
                          / (CONFIDENT_WARMTH - SUN_WARMTH_MIN)))
    floor_cos = math.cos(math.radians(RATIO_INLIER_DEG))
    tightness = max(0.0, min(1.0, (mean_cos - floor_cos) / (1.0 - floor_cos)))
    coherent = samples / float(samples + rival_warm)
    # how closely the chosen boundaries agree on a width, as an inter-quartile range of their
    # hardnesses. Robust to the tails a real frame always has; a spread of half the scale is
    # where the median stops describing anything.
    spread = hardnesses[(3 * samples) // 4] - hardnesses[samples // 4]
    agreement = max(0.0, 1.0 - spread / 0.5)
    label_margin = min(abs(median - HARD_LABEL), abs(median - SOFT_LABEL))
    # the WORST factor wins, the posture patchgeom.bearing_hint takes: a confident sub-measure
    # must not paper over a weak one
    confidence = min(support, 0.35 + 0.65 * warmth, 0.35 + 0.65 * tightness,
                     coherent, 0.25 + 0.75 * agreement)
    warm = True                     # a returned reading is warm by construction — see above
    widths.sort()
    return {
        "hardness": round(median, 4),
        "label": label,
        "confidence": round(confidence, 3),
        "samples": samples,
        "candidates": candidates,
        "inlier_frac": round(inlier_frac, 4),
        # True on every returned reading now — a cool cluster is declined outright rather
        # than reported with a caveat. Kept because it names the gate that was crossed.
        "ratio_is_warm": warm,
        # which cluster the sun turned out to be, largest first. 0 means the sun population
        # was also the biggest thing in frame; on real plates it usually is not, and a high
        # rank is the honest signal that most of this frame is not shadows at all.
        "cluster_rank": rank,
        "warmth_ratio": round(warmth_ratio, 3) if math.isfinite(warmth_ratio) else None,
        # the recovered sun-to-ambient spectral ratio, unit linear RGB — a by-product, but
        # a physically meaningful one: it is what separates the key from the fill
        "ratio_rgb": [round(c, 4) for c in axis],
        "width_px": round(widths[samples // 2], 2),
        "hard_frac": round(hard_frac, 4),
        "soft_frac": round(soft_frac, 4),
        # this frame carries a hard population AND a soft one rather than one typical width
        "bimodal": bimodal,
        # how far the median sits from the nearest label boundary. Deliberately NOT part of
        # confidence — it describes the categorical answer, not the measured one.
        "label_margin": round(label_margin, 4),
    }


# ============================================================================ haze
#: Dark-channel patch radius. He et al. use 15×15 on ~600 px images — 2.5% of the width. At
#: WORK_MAX_DIM that scales to 9×9, which is what this is. The size matters in both
#: directions: too small and the patch cannot contain a dark pixel even in clear air (haze
#: everywhere); too large and the patch spans real depth changes, so one near-field shadow
#: declares a whole foggy background clear.
DARK_RADIUS = 4

#: Label thresholds, derived from the rig they have to drive rather than fitted to plates.
#: The airlight model is I = J·t + A·(1-t) with transmission t = exp(-d / D); the dark
#: channel of a haze-free J is ~0, so what survives is J_dark ≈ A·(1-t), and with a typical
#: airlight A ≈ 0.9 that inverts to a visibility distance. Taking an architectural subject
#: at d ≈ 40 m and transfer.ATMOSPHERE_DISTANCE_M's own D values:
#:      D = 4000 m ("none")        t = 0.990 → J_dark ≈ 0.01
#:      D =  250 m ("light_haze")  t = 0.852 → J_dark ≈ 0.13
#:      D =   70 m ("heavy_haze")  t = 0.565 → J_dark ≈ 0.39
#:      D =   20 m ("fog")         t = 0.135 → J_dark ≈ 0.78
#: The boundaries below sit between those, roughly geometrically. They are consistent with
#: the prior's own statistic: He et al. report ~86% of haze-free outdoor patches under
#: 25/255 ≈ 0.098, which lands inside "none" with room to spare.
DARK_NONE = 0.12
DARK_LIGHT = 0.30
DARK_HEAVY = 0.58

#: Mean absolute luminance gradient (per pixel) below which a frame has genuinely lost its
#: local contrast — the other half of what haze does, and the cross-check that separates
#: real haze from a bright scene with no dark objects in it. Measured against the same
#: 8-bit floor the edge pass uses: 0.012 is about three code values per pixel, which a
#: normally-detailed photograph clears by a wide margin.
LOW_CONTRAST = 0.012

#: A bright-albedo frame (a white studio, a snowfield, a whiteout wall) raises the dark
#: channel exactly like haze does and is the prior's classic false positive. When the dark
#: channel is high but local contrast is NOT depressed, the reading is downgraded rather
#: than discarded: it is still the best evidence available, it just should not be defended.
BRIGHT_ALBEDO_PENALTY = 0.45

_SENTINEL = 2.0                              # above any possible dark-channel value


def _running_min(line: Sequence[float], radius: int) -> List[float]:
    """Sliding-window minimum over a 1-D line, O(len) exactly (van Herk / Gil-Werman).

    A summed-area table cannot do minima, and the naive window is O(n·k) — which for the
    dark channel is the same trap the sun-patch map avoided. Two running passes per block of
    k = 2r+1 (forward and backward) reduce every window to one comparison. The line is padded
    with a sentinel above any real value, which makes the border windows exactly the
    truncated windows without a special case."""
    n = len(line)
    if radius <= 0 or n == 0:
        return list(line)
    k = 2 * radius + 1
    padded = [_SENTINEL] * radius + list(line) + [_SENTINEL] * radius
    blocks = -(-len(padded) // k)
    total = blocks * k
    padded.extend([_SENTINEL] * (total - len(padded)))
    prefix = [0.0] * total
    suffix = [0.0] * total
    for b in range(blocks):
        s = b * k
        run = padded[s]
        prefix[s] = run
        for j in range(1, k):
            v = padded[s + j]
            if v < run:
                run = v
            prefix[s + j] = run
        e = s + k - 1
        run = padded[e]
        suffix[e] = run
        for j in range(e - 1, s - 1, -1):
            v = padded[j]
            if v < run:
                run = v
            suffix[j] = run
    out = [0.0] * n
    for i in range(n):
        c = i + radius
        # the two half-windows are exactly k-1 apart, so they meet on a block boundary and
        # their union is exactly [c-radius, c+radius]
        lo, hi = suffix[c - radius], prefix[c + radius]
        out[i] = lo if lo < hi else hi
    return out


def haze_estimate(source) -> Optional[Dict]:
    """How much air is between the camera and this scene? → dark-channel floor 0..1, or None.

    → ``{"dark_channel", "label", "confidence", "local_contrast", "clear_floor"}``, with
    ``label`` in the "none" | "light_haze" | "heavy_haze" | "fog" vocabulary
    `transfer.ATMOSPHERE_DISTANCE_M` scores against.

    ``dark_channel`` is the MEDIAN over the frame of the per-patch dark channel — the
    minimum over a DARK_RADIUS window and over the three colour channels. Median, because
    the two obvious alternatives both break: a low percentile is defeated by a single dark
    object in the foreground of a genuinely foggy landscape, and a high one by a single
    white wall in clear air. The median asks what the TYPICAL patch's floor is, which is the
    quantity the prior is actually about. ``clear_floor`` is the 10th percentile of the same
    map and is reported beside it: near zero means somewhere in this frame the air is
    demonstrably clear, which is what a depth-varying haze looks like and a uniform one
    does not.

    WHERE IT IS LEAST TRUSTWORTHY, and the reason ``confidence`` exists:

      * a frame with no dark content — a white studio, a snowfield, a bright overcast sky
        filling the frame, a high-key product shot — has a lifted dark channel with no haze
        at all. This is the dark-channel prior's classic false positive and it cannot be
        removed, only detected: haze ALSO destroys local contrast, so when the dark channel
        is high while the mean gradient is not depressed, confidence is cut by
        BRIGHT_ALBEDO_PENALTY.
      * a frame that is mostly sky. Sky has no dark channel by nature and the original
        prior excludes it; nothing here segments it out, so a landscape with a big bright
        sky reads hazier than its architecture does.
      * a heavily lifted / faded GRADE reads exactly like haze, because at the pixel level
        it is the same operation. That is arguably correct for our purposes — the rig has to
        reproduce the look — but it is not a claim about the air.
      * interiors generally. The prior is an outdoor statistic; an interior's "haze" is
        usually just a bright wall, and the confidence cross-check is doing all the work.

    None for an undecodable or degenerate frame. Never raises."""
    try:
        return _haze_estimate(source)
    except Exception:
        return None


def _haze_estimate(source) -> Optional[Dict]:
    loaded = _frame(source)
    if not loaded:
        return None
    pixels, w, h = loaded
    if w < 3 or h < 3:
        return None
    n = w * h
    # per-pixel dark channel: the minimum over the three channels. Luminance rides along in
    # the same pass because the contrast cross-check below needs it and the decode is paid.
    dark = [0.0] * n
    lum = [0.0] * n
    for i in range(n):
        r, g, b = pixels[i]
        m = r if r < g else g
        if b < m:
            m = b
        dark[i] = m / 255.0
        lum[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    # ---- windowed minimum, separably: rows then columns. O(n) both ways.
    rows_min: List[float] = []
    for y in range(h):
        rows_min.extend(_running_min(dark[y * w:(y + 1) * w], DARK_RADIUS))
    channel = [0.0] * n
    for x in range(w):
        col = [rows_min[y * w + x] for y in range(h)]
        col = _running_min(col, DARK_RADIUS)
        for y in range(h):
            channel[y * w + x] = col[y]

    # ---- local contrast: mean absolute luminance gradient, the second thing haze does
    grad_sum = 0.0
    grad_n = 0
    for y in range(1, h - 1):
        base = y * w
        for x in range(1, w - 1):
            i = base + x
            grad_sum += (abs(lum[i + 1] - lum[i - 1]) + abs(lum[i + w] - lum[i - w])) * 0.5
            grad_n += 1
    local_contrast = grad_sum / grad_n if grad_n else 0.0

    ordered = sorted(channel)
    median = ordered[n // 2]
    clear_floor = ordered[max(0, int(0.10 * n) - 1)]

    if median < DARK_NONE:
        label = "none"
    elif median < DARK_LIGHT:
        label = "light_haze"
    elif median < DARK_HEAVY:
        label = "heavy_haze"
    else:
        label = "fog"

    # how far the reading sits from the band boundary it is nearest — a value sitting on a
    # threshold is a real result and must be held loosely, not rounded into confidence
    edges = [DARK_NONE, DARK_LIGHT, DARK_HEAVY]
    gap = min(abs(median - e) for e in edges)
    confidence = min(1.0, gap / 0.06)
    if label != "none" and local_contrast > LOW_CONTRAST:
        # the dark channel says haze and the detail says otherwise — most likely a bright
        # scene, not thick air
        confidence *= BRIGHT_ALBEDO_PENALTY
    return {
        "dark_channel": round(median, 4),
        "label": label,
        "confidence": round(confidence, 3),
        "local_contrast": round(local_contrast, 5),
        "clear_floor": round(clear_floor, 4),
    }
