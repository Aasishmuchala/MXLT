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

#: Label bands. 0.62 is a width of 4.4 px, 0.35 is 7.7 px — so an edge that resolves inside
#: ~1.7% of the frame reads hard, one that takes more than ~3% reads soft, and the band
#: between them is where a source of intermediate size (a big window, a bright overcast
#: patch) genuinely sits. Chosen on the width geometry above, NOT fitted to a plate.
HARD_LABEL = 0.62
SOFT_LABEL = 0.35

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

#: THE MATERIAL TEST. A shadow is a change of ILLUMINATION over one albedo, so normalised
#: chromaticity is roughly preserved across it; a material boundary usually jumps. The
#: tolerance is |Δr̄| + |Δb̄| in normalised rgb and is deliberately loose, because real
#: shadows DO shift: lit by a 2500 K sun and filled by a 10000 K sky, a golden-hour shadow
#: edge moves ~0.08 in each coordinate, and a tighter tolerance would reject the very
#: reference this module was built for. See the docstring for what it therefore cannot do.
CHROMA_TOL = 0.22

#: Fewer boundary samples than this and the median is an anecdote → None.
MIN_SAMPLES = 40
#: ...and this many is a population; below it, confidence is scaled by what was found.
CONFIDENT_SAMPLES = 400

#: Both populations at least this large ⇒ "mixed" regardless of where the median landed.
#: A shaft of sun crossing a softly-filled room genuinely has both, and its median is an
#: average of two things that are not present in the middle.
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
      * MATERIAL. Normalised chromaticity must be preserved to CHROMA_TOL across the
        boundary. A shadow changes the illumination over ONE albedo; a paint line, a rug
        edge, a green plant against a grey wall changes the albedo itself.

    WHAT THAT DOES NOT CATCH, stated plainly: an OCCLUSION boundary between two objects of
    similar colour — a grey chair against a grey wall, a dark doorway, a window frame against
    sky — passes all three tests, and being a true step it reads maximally HARD. That is this
    estimator's dominant failure mode. Two things blunt it and neither removes it: the
    reported value is the MEDIAN over hundreds of samples, so a handful of silhouettes cannot
    swing a frame; and the material test does remove the coloured ones, which in practice is
    most of them. A scene built mostly of hard same-hue object edges under a soft sky — a
    grey bookshelf on an overcast day — will read harder than it is. Treat a "hard" reading
    from a cluttered monochrome interior as weak evidence, and note that the failure is
    ONE-SIDED: nothing here makes a genuinely hard-lit frame read soft, so a "soft" reading
    is the more trustworthy of the two.

    Diagonal boundaries are measured on a ramp, where the central-difference magnitude is
    exact at any orientation; only true steps (already clamped at the operator floor) carry
    an orientation bias. Boundaries within EDGE_MARGIN_PX of the frame border are not
    sampled at all, because their far probe would fall outside."""
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
            # MATERIAL — one albedo either side, so the chromaticity should survive
            if abs(a[1] - b[1]) + abs(a[2] - b[2]) > CHROMA_TOL:
                continue
            width = abs(near_step) / m       # Δ / slope: the edge's own width, in pixels
            widths.append(width)
            hardnesses.append(_hardness_from_width(width))
    samples = len(hardnesses)
    if samples < MIN_SAMPLES:
        return None

    hardnesses.sort()
    median = hardnesses[samples // 2]
    hard_frac = sum(1 for v in hardnesses if v >= HARD_LABEL) / samples
    soft_frac = sum(1 for v in hardnesses if v <= SOFT_LABEL) / samples
    if hard_frac >= MIXED_MINORITY and soft_frac >= MIXED_MINORITY:
        label = "mixed"                      # both populations are really present
    elif median >= HARD_LABEL:
        label = "hard"
    elif median <= SOFT_LABEL:
        label = "soft"
    else:
        label = "mixed"
    # confidence: how much evidence, and how far the answer sits from the band it is not in
    support = min(1.0, samples / float(CONFIDENT_SAMPLES))
    gap = min(abs(median - HARD_LABEL), abs(median - SOFT_LABEL))
    decisive = min(1.0, gap / 0.12)
    widths.sort()
    return {
        "hardness": round(median, 4),
        "label": label,
        "confidence": round(support * decisive, 3),
        "samples": samples,
        "width_px": round(widths[samples // 2], 2),
        "hard_frac": round(hard_frac, 4),
        "soft_frac": round(soft_frac, 4),
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
