"""Deterministic exposure + white-balance solve — math owns what math can own.

The single most reliable move in lighting matching: exposure and WB are *measurable*, so we
never let the LLM guess them (MaxDirector lesson: LLMs hallucinate spatial/metric precision;
anchor with computed values). Every iteration, before asking the LLM anything, we:

  EV  — compare the geometric-mean linear luminance ("key") of reference vs render.
        dEV = log2(key_ref / key_cur). V-Ray EV semantics: HIGHER EV = DARKER image, so a
        render darker than the reference (dEV > 0) needs new_ev = ev - dEV.  Center-weighted
        keys, per-iteration clamp and a deadband keep it from chasing noise.

  WB  — compare LAB b* means (blue-yellow axis). V-Ray white-balance temperature semantics:
        raising the WB kelvin renders WARMER (the camera compensates for a bluer assumed
        illuminant). If the reference is warmer than the render (db > 0) we raise kelvin.
        ~90 K per b* unit is an empirical slope; the visual sign-check lives in the on-box
        checklist and the slope is a module constant (WB_KELVIN_PER_B — edit it, or pass
        ``kelvin_per_b``, if the on-box #6 check ever demands a different slope).

Both return None inside their deadband so the caller can skip a no-op scene write.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from .genome import LightingState, clamp

EV_DEADBAND = 0.15
EV_MAX_STEP = 2.5
WB_DEADBAND_B = 1.5          # LAB b* units
WB_KELVIN_PER_B = 90.0
WB_MAX_STEP = 1500.0
# both images clipping ≥ this share of the highlight quartile = same-scene regime →
# switch to the inclusive (symmetric) highlight mean; see solve_wb
SYMMETRIC_CLIP_FRAC = 0.15


def _finite(value) -> Optional[float]:
    """float() + finiteness in one — junk stats (None, short lists, NaN) mean the solver
    has NO opinion, and a coerced blind slam is the worst possible answer."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _vec3(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def _finite(value) -> Optional[float]:
    """Coerce a stats field to a finite float, or None when the value is missing-shaped
    (None, non-numeric, NaN/inf) — stats cross the sidecar trust boundary unvalidated,
    and a malformed field must read as "metrics unavailable", never as a crash or a
    phantom full-step move."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def solve_ev(ref_stats: Dict, cur_stats: Dict, current_ev: float,
             tighten: float = 1.0) -> Optional[float]:
    """New EV that matches the render's key to the reference's, or None if close enough.
    ``tighten`` < 1 shrinks the deadband (and the per-step cap) as the match converges —
    a 0.15-stop tolerance is exploration slack, not a finishing standard."""
    key_ref = _finite(ref_stats.get("log_key"))
    key_cur = _finite(cur_stats.get("log_key"))
    if key_ref is None or key_cur is None:
        return None     # metrics unavailable (or absent) — skip, never invent a max-step move
    key_ref, key_cur = max(1e-5, key_ref), max(1e-5, key_cur)
    d_ev = math.log2(key_ref / key_cur)
    if abs(d_ev) < EV_DEADBAND * max(0.1, tighten):
        return None
    # NOTE: only the DEADBAND anneals — the correction cap stays full-size, because a
    # measured 2-stop error deserves a 2-stop fix regardless of how well the rest of the
    # match is going (the cap is a stability rail, not a convergence knob)
    d_ev = max(-EV_MAX_STEP, min(EV_MAX_STEP, d_ev))
    new = clamp("exposure.ev", current_ev - d_ev)
    # pinned at a genome bound (or a zeroed slope): the solved value IS the current one —
    # None means "skip a no-op scene write", not "keep solving"
    return None if abs(new - current_ev) < 1e-9 else new


def solve_wb(ref_stats: Dict, cur_stats: Dict, current_kelvin: float,
             kelvin_per_b: float = WB_KELVIN_PER_B, tighten: float = 1.0) -> Optional[float]:
    """New WB kelvin nudging the render's blue-yellow balance toward the reference.

    Prefers HIGHLIGHT chromaticity (top luminance quartile — the white-patch assumption:
    highlights carry the illuminant, the full mean carries the furniture). This is the
    direct counter to the albedo trap; falls back to full-frame means on old stats.

    The highlight path is chosen from the VALIDITY of both sides' highlight mean
    (never key presence — a ``null``/short field is as absent as a missing key), and
    the solve abstains (None) when the chosen source is unusable on either side:
    comparing a highlight mean against a full mean would mix two different quantities,
    and a missing mean must not silently read as b* = 0."""
    # Which highlight mean: the clip-EXCLUSIVE one is the cross-scene default (blown
    # pixels carry no chroma). But when BOTH images clip a large share of the quartile,
    # the match is same-scene (a relight/hero shot) and the populations must stay
    # SYMMETRIC — per-image exclusion measures different pixel sets on each side and
    # drags the WB anchor cool (measured: deep match stalled at 97.8 on a reachable
    # target; symmetric-inclusive lands 99+).
    hi_key = "lab_mean_hi"
    if (float(ref_stats.get("hi_clip_frac", 0.0) or 0.0) >= SYMMETRIC_CLIP_FRAC
            and float(cur_stats.get("hi_clip_frac", 0.0) or 0.0) >= SYMMETRIC_CLIP_FRAC):
        hi_key = "lab_mean_hi_full"
    # highlight-vs-highlight only when BOTH sides carry a well-formed one — comparing a
    # highlight mean against a full-frame mean is apples-to-oranges (different quantities)
    hi_both = _vec3(ref_stats.get(hi_key)) and _vec3(cur_stats.get(hi_key))

    def b_of(stats: Dict) -> Optional[float]:
        src = stats.get(hi_key) if hi_both else stats.get("lab_mean")
        if not _vec3(src):
            return None
        return _finite(src[2])

    b_ref, b_cur = b_of(ref_stats), b_of(cur_stats)
    if b_ref is None or b_cur is None:
        return None
    db = b_ref - b_cur
    if abs(db) < WB_DEADBAND_B * max(0.1, tighten):
        return None
    delta = max(-WB_MAX_STEP, min(WB_MAX_STEP, db * kelvin_per_b))
    new = clamp("exposure.wb_kelvin", current_kelvin + delta)
    return None if abs(new - current_kelvin) < 1e-9 else new


#: How far the shadow floor or the sun's own contribution may sit from the reference's
#: before it is worth a scene write. These are RATIOS, so 0.12 is roughly an eighth of a
#: stop — tighter than the EV deadband because this solve is separating two lights rather
#: than sliding both, and a small persistent bias here is what the search kept papering
#: over with exposure.
AMBIENT_DEADBAND = 0.12
#: Correction cap per iteration, in octaves. A stability rail, like EV_MAX_STEP.
AMBIENT_MAX_OCTAVES = 1.5


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def solve_ambient(ref_stats: Dict, cur_stats: Dict, current_dome: Optional[float],
                  current_sun: Optional[float],
                  tighten: float = 1.0) -> Dict[str, float]:
    """Separate the AMBIENT level from the SUN level, using the two ends of the histogram.

    The darkest part of a sunlit frame sees no direct sun — it is lit by ambient alone — so
    the 5th percentile measures the dome almost independently of the sun. The brightest
    part is where the sun lands, so (p95 - p5) measures the sun almost independently of the
    dome. Two measurements, two unknowns, and no search required.

    That separation is what was missing. Measured on-box 2026-07-26, a match that had the
    sun's DIRECTION right to 2.5 degrees still recovered dome 1.66 against a 0.55 target,
    with exposure compensating: nothing was solving the two lights apart, so the loop kept
    trading one against the other and the shadows came back lifted (p5 0.183 against the
    reference's 0.164). The ratio is deliberately scale-free — it corrects the two lights
    RELATIVE to each other and leaves the overall level to solve_ev, which is the one that
    owns it.

    → {"dome.intensity": v, "sun.intensity": v} for whichever is worth moving. Percentiles
    are display-referred, so they are linearised first: a ratio of display values is not a
    ratio of light."""
    out: Dict[str, float] = {}
    try:
        p_ref, p_cur = ref_stats.get("p"), cur_stats.get("p")
        if not isinstance(p_ref, dict) or not isinstance(p_cur, dict):
            return out
        lo_ref = _srgb_to_linear(max(0.0, float(p_ref["5"])))
        hi_ref = _srgb_to_linear(max(0.0, float(p_ref["95"])))
        lo_cur = _srgb_to_linear(max(0.0, float(p_cur["5"])))
        hi_cur = _srgb_to_linear(max(0.0, float(p_cur["95"])))
    except (KeyError, TypeError, ValueError):
        return out
    if min(lo_ref, lo_cur) <= 1e-6:
        return out          # a frame crushed to black carries no ambient measurement

    def _scaled(current: Optional[float], want: float, have: float,
                key: str) -> None:
        if current is None or have <= 1e-9 or want <= 1e-9:
            return
        octaves = math.log2(want / have)
        if abs(octaves) < AMBIENT_DEADBAND * max(0.1, tighten):
            return
        octaves = max(-AMBIENT_MAX_OCTAVES, min(AMBIENT_MAX_OCTAVES, octaves))
        new = clamp(key, float(current) * (2.0 ** octaves))
        if abs(new - float(current)) > 1e-9:
            out[key] = new

    _scaled(current_dome, lo_ref, lo_cur, "dome.intensity")
    # the sun's OWN contribution is what sits above the ambient floor at the bright end;
    # comparing raw p95 would credit the sun for ambient the dome is already supplying
    _scaled(current_sun, max(1e-9, hi_ref - lo_ref), max(1e-9, hi_cur - lo_cur),
            "sun.intensity")
    return out


def analytic_pass(
    state: LightingState,
    ref_stats: Dict,
    cur_stats: Dict,
    locks: Optional[set] = None,
    tighten: float = 1.0,
) -> Dict[str, float]:
    """The changes the solver wants this iteration ({} when everything is in the deadband).

    Capability-gated: a key absent from ``state`` means the rig has no host for it
    (read_state only includes supported params) — proposing it anyway would create a
    phantom parameter the bridge warns about every iteration and, worse, walk the leash
    into a false albedo diagnosis while changing nothing on screen."""
    locks = locks or set()
    changes: Dict[str, float] = {}
    if "exposure.ev" in state.values and "exposure.ev" not in locks:
        ev = solve_ev(ref_stats, cur_stats, state.get("exposure.ev"), tighten)
        if ev is not None:
            changes["exposure.ev"] = ev
    if "exposure.wb_kelvin" in state.values and "exposure.wb_kelvin" not in locks:
        wb = solve_wb(ref_stats, cur_stats, state.get("exposure.wb_kelvin"),
                      tighten=tighten)
        if wb is not None:
            changes["exposure.wb_kelvin"] = wb
    # ...and the two LIGHTS, separated from each other by the two ends of the histogram.
    # Capability-gated exactly like the tonal pair above: a key absent from the state means
    # the rig has no host for it.
    ambient = solve_ambient(
        ref_stats, cur_stats,
        state.get("dome.intensity") if "dome.intensity" in state.values
        and "dome.intensity" not in locks else None,
        state.get("sun.intensity") if "sun.intensity" in state.values
        and "sun.intensity" not in locks else None,
        tighten=tighten)
    changes.update(ambient)
    return changes
