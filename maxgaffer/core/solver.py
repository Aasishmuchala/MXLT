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
        checklist and the slope is config-tunable.

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


def solve_ev(ref_stats: Dict, cur_stats: Dict, current_ev: float,
             tighten: float = 1.0) -> Optional[float]:
    """New EV that matches the render's key to the reference's, or None if close enough.
    ``tighten`` < 1 shrinks the deadband (and the per-step cap) as the match converges —
    a 0.15-stop tolerance is exploration slack, not a finishing standard."""
    key_ref = _finite(ref_stats.get("log_key"))
    key_cur = _finite(cur_stats.get("log_key"))
    if key_ref is None or key_cur is None:
        return None
    key_ref = max(1e-5, key_ref)
    key_cur = max(1e-5, key_cur)
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
    direct counter to the albedo trap; falls back to full-frame means on old stats."""
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
    return changes
