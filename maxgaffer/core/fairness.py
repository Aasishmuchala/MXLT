"""Reference-vs-scene fairness — "can this reference actually constrain THIS scene?"

The match loop is honest about *tone* (the solver measures EV/WB, the critic scores mood),
but silent about a prior question: is the reference even a fair yardstick for the scene it is
being matched against? A white-room reference dragged onto a dark-walnut interior cannot be
satisfied by exposure/WB alone — the solver will chase its leash, the critic will bottom out
its tonal axes, and the run will "converge" on a compromise nobody asked for. The director
already *diagnoses* that after the fact (leash_hits >= 2 → "white room vs dark wood"); the
critic already flags it (content_gap). This module is the READ-ONLY pre/post estimate that
speaks the same language as both, so a caller can warn *before* burning a full match.

It owns NOTHING the solver or director owns. It reuses their exact numbers — the log2-key
ratio with the same ``max(1e-5, …)`` guard (director.py:253-255), the highlight-b* gap with
the same ``lab_mean_hi_full`` same-scene switch (solver.py:107-110), the same leashes and the
same WB_KELVIN_PER_B slope — passed in as kwargs that default to the live values. Given the
director/critic authoritative signals (leash_hits, content_gap) it can only ever read
AS-BAD-OR-WORSE than they do, never softer: that is what makes "fairness never contradicts
the director/critic" literally hold in the real wiring (controller D8 always feeds them). In
the predictive pre-match path (D9) those signals are absent, so the guarantee narrows to
"consistent with the numbers fairness can see" — stated in the returned ``notes``.

Route-A honesty: extra reference views are DENOISING/transparency data, not solve inputs, so
``n_references`` never raises ``constrainable`` and the single-frontal-view caveat
(back-hemisphere fill is not constrained by the reference view) is ALWAYS present. The advice
is only ever "LOCK exposure.ev / exposure.wb_kelvin and set them by eye" — this module never
proposes an exposure/WB value, exactly like the director's leash diagnosis.

Pure stdlib, no new deps. NEVER raises, NEVER returns None — missing/degenerate stats degrade
to a fully-shaped ``verdict == "unknown"`` result.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

# Always-true physical caveat of a single-view Route-A solve — the reference view constrains
# only what it can see; the far hemisphere is never observed and never reconstructed.
BACK_HEMI_CAVEAT = ("back-hemisphere / 360 fill is not constrained by the reference "
                    "view(s) in the solve")

# Same intent/wording family as director.py:453 and controller.py — LOCK, never a proposed
# value (do not emit a second competing "white room vs dark wood" verdict elsewhere).
REMEDY_LOCK = ("lock exposure.ev and exposure.wb_kelvin and set them by eye — reference and "
               "scene likely disagree in albedo (e.g. white room vs dark wood)")


def _finite(value) -> Optional[float]:
    """Finite float or None — stats cross the sidecar trust boundary unvalidated, so a
    present-but-mistyped field (None, "junk", NaN/inf) must read as ABSENT, never raise."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _default(value, fallback: float) -> float:
    """Finite float, or the live-value fallback — for the leash/slope kwargs, which the
    caller normally leaves at their defaults but must never be able to make assess raise."""
    v = _finite(value)
    return fallback if v is None else v


def _int_or(value, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _vec3(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def _clamp01(value) -> float:
    v = _finite(value)
    if v is None:
        return 0.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _clip_frac(stats: Dict) -> float:
    """``hi_clip_frac`` as a float, mirroring solver.py:108-109's ``float(x or 0.0)`` (a
    None/0 share reads as 0.0), hardened so a junk field can't raise."""
    try:
        return float(stats.get("hi_clip_frac", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _b_star(stats: Dict, hi_key: str, hi_both: bool) -> Optional[float]:
    """LAB b* of the chosen source, mirroring solver.py's ``b_of``: the highlight mean when
    BOTH sides carry a well-formed one, else the full-frame mean; None when the chosen source
    is unusable (comparing a highlight mean against a full mean would mix two quantities)."""
    src = stats.get(hi_key) if hi_both else stats.get("lab_mean")
    if not _vec3(src):
        return None
    return _finite(src[2])


def _cget(comps: Dict, key: str) -> float:
    """Component sub-score, defaulting to 1.0 when absent or junk — mirrors critic.py's
    ``clean.get(key, 1.0)`` (a missing axis carries no evidence of a gap on that axis),
    hardened so a mistyped field never raises mid-assessment."""
    v = _finite(comps.get(key, 1.0))
    return 1.0 if v is None else v


def _derived_low_score(comps: Dict) -> bool:
    """Proxy for the critic's ``score < 95`` ceiling clause when assess sees no raw score:
    the mean of the observable component sub-scores stands in for the weighted critic score
    (an equal-weight approximation, deliberately import-free so fairness stays standalone).
    Degrades to False when no usable component is present, per the frozen rule that the
    ceiling clause degrades to False when a score signal isn't available."""
    vals = [x for x in (_finite(v) for v in comps.values()) if x is not None]
    if not vals:
        return False
    return (sum(vals) / len(vals)) < 0.95


def _unknown(coverage_report: float, n_references: int, note: str) -> Dict:
    """A fully-shaped 'unknown' result (all B3 keys) for missing/degenerate stats. Numeric
    fields are 0.0 (assess's own return type stays float on every path); the back-hemisphere
    caveat is present because it is a property of the solve method, not of the stats."""
    return {
        "verdict": "unknown",
        "constrainable": 0.0,
        "albedo_risk": 0.0,
        "predicted_ev_gap": 0.0,
        "predicted_wb_gap": 0.0,
        "same_scene": False,
        "coverage": coverage_report,
        "n_references": n_references,
        "unreconstructable": [BACK_HEMI_CAVEAT],
        "predicts_leash_trip": False,
        "remedy": "",
        "notes": note,
    }


def assess(
    ref_stats: Optional[Dict],
    cur_stats: Optional[Dict],
    *,
    components: Optional[Dict[str, float]] = None,   # critic Verdict.components (0..1)
    coverage: Optional[float] = None,                # scorecard metric observability
    n_references: int = 1,
    roles: Optional[List[str]] = None,               # accepted for the caller's convenience;
    ceiling_proven: bool = False,                    #   Route A: role labels do not tighten
    content_gap: Optional[bool] = None,              #   the solve, so they do not affect the
    leash_hits: Optional[int] = None,                #   assessment (reported via n_references)
    ev_leash: float = 4.0,                           # == MatchConfig.ev_leash
    wb_leash: float = 3000.0,                        # == MatchConfig.wb_leash
    wb_kelvin_per_b: float = 90.0,                   # == solver.WB_KELVIN_PER_B
    same_scene_clip_frac: float = 0.15,              # == solver.SYMMETRIC_CLIP_FRAC
) -> Dict:
    """Estimate how fairly ``ref_stats`` can constrain the scene behind ``cur_stats``.

    NEVER raises, NEVER returns None. Missing/degenerate stats → a fully-shaped result with
    ``verdict == "unknown"``. When the caller supplies the authoritative director/critic
    signals (``leash_hits``, ``content_gap``) the verdict can only read as-bad-or-worse than
    they do; without them the verdict is scoped to the numbers fairness can see (see notes).
    """
    # coverage: reported clamped (0.0 stays 0.0); as a constrainable factor a 0.0/None reads
    # as "full" (clamp01(coverage or 1.0)), matching the frozen B4 formula exactly.
    cov = _finite(coverage)
    cov_report = 1.0 if cov is None else _clamp01(cov)
    cov_factor = 1.0 if (cov is None or cov == 0.0) else _clamp01(cov)
    n_refs = _int_or(n_references, 1)

    ev_leash_f = _default(ev_leash, 4.0)
    wb_leash_f = _default(wb_leash, 3000.0)
    kpb = _default(wb_kelvin_per_b, 90.0)
    ssc = _default(same_scene_clip_frac, 0.15)

    # --- gate on usable stats: any missing/None input key → "unknown" (never KeyError) ---
    if not isinstance(ref_stats, dict) or not isinstance(cur_stats, dict):
        return _unknown(cov_report, n_refs,
                        "reference/current stats unavailable — fairness undetermined")

    lk_ref, lk_cur = _finite(ref_stats.get("log_key")), _finite(cur_stats.get("log_key"))
    if lk_ref is None or lk_cur is None:
        return _unknown(cov_report, n_refs,
                        "exposure key (log_key) missing on a side — fairness undetermined")
    # verbatim from director.py:253-255 — the same log2 key ratio with the same 1e-5 floor
    predicted_ev_gap = abs(math.log2(max(1e-5, lk_ref) / max(1e-5, lk_cur)))

    # same-scene regime: BOTH frames clip a large share of the highlight quartile (a relight
    # / hero shot) — identical predicate to solver.py:108-109
    same_scene = (_clip_frac(ref_stats) >= ssc and _clip_frac(cur_stats) >= ssc)

    # highlight-b* source: the same lab_mean_hi_full ← same_scene switch the solver uses
    # (solver.py:107-113), with the lab_mean fallback when the highlight mean is absent on
    # either side. The draft's fixed lab_mean_hi was wrong precisely in the same_scene regime.
    hi_key = "lab_mean_hi_full" if same_scene else "lab_mean_hi"
    hi_both = _vec3(ref_stats.get(hi_key)) and _vec3(cur_stats.get(hi_key))
    b_ref = _b_star(ref_stats, hi_key, hi_both)
    b_cur = _b_star(cur_stats, hi_key, hi_both)
    if b_ref is None or b_cur is None:
        return _unknown(cov_report, n_refs,
                        "highlight/mean chromaticity missing on a side — fairness undetermined")
    predicted_wb_gap = abs(b_ref - b_cur) * kpb

    # --- deterministic content-gap condition (same clauses as critic.py:79-82) ---
    comps_present = isinstance(components, dict)
    content_gap_condition = bool(
        (content_gap is True)
        # 1st critic clause (ceiling): degrades to False without a score signal
        or (ceiling_proven and comps_present and _derived_low_score(components))
        # 2nd critic clause: light axes strong but tonal shape weak → content, not lighting
        or (comps_present
            and min(_cget(components, "key"), _cget(components, "color"),
                    _cget(components, "direction")) >= 0.78
            and min(_cget(components, "envelope"),
                    _cget(components, "histogram")) < 0.62)
    )

    # --- risk (no unspecified boosts; the content-gap floor guarantees >= "marginal") ---
    base = (0.5 * min(1.0, predicted_ev_gap / max(1e-9, ev_leash_f))
            + 0.5 * min(1.0, predicted_wb_gap / max(1e-9, wb_leash_f)))
    if content_gap_condition:
        base = max(base, 0.50)                       # deterministic floor
    albedo_risk = 0.0 if same_scene else _clamp01(base)

    # same_scene EXEMPTION also gates the leash prediction (explicit precedence)
    predicts_leash_trip = (not same_scene) and (
        predicted_ev_gap >= ev_leash_f or predicted_wb_gap >= wb_leash_f)

    # --- authoritative override: fairness can be as-bad-or-worse than the director/critic,
    # NEVER better — this is what makes "never contradicts" literally hold ---
    director_unfair = (leash_hits is not None and _int_or(leash_hits, 0) >= 2)
    if director_unfair or predicts_leash_trip or albedo_risk >= 0.66:
        verdict = "unfair"
    elif albedo_risk >= 0.33 or content_gap_condition:
        verdict = "marginal"
    else:
        verdict = "fair"                             # same_scene w/ no overriding signal

    # ROUTE-A HONESTY: n_references does NOT raise constrainable (extra views never reach the
    # solve — D6/D7). Only the albedo risk and metric observability bound it.
    constrainable = _clamp01((1.0 - albedo_risk) * cov_factor)

    # single-view caveat ALWAYS; the escalating caveats are honestly gated
    unreconstructable = [BACK_HEMI_CAVEAT]
    if not same_scene:
        unreconstructable.append("off-frame key-source direction is inferred, not observed")
    if not same_scene and albedo_risk >= 0.5:
        unreconstructable.append("reference albedo/material family differs from the scene")

    remedy = REMEDY_LOCK if verdict == "unfair" else ""

    # notes document the guarantee SCOPE (B8): full when the authoritative signals are fed,
    # reduced (numbers-only) in the predictive pre-match path
    fragments: List[str] = []
    if leash_hits is not None and content_gap is not None:
        fragments.append("aligned with the director leash and critic content-gap signals "
                         "(cannot read less severe than either)")
    else:
        fragments.append("predictive estimate from stats only — without the director "
                         "leash_hits / critic content_gap signals this is consistent with "
                         "the numbers fairness can see, not the genome bounds")
    if director_unfair:
        fragments.append("director hit its exposure/WB leash twice — authoritative unfair")
    if same_scene:
        fragments.append("same-scene regime (both frames clip the highlight quartile) — "
                         "albedo risk waived")
    if n_refs > 1:
        fragments.append(f"{n_refs} reference views reported for transparency; under Route A "
                         "they denoise the primary ANALYZE but do not tighten the solve, so "
                         "constrainability is unchanged")

    return {
        "verdict": verdict,
        "constrainable": constrainable,
        "albedo_risk": albedo_risk,
        "predicted_ev_gap": predicted_ev_gap,
        "predicted_wb_gap": predicted_wb_gap,
        "same_scene": same_scene,
        "coverage": cov_report,
        "n_references": n_refs,
        "unreconstructable": unreconstructable,
        "predicts_leash_trip": predicts_leash_trip,
        "remedy": remedy,
        "notes": "; ".join(fragments),
    }
