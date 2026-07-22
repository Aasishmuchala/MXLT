"""Tonal critic — a deterministic 0-100 "how close is the lighting mood" score.

Reference and render are DIFFERENT SCENES, so no SSIM/feature matching: the score is built
only from statistics that transfer across scenes — exposure key, tonal envelope, chromatic
mood. It is the loop's accept/revert arbiter and convergence signal, not a beauty judge; the
LLM (and the human) own the last mile of taste, exactly like MaxDirector's geometric critic
gates its storyboards.

Components (weights config-tunable; renormalized over whatever was measurable):
  key       exposure match — log2 distance between geometric-mean linear luminances
  envelope  shadow/highlight placement — p5 + p95 luminance deltas
  histogram luminance distribution shape — 1-D EMD
  color     chromatic mood — LAB mean distance (a*, b* weighted over L)
  hue       hue distribution — chroma-weighted cosine similarity
  direction WHERE the light lives — cosine of mean-centered 3×3 luminance grids (the one
            spatial signal that transfers across different scenes lit the same way)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from .metrics import cosine, hist_emd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "key": 0.19,
    "envelope": 0.15,
    "histogram": 0.17,
    "color": 0.21,
    "hue": 0.13,
    "direction": 0.15,   # 3×3 luminance-grid cosine — WHERE the light lives
}

PREFERENCE_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": dict(DEFAULT_WEIGHTS),
    "direction": {"key": 0.14, "envelope": 0.14, "histogram": 0.13,
                  "color": 0.16, "hue": 0.10, "direction": 0.33},
    "color_mood": {"key": 0.13, "envelope": 0.10, "histogram": 0.11,
                   "color": 0.34, "hue": 0.22, "direction": 0.10},
    "tonal": {"key": 0.29, "envelope": 0.24, "histogram": 0.24,
              "color": 0.10, "hue": 0.05, "direction": 0.08},
}


def weights_for(preference: str = "balanced", overrides: Dict[str, float] = None) -> Dict[str, float]:
    """Artist-facing taste profile plus optional expert per-component overrides."""
    weights = dict(PREFERENCE_PROFILES.get(str(preference or "balanced"), DEFAULT_WEIGHTS))
    for key, value in (overrides or {}).items():
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if key in weights and math.isfinite(value) and value >= 0:
            weights[key] = value
    return weights


def scorecard(score_value: float, components: Dict[str, float], *,
              ceiling_proven: bool = False, ceiling_converged: bool = False) -> Dict:
    """Honest human-readable interpretation of the proxy score and its likely gap."""
    clean = {key: max(0.0, min(1.0, _num(value)))
             for key, value in (components or {}).items() if key in DEFAULT_WEIGHTS}
    measured_weight = sum(DEFAULT_WEIGHTS[key] for key in clean)
    total_weight = sum(DEFAULT_WEIGHTS.values()) or 1.0
    coverage = measured_weight / total_weight
    ordered = sorted(clean.items(), key=lambda item: item[1])
    weakest = [key for key, value in ordered if value < 0.72][:3]
    strengths = [key for key, value in reversed(ordered) if value >= 0.85][:3]
    likely = []
    if clean.get("direction", 1.0) < 0.65:
        likely.append("light direction/shadow placement")
    if min(clean.get("color", 1.0), clean.get("hue", 1.0)) < 0.65:
        likely.append("white balance or source color")
    if clean.get("key", 1.0) < 0.65:
        likely.append("exposure/key level")
    tonal_shape = min(clean.get("envelope", 1.0), clean.get("histogram", 1.0))
    light_axes = min(clean.get("key", 1.0), clean.get("color", 1.0),
                     clean.get("direction", 1.0))
    content_gap = bool((ceiling_proven and (score_value or 0) < 95.0)
                       or (light_axes >= 0.78 and tonal_shape < 0.62))
    if content_gap:
        likely.append("scene content/albedo/material distribution—not lighting alone")
    confidence = "high" if coverage >= 0.90 else "medium" if coverage >= 0.65 else "low"
    return {
        "score": score_value,
        "components": clean,
        "coverage": round(coverage, 3),
        "confidence": confidence,
        "strengths": strengths,
        "weakest": weakest,
        "likely_gap": likely,
        "content_gap": content_gap,
        "ceiling": "proven" if ceiling_proven else
                   "plateau" if ceiling_converged else "not_tested",
        "disclaimer": ("This measures tonal, color, and direction statistics; it is not "
                       "a guarantee of artistic or semantic visual equivalence."),
    }


@dataclass
class Verdict:
    score: float                      # 0..100
    components: Dict[str, float] = field(default_factory=dict)   # each 0..1

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in sorted(self.components.items()))
        return f"{self.score:.1f}/100 ({parts})"


def _sub(d: Dict, *path, default=0.0):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _num(value, default: float = 0.0) -> float:
    """Finite-float coercion for stats fields — stats cross the sidecar trust boundary
    unvalidated, so a present-but-mistyped field (None, "junk", NaN) must degrade to the
    default, not raise on Max's main thread mid-loop."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _finite(value):
    """Finite float or None — for component gates, where a junk field must read as
    ABSENT (component skipped + renormalized), never as a fabricated measurement."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _seq(value) -> List[float]:
    """Numeric-list coercion (histograms, grids): anything else → empty list."""
    if not isinstance(value, (list, tuple)):
        return []
    return [_num(v) for v in value]


def _lab(value) -> List[float]:
    """LAB mean coercion: exactly-3 numeric, else the neutral default."""
    if not (isinstance(value, (list, tuple)) and len(value) == 3):
        return [0.0, 0.0, 0.0]
    return [_num(v) for v in value]


def score(ref: Dict, cur: Dict, weights: Dict[str, float] = None) -> Verdict:
    w = dict(DEFAULT_WEIGHTS)
    if isinstance(weights, dict):
        # config.json is user-editable — accept only known keys with finite, non-negative
        # floats (a NaN weight makes every accept/revert comparison silently False);
        # anything else keeps the default weight

        for k, v in weights.items():
            if k not in w:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(fv) and fv >= 0.0:
                w[k] = fv

    # every component joins ONLY when its source data is present AND coercible on both
    # sides — absent data must never default to a perfect sub-score (a present-but-empty
    # stats dict once scored a false 100.0 and stopped a match as target_reached), and a
    # present-but-mistyped field reads as UNMEASURED, never raising mid-loop
    comps: Dict[str, float] = {}

    key_ref, key_cur = _finite(ref.get("log_key")), _finite(cur.get("log_key"))
    if key_ref is not None and key_cur is not None:
        comps["key"] = 1.0 - min(1.0, abs(
            math.log2(max(1e-5, key_ref) / max(1e-5, key_cur))) / 3.0)

    if isinstance(ref.get("p"), dict) and isinstance(cur.get("p"), dict):
        p_vals = (_finite(_sub(ref, "p", "5")), _finite(_sub(cur, "p", "5")),
                  _finite(_sub(ref, "p", "95")), _finite(_sub(cur, "p", "95")))
        if None not in p_vals:
            comps["envelope"] = 1.0 - min(
                1.0, (abs(p_vals[0] - p_vals[1]) + abs(p_vals[2] - p_vals[3])) / 0.5)

    hist_ref, hist_cur = _seq(ref.get("lum_hist")), _seq(cur.get("lum_hist"))
    if hist_ref and hist_cur:
        comps["histogram"] = 1.0 - min(1.0, hist_emd(hist_ref, hist_cur) * 4.0)

    lr, lc = ref.get("lab_mean"), cur.get("lab_mean")
    if (isinstance(lr, (list, tuple)) and len(lr) == 3
            and isinstance(lc, (list, tuple)) and len(lc) == 3):
        lab_vals = [_finite(x) for x in (*lr, *lc)]
        if None not in lab_vals:
            d_col = math.sqrt(0.4 * (lab_vals[0] - lab_vals[3]) ** 2
                              + (lab_vals[1] - lab_vals[4]) ** 2
                              + (lab_vals[2] - lab_vals[5]) ** 2)
            comps["color"] = 1.0 - min(1.0, d_col / 30.0)

    # hue joins when EITHER side is chromatic: a grey render of a colorful reference is
    # real signal (cosine 0 — penalized toward color); two achromatic images carry no
    # hue information and would inflate exactly the degenerate pairs — skipped
    hue_ref, hue_cur = _seq(ref.get("hue_hist")), _seq(cur.get("hue_hist"))
    if ((hue_ref or hue_cur) and (any(abs(v) > 1e-9 for v in hue_ref)
                                  or any(abs(v) > 1e-9 for v in hue_cur))):
        comps["hue"] = max(0.0, cosine(hue_ref, hue_cur))
    # prefer the finer 5×5 grid when both sides carry it (better azimuth acuity);
    # 3×3 remains for stats produced by older engine versions
    g_ref, g_cur = _seq(ref.get("grid5")), _seq(cur.get("grid5"))
    if not (g_ref and g_cur and len(g_ref) == len(g_cur)):
        g_ref, g_cur = _seq(ref.get("grid")), _seq(cur.get("grid"))
    if g_ref and g_cur and len(g_ref) == len(g_cur) and any(abs(v) > 1e-6 for v in g_ref):
        comps["direction"] = max(0.0, (cosine(g_ref, g_cur) + 1.0) / 2.0)
    # only weigh what was measurable — old stats without a grid renormalize cleanly;
    # NOTHING measurable = unmeasurable comparison, scored 0 (never a false perfect)
    if not comps:
        return Verdict(score=0.0, components={})
    total_w = sum(w[k] for k in comps) or 1.0
    total = sum(w[k] * comps[k] for k in comps) / total_w
    return Verdict(score=round(100.0 * total, 2), components=comps)
