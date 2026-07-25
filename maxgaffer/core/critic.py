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

from .metrics import (cosine, highlight_similarity, hist_emd,
                      illuminant_similarity)

#: How hard the aggregate is pulled toward its WORST component. 0 = the old plain
#: weighted mean; 1 = the score IS the worst component. 0.35 leaves the mean in charge of
#: ordinary trade-offs while making a single collapsed dimension impossible to average
#: away — measured need: a sunless impersonation scored ~92, and a hue collapse to 0.06
#: still scored 79.
_WEAKEST_LINK = 0.35

# Weight moved from "direction" to "highlight" (2026-07-25). Both ask where the light is;
# only one of them can answer. Measured on-box on a golden-hour interior, "direction"
# returned 0.922 for a sun 171 degrees out and 0.917 for one 13.5 degrees out — the same
# number for a match that works and one that does not. It keeps a reduced share because it
# still discriminates on scenes with broad tonal gradients (0.998 on a 5-degree miss in the
# sun+sky archetype); it just cannot see a patch.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "key": 0.18,
    "envelope": 0.14,
    "histogram": 0.16,
    "color": 0.20,
    "hue": 0.12,
    "direction": 0.08,   # 3×3/5×5 luminance-grid cosine — broad gradients only
    "highlight": 0.12,   # sun-patch presence and placement — the directional light itself
}

# Report-only note the scorecard fairness fallback carries when a content gap is suspected.
# Kept distinct from the director's "white room vs dark wood" remedy so the two never read as
# competing verdicts (the remedy belongs to fairness.assess / the director diagnosis).
_ALBEDO_NOTE = ("reference and scene may differ in albedo/material distribution, "
                "not lighting alone")

PREFERENCE_PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": dict(DEFAULT_WEIGHTS),
    # "direction" the PROFILE means "I care where the light is" — so most of its emphasis
    # belongs on the component that can actually tell, not the one that reported the same
    # number for a 13-degree and a 171-degree miss
    "direction": {"key": 0.12, "envelope": 0.12, "histogram": 0.11,
                  "color": 0.14, "hue": 0.09, "direction": 0.14, "highlight": 0.28},
    "color_mood": {"key": 0.12, "envelope": 0.09, "histogram": 0.10,
                   "color": 0.32, "hue": 0.21, "direction": 0.07, "highlight": 0.09},
    "tonal": {"key": 0.28, "envelope": 0.23, "histogram": 0.23,
              "color": 0.09, "hue": 0.05, "direction": 0.05, "highlight": 0.07},
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
              ceiling_proven: bool = False, ceiling_converged: bool = False,
              fairness: Dict = None) -> Dict:
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
    # A full fairness.assess() result rides through verbatim; when absent we synthesize a
    # self-contained fallback carrying the SAME keys assess() returns, so the dock's
    # per-field reads never KeyError on older/ungraded runs. Opaque data downstream.
    if isinstance(fairness, dict):
        fairness_card = fairness
    else:
        fairness_card = {
            "verdict": "unknown",
            "constrainable": None,
            "albedo_risk": 0.0,
            "same_scene": False,
            "predicted_ev_gap": 0.0,
            "predicted_wb_gap": 0.0,
            "unreconstructable": [],
            "predicts_leash_trip": False,
            "remedy": "",
            "albedo_suspect": bool(content_gap),
            "notes": (_ALBEDO_NOTE if content_gap else ""),
        }
    return {
        "score": score_value,
        "components": clean,
        "coverage": round(coverage, 3),
        "confidence": confidence,
        "strengths": strengths,
        "weakest": weakest,
        "likely_gap": likely,
        "content_gap": content_gap,
        "fairness": fairness_card,
        "ceiling": "proven" if ceiling_proven else
                   "plateau" if ceiling_converged else "not_tested",
        "disclaimer": ("This measures tonal, color, and direction statistics; it is not "
                       "a guarantee of artistic or semantic visual equivalence."),
    }


@dataclass
class Verdict:
    score: float                      # 0..100
    components: Dict[str, float] = field(default_factory=dict)   # each 0..1
    # diagnostic ONLY (e.g. {"illuminant_match": …}); NEVER enters the score. Populating it
    # diverges two otherwise-equal Verdicts, so Verdict must never be compared by == / asdict.
    fairness: Dict = field(default_factory=dict)

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
    # SUN PATCHES — is the reference's bright directional light present, and in the same
    # places? "direction" above cannot answer this: its grids are AVERAGED cell luminances,
    # and averaging is exactly what erases a patch, which is small and very bright.
    # Measured on-box 2026-07-25 on one golden-hour interior it returned 0.922 for a sun
    # 171 degrees out and 0.917 for one 13.5 degrees out — the same number for a match that
    # works and one that does not, i.e. no information at all. The pair it called 0.92 was
    # a reference covered in golden floor patches against a match with no directional light
    # anywhere in the room.
    #
    # Absent only when the STATS lack the map (older engine versions) — never because of
    # anything the search can change. A rig with no patches against a reference full of
    # them scores 0.0 here, and the weakest-link term below turns that into a 43-point
    # penalty on an otherwise tonally perfect answer.
    hi = highlight_similarity(ref, cur)
    if hi is not None:
        comps["highlight"] = hi
    # diagnostic ONLY — the illuminant estimate is absent from DEFAULT_WEIGHTS and from
    # comps, so it NEVER enters the weighted score; None on legacy stats without illum_*
    fairness_diag = {"illuminant_match": illuminant_similarity(ref, cur)}
    # only weigh what was measurable — old stats without a grid renormalize cleanly;
    # NOTHING measurable = unmeasurable comparison, scored 0 (never a false perfect)
    if not comps:
        return Verdict(score=0.0, components={}, fairness=fairness_diag)
    total_w = sum(w[k] for k in comps) or 1.0
    total = sum(w[k] * comps[k] for k in comps) / total_w
    # WEAKEST-LINK PENALTY. A weighted mean lets one catastrophically wrong dimension hide
    # behind five good ones, and the search exploits exactly that: measured on-box
    # 2026-07-25, a SUNLESS room impersonating a golden-hour reference scored ~92 because
    # a 150-degree sun error costs "direction" alone, and a 400-lamp city scored 79 with
    # hue collapsed to 0.06. Both are structurally wrong answers wearing a good average.
    #
    # So the aggregate is pulled toward its worst measured component. A CORRECT match is
    # unaffected — every component sits near 1.0, so the gap term vanishes and a perfect
    # answer still scores 100 — but a lopsided one can no longer average its failure away.
    # This makes the score mean "no dimension is badly wrong", which is what a lighting
    # match has to guarantee.
    if len(comps) >= 3:
        worst = min(comps.values())
        total -= _WEAKEST_LINK * max(0.0, total - worst)
    return Verdict(score=round(100.0 * max(0.0, total), 2), components=comps,
                   fairness=fairness_diag)
