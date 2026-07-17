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

from dataclasses import dataclass, field
from typing import Dict

from .metrics import cosine, hist_emd

DEFAULT_WEIGHTS: Dict[str, float] = {
    "key": 0.19,
    "envelope": 0.15,
    "histogram": 0.17,
    "color": 0.21,
    "hue": 0.13,
    "direction": 0.15,   # 3×3 luminance-grid cosine — WHERE the light lives
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


def score(ref: Dict, cur: Dict, weights: Dict[str, float] = None) -> Verdict:
    import math

    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    # every component joins ONLY when its source data is present on both sides —
    # absent data must never default to a perfect sub-score (a present-but-empty
    # stats dict once scored a false 100.0 and stopped a match as target_reached)
    comps: Dict[str, float] = {}

    if "log_key" in ref and "log_key" in cur:
        key_ref = max(1e-5, float(ref["log_key"]))
        key_cur = max(1e-5, float(cur["log_key"]))
        comps["key"] = 1.0 - min(1.0, abs(math.log2(key_ref / key_cur)) / 3.0)

    if isinstance(ref.get("p"), dict) and isinstance(cur.get("p"), dict):
        d5 = abs(_sub(ref, "p", "5") - _sub(cur, "p", "5"))
        d95 = abs(_sub(ref, "p", "95") - _sub(cur, "p", "95"))
        comps["envelope"] = 1.0 - min(1.0, (d5 + d95) / 0.5)

    if ref.get("lum_hist") and cur.get("lum_hist"):
        comps["histogram"] = 1.0 - min(
            1.0, hist_emd(ref["lum_hist"], cur["lum_hist"]) * 4.0)

    lr, lc = ref.get("lab_mean"), cur.get("lab_mean")
    if (isinstance(lr, (list, tuple)) and len(lr) >= 3
            and isinstance(lc, (list, tuple)) and len(lc) >= 3):
        d_col = math.sqrt(0.4 * (lr[0] - lc[0]) ** 2 + (lr[1] - lc[1]) ** 2
                          + (lr[2] - lc[2]) ** 2)
        comps["color"] = 1.0 - min(1.0, d_col / 30.0)

    hr, hc = ref.get("hue_hist") or [], cur.get("hue_hist") or []
    if (hr or hc) and (any(v > 1e-9 for v in hr) or any(v > 1e-9 for v in hc)):
        comps["hue"] = max(0.0, cosine(hr, hc))
    # prefer the finer 5×5 grid when both sides carry it (better azimuth acuity);
    # 3×3 remains for stats produced by older engine versions
    g_ref, g_cur = ref.get("grid5"), cur.get("grid5")
    if not (g_ref and g_cur and len(g_ref) == len(g_cur or [])):
        g_ref, g_cur = ref.get("grid"), cur.get("grid")
    if g_ref and g_cur and len(g_ref) == len(g_cur) and any(abs(v) > 1e-6 for v in g_ref):
        comps["direction"] = max(0.0, (cosine(g_ref, g_cur) + 1.0) / 2.0)
    # only weigh what was measurable — old stats without a grid renormalize cleanly;
    # NOTHING measurable = unmeasurable comparison, scored 0 (never a false perfect)
    if not comps:
        return Verdict(score=0.0, components={})
    total_w = sum(w[k] for k in comps) or 1.0
    total = sum(w[k] * comps[k] for k in comps) / total_w
    return Verdict(score=round(100.0 * total, 2), components=comps)
