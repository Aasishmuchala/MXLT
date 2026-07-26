"""What the reference IMAGE says about its own lighting — measured, not guessed.

ANALYZE asks a vision model to read a reference photo and report its lighting: whether the
sun is active, which way it bears, how warm it is, how hard its shadows are, how hazy the
air. That is one model looking at one picture and naming absolute quantities, and measured
on-box it is the least reliable component in the plugin — four reads of a SINGLE reference
gave sun bearings of 45.0, -52.5, 77.6 and 64.9 degrees, and on another scene the sign came
back inverted, putting the sun exactly twice its reported bearing away from the truth.

The fix is not a better prompt. Most of what is being guessed is measurable from the pixels:
whether bright directional patches exist at all, what colour the illuminant is, how sharply
shadows terminate, how much the air has lifted the blacks. This module measures what can be
measured and hands the model's reading only the questions that genuinely need judgement —
time of day, sky character, what the picture is OF.

The discipline throughout, learned expensively elsewhere in this codebase: a measurement is
strongest at REFUTING and weakest at asserting. "There is not one bright directional patch
in this frame" is nearly certain evidence against a sunlit reading. "There are bright
patches" is NOT proof of sun — it could be lamps, a bright window, a specular table. So the
measurements veto; they do not overrule wholesale, and every override is logged with what
it saw.

Pure stdlib, sans-IO: callers pass stats dicts from metrics.compute_stats.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from . import cct

#: Share of frame in locally-contrasty bright patches below which a frame carries no
#: directional light worth speaking of. Measured across every reference this session:
#: sunlit plates sat at 0.0251 to 0.0397 (golden-hour interior, a real dawn photograph, a
#: sun+sky exterior) and genuinely sunless ones at exactly 0.0000. The threshold sits far
#: below the lowest true positive on purpose — this signal is used to REFUTE a sunlit
#: reading, so it must never fire on a real sun.
SUN_ACTIVE_HOT_FRAC = 0.004


def measure(stats: Optional[Dict]) -> Dict:
    """→ {field: {"value", "confidence", "why"}} for everything the pixels can answer.

    Only fields with real evidence appear. An empty dict means the image told us nothing,
    which is a fine answer and quite different from telling us something uncertain."""
    out: Dict[str, Dict] = {}
    if not isinstance(stats, dict):
        return out

    # ---- illuminant colour temperature, measured off the pixels
    measured_k = cct.illuminant_kelvin(stats)
    if measured_k and measured_k.get("kelvin"):
        out["wb_kelvin_estimate"] = {
            "value": float(measured_k["kelvin"]),
            "confidence": float(measured_k.get("confidence") or 0.0),
            "why": ("illuminant measured at %.0f K (Duv %+.4f, %s)"
                    % (measured_k["kelvin"], measured_k.get("duv") or 0.0,
                       measured_k.get("source"))),
            # BLENDED, not switched. cct's confidence is built from off-locus distance and
            # estimator disagreement, which is honest but conservative about its own
            # accuracy: measured against ground truth on this session's references it read
            # 4846.7 K against a true 4900 (53 K out) at confidence 0.72, and 5024.7
            # against a true 4800 (225 K out) at confidence 0.25 — while the model's guess
            # for that same image was 5500, a 700 K miss. So even a diffident measurement
            # has outperformed the guess, and a hard threshold would throw the good ones
            # away with the doubtful ones.
            "blend": True,
        }

    hot = stats.get("hot_frac")
    if isinstance(hot, (int, float)):
        lit = float(hot) >= SUN_ACTIVE_HOT_FRAC
        out["sun_active"] = {
            "value": lit,
            # Asymmetric on purpose. An empty frame is strong evidence AGAINST direct sun;
            # a full one is weak evidence FOR it, because lamps, windows and speculars all
            # make bright patches too. The fusion below only lets the strong direction act.
            "confidence": 0.9 if not lit else 0.35,
            "why": ("no bright directional patch anywhere in frame (%.4f of it)" % hot
                    if not lit else
                    "%.1f%% of frame is bright directional patch" % (100.0 * hot)),
        }
    return out


def _blend_field(fused: Dict, field: str, m: Dict,
                 log: Callable[[str], None]) -> None:
    """Weighted average of guess and measurement, rather than one replacing the other.

    The measurement starts at equal weight and grows from there — never below half, because
    on every reference this session where ground truth was known the measurement beat the
    model's guess by a factor of three or more. Colour temperature is averaged in MIREDS,
    the scale on which equal steps look equal: 4000 to 5000 K is an obvious shift and 9000
    to 10000 K is barely visible, so a kelvin average would quietly favour the cool end."""
    was = fused.get(field)
    got = float(m["value"])
    weight = 0.5 + 0.5 * max(0.0, min(1.0, float(m.get("confidence", 0.0))))
    if not isinstance(was, (int, float)):
        fused[field] = got
    elif field.endswith("kelvin_estimate"):
        mired_was, mired_got = 1.0e6 / max(1000.0, float(was)), 1.0e6 / max(1000.0, got)
        fused[field] = 1.0e6 / ((1.0 - weight) * mired_was + weight * mired_got)
    else:
        fused[field] = (1.0 - weight) * float(was) + weight * got
    fused.setdefault("measured_fields", [])
    if field not in fused["measured_fields"]:
        fused["measured_fields"].append(field)
    if isinstance(was, (int, float)) and abs(fused[field] - was) > 1e-6:
        log(f"reference reading: {field} {was:.0f} → {fused[field]:.0f} "
            f"({weight:.0%} measurement) — {m['why']}")


def fuse(reading: Optional[Dict], measured: Optional[Dict],
         log: Callable[[str], None] = lambda _m: None,
         min_confidence: float = 0.75) -> Dict:
    """Fold measurements into the model's reading. Measurement wins where it is confident.

    Returns a NEW dict — the caller's cached reading is the record of what ANALYZE actually
    said and must not be rewritten, the same rule the sun sweep had to learn: laundering a
    measurement into the reading would make a contested read look authoritative to every
    later run.

    ``min_confidence`` is deliberately high. A measurement that overrules the model on thin
    evidence is not an improvement, it is a second guess wearing a lab coat."""
    fused = dict(reading or {})
    for field, m in (measured or {}).items():
        if not isinstance(m, dict) or "value" not in m:
            continue
        if m.get("blend"):
            _blend_field(fused, field, m, log)
            continue
        if float(m.get("confidence", 0.0)) < min_confidence:
            continue
        was = fused.get(field)
        fused[field] = m["value"]
        fused.setdefault("measured_fields", [])
        if field not in fused["measured_fields"]:
            fused["measured_fields"].append(field)
        if was is not None and was != m["value"]:
            log(f"reference reading: {field} measured as {m['value']!r}, not "
                f"{was!r} as read — {m['why']}")
    return fused
