"""Global sun-angle solve — find the light's direction instead of searching for it.

Sun azimuth was the plugin's worst failure and it was being attacked with local search:
a first guess, a 1-D azimuth sweep, then coordinate descent. Measured on-box across four
runs of ONE golden-hour interior, that returned errors of 2.8, 171, 168 and 78 degrees.
The problem was never the search strategy. It was that nothing could TELL: the critic's
only spatial descriptor was an averaged luminance grid, which returned 0.922 for a sun 171
degrees out and 0.917 for one 13.5 degrees out — the same number for an answer that works
and one that does not. A local search cannot climb a gradient that is not there.

`metrics.highlight_similarity` changed that. It compares where the locally-contrasty bright
patches are, which is what a sun patch physically is, and it separates those same states
0.912 against 0.546. With a discriminating measure in hand the honest move is not a better
hill-climb — it is to stop hill-climbing. Sun direction is TWO bounded parameters, azimuth
and altitude, and a coarse grid over both costs a few dozen small renders. That is a global
solve: it cannot land in a local optimum, because it looks everywhere first.

Deliberately scored on patch agreement ALONE, not the weighted critic. This stage answers
"where is the light", and letting exposure and white balance into that vote is exactly how
the search kept buying a wrong sun with a bright dome and a warm camera. Tone is the loop's
job, afterwards.

Sans-IO: the caller supplies apply/render/stats, so the whole solve is unit-tested off-Max.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from .genome import LightingState
from .metrics import highlight_similarity

#: Coarse grid. 12 azimuths is 30-degree resolution — half a step is 15 degrees, which is
#: inside a lighting artist's own margin and is where the fine pass takes over. Three
#: altitudes bracket the plausible range for a sun that is visibly shaping a frame.
COARSE_AZIMUTHS = 12
COARSE_ALTITUDES = (8.0, 26.0, 52.0)

#: Fine pass: half a coarse step either side of the winner, so the two passes tile the
#: space without a gap. Resolution after refinement is ~7 degrees of azimuth.
FINE_AZIMUTH_OFFSETS = (-15.0, -7.0, 7.0, 15.0)
FINE_ALTITUDE_OFFSETS = (-12.0, -5.0, 5.0, 12.0)

#: A solve is only worth trusting if the winner actually stood out. Same idea as the
#: sweep's margin: a flat landscape means every direction lights this scene alike, and the
#: honest answer is to say so rather than to report the arbitrary winner as a measurement.
DECISIVE_MARGIN = 0.06


def _clamp_altitude(value: float) -> float:
    return max(-8.0, min(88.0, float(value)))


def solve_sun_angles(
    state: LightingState,
    ref_stats: Dict,
    apply: Callable[[LightingState], None],
    render: Callable[[str], Optional[str]],
    stats: Callable[[str], Optional[Dict]],
    log: Callable[[str], None] = lambda _m: None,
    should_cancel: Callable[[], bool] = lambda: False,
    max_probes: int = 56,
    locks: Optional[set] = None,
) -> Optional[Dict]:
    """Grid-search sun azimuth and altitude on patch agreement alone.

    → ``{"azimuth_deg", "altitude_deg", "score", "margin", "confidence", "probes",
    "table"}`` or None when the scene cannot answer (no sun axis, no patches anywhere, a
    dead renderer, or the caller cancelled).

    ``confidence`` is how far the winner led the runner-up, normalised on
    ``DECISIVE_MARGIN``. A low value is a real result and must be reported, not hidden: it
    means every direction lights this scene about equally, and the caller should hold the
    answer loosely rather than defend it."""
    locks = set(locks or ())
    if "sun.azimuth_deg" not in state.values or "sun.azimuth_deg" in locks:
        return None                     # no sun axis to solve, or the artist already knows
    if not isinstance(ref_stats, dict) or "hot_frac" not in ref_stats:
        return None                     # legacy stats without the patch map
    if float(ref_stats.get("hot_frac") or 0.0) <= 0.0:
        # The reference carries no directional patch at all — overcast, night, a flat
        # interior. There is genuinely nothing here to solve toward, and grinding 50
        # renders to discover that is worse than saying so.
        log("sun solve: the reference has no directional light to place — skipped")
        return None

    has_altitude = ("sun.altitude_deg" in state.values
                    and "sun.altitude_deg" not in locks)
    probes = 0
    table: List[Tuple[float, float, float]] = []

    def probe(azimuth: float, altitude: Optional[float], tag: str) -> Optional[float]:
        nonlocal probes
        if probes >= max_probes or should_cancel():
            return None
        cand = state.copy()
        cand.set("sun.azimuth_deg", float(azimuth) % 360.0)
        if altitude is not None and has_altitude:
            cand.set("sun.altitude_deg", _clamp_altitude(altitude))
        apply(cand)
        path = render(tag)
        if path is None:
            return None
        cur = stats(path)
        if cur is None:
            return None
        probes += 1
        value = highlight_similarity(ref_stats, cur)
        if value is None:
            return None
        table.append((float(azimuth) % 360.0,
                      float(altitude) if altitude is not None else -999.0, value))
        return value

    # ---- coarse pass over the whole sphere of plausible sun positions
    altitudes = COARSE_ALTITUDES if has_altitude else (None,)
    best: Optional[Tuple[float, Optional[float], float]] = None
    for i in range(COARSE_AZIMUTHS):
        az = i * 360.0 / COARSE_AZIMUTHS
        for alt in altitudes:
            got = probe(az, alt, f"sunsolve_a{int(az):03d}"
                                 + (f"_e{int(alt):02d}" if alt is not None else ""))
            if got is None:
                continue
            if best is None or got > best[2]:
                best = (az, alt, got)
    if best is None:
        log("sun solve: no probe could be measured — leaving the sun where it was")
        return None

    # ---- fine pass around the winner
    az0, alt0, score0 = best
    for d_az in FINE_AZIMUTH_OFFSETS:
        got = probe(az0 + d_az, alt0, f"sunsolve_fine_a{int(az0 + d_az) % 360:03d}")
        if got is not None and got > best[2]:
            best = ((az0 + d_az) % 360.0, alt0, got)
    if has_altitude:
        az1, alt1 = best[0], best[1] if best[1] is not None else alt0
        for d_alt in FINE_ALTITUDE_OFFSETS:
            got = probe(az1, (alt1 or 0.0) + d_alt,
                        f"sunsolve_fine_e{int((alt1 or 0.0) + d_alt):02d}")
            if got is not None and got > best[2]:
                best = (az1, _clamp_altitude((alt1 or 0.0) + d_alt), got)

    # ---- how decisive was it? A flat table is a finding, not a failure.
    rivals = [v for az, alt, v in table
              if abs((az - best[0] + 180.0) % 360.0 - 180.0) > 25.0]
    margin = best[2] - max(rivals) if rivals else best[2]
    confidence = max(0.0, min(1.0, margin / DECISIVE_MARGIN))
    result = {
        "azimuth_deg": round(best[0], 2),
        "altitude_deg": (round(best[1], 2) if best[1] is not None else None),
        "score": round(best[2], 4),
        "margin": round(margin, 4),
        "confidence": round(confidence, 3),
        "probes": probes,
        "table": [(round(a, 1), round(e, 1), round(v, 4)) for a, e, v in table],
    }
    if confidence < 0.5:
        log(f"sun solve: azimuth {result['azimuth_deg']:.0f}° leads by only "
            f"{margin:.3f} — every direction lights this scene about equally, so the "
            f"answer is held loosely (lock sun.azimuth_deg if you know better)")
    else:
        log(f"sun solve: azimuth {result['azimuth_deg']:.0f}°"
            + (f", altitude {result['altitude_deg']:.0f}°"
               if result["altitude_deg"] is not None else "")
            + f" — patch agreement {best[2]:.3f} over {probes} probes "
              f"(confidence {confidence:.0%})")
    return result
