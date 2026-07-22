"""Lighting-state interpolation and deterministic animation sampling."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

from .genome import GROUP_PREFIX, LightingState, spec_for


def _blend_value(key: str, a: float, b: float, t: float) -> float:
    spec = spec_for(key)
    if spec is not None and spec.wrap:
        delta = (b - a + 180.0) % 360.0 - 180.0
        if delta == -180.0:
            delta = 180.0
        return a + delta * t
    if spec is not None and spec.log_scale and a > 1e-8 and b > 1e-8:
        return 2.0 ** (math.log2(a) + (math.log2(b) - math.log2(a)) * t)
    return a + (b - a) * t


def interpolate(a: LightingState, b: LightingState, t: float,
                easing: str = "smooth") -> LightingState:
    """Blend two states; angles take the short arc and intensities blend in log space."""
    t = min(1.0, max(0.0, float(t)))
    if easing == "smooth":
        t = t * t * (3.0 - 2.0 * t)
    elif easing not in ("linear", "step"):
        raise ValueError("easing must be smooth, linear, or step")
    if easing == "step":
        return (a if t < 1.0 else b).copy()

    out = LightingState()
    for key in sorted(set(a.values) | set(b.values)):
        if key not in a.values:
            out.set(key, b.get(key))
        elif key not in b.values:
            out.set(key, a.get(key))
        else:
            out.set(key, _blend_value(key, a.get(key), b.get(key), t))
    for group in sorted(set(a.groups) | set(b.groups)):
        key = GROUP_PREFIX + group
        av = a.groups.get(group, b.groups.get(group, 1.0))
        bv = b.groups.get(group, av)
        out.set(key, _blend_value(key, av, bv, t))
    return out


def sample_keyframes(keyframes: Sequence[Tuple[int, LightingState]], step: int = 1,
                     easing: str = "smooth") -> List[Tuple[int, LightingState]]:
    """Bake sparse key states into an inclusive, ordered sequence of frame samples."""
    if not keyframes:
        return []
    step = max(1, int(step))
    ordered = sorted(((int(frame), state.copy()) for frame, state in keyframes),
                     key=lambda item: item[0])
    dedup = {}
    for frame, state in ordered:
        dedup[frame] = state                 # last declaration wins at duplicate frames
    ordered = sorted(dedup.items())
    if len(ordered) == 1:
        return ordered
    out: List[Tuple[int, LightingState]] = []
    for index, ((f0, s0), (f1, s1)) in enumerate(zip(ordered, ordered[1:])):
        if f1 <= f0:
            continue
        frames = list(range(f0, f1, step))
        if index > 0 and frames and frames[0] == out[-1][0]:
            frames.pop(0)
        for frame in frames:
            out.append((frame, interpolate(s0, s1, (frame - f0) / (f1 - f0), easing)))
    out.append((ordered[-1][0], ordered[-1][1].copy()))
    return out
