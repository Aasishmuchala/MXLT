"""Software exposure — apply EV + white-balance as a known transform, so the match
loop never depends on the RENDERER applying them.

Why this exists: V-Ray GPU applies camera/exposure-control exposure only at the VFB
display stage — ``render()`` hands back the pre-exposure buffer, so on-box loop
renders don't reflect the EV/WB the solver sets (measured 2026-07-18: EV moved 5
stops, saved-render key moved < 2e-5). Exposure is a pure operation, though: a
linear-light multiply for EV, a chromatic-adaptation gain for WB. So we apply it in
software to the rendered frame before the critic scores it. The analytic solver then
converges EV/WB with no re-render, on any renderer.

``base_ev``/``base_wb`` are the exposure the raw render corresponds to (captured at
match start): at ev == base_ev, wb == base_wb the transform is the identity.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .metrics import _srgb_to_linear


def _linear_to_srgb(c: float) -> float:
    c = 0.0 if c < 0.0 else c
    return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _blackbody_rgb(kelvin: float) -> Tuple[float, float, float]:
    """Approx sRGB color of a ``kelvin`` blackbody (Tanner Helland fit), normalized so
    green == 1. Redder as K falls, bluer as K rises."""
    t = max(1000.0, min(40000.0, kelvin)) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307
    r = max(1.0, min(255.0, r))
    g = max(1.0, min(255.0, g))
    b = max(1.0, min(255.0, b))
    return r / g, 1.0, b / g


def wb_gain(kelvin: float, base_kelvin: float) -> Tuple[float, float, float]:
    """RGB multipliers that re-white-balance FROM base_kelvin TO kelvin, green-neutral.

    V-Ray WB semantics: a HIGHER kelvin renders WARMER (the camera compensates for a
    bluer assumed illuminant). So the compensation gain is the base blackbody color
    divided by the target's — raising kelvin boosts red, cuts blue."""
    br, _, bb = _blackbody_rgb(base_kelvin)
    tr, _, tb = _blackbody_rgb(kelvin)
    return br / tr, 1.0, bb / tb


def expose_pixels(
    pixels: Sequence[Tuple[int, int, int]],
    ev: float,
    base_ev: float,
    wb_kelvin: float,
    base_wb: float,
) -> List[Tuple[int, int, int]]:
    """Apply EV (linear multiply) + WB (chromatic gain) in linear light. Pure."""
    scale = 2.0 ** (base_ev - ev)          # V-Ray: higher EV = darker
    gr, gg, gb = wb_gain(wb_kelvin, base_wb)
    gr *= scale
    gg *= scale
    gb *= scale
    out: List[Tuple[int, int, int]] = []
    for px in pixels:
        r, g, b = px[0], px[1], px[2]
        lr = _srgb_to_linear(r / 255.0) * gr
        lg = _srgb_to_linear(g / 255.0) * gg
        lb = _srgb_to_linear(b / 255.0) * gb
        out.append((
            max(0, min(255, int(round(_linear_to_srgb(lr) * 255.0)))),
            max(0, min(255, int(round(_linear_to_srgb(lg) * 255.0)))),
            max(0, min(255, int(round(_linear_to_srgb(lb) * 255.0)))),
        ))
    return out


def expose_image_file(src: str, dst: str, ev: float, base_ev: float,
                      wb_kelvin: float, base_wb: float) -> Optional[str]:
    """Load an image, apply software exposure, write it back. Pillow-only (writing an
    8-bit PNG without a dependency isn't worth a hand-rolled encoder here); returns None
    if Pillow is absent so the caller cleanly falls back to the un-exposed frame.

    A near-identity call (ev≈base_ev, wb≈base_wb) is a no-op — skips the rewrite."""
    if abs(ev - base_ev) < 1e-3 and abs(wb_kelvin - base_wb) < 1.0:
        return src
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            exposed = expose_pixels(list(im.getdata()), ev, base_ev,
                                    wb_kelvin, base_wb)
            out = Image.new("RGB", im.size)
            out.putdata(exposed)
            out.save(dst)
        return dst
    except Exception:
        return None
