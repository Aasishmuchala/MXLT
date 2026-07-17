"""Dome seed — a reference-derived HDR environment for the dome light.

The parametric loop's honest ceiling is the UNREACHABLE reference: when the stock HDRI's
sky character simply isn't in the reference's family (neon dusk over a daylight dome), no
amount of EV/rotation polish closes the gap. The seed attacks that directly: synthesize an
equirect HDR panorama FROM the reference itself — its colors become the ambient light from
every direction, and a high-energy sun disc is injected at the solved sun position — then
bind it to the dome and let the existing solver/critic refine EV/WB/rotation on top.

This is the same move Chaos's AI Mood Match makes (reference → image-based lighting), built
deterministic and local: no cloud, no model, reproducible to the pixel. For a generative
pano (DiffusionLight-class chrome-ball estimation, LuxDiT, any outpainted 360), pass
``pano_path`` — the synthesis step is skipped and the external pano is ingested, oriented,
and sun-injected through the identical pipeline. That seam is the upgrade path; the
synthesis below is the always-available floor.

Geometry conventions (match genome + rules):
  * pano is WORLD-oriented: column azimuth 0 = +Y north, clockwise, top row = zenith;
    ``dome.rotation_deg`` stays a live genome param and absorbs any constant u-origin
    offset in V-Ray's spherical mapping (measured once on-box, checklist item);
  * the reference occupies a horizontal FOV window centered on the CAMERA's yaw (the
    reference is what the camera sees); outside the window the image mirror-folds, so
    every direction inherits reference color and the ±180° back seam stays continuous
    (fov snaps to a divisor of 180° to guarantee that);
  * vertically the reference spans a typical photo altitude band; above it blends to the
    reference's own sky tone (constant at the zenith pole), below to its ground tone.

Pure python + stdlib. Pixels come from metrics' loader (Pillow if present, png floor);
output goes through hdr_min. Everything is deterministic.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from . import hdr_min, metrics
from .colortemp import kelvin_to_rgb

Rows = List[List[Tuple[float, float, float]]]

# reference vertical span in degrees of altitude — a normal photo sees a slice around the
# horizon, not the zenith; sky/ground extensions take over outside this band
BAND_LO_DEG = -30.0
BAND_HI_DEG = 40.0

# sun-disc color by altitude (illuminant temperature — kelvin_to_rgb gives its own color)
_DISC_KELVIN = ((10.0, 3500.0), (25.0, 4300.0), (90.1, 5300.0))

_SRGB_LUT = [((v / 255.0) / 12.92 if (v / 255.0) <= 0.04045
              else (((v / 255.0) + 0.055) / 1.055) ** 2.4) for v in range(256)]


def _lum(px: Tuple[float, float, float]) -> float:
    return 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def snap_fov(fov_deg: float) -> float:
    """Nearest FOV that divides 180° evenly — the mirror-fold is only continuous at the
    ±180° back seam when the fold period tiles the half-circle exactly."""
    k = max(1, round(180.0 / max(30.0, min(180.0, float(fov_deg)))))
    return 180.0 / k


# --------------------------------------------------------------------------- synthesis
def synthesize_pano(
    pixels: Sequence[Tuple[int, int, int]],
    w: int,
    h: int,
    cam_yaw_deg: float,
    out_w: int = 256,
    out_h: int = 128,
    fov_deg: float = 90.0,
) -> Rows:
    """Reference pixels (8-bit RGB, row-major) → world-oriented equirect linear rows."""
    fov = snap_fov(fov_deg)
    half = fov / 2.0
    lin = [( _SRGB_LUT[p[0]], _SRGB_LUT[p[1]], _SRGB_LUT[p[2]]) for p in pixels]

    def ref_px(u: float, v: float) -> Tuple[float, float, float]:
        ix = min(w - 1, max(0, int(u * (w - 1) + 0.5)))
        iy = min(h - 1, max(0, int(v * (h - 1) + 0.5)))
        return lin[iy * w + ix]

    # sky / ground anchor tones — means of the top and bottom 10% of reference rows
    strip = max(1, h // 10)
    def _strip_mean(y0: int, y1: int) -> Tuple[float, float, float]:
        acc = [0.0, 0.0, 0.0]
        n = 0
        for y in range(y0, y1):
            for x in range(0, w, max(1, w // 64)):
                p = lin[y * w + x]
                for i in range(3):
                    acc[i] += p[i]
                n += 1
        return (acc[0] / n, acc[1] / n, acc[2] / n) if n else (0.5, 0.5, 0.5)
    sky_mean = _strip_mean(0, strip)
    ground_mean = _strip_mean(h - strip, h)

    rows: Rows = []
    for y in range(out_h):
        alt = 90.0 - (y + 0.5) * 180.0 / out_h
        row: List[Tuple[float, float, float]] = []
        for x in range(out_w):
            az = (x + 0.5) * 360.0 / out_w
            d = _wrap180(az - cam_yaw_deg)
            # triangle-fold |d| into the window; sign gives left/right of center
            t = abs(d) % (2.0 * half)
            folded = (2.0 * half - t) if t > half else t
            u = 0.5 + math.copysign(folded, d) / fov
            if alt > BAND_HI_DEG:                      # sky extension → zenith pole
                z = min(1.0, (alt - BAND_HI_DEG) / (90.0 - BAND_HI_DEG))
                base = ref_px(u, 0.0)
                row.append(tuple(base[i] + (sky_mean[i] - base[i]) * z for i in range(3)))
            elif alt < BAND_LO_DEG:                    # ground extension → nadir pole
                z = min(1.0, (BAND_LO_DEG - alt) / (BAND_LO_DEG + 90.0))
                base = ref_px(u, 1.0)
                row.append(tuple(base[i] + (ground_mean[i] - base[i]) * z
                                 for i in range(3)))
            else:                                      # the reference band itself
                v = (BAND_HI_DEG - alt) / (BAND_HI_DEG - BAND_LO_DEG)
                row.append(ref_px(u, v))
        rows.append(row)
    return rows


def ingest_pano(
    pixels: Sequence[Tuple[int, int, int]],
    w: int,
    h: int,
    cam_yaw_deg: float,
    out_w: int = 256,
    out_h: int = 128,
) -> Rows:
    """An EXTERNAL equirect pano (camera-forward at u=0.5, e.g. DiffusionLight output) →
    world-oriented linear rows: resample + rotate so column azimuth 0 lands at north."""
    rows: Rows = []
    for y in range(out_h):
        v = (y + 0.5) / out_h
        iy = min(h - 1, int(v * h))
        row: List[Tuple[float, float, float]] = []
        for x in range(out_w):
            az = (x + 0.5) * 360.0 / out_w
            u = (0.5 + _wrap180(az - cam_yaw_deg) / 360.0) % 1.0
            ix = min(w - 1, int(u * w))
            p = pixels[iy * w + ix]
            row.append((_SRGB_LUT[p[0]], _SRGB_LUT[p[1]], _SRGB_LUT[p[2]]))
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- filtering
def blur_pano(rows: Rows, radius_x: Optional[int] = None,
              radius_y: Optional[int] = None, passes: int = 2) -> Rows:
    """Separable box blur — X wraps (it's a panorama), Y clamps. Two passes ≈ gaussian.
    The goal is ILLUMINATION, not picture: structure melts, color and direction stay."""
    h = len(rows)
    w = len(rows[0]) if h else 0
    if not w:
        return rows
    rx = max(1, w // 32) if radius_x is None else max(0, radius_x)
    ry = max(1, h // 48) if radius_y is None else max(0, radius_y)
    cur = [list(r) for r in rows]
    for _ in range(passes):
        if rx > 0:                                     # horizontal, wrapped
            nxt = [[(0.0, 0.0, 0.0)] * w for _ in range(h)]
            span = 2 * rx + 1
            for y in range(h):
                row = cur[y]
                acc = [0.0, 0.0, 0.0]
                for i in range(-rx, rx + 1):
                    p = row[i % w]
                    for c in range(3):
                        acc[c] += p[c]
                for x in range(w):
                    nxt[y][x] = (acc[0] / span, acc[1] / span, acc[2] / span)
                    out_p = row[(x - rx) % w]
                    in_p = row[(x + rx + 1) % w]
                    acc[0] += in_p[0] - out_p[0]
                    acc[1] += in_p[1] - out_p[1]
                    acc[2] += in_p[2] - out_p[2]
            cur = nxt
        if ry > 0:                                     # vertical, clamped
            nxt = [[(0.0, 0.0, 0.0)] * w for _ in range(h)]
            for x in range(w):
                for y in range(h):
                    acc = [0.0, 0.0, 0.0]
                    n = 0
                    for i in range(max(0, y - ry), min(h, y + ry + 1)):
                        p = cur[i][x]
                        for c in range(3):
                            acc[c] += p[c]
                        n += 1
                    nxt[y][x] = (acc[0] / n, acc[1] / n, acc[2] / n)
            cur = nxt
    return cur


def normalize_key(rows: Rows, target_mean: float = 0.35) -> Tuple[Rows, float]:
    """Scale so mean linear luminance hits ``target_mean`` — dome.intensity then means the
    same thing regardless of how bright the reference photo happened to be. Returns
    (rows, scale_applied)."""
    total = sum(_lum(p) for r in rows for p in r)
    n = sum(len(r) for r in rows)
    mean = total / n if n else 0.0
    if mean <= 1e-6:
        return rows, 1.0
    s = target_mean / mean
    return [[(p[0] * s, p[1] * s, p[2] * s) for p in r] for r in rows], s


# --------------------------------------------------------------------------- sun + sky
def disc_kelvin_for_altitude(alt_deg: float) -> float:
    for hi, k in _DISC_KELVIN:
        if alt_deg < hi:
            return k
    return _DISC_KELVIN[-1][1]


def inject_sun(rows: Rows, az_deg: float, alt_deg: float,
               strength: float = 200.0, size_deg: float = 4.0) -> Dict:
    """Add a gaussian HDR disc at (az, alt) — the energy that makes the pano a LIGHT.
    In hybrid rigs the crisp shadows still come from the parametric VRaySun; the disc
    matters for sun-off looks, reflections, and bounce color. Mutates rows in place."""
    h = len(rows)
    w = len(rows[0]) if h else 0
    if not w:
        return {}
    tint = kelvin_to_rgb(disc_kelvin_for_altitude(alt_deg))
    sigma = math.radians(max(0.5, size_deg))
    a0 = math.radians(az_deg)
    e0 = math.radians(alt_deg)
    sun_dir = (math.sin(a0) * math.cos(e0), math.cos(a0) * math.cos(e0), math.sin(e0))
    reach = math.degrees(sigma) * 4.0
    touched = 0
    for y in range(h):
        alt = 90.0 - (y + 0.5) * 180.0 / h
        if abs(alt - alt_deg) > reach:
            continue
        for x in range(w):
            az = math.radians((x + 0.5) * 360.0 / w)
            el = math.radians(alt)
            d = (math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el))
            dot = min(1.0, max(-1.0,
                               d[0] * sun_dir[0] + d[1] * sun_dir[1] + d[2] * sun_dir[2]))
            ang = math.acos(dot)
            g = math.exp(-0.5 * (ang / sigma) ** 2)
            if g < 1e-3:
                continue
            p = rows[y][x]
            rows[y][x] = (p[0] + strength * g * tint[0],
                          p[1] + strength * g * tint[1],
                          p[2] + strength * g * tint[2])
            touched += 1
    return {"azimuth_deg": az_deg, "altitude_deg": alt_deg, "strength": strength,
            "size_deg": size_deg, "kelvin": disc_kelvin_for_altitude(alt_deg),
            "pixels": touched}


def lift_sky(rows: Rows, above_alt_deg: float = 10.0, factor: float = 1.3) -> None:
    """Overcast: no disc — the sky IS the key light, so raise it uniformly (in place)."""
    h = len(rows)
    for y in range(h):
        if 90.0 - (y + 0.5) * 180.0 / h > above_alt_deg:
            rows[y] = [(p[0] * factor, p[1] * factor, p[2] * factor) for p in rows[y]]


# --------------------------------------------------------------------------- entry point
def build_seed(
    out_path: str,
    ref_path: Optional[str] = None,
    pano_path: Optional[str] = None,
    semantics: Optional[Dict] = None,
    cam_yaw_deg: float = 0.0,
    sun_az_deg: Optional[float] = None,
    sun_alt_deg: Optional[float] = None,
    out_w: int = 256,
    out_h: int = 128,
    fov_deg: float = 90.0,
    sun_strength: float = 200.0,
    sun_size_deg: float = 4.0,
    ambient_key: float = 0.35,
    parametric_sun_active: bool = False,
) -> Optional[Dict]:
    """Reference (or external pano) → seeded .hdr on disk. → meta dict, None on failure.

    Sun placement: explicit (az, alt) wins (pass the solved/matched values); otherwise it
    derives from semantics (camera yaw + bearing, altitude band table) exactly like the
    first-guess rules do. ``semantics`` also decides disc-vs-overcast.

    ``parametric_sun_active``: the hybrid-rig energy rule. When a live VRaySun already
    provides the direct light, a second sun baked into the dome would DOUBLE the direct
    energy and cast a second soft shadow — exactly the mistake sunless commercial HDRIs
    exist to avoid — so the disc is skipped and the seed stays ambient-only. Disc-bearing
    seeds are for sunless/disabled-sun rigs, where the dome IS the direction."""
    sem = semantics or {}
    src = pano_path or ref_path
    if not src:
        return None
    loaded = metrics._load_pixels(src, max_dim=384)
    if not loaded:
        return None
    pixels, w, h = loaded
    if pano_path:
        rows = ingest_pano(pixels, w, h, cam_yaw_deg, out_w, out_h)
        source = "pano"
    else:
        rows = synthesize_pano(pixels, w, h, cam_yaw_deg, out_w, out_h, fov_deg)
        source = "reference"
    rows = blur_pano(rows)
    rows, scale = normalize_key(rows, ambient_key)

    sky = sem.get("sky", "clear")
    time_of_day = sem.get("time_of_day", "afternoon")
    sun_active = bool(sem.get("sun_active", True)) and sky != "overcast" \
        and time_of_day != "night"
    sun_meta: Optional[Dict] = None
    disc_policy = "skipped_parametric_sun" if (sun_active and parametric_sun_active) \
        else ("disc" if sun_active else "no_sun")
    if sun_active and parametric_sun_active:
        sun_active = False                     # ambient-only seed; VRaySun owns direction
    if sun_active:
        if sun_az_deg is None:
            sun_az_deg = cam_yaw_deg + float(sem.get("sun_bearing_deg", 0.0))
        if sun_alt_deg is None:
            from .rules import ALTITUDE_DEG, TIME_FALLBACK_ALTITUDE

            band = sem.get("sun_altitude_band", "na")
            sun_alt_deg = ALTITUDE_DEG.get(band, 35.0)
            if band == "na":
                sun_alt_deg = TIME_FALLBACK_ALTITUDE.get(time_of_day, 35.0)
        sun_meta = inject_sun(rows, sun_az_deg % 360.0, max(-4.0, sun_alt_deg),
                              sun_strength, sun_size_deg)
    elif sky == "overcast":
        lift_sky(rows)

    if not hdr_min.write_hdr(out_path, rows):
        return None
    return {
        "path": out_path,
        "source": source,
        "width": out_w,
        "height": out_h,
        "cam_yaw_deg": cam_yaw_deg,
        "fov_deg": snap_fov(fov_deg),
        "normalize_scale": round(scale, 5),
        "sun": sun_meta,
        "disc_policy": disc_policy,
        "overcast_lift": (sky == "overcast"),
    }


def seed_filename(camera_name: str, token: str = "") -> str:
    """``token`` (input fingerprint) goes into the NAME deliberately: Max caches bitmaps
    by path, so re-seeding into the same filename can render the STALE pano — a changed
    input must produce a changed path."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in camera_name) or "cam"
    return f"seed_{safe}_{token}.hdr" if token else f"seed_{safe}.hdr"
