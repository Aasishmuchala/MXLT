"""Vantage measurement-surface calibration — run INSIDE 3ds Max 2026 with V-Ray and the
Chaos Vantage live link streaming the scene you care about.

One run answers, in order, every question the 2026-07-31 stress-test said must be
MEASURED rather than assumed — and the order is load-bearing: each experiment gates the
ones after it, so a box that fails early fails cheaply.

  E0  environment      — link up? window found? renderer class (V-Ray GPU refuses the
                         render legs: checklist #14 is a documented Max-crash
                         configuration when GPU renders race the live link)? colour
                         management mode? window client size (provenance).
  E1  AUTO-EXPOSURE    — sun intensity ×0.5/×1/×2/×4 with everything else frozen: the
                         grabs' centre means must track roughly multiplicatively. A flat
                         response means Vantage's auto-exposure is renormalising the
                         image ("ignores set camera exposure" — Chaos docs 125272932),
                         no static tone curve exists, and the fix is one artist click
                         (disable auto-exposure in Vantage) followed by a re-run.
  E2  EV / WB REACH    — Max-side exposure ±2 EV and white balance +2000 K: does the
                         Vantage image move? The vendor support table (124621427) says
                         Physical Camera EV/WB stream; this measures it on THIS box,
                         because an axis that does not reach the deliverable must never
                         generate fit pairs (it would poison the residual) and must be
                         reported as V-Ray-only.
  E3  DWELL            — one large change, then grabs every ~100 ms for 10 s: time to
                         first motion and time to stationarity. This is the REAL
                         per-probe cost of a measurement-grade grab (the honest speedup
                         is convergence-bound, not grab-bound), and it calibrates
                         vgrab.CONVERGE_LIMIT_S for this scene class.
  E4  AXIS CENSUS      — wiggle each genome axis the matcher probes (sun intensity,
                         dome multiplier, first practical light, EV, WB) and require
                         the settle signature to move. An axis that does not stream is
                         VANTAGE-BLIND: a search probing it through grabs cannot see
                         its own moves — the exact confidently-wrong failure vgrab's
                         contract exists to prevent. Vantage 3.3 preserves properties
                         edited inside Vantage (changelog 908558346), so an axis can be
                         blind on one box and streaming on another.
  E5  PAIRED FIT       — the states factorial (sun intensity × EV when E2 says EV
                         reaches, plus a dome-only state and a WB-shifted state): each
                         rendered in V-Ray (K repeats averaged — the V-Ray side is the
                         noisy side) AND grabbed converged from Vantage. Per-channel
                         QUANTILE pairing (registration-free — the two plates share no
                         pixel grid), a per-state fit family whose DISPERSION is the
                         second auto-exposure detector, a pooled fit on the training
                         states, and the HELD-OUT residual on states the fit never saw
                         as the headline number (in-sample residual is inverted
                         evidence — see core/vtone.py's module docstring).
                         Finishes with metric-space certification: the corrected grab
                         and the V-Ray plate must agree on the numbers downstream
                         stages actually decide on (log_key, p5/p95, hot_frac, grid),
                         because a small pixel residual can coexist with a wrong
                         hot_frac — bloom and denoisers live exactly at the hot pixels.

The fit is SAVED regardless of verdict (evidence is never thrown away) — the controller's
arming gates (controller._arm_tone_fit) re-check residual, dispersion and build
provenance on every run and refuse anything untrustworthy, out loud.

Launched from the MAXScript listener:
    python.ExecuteFile @"<repo>\\scripts\\vantage_calibrate.py"
Writes vantage_calibrate_report.json next to itself and the fit to
%LOCALAPPDATA%/MaxGaffer/vantage_tone.json.

SAFE: every touched value (sun intensity/enabled, dome multiplier, light multiplier, EV,
WB) is snapshotted first and restored in a finally; nothing is created in the scene; the
live link is never toggled. Off-box (no pymxs) the module imports cleanly and runs
nothing — the pure helpers at the top are unit-tested by tests/test_vantage_calibrate.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "vantage_calibrate_report.json")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

#: Module-level on purpose — the same shape onbox_effects_smoke.py uses. The tail runner
#: restores the scene and writes the report from OUTSIDE main(), so a crash anywhere in
#: the experiments still leaves the artist's scene intact and the findings on disk.
results: list = []
verdicts: dict = {}
restores: list = []          # (label, callable), run LIFO on every exit path

#: V-Ray renders averaged per paired state — the harness is once-per-install, so its
#: probes may be slow; what they may not be is noisy, because V-Ray probe noise lands in
#: the fitted curve as error. Override with MAXGAFFER_CAL_K.
K_RENDERS = max(1, int(os.environ.get("MAXGAFFER_CAL_K", "3") or 3))

#: Sun-intensity multipliers for the AE gate and the fit factorial. Two octaves up and
#: one down spans enough dynamic range that an adaptive stage cannot hide inside it.
AE_SWEEP = (0.5, 1.0, 2.0, 4.0)
FIT_SUN_SWEEP = (0.5, 1.0, 2.0)
FIT_EV_SWEEP = (-2.0, 0.0, 2.0)

#: Flat-response threshold for the AE verdict: over a ×8 intensity span, grab means
#: moving by less than this ratio between the dimmest and brightest state cannot be a
#: static transfer of the scene's light.
AE_FLAT_RATIO = 1.15


# =============================================================================== pure
# Everything above the pymxs guard is importable and unit-tested off-box.
def avg_rows(rows_list):
    """Element-wise mean of same-shaped png_min row lists → one row list (ints).
    → None on empty input or shape mismatch — averaging unlike frames would silently
    blur two different pictures into one measurement."""
    rows_list = [r for r in rows_list if r]
    if not rows_list:
        return None
    h, w = len(rows_list[0]), len(rows_list[0][0])
    for r in rows_list:
        # EVERY row's width, not just the first — a ragged inner row IndexErrored here
        # under the 2026-07-31 fuzz gauntlet, and this function runs mid-E5 where a
        # crash forfeits the whole calibration run
        if len(r) != h or any(len(row) != w for row in r):
            return None
    n = float(len(rows_list))
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            sr = sg = sb = 0
            for r in rows_list:
                px = r[y][x]
                sr += px[0]
                sg += px[1]
                sb += px[2]
            row.append((int(round(sr / n)), int(round(sg / n)), int(round(sb / n))))
        out.append(row)
    return out


def center_mean(rows, crop=0.6):
    """Mean luminance (0..255) of the middle ``crop`` of a plate — the same window every
    vgrab test uses, so the harness and the grab path agree about what 'the picture'
    is. → None on empty input."""
    if not rows or not rows[0]:
        return None
    h, w = len(rows), len(rows[0])
    margin = max(0.0, min(0.49, (1.0 - crop) / 2.0))
    y0, y1 = int(h * margin), max(int(h * margin) + 1, int(h * (1.0 - margin)))
    x0, x1 = int(w * margin), max(int(w * margin) + 1, int(w * (1.0 - margin)))
    total = 0.0
    count = 0
    for row in rows[y0:y1]:
        for px in row[x0:x1]:
            total += 0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]
            count += 1
    return total / count if count else None


def ae_verdict(means):
    """The auto-exposure gate's judgement over the AE_SWEEP grab means, dimmest first.

    → ("TRACKS", ratio)      means rise monotonically and span ≥ AE_FLAT_RATIO — a
                             static transfer of a ×8 light sweep behaves like this;
    → ("FLAT", ratio)        the span collapsed — auto-exposure is renormalising;
    → ("NON_MONOTONE", r)    means moved but not in the light's direction — something
                             content-adaptive beyond plain AE; treat as FLAT for arming;
    → ("UNMEASURED", None)   fewer than two usable means.
    """
    usable = [m for m in means if m is not None and m > 0.0]
    if len(usable) < 2:
        return "UNMEASURED", None
    ratio = max(usable) / max(1e-6, min(usable))
    monotone = all(b >= a - 1.0 for a, b in zip(usable, usable[1:]))
    if not monotone:
        return "NON_MONOTONE", ratio
    if ratio < AE_FLAT_RATIO:
        return "FLAT", ratio
    return "TRACKS", ratio


def split_holdout(state_keys):
    """(train_keys, holdout_keys). Held-out = every state tagged 'holdout:' plus the
    MIDDLE factorial state — so the residual is measured both on axes the fit never swept
    (dome-only, WB) and on an interior point of the axes it did. A fit validated only on
    its own sweep is validated on nothing (the stress-test's A9)."""
    tagged = [k for k in state_keys if k.startswith("holdout:")]
    factorial = [k for k in state_keys if not k.startswith("holdout:")]
    train = list(factorial)
    extra = ""
    if len(factorial) >= 3:
        extra = factorial[len(factorial) // 2]
        train.remove(extra)
    holdout = tagged + ([extra] if extra else [])
    return train, holdout


def log2_ratio(a, b):
    """|log2(a/b)| with degenerate guards — the 'how many stops apart' number."""
    if not a or not b or a <= 0 or b <= 0:
        return None
    return abs(math.log2(a / b))


def fmt_row(name, verdict, detail=""):
    """One aligned console line — the harness's on-box output is read in Max's listener,
    where a table survives and prose wraps into soup."""
    return "%-28s %-14s %s" % (name[:28], verdict, detail)


# ========================================================================= on-box only
def _running_in_max() -> bool:
    try:
        import pymxs  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 — off-box import must stay silent and side-effect-free
        return False


def main() -> int:
    from pymxs import runtime as rt

    from maxgaffer.core import metrics, png_min, vtone
    from maxgaffer.core.vtone import VTone
    from maxgaffer.maxbridge import config as cfgmod
    from maxgaffer.maxbridge import render as rd
    from maxgaffer.maxbridge import vantage as vt
    from maxgaffer.maxbridge import vgrab
    from maxgaffer.maxbridge.controller import _vantage_exe_stamp
    from maxgaffer.maxbridge.exposure import ExposureHost

    cfg = cfgmod.load()
    W, H = int(cfg.loop_width), int(cfg.loop_height)
    tmpdir = os.path.join(cfgmod.sessions_dir(), "_calibrate")
    try:
        os.makedirs(tmpdir, exist_ok=True)
    except OSError:
        pass

    def record(step, ok, detail):
        results.append({"step": step, "ok": bool(ok), "detail": detail})
        print(fmt_row(step, "PASS" if ok else "FAIL",
                      json.dumps(detail, default=str)[:180]))

    def _first_attr(obj, names):
        for n in names:
            try:
                if getattr(obj, n, None) is not None:
                    return n
            except Exception:  # noqa: BLE001 — an unreadable candidate is not a finding
                continue
        return None

    def _snapshot(obj, prop, label):
        value = getattr(obj, prop)
        restores.append((label, lambda o=obj, p=prop, v=value: setattr(o, p, v)))
        return value

    def _grab(path, converged=True):
        """One harness grab: settle chain reset first, so every grab is judged fresh on
        its own dwell rather than against the previous experiment's picture."""
        vgrab.reset_settle()
        return vgrab.capture_window_png(vgrab.VANTAGE_TITLE, path, W, H,
                                        converged=converged)

    def _grab_rows_or_none(tag, converged=True):
        p = os.path.join(tmpdir, tag + "_grab.png")
        got = _grab(p, converged=converged)
        if not got:
            return None, vgrab.last_error()
        return png_min.read_png_rgb(got, max_dim=512), ""

    # ---------------------------------------------------------------- E0 environment
    port = vt.link_running()
    hwnd = vgrab.find_window(vgrab.VANTAGE_TITLE) if port else None
    client = vgrab._client_rect(int(hwnd)) if hwnd else None
    renderer = ""
    try:
        renderer = str(rt.classOf(rt.renderers.current))
    except Exception:  # noqa: BLE001 — a paranoid read; absence is reported, not raised
        pass
    gpu = "GPU" in renderer.upper() or "RT" in renderer.upper().split()
    colorspace = {}
    try:
        colorspace = rd.probe_colorspace()
    except Exception as e:  # noqa: BLE001
        colorspace = {"error": str(e)}
    cam = None
    try:
        cam = rt.viewport.getCamera()
    except Exception:  # noqa: BLE001
        cam = None
    env_ok = bool(port and hwnd and client)
    record("E0 environment", env_ok, {
        "live_link_port": port, "window": bool(hwnd),
        "client_px": client and [client[2] - client[0], client[3] - client[1]],
        "renderer": renderer, "gpu": gpu, "colorspace": colorspace,
        "camera": bool(cam), "probe_size": [W, H], "k_renders": K_RENDERS})
    if not env_ok:
        verdicts["ARMING"] = ("BLOCKED", "no live link / window — nothing to calibrate")
        return 1                # the tail runner still restores and writes the report
    if gpu:
        # checklist #14: V-Ray GPU frames racing the live link on one card is a
        # DOCUMENTED Max-crash configuration — the render legs (E5) refuse rather than
        # roll the dice on the artist's session. Everything grab-only still runs.
        verdicts["E5"] = ("BLOCKED",
                          "V-Ray GPU + live link is the checklist-#14 crash config — "
                          "switch matching to CPU for the calibration run")

    sun = None
    try:
        suns = list(rt.getClassInstances(rt.VRaySun))
        sun = suns[0] if suns else None
    except Exception:  # noqa: BLE001
        sun = None
    dome = None
    practical = None
    try:
        for lt in rt.getClassInstances(rt.VRayLight):
            if int(getattr(lt, "type", -1)) == 1 and dome is None:
                dome = lt
            elif practical is None:
                practical = lt
    except Exception:  # noqa: BLE001
        pass
    host = ExposureHost(cam)
    sun_int_prop = sun and _first_attr(sun, ("intensity_multiplier", "intensity"))
    sun_on_prop = sun and _first_attr(sun, ("enabled", "on"))

    # ---------------------------------------------------------------- E1 auto-exposure
    if sun is not None and sun_int_prop:
        base = _snapshot(sun, sun_int_prop, "sun intensity")
        means = []
        for m in AE_SWEEP:
            setattr(sun, sun_int_prop, base * m)
            rows, why = _grab_rows_or_none("e1_x%g" % m)
            means.append(center_mean(rows) if rows else None)
        verdict, ratio = ae_verdict(means)
        verdicts["E1_AE"] = (verdict, ratio)
        record("E1 auto-exposure gate", verdict == "TRACKS",
               {"verdict": verdict, "span_ratio": ratio, "means": means,
                "hint": "" if verdict == "TRACKS" else
                "disable Vantage auto-exposure (viewport bar) and re-run"})
    else:
        verdicts["E1_AE"] = ("UNMEASURED", None)
        record("E1 auto-exposure gate", False,
               {"verdict": "UNMEASURED", "why": "no VRaySun with an intensity prop"})

    # ---------------------------------------------------------------- E2 EV/WB reach
    ev_reach = wb_reach = "UNMEASURED"
    base_ev = host.read_ev()
    if base_ev is not None:
        restores.append(("exposure EV", lambda v=base_ev: host.write_ev(v)))
        rows0, _ = _grab_rows_or_none("e2_ev_base")
        host.write_ev(base_ev + 2.0)
        rows2, _ = _grab_rows_or_none("e2_ev_plus2")
        host.write_ev(base_ev)
        m0, m2 = center_mean(rows0), center_mean(rows2)
        shift = log2_ratio(m0, m2)      # V-Ray: higher EV = darker → m2 < m0 when live
        ev_reach = ("COUPLED" if shift is not None and shift > 0.5 else
                    "DECOUPLED" if shift is not None else "UNMEASURED")
        record("E2 EV reach", ev_reach == "COUPLED",
               {"verdict": ev_reach, "stops_measured": shift,
                "consequence": "" if ev_reach == "COUPLED" else
                "exposure.ev never reaches the Vantage deliverable — the report must "
                "hand the artist an explicit Vantage-side exposure delta instead"})
    else:
        record("E2 EV reach", False, {"verdict": "UNMEASURED",
                                      "why": "no exposure host on this camera/scene"})
    verdicts["E2_EV"] = ev_reach
    base_wb = host.read_wb_kelvin()
    if base_wb is not None:
        restores.append(("white balance", lambda v=base_wb: host.write_wb_kelvin(v)))
        rows0, _ = _grab_rows_or_none("e2_wb_base")
        host.write_wb_kelvin(base_wb + 2000.0)
        rows2, _ = _grab_rows_or_none("e2_wb_plus2000")
        host.write_wb_kelvin(base_wb)

        def _rb(rows):
            if not rows:
                return None
            rs = center_mean([[(p[0], p[0], p[0]) for p in r] for r in rows])
            bs = center_mean([[(p[2], p[2], p[2]) for p in r] for r in rows])
            return (rs / bs) if rs and bs else None

        rb0, rb2 = _rb(rows0), _rb(rows2)
        moved = rb0 is not None and rb2 is not None and abs(rb2 - rb0) / rb0 > 0.05
        wb_reach = "COUPLED" if moved else ("DECOUPLED" if rb0 and rb2 else "UNMEASURED")
        record("E2 WB reach", wb_reach == "COUPLED",
               {"verdict": wb_reach, "rb_ratio": [rb0, rb2]})
    verdicts["E2_WB"] = wb_reach

    # ---------------------------------------------------------------- E3 dwell
    if sun is not None and sun_int_prop:
        base = getattr(sun, sun_int_prop)
        rect = vgrab._aspect_crop(client, W, H)
        before = vgrab._grab_rows(rect, W, H)
        sig_before = vgrab._signature(before) if before else []
        setattr(sun, sun_int_prop, base * 3.0)
        t0 = time.monotonic()
        first_motion = stationary_at = None
        prev_sig = sig_before
        while time.monotonic() - t0 < 10.0:
            time.sleep(0.1)
            rows = vgrab._grab_rows(rect, W, H)
            if not rows:
                continue
            sig = vgrab._signature(rows)
            if first_motion is None and sig_before and vgrab._moved(sig, sig_before):
                first_motion = time.monotonic() - t0
            if first_motion is not None and prev_sig and vgrab._stationary(sig, prev_sig):
                stationary_at = time.monotonic() - t0
                break
            prev_sig = sig
        setattr(sun, sun_int_prop, base)
        verdicts["E3_DWELL"] = {"first_motion_s": first_motion,
                                "stationary_s": stationary_at}
        record("E3 dwell", first_motion is not None,
               {"first_motion_s": first_motion, "stationary_s": stationary_at,
                "note": "stationary_s is the REAL per-probe cost of a measurement-grade "
                        "grab; compare vgrab.CONVERGE_LIMIT_S"})

    # ---------------------------------------------------------------- E4 axis census
    axes = []
    if sun is not None and sun_int_prop:
        axes.append(("sun.intensity", sun, sun_int_prop,
                     lambda v: v * 4.0))
    if dome is not None and _first_attr(dome, ("multiplier",)):
        axes.append(("dome.intensity", dome, "multiplier", lambda v: v * 4.0))
    if practical is not None and _first_attr(practical, ("multiplier",)):
        axes.append(("light.multiplier", practical, "multiplier", lambda v: v * 4.0))
    census = {}
    rect = vgrab._aspect_crop(client, W, H)
    for name, obj, prop, bump in axes:
        base = getattr(obj, prop)
        before = vgrab._grab_rows(rect, W, H)
        sig0 = vgrab._signature(before) if before else []
        setattr(obj, prop, bump(base))
        moved = False
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0 and not moved:
            time.sleep(0.15)
            rows = vgrab._grab_rows(rect, W, H)
            if rows and sig0 and vgrab._moved(vgrab._signature(rows), sig0):
                moved = True
        setattr(obj, prop, base)
        census[name] = "STREAMS" if moved else "BLIND"
    if ev_reach in ("COUPLED", "DECOUPLED"):
        census["exposure.ev"] = "STREAMS" if ev_reach == "COUPLED" else "BLIND"
    if wb_reach in ("COUPLED", "DECOUPLED"):
        census["exposure.wb_kelvin"] = "STREAMS" if wb_reach == "COUPLED" else "BLIND"
    verdicts["E4_AXES"] = census
    record("E4 axis census", all(v == "STREAMS" for v in census.values()) if census
           else False, census or {"why": "no axes found to wiggle"})

    # ---------------------------------------------------------------- E5 paired fit
    if verdicts.get("E5", ("",))[0] == "BLOCKED":
        record("E5 paired fit", False, {"why": verdicts["E5"][1]})
    elif cam is None:
        verdicts["E5"] = ("BLOCKED", "no active camera to render")
        record("E5 paired fit", False, {"why": "no active camera"})
    elif sun is None or not sun_int_prop:
        verdicts["E5"] = ("BLOCKED", "no VRaySun to sweep")
        record("E5 paired fit", False, {"why": "no VRaySun"})
    else:
        sun_base = getattr(sun, sun_int_prop)
        ev_sweep = FIT_EV_SWEEP if (ev_reach == "COUPLED"
                                    and base_ev is not None) else (0.0,)
        states = []
        for sm in FIT_SUN_SWEEP:
            for dev in ev_sweep:
                states.append(("sun%g_ev%+g" % (sm, dev), sm, dev, None))
        if dome is not None and sun_on_prop:
            states.append(("holdout:dome_only", 1.0, 0.0, "dome"))
        if base_wb is not None:
            states.append(("holdout:wb_plus2000", 1.0, 0.0, "wb"))
        mode = str(colorspace.get("mode", "")).lower()
        needs_encode = mode and "gamma" not in mode
        per_state = {}
        errors = {}
        for key, sm, dev, special in states:
            setattr(sun, sun_int_prop, sun_base * sm)
            if dev and base_ev is not None:
                host.write_ev(base_ev + dev)
            if special == "dome":
                setattr(sun, sun_on_prop, False)
            if special == "wb":
                host.write_wb_kelvin(base_wb + 2000.0)
            try:
                renders = []
                for k in range(K_RENDERS):
                    p = os.path.join(tmpdir, "%s_vray%d.png" % (key.replace(":", "_"), k))
                    got = rd.render_frame(cam, p, W, H)
                    if got and needs_encode:
                        from maxgaffer.core import expose
                        expose.display_encode_png(got, got)
                    if got:
                        rows = png_min.read_png_rgb(got, max_dim=512)
                        if rows:
                            renders.append(rows)
                vray_rows = avg_rows(renders)
                grab_rows, why = _grab_rows_or_none("e5_" + key.replace(":", "_"))
                if vray_rows and grab_rows:
                    per_state[key] = vtone.quantile_pairs(grab_rows, vray_rows)
                    if per_state[key] is None:
                        errors[key] = "quantile pairing refused (empty plate)"
                else:
                    errors[key] = why or "render or read failed"
            finally:
                setattr(sun, sun_int_prop, sun_base)
                if dev and base_ev is not None:
                    host.write_ev(base_ev)
                if special == "dome":
                    setattr(sun, sun_on_prop, True)
                if special == "wb":
                    host.write_wb_kelvin(base_wb)
        paired = {k: v for k, v in per_state.items() if v}
        train_keys, holdout_keys = split_holdout(list(paired))
        state_fits = [VTone.fit(paired[k]) for k in train_keys]
        dispersion = vtone.curve_dispersion(state_fits)
        pooled = VTone.fit(vtone.merge_samples([paired[k] for k in train_keys]))
        if pooled is None:
            verdicts["E5"] = ("FAILED", "pooled fit refused — too sparse; errors: %s"
                              % errors)
            record("E5 paired fit", False, {"errors": errors,
                                            "paired_states": len(paired)})
        else:
            holdout = vtone.merge_samples([paired[k] for k in holdout_keys])
            residual = pooled.residual_on(holdout)
            pooled.residual = residual
            pooled.dispersion = dispersion
            pooled.provenance = {
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "vantage_exe_stamp": _vantage_exe_stamp(cfg.vantage_exe),
                "window_client_px": client and [client[2] - client[0],
                                                client[3] - client[1]],
                "renderer": renderer, "colorspace": colorspace,
                "display_encoded": bool(needs_encode),
                "k_renders": K_RENDERS, "probe_size": [W, H],
                "train_states": train_keys, "holdout_states": holdout_keys,
                "ae_verdict": verdicts.get("E1_AE", ("?",))[0],
                "ev_reach": ev_reach, "wb_reach": wb_reach,
            }
            saved = cfgmod.save_tone_fit(pooled.to_dict())
            # certification in METRIC space: the numbers downstream stages decide on
            cert = {}
            for key in holdout_keys:
                if key not in paired:
                    continue
                gpath = os.path.join(tmpdir, "e5_%s_grab.png" % key.replace(":", "_"))
                vpath = os.path.join(tmpdir, "%s_vray0.png" % key.replace(":", "_"))
                grows = png_min.read_png_rgb(gpath, max_dim=512)
                if not grows or not os.path.exists(vpath):
                    continue
                cpath = os.path.join(tmpdir, "e5_%s_corr.png" % key.replace(":", "_"))
                png_min.write_png_rgb(cpath, pooled.to_vray_space(grows))
                sc_, sv = metrics.compute_stats(cpath), metrics.compute_stats(vpath)
                if sc_ and sv:
                    cert[key] = {
                        "d_log_key_stops": log2_ratio(sc_.get("log_key"),
                                                      sv.get("log_key")),
                        "d_hot_frac": abs(float(sc_.get("hot_frac") or 0)
                                          - float(sv.get("hot_frac") or 0)),
                        "d_p5": abs(float(sc_.get("p5") or 0)
                                    - float(sv.get("p5") or 0)),
                        "d_p95": abs(float(sc_.get("p95") or 0)
                                     - float(sv.get("p95") or 0)),
                    }
            gap = pooled.delivery_gap()
            ok = (residual is not None and residual <= vtone.RESIDUAL_LIMIT
                  and (dispersion is None or dispersion <= vtone.DISPERSION_LIMIT)
                  and verdicts.get("E1_AE", ("?",))[0] == "TRACKS")
            verdicts["E5"] = ("FIT_OK" if ok else "FIT_UNTRUSTED", {
                "heldout_residual": residual, "dispersion": dispersion,
                "delivery_gap": gap, "saved": saved})
            record("E5 paired fit", ok, {
                "heldout_residual": residual, "dispersion": dispersion,
                "in_sample_note": "in-sample residual is deliberately NOT reported — "
                                  "it is inverted evidence (core/vtone.py)",
                "paired_states": len(paired), "train": len(train_keys),
                "holdout": len(holdout_keys), "errors": errors,
                "certification": cert, "delivery_gap": gap,
                "saved_to": cfgmod.TONE_FIT_PATH if saved else "SAVE FAILED"})

    # ---------------------------------------------------------------- verdict
    arm = (verdicts.get("E5", ("",))[0] == "FIT_OK"
           and verdicts.get("E1_AE", ("?",))[0] == "TRACKS")
    verdicts.setdefault("ARMING", (
        "READY — the next probe_backend:vantage run arms tone-corrected grabs"
        if arm else
        "NOT READY — grabs stay ordinal-only (sun solve licence); see the failed rows"))
    print(fmt_row("ARMING", "READY" if arm else "ORDINAL-ONLY",
                  str(verdicts["ARMING"])[:150]))
    return 0


def _run_restores() -> None:
    """LIFO, one failure never eats the rest — the artist's scene comes back whatever
    happened in between."""
    for label, fn in reversed(restores):
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — report the failed restore, run the others
            results.append({"step": "restore: " + label, "ok": False,
                            "detail": {"error": str(e)}})
            print(fmt_row("restore: " + label, "FAIL", str(e)[:120]))
    del restores[:]


def _write_report() -> None:
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump({"repo": REPO, "results": results, "verdicts": verdicts},
                      f, indent=1, default=str)
        print("report:", REPORT)
    except OSError as e:
        print("report could not be written:", e)


if _running_in_max():
    try:
        main()
    except Exception:  # noqa: BLE001 — the report IS the product; a crash is a finding
        traceback.print_exc()
        results.append({"step": "harness", "ok": False,
                        "detail": {"trace": traceback.format_exc()[-1500:]}})
    finally:
        _run_restores()
        _write_report()
elif __name__ == "__main__":
    print("vantage_calibrate: no pymxs — run inside 3ds Max via "
          'python.ExecuteFile @"<repo>\\scripts\\vantage_calibrate.py"')
