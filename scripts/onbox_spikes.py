"""Automated on-box P0 — run INSIDE 3ds Max 2026's Python, on a THROWAWAY scene copy
that has a VRaySun, a dome VRayLight and at least one camera.

    MAXScript listener:   python.ExecuteFile @"C:\\<repo>\\scripts\\onbox_spikes.py"

Measures (not eyeballs) checklist #1-#8, #10-#13, #15-#16: property names, dome enum,
exposure host, WB/EV/azimuth DIRECTIONS via tiny probe renders, state roundtrip, vrscene
export, vantage_console presence, live-link probe, draft-prop names, sidecar. Manual
leftovers: #9 only if the live-link probe fails (click the V-Ray menu once and note the
label), #14 (watch VRAM with the link up on a heavy scene).

Scene lighting is snapshotted first and restored in a finally. Report is printed AND
written to %LOCALAPPDATA%/MaxGaffer/spike_report.txt.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RESULTS = []


def check(cid, name, fn):
    try:
        detail = fn()
        RESULTS.append((cid, name, "PASS", str(detail)))
        print(f"  [PASS] {cid:>4} {name} — {detail}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((cid, name, "FAIL", f"{type(e).__name__}: {e}"))
        print(f"  [FAIL] {cid:>4} {name} — {e}")
        if os.environ.get("MAXGAFFER_SPIKE_TRACE"):
            traceback.print_exc()


def matched(obj, tuples):
    """{logical: matched_property_name_or_MISSING} for candidate tuples."""
    from maxgaffer.maxbridge.scene import get_prop

    out = {}
    for label, cands in tuples.items():
        hit = next((n for n in cands if get_prop(obj, (n,)) is not None), None)
        out[label] = hit or "MISSING"
    return out


def main():
    from maxgaffer.core import metrics
    from maxgaffer.maxbridge import config as cfgmod
    from maxgaffer.maxbridge import draft as df
    from maxgaffer.maxbridge import scene as sc
    from maxgaffer.maxbridge import vantage as vt
    from maxgaffer.maxbridge.apply import apply_state, capture_baselines, read_state
    from maxgaffer.maxbridge.exposure import (CAM_FNUM, CAM_ISO, CAM_SHUTTER_SECONDS,
                                              CAM_SHUTTER_SPEED, EC_EV, EC_WB_KELVIN,
                                              ExposureHost)
    from maxgaffer.maxbridge.render import render_frame
    from pymxs import runtime as rt

    cfg = cfgmod.load()
    tmp = tempfile.mkdtemp(prefix="maxgaffer_spike_")
    print(f"\n=== MaxGaffer on-box spikes · probes → {tmp} ===\n")

    # ---------- 0 CRASH-risk advisory (#14): this script renders ~15 probes — with a
    # Vantage live link up AND V-Ray GPU on one card, that's the documented VRAM crash
    try:
        from maxgaffer.maxbridge import vantage as _vt

        _port = _vt.link_running()
        _rcls = str(rt.classOf(rt.renderers.current)).lower()
        if _port is not None and "gpu" in _rcls:
            print("  [!!] advisory: Vantage live link is UP (port "
                  f"{_port}) and the renderer is V-Ray GPU — probe renders while the "
                  "link streams can VRAM-crash Max (checklist #14). Close the link or "
                  "switch V-Ray to CPU before continuing.\n")
    except Exception:
        pass

    # ---------- A environment
    check("A", "V-Ray is the active renderer", lambda: (
        str(rt.classOf(rt.renderers.current)) if "vray" in
        str(rt.classOf(rt.renderers.current)).lower().replace("_", "")
        else (_ for _ in ()).throw(
            RuntimeError(f"renderer is {rt.classOf(rt.renderers.current)}"))))

    # ---------- B cameras
    cams = sc.list_cameras()
    check("B", "cameras present", lambda: ", ".join(
        f"{c['name']}(yaw {c['yaw_deg']:.0f}°)" for c in cams[:5]) or
        (_ for _ in ()).throw(RuntimeError("no cameras in scene")))
    cam = sc.get_camera(cams[0]["name"]) if cams else None

    # ---------- C rig + property names (#1 #2 #3 #11 #16)
    rig = sc.classify_rig()
    baselines = capture_baselines(rig)
    check("C1", "rig classified", lambda: (
        f"sun={'yes' if rig['sun'] else 'NO'} dome={'yes' if rig['dome'] else 'NO'} "
        f"groups={list(rig['groups'])} notes={rig['notes'] or 'none'}"))
    if rig["sun"] is not None:
        check("C2/#1", "VRaySun property names", lambda: matched(rig["sun"], {
            "intensity": sc.SUN_INTENSITY, "size": sc.SUN_SIZE,
            "turbidity": sc.SUN_TURBIDITY, "on": sc.LIGHT_ON}))
    if rig["dome"] is not None:
        check("C3/#2", "dome .type enum (expect 1)", lambda: sc.get_prop(rig["dome"], ("type",)))
        check("C4/#3", "dome texmap rotation prop", lambda: matched(
            sc.get_prop(rig["dome"], ("texmap",)) or rig["dome"],
            {"h_rotation": sc.DOME_TEX_ROT, "file": sc.DOME_TEX_FILE}))

    # ---------- D exposure host (#4 #5)
    host = ExposureHost(cam)
    check("D/#4-5", "exposure host", lambda: (
        host.describe() if host.kind != "none" else (_ for _ in ()).throw(
            RuntimeError("no host — add a V-Ray exposure control or use a physical cam"))))
    if host.kind == "exposure_control":
        check("D2", "EC property names", lambda: matched(host.ec, {
            "ev": EC_EV, "wb_kelvin": EC_WB_KELVIN}))
    elif host.kind == "physical_cam":
        # one of the two shutter conventions will be MISSING — that identifies the camera
        # generation (native Physical = seconds, legacy VRayPhysical = 1/s speed)
        check("D2", "camera exposure property names", lambda: matched(host.cam, {
            "iso": CAM_ISO, "f": CAM_FNUM, "shutter_s": CAM_SHUTTER_SECONDS,
            "shutter_speed": CAM_SHUTTER_SPEED}))

    # ---------- snapshot before anything mutates
    snapshot = read_state(rig, baselines, cam)
    try:
        # ---------- E state roundtrip (#10-ish: setters exist + idempotent)
        def roundtrip():
            warnings = apply_state(rig, baselines, snapshot, cam)
            back = read_state(rig, baselines, cam)
            drift = {k: v for k, v in back.diff(snapshot).items()
                     if abs(v[0] - v[1]) > 0.51}          # loose: unit quirks surface here
            if drift:
                raise RuntimeError(f"read-back drift: {drift} (warnings: {warnings})")
            return f"clean roundtrip · warnings: {warnings or 'none'}"

        check("E", "state read→apply→read-back", roundtrip)

        # ---------- F render probe + stdlib stats (#10)
        def probe(tag):
            p = render_frame(cam, os.path.join(tmp, f"{tag}.png"), 160, 90)
            if not p:
                raise RuntimeError("render_frame returned None")
            s = metrics.compute_stats(p)
            if not s:
                raise RuntimeError("stats unreadable (stdlib PNG floor)")
            return s

        check("F/#10", "loop render + stdlib stats", lambda: (
            f"key={probe('base')['log_key']:.4f}"))

        # ---------- G WB direction (#6) — THE sign check, measured
        def wb_direction():
            st = snapshot.copy()
            if "exposure.wb_kelvin" not in st.values:
                raise RuntimeError("no WB host — skipped (lock WB in the UI)")
            st.set("exposure.wb_kelvin", 4500.0)
            apply_state(rig, baselines, st, cam)
            cool = probe("wb4500")["mean_rgb"]
            st.set("exposure.wb_kelvin", 9000.0)
            apply_state(rig, baselines, st, cam)
            warm = probe("wb9000")["mean_rgb"]
            ratio_c = cool[0] / max(1e-4, cool[2])
            ratio_w = warm[0] / max(1e-4, warm[2])
            if ratio_w <= ratio_c:
                raise RuntimeError(
                    f"INVERTED: r/b at 9000K ({ratio_w:.3f}) ≤ at 4500K ({ratio_c:.3f}) — "
                    "flip the sign in core/solver.solve_wb (report first!)")
            return f"9000K warmer than 4500K ✓ (r/b {ratio_c:.3f} → {ratio_w:.3f})"

        check("G/#6", "WB kelvin direction (measured)", wb_direction)

        # ---------- H EV direction — measured
        def ev_direction():
            st = snapshot.copy()
            if "exposure.ev" not in st.values:
                raise RuntimeError("no EV host")
            base_ev = st.get("exposure.ev")
            apply_state(rig, baselines, st, cam)
            k1 = probe("ev_base")["log_key"]
            st.set("exposure.ev", base_ev + 2.0)
            apply_state(rig, baselines, st, cam)
            k2 = probe("ev_plus2")["log_key"]
            if k2 >= k1:
                raise RuntimeError(f"INVERTED: key rose {k1:.4f}→{k2:.4f} after +2 EV")
            import math as _m

            moved = _m.log2(max(1e-5, k1) / max(1e-5, k2))
            if moved < 1.0:   # sign alone can pass on a near-zero delta (V-Ray GPU did)
                raise RuntimeError(
                    f"INERT: +2 EV moved the render only {moved:.2f} stops — this "
                    "renderer's exposure is display-stage only; enable "
                    "software_exposure (Settings) — the runtime check also auto-flips it")
            return f"+2 EV darkened key {k1:.4f} → {k2:.4f} ({moved:.2f} stops) ✓"

        check("H", "EV direction (measured)", ev_direction)

        # ---------- I sun azimuth affects the frame — measured
        def azimuth_effect():
            if rig["sun"] is None:
                raise RuntimeError("no sun")
            st = snapshot.copy()
            apply_state(rig, baselines, st, cam)
            a = probe("azA")
            st.set("sun.azimuth_deg", (st.get("sun.azimuth_deg") + 120.0) % 360.0)
            apply_state(rig, baselines, st, cam)
            b = probe("azB")
            delta = sum(abs(x - y) for x, y in zip(a["mean_rgb"], b["mean_rgb"]))
            if delta < 0.002 and abs(a["log_key"] - b["log_key"]) < 1e-4:
                raise RuntimeError("frame unchanged after 120° sun swing — "
                                   "sun transform may be controller-locked (#11)")
            return f"frame responds to sun swing (Δrgb {delta:.4f}) ✓"

        check("I/#11", "sun azimuth affects render", azimuth_effect)

        # ---------- J dome rotation write (#3)
        if rig["dome"] is not None:
            def dome_rot():
                before = sc.read_dome_rotation(rig["dome"])
                how = sc.write_dome_rotation(rig["dome"], (before + 90.0) % 360.0)
                after = sc.read_dome_rotation(rig["dome"])
                sc.write_dome_rotation(rig["dome"], before)
                if how == "failed":
                    raise RuntimeError("no writable rotation path")
                return f"{how} · {before:.0f}°→{after:.0f}° ✓"

            check("J/#3", "dome rotation write", dome_rot)

        # ---------- K sun-off vs VRaySky (#13) — measured
        def sun_off_sky():
            if rig["sun"] is None or not rig.get("sky_env"):
                return "n/a (no sun+VRaySky pair)"
            st = snapshot.copy()
            st.set("sun.enabled", 0)
            apply_state(rig, baselines, st, cam)
            dark = probe("sunoff")["log_key"]
            apply_state(rig, baselines, snapshot, cam)
            if dark < 1e-4:
                return ("sky DIES with sun off → set overcast_sun_mode:'dim' in config "
                        f"(key {dark:.5f})")
            return f"sky survives sun-off (key {dark:.4f}) — 'disable' mode fine"

        check("K/#13", "sun-off vs VRaySky", sun_off_sky)

        # ---------- Q dome seed end-to-end (#17, v0.9) — fully offline, no gateway
        if rig["dome"] is not None and cam is not None:
            def dome_seed():
                from maxgaffer.core import domeseed, hdr_min

                pre_file = sc.get_dome_texture(rig["dome"])
                pre_rot = sc.read_dome_rotation(rig["dome"])
                pre_texmap = sc.get_prop(rig["dome"], ("texmap",))
                pre_tex_on = sc.get_prop(rig["dome"], sc.DOME_TEX_ON)
                ref_png = os.path.join(tmp, "base.png")
                if not os.path.exists(ref_png) and not render_frame(cam, ref_png, 160, 90):
                    raise RuntimeError("no probe render to seed from")
                out = os.path.join(tmp, "spike_seed.hdr")
                meta = domeseed.build_seed(
                    out, ref_path=ref_png,
                    semantics={"sky": "clear", "sun_active": True,
                               "time_of_day": "afternoon"},
                    cam_yaw_deg=cams[0]["yaw_deg"] if cams else 0.0,
                    sun_az_deg=210.0, sun_alt_deg=35.0, out_w=128, out_h=64)
                if meta is None:
                    raise RuntimeError("build_seed returned None (reference unreadable)")
                if hdr_min.read_hdr(out) is None:
                    raise RuntimeError("written .hdr failed round-trip read")
                try:
                    how = sc.set_dome_texture(rig["dome"], out)
                    if how == "failed":
                        raise RuntimeError("dome texture not writable (#16)")
                    bound = sc.get_dome_texture(rig["dome"])
                    seeded = render_frame(cam, os.path.join(tmp, "seeded.png"), 160, 90)
                finally:   # the artist's dome comes back even if the check raises
                    if pre_file:
                        sc.set_dome_texture(rig["dome"], pre_file)
                    elif pre_texmap is not None:
                        # the spike only overwrote the FILE on the artist's own texmap —
                        # blank it back and restore the on/off switch
                        sc.set_prop(pre_texmap, sc.DOME_TEX_FILE, "")
                        if pre_tex_on is not None:
                            sc.set_prop(rig["dome"], sc.DOME_TEX_ON, pre_tex_on)
                    else:
                        # the spike CREATED the texmap — merely disabling it leaves the
                        # tmp .hdr bound (a later re-enable resurrects the spike's sky)
                        sc.set_prop(rig["dome"], ("texmap",), None)
                        sc.set_prop(rig["dome"], sc.DOME_TEX_ON, False)
                    sc.write_dome_rotation(rig["dome"], pre_rot)
                if bound != out:
                    raise RuntimeError(f"texture readback mismatch: {bound!r}")
                if not seeded:
                    raise RuntimeError("render with seeded dome failed — check .hdr "
                                       "load / gamma (#17)")
                return (f"seed {meta['width']}x{meta['height']} bound via {how}, "
                        "rendered ✓ (u-origin #18 stays a visual check)")

            check("Q/#17", "dome seed end-to-end (v0.9)", dome_seed)

        # ---------- S color management detection (#19) — the judgment stays visual, but
        # the MODE is queryable (Max 2024+ ColorPipelineMgr): OCIO/ACES means the saved
        # loop PNG may lack the display transform → biased solver (docs/STRESS.md §1)
        def color_mode():
            mgr = getattr(rt, "ColorPipelineMgr", None)
            if mgr is None:
                return ("no ColorPipelineMgr (classic gamma workflow) — #19 low risk; "
                        "still eyeball one probe PNG vs the VFB")
            out = {}
            for prop in ("Mode", "mode", "OCIOConfigPath", "displayName", "viewName",
                         "defaultDisplay", "defaultView", "outputConversion"):
                try:
                    v = getattr(mgr, prop)
                except Exception:
                    continue
                if v is not None:
                    out[prop] = str(v)[:60]
            return (f"{out} — if Mode says OCIO/ACES, hold a probe PNG next to the VFB "
                    "BEFORE trusting any score (#19)")

        check("S/#19", "color management mode (detection)", color_mode)

        # ---------- R scenario board core (v0.9) — pure, no renders
        def board_core():
            from maxgaffer.core import scenarios as scen

            board = scen.build_scenarios(None, snapshot,
                                         cams[0]["yaw_deg"] if cams else 0.0)
            if not board:
                raise RuntimeError("no candidates from a live rig — rig read is empty?")
            return f"{len(board)} candidates: " + ", ".join(b["key"] for b in board)

        check("R", "scenario board candidates (v0.9)", board_core)

    finally:
        apply_state(rig, baselines, snapshot, cam)   # always leave the scene as found

    # ---------- L vrscene export (#7) — the export switches the active viewport camera;
    # put the artist's back afterwards
    pre_vp_cam = None
    try:
        pre_vp_cam = rt.viewport.getCamera()
    except Exception:
        pass
    check("L/#7", "vrscene export", lambda: (
        vt.export_vrscene(os.path.join(tmp, "spike.vrscene"),
                          cams[0]["name"] if cams else None)
        or (_ for _ in ()).throw(RuntimeError("vrayExportVRScene missing/failed"))))
    if pre_vp_cam is not None:
        try:
            rt.viewport.setCamera(pre_vp_cam)
        except Exception:
            pass

    # ---------- M vantage executable (#8) — stock 3.x has NO render CLI; finals default
    # to the V-Ray backend, vantage.exe is only needed for live link / manual batch queue
    check("M/#8", "vantage.exe (handoff)", lambda: (
        cfg.vantage_exe if os.path.exists(cfg.vantage_exe)
        else (_ for _ in ()).throw(RuntimeError(
            f"not found: {cfg.vantage_exe} — set config.vantage_exe; live link can still "
            "start Vantage via V-Ray's toolbar action"))))
    if cfg.final_render_backend == "vantage_cli":
        check("M2", "vantage_console.exe (Developer Edition CLI)", lambda: (
            cfg.vantage_console if os.path.exists(cfg.vantage_console)
            else (_ for _ in ()).throw(RuntimeError(f"not found: {cfg.vantage_console}"))))

    # ---------- N live link probe (#9) — the V-Ray action is a TOGGLE: probing while an
    # artist's link is up would KILL it. The link streams on port 20701 (20703 on V-Ray
    # 7.3 DR2) — a listener there means a link is already running: report it, never
    # touch it. With nothing listening the toggle can only START a link, which is what
    # the probe exists to verify — and the manual stop instruction is printed.
    def _link_listening(ports=(20701, 20703)):
        import socket

        for port in ports:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    return port
            except OSError:
                continue
        return None

    live_port = _link_listening()
    if live_port is not None:
        detail = f"port {live_port} already listening — a link is running; left untouched"
        RESULTS.append(("N/#9", "vantage live link (toggle action)", "PASS", detail))
        print(f"  [PASS] N/#9 vantage live link — {detail}")
    else:
        ok, how = vt.start_live_link()
        RESULTS.append(("N/#9", "vantage live link (toggle action)",
                        "PASS" if ok else "MANUAL", how))
        print(f"  [{'PASS' if ok else 'MANUAL'}] N/#9 vantage live link — {how}")
        if ok:
            print("       note: the probe STARTED a link — to stop it, run V-Ray's "
                  "'Initiate a Live-Link to Chaos Vantage' action once more")

    # ---------- O draft sampler props (#15) — names only, nothing changed
    check("O/#15", "draft sampler property names", lambda: {
        cands[0]: next((n for n in cands
                        if sc.get_prop(rt.renderers.current, (n,)) is not None), "MISSING")
        for cands, _v in df.DRAFT_PROPS})

    # ---------- P sidecar (#12)
    check("P/#12", "sidecar python (optional)", lambda: (
        f"{cfg.system_python} ok" if cfg.system_python and os.path.exists(cfg.system_python)
        else "not configured — stdlib floor + Max transcode cover everything"))

    # ---------- report
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    lines = [f"MaxGaffer spike report — {len(RESULTS)} checks, {len(fails)} FAIL", ""]
    lines += [f"[{s:^6}] {cid:>6} {name}: {detail}" for cid, name, s, detail in RESULTS]
    report = "\n".join(lines)
    path = os.path.join(os.path.dirname(cfgmod.CONFIG_PATH), "spike_report.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
    except OSError:
        path = "(could not write report file)"
    print(f"\n=== {len(RESULTS)} checks · {len(fails)} FAIL · report: {path} ===")
    if not fails:
        print("CHECKPOINT 0: GREEN — run a real match (tasks/plan.md P1).")


# pymxs alone decides the "inside Max?" question — main() runs OUTSIDE this guard so a
# genuine coding error (a stale import once printed this exact message ON the box)
# tracebacks loudly instead of masquerading as "you're not inside Max"
try:
    import pymxs  # noqa: F401
except ImportError:
    print("onbox_spikes.py must run INSIDE 3ds Max (pymxs not available here).")
else:
    main()
