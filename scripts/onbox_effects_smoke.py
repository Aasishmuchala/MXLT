"""Boundary-2 effects presence-harness — run INSIDE 3ds Max 2026 with V-Ray.

Proves the atmospheric / keyframe / multi-light / colorspace / Vantage-probe surface on a
REAL box, and records the ground truth the frozen contract left as on-box calibration items:
  * whether a VRayEnvironmentFog / VRayAerialPerspective exposes an on/off toggle (FOG_ON)
    and which *_REPORT candidate resolves for distance / height (scene.report_aliases, G2/G3);
  * that a two-key lighting bake actually INTERPOLATES at an intermediate frame;
  * the VRayIES alias reality (power vs multiplier) and that sun/dome/IES are distinct nodes;
  * the ColorPipelineMgr mode + a saved-frame swatch (render.probe_colorspace, I1);
  * the non-destructive Vantage entry points (vantage.probe_entrypoints / vrscene_export_
    available, H1/H2) and which compression / frame keywords vrayExportVRScene accepts (H3).

Launched from the MAXScript listener via:  python.ExecuteFile @"<repo>\\scripts\\onbox_effects_smoke.py"
Writes a JSON report next to itself: onbox_effects_report.json

SAFE: every created node is a THROWAWAY that is deleted in a finally, the render size / active
camera / environment map / current time are captured and restored, atmospheric probe objects
are never added to the scene's atmospheric stack, and the Vantage live link is NEVER toggled —
only the non-destructive read-only probes are called. Every step is isolated and non-fatal;
a step failure is recorded, never propagated. Degrades cleanly when V-Ray is absent (each
section self-reports ``status: degraded`` instead of crashing). Off-box it is import/compile-only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "onbox_effects_report.json")

results = []


def _short(detail) -> str:
    """A compact one-line console echo of a step's self-reported detail."""
    try:
        return json.dumps(detail, default=str)[:220]
    except Exception:
        return str(detail)[:220]


def step(name):
    """Run one isolated section. Records PASS + its self-reported detail, or FAIL + a short
    trace. NEVER fatal — surviving and logging a crash here is the whole point of the harness.
    A section that cannot run because V-Ray/an object is absent returns ``{"status":
    "degraded", ...}`` and still counts as PASS: only a genuine defect raises."""
    def deco(fn):
        try:
            detail = fn()
            results.append({"step": name, "ok": True, "detail": detail})
            print("[EFFECTS PASS]", name, "->", _short(detail))
        except Exception as e:  # noqa: BLE001 — a crash here is exactly what we hunt
            results.append({"step": name, "ok": False,
                            "error": "%s: %s" % (type(e).__name__, e),
                            "trace": traceback.format_exc()[-2000:]})
            print("[EFFECTS FAIL]", name, "->", type(e).__name__, e)
    return deco


# ---------------------------------------------------------------- small pymxs helpers
def _make(rt, names, **kw):
    """First constructable class from a candidate name tuple, or None (V-Ray absent).
    Mirrors the codebase's candidates philosophy so a casing/availability drift degrades."""
    for n in names:
        try:
            ctor = getattr(rt, n)
        except Exception:
            continue
        if ctor is None:
            continue
        try:
            return ctor(**kw)
        except Exception:
            continue
    return None


def _cls(rt, obj) -> str:
    try:
        return str(rt.classOf(obj))
    except Exception:
        return ""


def _handle(rt, obj):
    try:
        return int(rt.getHandleByAnim(obj))
    except Exception:
        try:
            return int(obj.handle)
        except Exception:
            return None


def _make_camera(rt):
    """A target camera aims reliably at the content; fall back to a free camera. Returns
    (cam, target_node_or_None) so the auto-created target can be cleaned up too."""
    try:
        cam = rt.targetcamera(pos=rt.Point3(0.0, -240.0, 110.0),
                              target=rt.targetobject(pos=rt.Point3(0.0, 0.0, 30.0)))
        return cam, getattr(cam, "target", None)
    except Exception:
        pass
    try:
        return rt.freecamera(pos=rt.Point3(0.0, -240.0, 110.0)), None
    except Exception:
        return None, None


def _atmo_rows(scene, atmo):
    """The three atmosphere.* alias rows for one atmospheric, via report_aliases when the
    sibling G2 has landed, else matched_prop directly, else 'unavailable' — so the ground
    truth for the FOG_ON toggle + *_REPORT resolutions is captured either way."""
    report = getattr(scene, "report_aliases", None)
    if report is not None:
        r = report({"sun": None, "dome": None, "atmosphere": atmo})
        return {"enabled": r.get("atmosphere.enabled"),
                "distance": r.get("atmosphere.distance"),
                "height": r.get("atmosphere.height"), "via": "report_aliases"}
    mp = getattr(scene, "matched_prop", None)
    if mp is not None:
        fog_on = getattr(scene, "FOG_ON", ("enabled", "on", "active"))
        fdr = getattr(scene, "FOG_DISTANCE_REPORT", getattr(scene, "FOG_DISTANCE", ()))
        fhr = getattr(scene, "FOG_HEIGHT_REPORT", getattr(scene, "FOG_HEIGHT", ()))
        return {"enabled": mp(atmo, fog_on), "distance": mp(atmo, fdr),
                "height": mp(atmo, fhr), "via": "matched_prop"}
    return {"via": "unavailable (scene.report_aliases / matched_prop not present)"}


def main():
    import pymxs
    from pymxs import runtime as rt

    if REPO not in sys.path:
        sys.path.insert(0, REPO)

    from maxgaffer.core import metrics
    from maxgaffer.maxbridge import config as cfgmod
    from maxgaffer.maxbridge import render as render_mod
    from maxgaffer.maxbridge import scene as scene_mod
    from maxgaffer.maxbridge import vantage as vantage_mod
    from maxgaffer.maxbridge.apply import bake_animation, capture_baselines
    from maxgaffer.core.genome import LightingState

    cfg = cfgmod.load()
    tmp = tempfile.mkdtemp(prefix="maxgaffer_effects_")
    try:
        vray_active = "vray" in str(rt.classOf(rt.renderers.current)).lower().replace("_", "")
    except Exception:
        vray_active = False
    print("\n=== MaxGaffer on-box effects smoke · %s · V-Ray active=%s ===\n"
          % (tmp, vray_active))

    throwaway = []            # every node created here is deleted in the finally
    # ---------- capture everything the harness touches, to restore in the finally
    render_size = None
    env_before = None
    cam_before = None
    time_before = None
    try:
        render_size = (int(rt.renderWidth), int(rt.renderHeight))
    except Exception:
        pass
    try:
        env_before = rt.environmentMap
    except Exception:
        pass
    try:
        cam_before = rt.viewport.getCamera()
    except Exception:
        pass
    try:
        time_before = rt.sliderTime
    except Exception:
        pass

    try:
        # ------------------------------------------------- setup: throwaway rig
        sphere = _make(rt, ("Sphere",), radius=40.0)   # something for the render to see
        if sphere is not None:
            throwaway.append(sphere)
        sun = _make(rt, ("VRaySun",))
        if sun is not None:
            throwaway.append(sun)
            tgt = getattr(sun, "target", None)
            if tgt is not None:
                throwaway.append(tgt)
        dome = _make(rt, ("VRayLight",))
        if dome is not None:
            throwaway.append(dome)
            try:
                dome.type = 1              # 1 == dome (classify_rig's dome discriminator)
            except Exception:
                pass
        ies = _make(rt, ("VRayIES",))
        if ies is not None:
            throwaway.append(ies)
        # keyframe light: the dome when present, else a stdlib omni (bake is not V-Ray-only)
        kf_light = dome if dome is not None else _make(rt, ("Omnilight", "OmniLight"))
        if kf_light is not None and kf_light is not dome:
            throwaway.append(kf_light)
        cam, cam_target = _make_camera(rt)
        if cam is not None:
            throwaway.append(cam)
        if cam_target is not None:
            throwaway.append(cam_target)

        # a deterministic rig dict pointed at OUR nodes (never the artist's), plus an
        # informational classify_rig() read of the whole scene
        my_rig = {"sun": sun, "suns": [sun] if sun else [],
                  "dome": dome, "domes": [dome] if dome else [],
                  "atmosphere": None,
                  "groups": {"practicals": [ies]} if ies else {},
                  "sky_env": False, "notes": []}
        baselines = capture_baselines(my_rig)

        # ============================================================ 1 FOG ALIASES
        @step("fog aliases: report_aliases on VRayEnvironmentFog + VRayAerialPerspective (G2/G3)")
        def _():
            fog = _make(rt, ("VRayEnvironmentFog",))
            aerial = _make(rt, ("VRayAerialPerspective",))
            if fog is None and aerial is None:
                return {"status": "degraded",
                        "reason": "neither VRayEnvironmentFog nor VRayAerialPerspective "
                                  "constructable (V-Ray absent)"}
            detail = {"status": "pass"}
            if fog is not None:
                rows = _atmo_rows(scene_mod, fog)
                detail["fog"] = {"class": _cls(rt, fog), "rows": rows,
                                 "fog_on_toggle_exists": rows.get("enabled") is not None}
            if aerial is not None:
                rows = _atmo_rows(scene_mod, aerial)
                detail["aerial"] = {"class": _cls(rt, aerial), "rows": rows,
                                    "fog_on_toggle_exists": rows.get("enabled") is not None}
            # the probe objects were never added to the atmospheric stack — drop the refs
            fog = aerial = None
            return detail

        # ============================================================ 2 KEYFRAME BAKING
        @step("keyframe baking: bake_animation writes 2 keys, intermediate frame interpolates")
        def _():
            if kf_light is None:
                return {"status": "degraded", "reason": "no light to key (V-Ray absent, "
                        "and no stdlib omni could be created)"}
            k0, kn, mid = 0, 20, 10
            v0, vn = 1.0, 6.0
            s0 = LightingState(); s0.set("dome.intensity", v0)
            sn = LightingState(); sn.set("dome.intensity", vn)
            warnings = bake_animation(my_rig, baselines, [(k0, s0), (kn, sn)], cam)

            def read_at(frame):
                with pymxs.attime(frame):
                    v = scene_mod.get_prop(kf_light, scene_mod.LIGHT_MULT)
                return None if v is None else float(v)

            a0, an, amid = read_at(k0), read_at(kn), read_at(mid)
            if a0 is None or an is None or amid is None:
                raise RuntimeError("light multiplier unreadable through attime — "
                                   "cannot verify the bake (values %r)" % [a0, an, amid])
            if abs(a0 - v0) > 0.5 or abs(an - vn) > 0.5:
                raise RuntimeError("keys not written as authored: f0=%.3f (want %.1f) "
                                   "f%d=%.3f (want %.1f)" % (a0, v0, kn, an, vn))
            lo, hi = min(v0, vn), max(v0, vn)
            interpolating = (lo + 1e-3) < amid < (hi - 1e-3)
            if not interpolating:
                raise RuntimeError("NO INTERPOLATION: mid frame %d reads %.4f, pinned to an "
                                   "endpoint (step controller / single key?)" % (mid, amid))
            return {"status": "pass", "v_frame0": a0, "v_frame%d" % kn: an,
                    "v_mid_frame%d" % mid: amid, "interpolating": True,
                    "keyed_prop": "dome.intensity", "bake_warnings": warnings or []}

        # ============================================================ 3 MULTI-LIGHT
        @step("multi-light: dome + sun + power-only VRayIES distinct, aliases resolved")
        def _():
            if sun is None and dome is None and ies is None:
                return {"status": "degraded", "reason": "no V-Ray lights constructable"}
            # give the dome a texmap so report_aliases' dome.tex_* rows resolve
            dome_tex_how = None
            if dome is not None:
                try:
                    dome_tex_how = scene_mod.set_dome_texture(
                        dome, os.path.join(tmp, "throwaway_dome.hdr"))
                except Exception as e:  # noqa: BLE001
                    dome_tex_how = "error: %s" % e
            handles = [h for h in (_handle(rt, sun), _handle(rt, dome), _handle(rt, ies))
                       if h is not None]
            distinct = len(set(handles)) == len(handles) and len(handles) >= 2
            detail = {"status": "pass",
                      "sun_class": _cls(rt, sun) if sun else None,
                      "dome_class": _cls(rt, dome) if dome else None,
                      "dome_type_enum": scene_mod.get_prop(dome, ("type",)) if dome else None,
                      "dome_tex_bind": dome_tex_how,
                      "ies_class": _cls(rt, ies) if ies else None,
                      "distinct_handles": distinct, "handles": handles}
            # VRayIES alias reality: expected to carry "power" (lumens), NOT multiplier/intensity
            mp = getattr(scene_mod, "matched_prop", None)
            light_mult_report = getattr(scene_mod, "LIGHT_MULT_REPORT", scene_mod.LIGHT_MULT)
            if ies is not None and mp is not None:
                detail["ies_multiplier_alias"] = mp(ies, scene_mod.LIGHT_MULT)
                detail["ies_report_alias"] = mp(ies, light_mult_report)
                detail["ies_on_alias"] = mp(ies, scene_mod.LIGHT_ON)
            elif ies is not None:
                detail["ies_alias_note"] = "scene.matched_prop not present (sibling G1)"
            # full report_aliases over the real sun+dome (record=True exercises LAST_ALIASES)
            report = getattr(scene_mod, "report_aliases", None)
            if report is not None:
                aliases = report({"sun": sun, "dome": dome, "atmosphere": None}, record=True)
                detail["sun_dome_aliases"] = aliases
                detail["last_aliases_recorded"] = bool(
                    getattr(scene_mod, "LAST_ALIASES", None))
            else:
                detail["report_aliases_note"] = "scene.report_aliases not present (sibling G2)"
            # informational: what the whole-scene classifier makes of this rig
            try:
                rig = scene_mod.classify_rig()
                detail["classify"] = {"sun": rig["sun"] is not None,
                                      "dome": rig["dome"] is not None,
                                      "groups": list(rig.get("groups") or {})}
            except Exception as e:  # noqa: BLE001
                detail["classify"] = "error: %s" % e
            if not distinct:
                raise RuntimeError("lights are NOT distinct nodes (energy double?): %r"
                                   % handles)
            return detail

        # ============================================================ 4 RENDER + COLORSPACE
        @step("render + colorspace: V-Ray frame + probe_colorspace mode/swatch (I1)")
        def _():
            probe = getattr(render_mod, "probe_colorspace", None)
            cs = probe() if probe is not None else {
                "note": "render.probe_colorspace not present (sibling I1)"}
            if not vray_active or cam is None:
                return {"status": "degraded",
                        "reason": ("renderer is not V-Ray" if not vray_active
                                   else "no camera to render through"),
                        "colorspace": cs}
            png = render_mod.render_frame(cam, os.path.join(tmp, "effects.png"), 160, 90)
            if not png:
                raise RuntimeError("render_frame returned None while V-Ray is active — "
                                   "the loop render path is broken")
            stats = metrics.compute_stats(png)
            if not stats:
                raise RuntimeError("compute_stats could not read the saved frame")
            return {"status": "pass", "rendered": True,
                    "swatch_mean_rgb": stats.get("mean_rgb"),
                    "log_key": stats.get("log_key"),
                    "colorspace": cs}

        # ============================================================ 5 VANTAGE PROBES
        @step("vantage probes: probe_entrypoints + vrscene_export_available, non-destructive (H1/H2)")
        def _():
            link_before = vantage_mod.link_running()
            pe = getattr(vantage_mod, "probe_entrypoints", None)
            ep = pe() if pe is not None else {
                "note": "vantage.probe_entrypoints not present (sibling H1)"}
            link_after = vantage_mod.link_running()
            # the probe is READ-ONLY — it must never attach or detach the live link
            if (link_before is None) != (link_after is None):
                raise RuntimeError("probe_entrypoints CHANGED the live-link attach state "
                                   "(%r -> %r) — it fired the toggle!" % (link_before, link_after))
            va = getattr(vantage_mod, "vrscene_export_available", None)
            avail = va(cfg) if va is not None else {
                "note": "vantage.vrscene_export_available not present (sibling H2)"}
            return {"status": "pass", "link_attached": link_after is not None,
                    "entrypoints": ep, "export_available": avail}

        # ============================================================ 6 EXPORT KEYWORD PROBE
        @step("export_vrscene keyword-probe: record accepted compression/frame keywords (H3)")
        def _():
            accepted = {}
            if hasattr(rt, "vrayExportVRScene"):
                # each keyword combo is exported to its OWN throwaway .vrscene (non-destructive
                # to the scene); TypeError == a keyword this build rejects (H3's contract).
                combos = [
                    ("bare", {}),
                    ("exportCompressed", {"exportCompressed": True}),
                    ("compressed", {"compressed": True}),
                    ("compression", {"compression": True}),
                    ("startEndFrame", {"startFrame": 0, "endFrame": 0}),
                    ("compressed+frames", {"exportCompressed": True,
                                           "startFrame": 0, "endFrame": 0}),
                ]
                for label, kw in combos:
                    path = os.path.join(tmp, "kwprobe_%s.vrscene" % label.replace("+", "_"))
                    try:
                        rt.vrayExportVRScene(path, **kw)
                        accepted[label] = ("accepted+file" if os.path.exists(path)
                                           else "accepted+nofile")
                    except TypeError as e:
                        accepted[label] = "rejected(TypeError): %s" % e
                    except Exception as e:  # noqa: BLE001
                        accepted[label] = "error(%s): %s" % (type(e).__name__, e)
            else:
                accepted = {"note": "rt.vrayExportVRScene absent (V-Ray missing)"}
            # also exercise the bridge's own safe keyword-probing export (H3)
            export = getattr(vantage_mod, "export_vrscene", None)
            bridge_result = None
            if export is not None:
                try:
                    bridge_result = export(os.path.join(tmp, "effects_safe.vrscene"),
                                           cam.name if cam is not None else None)
                except Exception as e:  # noqa: BLE001
                    bridge_result = "error: %s" % e
            return {"status": "pass" if hasattr(rt, "vrayExportVRScene") else "degraded",
                    "accepted_keywords": accepted, "bridge_export_result": bridge_result}

    finally:
        # ---------- restore everything, then delete every throwaway node
        if render_size is not None:
            try:
                rt.renderWidth, rt.renderHeight = render_size
            except Exception:
                pass
        try:
            rt.environmentMap = env_before
        except Exception:
            pass
        if cam_before is not None:
            try:
                rt.viewport.setCamera(cam_before)
            except Exception:
                pass
        if time_before is not None:
            try:
                rt.sliderTime = time_before
            except Exception:
                pass
        for node in throwaway:
            try:
                rt.delete(node)
            except Exception:
                pass

    # ---------- report
    passed = sum(1 for r in results if r["ok"])
    failed = sum(1 for r in results if not r["ok"])
    try:
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump({"repo": REPO, "vray_active": vray_active, "in_max": True,
                       "results": results, "passed": passed, "failed": failed},
                      f, indent=1, default=str)
        where = REPORT
    except OSError as e:
        where = "(could not write report: %s)" % e
    print("\n=== effects smoke DONE · passed=%d failed=%d · report: %s ===\n"
          % (passed, failed, where))


# pymxs alone decides the "inside Max?" question — main() runs OUTSIDE this guard so a genuine
# coding error tracebacks loudly instead of masquerading as "you're not inside Max".
try:
    import pymxs  # noqa: F401
except ImportError:
    print("onbox_effects_smoke.py must run INSIDE 3ds Max with V-Ray "
          "(pymxs not available here) — off-box this file is import/compile-only.")
else:
    main()
