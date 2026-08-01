"""Build the MXLT test scene — a deliberately HARD lighting rig for exercising the
Vantage measurement surface (scripts/vantage_calibrate.py) and a real MATCH.

Run headless:
    "C:\\Program Files\\Autodesk\\3ds Max 2026\\3dsmaxbatch.exe" scripts\\build_test_scene.ms
or from the MAXScript listener:
    python.ExecuteFile @"C:\\Users\\aasis\\MXLT\\scripts\\build_test_scene.py"

WHAT IT BUILDS AND WHY EACH PIECE IS THERE — every element below exists because some
stage of the calibration harness or the matcher needs it, not for looks:

  * VRaySun, TARGETED, low altitude, aimed through a window aperture. Targeted because
    classify_rig warns an untargeted sun "aims by node rotation, so azimuth/altitude
    writes will not re-aim it" — an untargeted sun would make the sun solve untestable.
    Low altitude + an aperture produces the thing the whole hot-patch metric is built
    on: a small, bright, DIRECTIONAL patch with a nameable position.
  * A VRayLight DOME (type 1) — classify_rig's dome detection, and the E4 census's
    dome.intensity axis.
  * Practical lights on FOUR NAMED LAYERS. classify_rig groups by LAYER name ("archviz
    scenes organize practicals by layer, which makes layers the natural dimmer boards"),
    so four layers means four independently solvable group dimmers instead of one blob.
  * A PHYSICAL CAMERA as the exposure host, and deliberately NO V-Ray Exposure Control:
    ExposureHost prefers a scene exposure control when one exists, but the Chaos support
    table (124621427) says it is the PHYSICAL CAMERA whose ISO/f-number/shutter/EV/WB
    stream over the live link. Testing EV reach against the host that is documented to
    stream is the honest experiment; a scene-level exposure control would likely measure
    DECOUPLED for an uninteresting reason.
  * HIGH ALBEDO CONTRAST — near-black floor, white walls, a saturated rug. The albedo
    trap is a named failure mode of this tool, and a scene that cannot express it cannot
    test defences against it.
  * DEEP DYNAMIC RANGE — a sunlit patch near clipping, an unlit alcove near black. The
    tone fit bins samples by input LEVEL and refuses under MIN_LEVELS distinct levels,
    so a flat scene would produce a sparse, untrustworthy curve. Range here is a
    measurement requirement, not an aesthetic one.

Geometry is kept MODERATE on purpose. The lighting is what needs to be hard; the
calibration harness fires K x states renders, and an 18M-triangle scene would make each
iteration of testing cost an hour without testing anything the lighting does not already.
"""

from __future__ import annotations

import os
import sys
import traceback

OUT_DIR = os.environ.get("MXLT_TEST_DIR", r"E:\MXLT-test")
OUT_FILE = os.path.join(OUT_DIR, "mxlt_hard_lighting.max")

log = []


def say(msg):
    log.append(str(msg))
    print("[build] " + str(msg))


def _first_class(rt, names):
    """First constructable class from a candidate tuple — the codebase's own
    candidates philosophy, so a casing or availability drift degrades instead of
    crashing the build halfway through."""
    for n in names:
        try:
            c = getattr(rt, n)
        except Exception:
            continue
        if c is not None:
            return c, n
    return None, ""


def _set_vray_cpu(rt):
    """Assign the V-Ray CPU production renderer.

    CPU deliberately, not GPU: checklist #14 says V-Ray GPU rendering while the Vantage
    live link streams is a documented Max-crash configuration, and vantage_calibrate.py
    refuses its render legs on a GPU renderer. A scene saved with GPU active would block
    the very harness it exists to feed.
    """
    chosen = None
    try:
        classes = list(rt.RendererClass.classes)
    except Exception:
        classes = []
    named = []
    for c in classes:
        try:
            nm = str(c)
        except Exception:
            continue
        named.append((nm, c))
    # prefer a plain "V_Ray_<version>" over GPU/RT/Express variants
    def _score(nm):
        low = nm.lower()
        if not low.startswith("v_ray"):
            return -1
        if "gpu" in low or low.endswith("_rt") or "_rt_" in low:
            return 0
        return 2
    best = None
    for nm, c in named:
        s = _score(nm)
        if s > 0 and (best is None or s > best[0]):
            best = (s, nm, c)
    if best is None:
        say("WARNING: no V-Ray CPU renderer class found; leaving the renderer as-is")
        say("  available: " + ", ".join(nm for nm, _c in named[:12]))
        return ""
    try:
        rt.renderers.production = best[2]()
        chosen = best[1]
        say("renderer: " + chosen)
    except Exception as e:
        say("WARNING: could not assign %s (%s)" % (best[1], e))
        return ""
    return chosen


def _layer(rt, name):
    """Create (or fetch) a named layer. classify_rig keys dimmer GROUPS off layer
    names, so this is what makes the practicals separately solvable."""
    lay = None
    try:
        lay = rt.LayerManager.getLayerFromName(name)
    except Exception:
        lay = None
    if lay is None:
        try:
            lay = rt.LayerManager.newLayerFromName(name)
        except Exception as e:
            say("WARNING: layer %s could not be created (%s)" % (name, e))
            return None
    return lay


def _put(lay, node):
    if lay is None or node is None:
        return
    try:
        lay.addNode(node)
    except Exception:
        pass


def _mtl(rt, name, rgb, refl=0):
    """A VRayMtl with a plain diffuse colour (and optional reflection)."""
    cls, _n = _first_class(rt, ("VRayMtl",))
    if cls is None:
        return None
    try:
        m = cls()
        m.name = name
        m.diffuse = rt.color(*rgb)
        if refl:
            m.reflection = rt.color(refl, refl, refl)
            m.reflection_glossiness = 0.85
        return m
    except Exception as e:
        say("WARNING: material %s failed (%s)" % (name, e))
        return None


def build():
    from pymxs import runtime as rt

    rt.resetMaxFile(rt.Name("noPrompt"))
    say("scene reset")
    renderer = _set_vray_cpu(rt)

    # ---------------------------------------------------------------- materials
    # Albedo contrast is a TEST REQUIREMENT: the albedo trap (a bright reference
    # matched inside a dark scene biasing the EV/WB solver) is a named failure mode,
    # and a uniformly mid-grey scene cannot exercise the defences against it.
    m_wall = _mtl(rt, "MXLT_wall_white", (232, 230, 225))
    m_floor = _mtl(rt, "MXLT_floor_darkwood", (38, 26, 18), refl=42)
    m_ceil = _mtl(rt, "MXLT_ceiling", (245, 245, 242))
    m_rug = _mtl(rt, "MXLT_rug_saturated", (150, 40, 30))
    m_metal = _mtl(rt, "MXLT_metal", (180, 180, 185), refl=180)
    m_wood = _mtl(rt, "MXLT_wood_mid", (120, 82, 46))

    # ---------------------------------------------------------------- shell
    # Metres-ish in generic units: a 9 x 7 x 3.2 room. The aperture is built from FOUR
    # wall slabs rather than a boolean — booleans are the classic headless-batch
    # failure, and a hole made of slabs cannot fail silently.
    W, D, H = 9000.0, 7000.0, 3200.0
    T = 120.0
    geo = []

    def box(name, w, d, h, pos, mtl=None):
        try:
            b = rt.Box(width=w, length=d, height=h, name=name)
            b.pos = rt.Point3(*pos)
            if mtl is not None:
                b.material = mtl
            geo.append(b)
            return b
        except Exception as e:
            say("WARNING: box %s failed (%s)" % (name, e))
            return None

    box("MXLT_floor", W, D, T, (0, 0, -T), m_floor)
    box("MXLT_ceiling", W, D, T, (0, 0, H), m_ceil)
    box("MXLT_wall_back", W, T, H, (0, D / 2, 0), m_wall)
    box("MXLT_wall_left", T, D, H, (-W / 2, 0, 0), m_wall)
    box("MXLT_wall_right", T, D, H, (W / 2, 0, 0), m_wall)

    # front wall with a 4.2m x 2.0m aperture at sill height 0.9m — the sun's doorway
    SILL, WIN_H, WIN_W = 900.0, 2000.0, 4200.0
    side = (W - WIN_W) / 2.0
    box("MXLT_wall_front_sill", W, T, SILL, (0, -D / 2, 0), m_wall)
    box("MXLT_wall_front_head", W, T, H - SILL - WIN_H,
        (0, -D / 2, SILL + WIN_H), m_wall)
    box("MXLT_wall_front_L", side, T, WIN_H, (-(W - side) / 2, -D / 2, SILL), m_wall)
    box("MXLT_wall_front_R", side, T, WIN_H, ((W - side) / 2, -D / 2, SILL), m_wall)

    # an ALCOVE — a deliberately unlit pocket. The tone fit bins by input level and
    # refuses under MIN_LEVELS; without near-black pixels the curve's shadow end is
    # extrapolated flat, which is exactly where hot_frac's complement lives.
    box("MXLT_alcove_wall", T, 2400.0, H, (-W / 2 + 2200.0, D / 2 - 1200.0, 0), m_wall)

    # ---------------------------------------------------------------- furniture
    # Occluders exist to CAST STRUCTURE into the sun patch: the hot-patch metric scores
    # placement on a 5x5 grid, so a featureless floor would give the sun solve a
    # rotationally ambiguous blob to rank.
    box("MXLT_table", 2400.0, 1100.0, 90.0, (600.0, -500.0, 740.0), m_wood)
    for i, x in enumerate((-500.0, 250.0, 1000.0, 1750.0)):
        box("MXLT_leg%02d" % i, 90.0, 90.0, 740.0, (x, -500.0, 0), m_wood)
    box("MXLT_rug", 3600.0, 2600.0, 8.0, (400.0, -300.0, 4.0), m_rug)
    box("MXLT_sideboard", 2000.0, 500.0, 800.0, (-2600.0, 2600.0, 0), m_wood)
    try:
        col = rt.Cylinder(radius=140.0, height=2100.0, name="MXLT_column")
        col.pos = rt.Point3(2900.0, 1400.0, 0)
        if m_wall is not None:
            col.material = m_wall
        geo.append(col)
    except Exception:
        pass
    for i, (x, y, r) in enumerate(((-1800.0, -1900.0, 320.0),
                                   (2200.0, -2100.0, 260.0),
                                   (-3000.0, 900.0, 200.0))):
        try:
            s = rt.Sphere(radius=r, name="MXLT_prop%02d" % i)
            s.pos = rt.Point3(x, y, r)
            s.material = m_metal if i == 1 else m_wood
            geo.append(s)
        except Exception:
            pass
    say("geometry: %d objects" % len(geo))

    # ---------------------------------------------------------------- SUN
    # Targeted, and low. classify_rig flags an untargeted VRaySun as unaimable; a low
    # altitude is what drives a long, unambiguous patch across the floor.
    sun = None
    sun_cls, _n = _first_class(rt, ("VRaySun",))
    if sun_cls is not None:
        try:
            tgt = rt.targetObject(pos=rt.Point3(500.0, 1200.0, 300.0),
                                  name="MXLT_sun.Target")
            sun = sun_cls(target=tgt, name="MXLT_sun")
            # azimuth ~ from front-left, altitude ~14 deg: golden-hour rake
            sun.pos = rt.Point3(-7000.0, -12000.0, 3400.0)
            for prop, val in (("intensity_multiplier", 0.045),
                              ("size_multiplier", 3.0),
                              ("turbidity", 3.2), ("ozone", 0.35)):
                try:
                    setattr(sun, prop, val)
                except Exception:
                    pass
            try:
                sun.enabled = True
            except Exception:
                pass
            say("sun: MXLT_sun (targeted, low altitude)")
        except Exception as e:
            say("WARNING: VRaySun failed (%s)" % e)
    else:
        say("WARNING: no VRaySun class — the sun solve cannot be tested")

    # VRaySky in the environment slot: it auto-binds to the first ENABLED VRaySun, so
    # the sky follows every sun move the matcher makes instead of contradicting it.
    sky_cls, _n = _first_class(rt, ("VRaySky",))
    if sky_cls is not None:
        try:
            rt.environmentMap = sky_cls()
            rt.useEnvironmentMap = True
            say("environment: VRaySky (auto-binds to the enabled sun)")
        except Exception as e:
            say("WARNING: VRaySky failed (%s)" % e)

    # ---------------------------------------------------------------- lights
    vl_cls, _n = _first_class(rt, ("VRayLight",))
    made = {"dome": 0, "pendants": 0, "wall_wash": 0, "accent": 0, "exterior": 0}

    def vlight(name, ltype, pos, **kw):
        if vl_cls is None:
            return None
        try:
            lt = vl_cls(name=name)
            lt.type = ltype
            lt.pos = rt.Point3(*pos)
            for k, v in kw.items():
                try:
                    setattr(lt, k, v)
                except Exception:
                    pass
            return lt
        except Exception as e:
            say("WARNING: light %s failed (%s)" % (name, e))
            return None

    # DOME (type 1) — classify_rig's dome detection and the E4 dome.intensity axis
    dome = vlight("MXLT_dome", 1, (0, 0, 0), multiplier=0.55)
    if dome is not None:
        made["dome"] = 1
        _put(_layer(rt, "MXLT_dome"), dome)

    # four dimmer boards, one per LAYER — this is what classify_rig turns into groups
    lay_pend = _layer(rt, "MXLT_pendants")
    for i, x in enumerate((-200.0, 600.0, 1400.0)):
        lt = vlight("MXLT_pendant%02d" % i, 2, (x, -500.0, 2250.0),
                    multiplier=14.0, size0=90.0)
        if lt is not None:
            made["pendants"] += 1
            _put(lay_pend, lt)

    lay_wash = _layer(rt, "MXLT_wall_wash")
    for i, x in enumerate((-3000.0, -1000.0, 1000.0, 3000.0)):
        lt = vlight("MXLT_wash%02d" % i, 0, (x, D / 2 - 400.0, 2600.0),
                    multiplier=3.2, size0=1500.0, size1=400.0)
        if lt is not None:
            made["wall_wash"] += 1
            _put(lay_wash, lt)

    lay_acc = _layer(rt, "MXLT_accent")
    for i, (x, y, z) in enumerate(((-2600.0, 2600.0, 950.0),
                                   (2900.0, 1400.0, 1900.0))):
        lt = vlight("MXLT_accent%02d" % i, 2, (x, y, z), multiplier=9.0, size0=55.0)
        if lt is not None:
            made["accent"] += 1
            _put(lay_acc, lt)

    # a bounce card OUTSIDE the aperture: it lifts the shadow side without touching the
    # sun, which is precisely the dome-vs-sun metamer the matcher has to tell apart
    lay_ext = _layer(rt, "MXLT_exterior")
    lt = vlight("MXLT_bounce", 0, (0, -D / 2 - 2500.0, 1600.0),
                multiplier=1.6, size0=6000.0, size1=3000.0)
    if lt is not None:
        made["exterior"] = 1
        _put(lay_ext, lt)
    say("lights: " + ", ".join("%s=%d" % kv for kv in sorted(made.items())))

    # ---------------------------------------------------------------- camera
    # PHYSICAL CAMERA, and NO V-Ray Exposure Control anywhere in the scene.
    # ExposureHost prefers a scene exposure control when one exists — but the Chaos
    # support table says it is the physical camera whose EV/WB stream over the live
    # link, so this is the host the E2 reach experiment must interrogate.
    cam = None
    cam_cls, cam_name = _first_class(rt, ("Physical", "VRayPhysicalCamera",
                                          "Freecamera"))
    if cam_cls is not None:
        try:
            ctgt = rt.targetObject(pos=rt.Point3(300.0, 800.0, 1100.0),
                                   name="MXLT_cam.Target")
            cam = cam_cls(target=ctgt, name="MXLT_cam")
            cam.pos = rt.Point3(-2400.0, -2900.0, 1650.0)
            # Target exposure mode + a direct EV is the combination ExposureHost reads
            # as `physical_cam`; without it read_ev falls back to the exposure triangle
            # names taken from exposure.py's VERIFIED candidate tuples (CAM_EV,
            # CAM_EV_TYPE, CAM_ISO, CAM_FNUM, CAM_WB_*) rather than guessed — the first
            # build guessed "film_speed_iso" and "focal_length" and Max rejected both
            for prop, val in (("exposure_gain_type", 1), ("exposure_value", 11.5),
                              ("white_balance_type", 1),
                              ("white_balance_kelvin", 6500.0),
                              ("f_number", 4.0), ("iso", 200.0),
                              ("focal_length_mm", 24.0)):
                try:
                    setattr(cam, prop, val)
                except Exception:
                    say("  (camera has no '%s' — left at its default)" % prop)
            say("camera: MXLT_cam (%s), EV 11.5, WB 6500K" % cam_name)
        except Exception as e:
            say("WARNING: camera failed (%s)" % e)
    if cam is not None:
        try:
            rt.viewport.setCamera(cam)
            say("viewport: MXLT_cam is active (vantage_calibrate reads "
                "rt.viewport.getCamera)")
        except Exception as e:
            say("WARNING: could not set the viewport camera (%s) — set it by hand "
                "before running the harness" % e)

    # ---------------------------------------------------------------- render setup
    try:
        rt.renderWidth = 1920
        rt.renderHeight = 1080
    except Exception:
        pass

    # ---------------------------------------------------------------- save
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except OSError as e:
        say("FATAL: could not create %s (%s)" % (OUT_DIR, e))
        return 2
    ok = False
    try:
        ok = bool(rt.saveMaxFile(OUT_FILE, quiet=True))
    except Exception as e:
        say("FATAL: save raised (%s)" % e)
        return 2
    if not ok or not os.path.exists(OUT_FILE):
        say("FATAL: %s was not written" % OUT_FILE)
        return 2
    size = os.path.getsize(OUT_FILE)
    say("SAVED %s (%.1f MB)" % (OUT_FILE, size / 1048576.0))

    # ---------------------------------------------------------------- self-check
    # Verify through the PLUGIN'S OWN eyes, not the builder's: what matters is what
    # classify_rig and ExposureHost see, because those are what the harness uses.
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from maxgaffer.maxbridge import scene as sc
        from maxgaffer.maxbridge.exposure import ExposureHost

        rig = sc.classify_rig()
        groups = {k: len(v) for k, v in (rig.get("groups") or {}).items()}
        host = ExposureHost(cam)
        say("classify_rig: sun=%s dome=%s suns=%d domes=%d groups=%s sky_env=%s"
            % (bool(rig.get("sun")), bool(rig.get("dome")), len(rig.get("suns") or []),
               len(rig.get("domes") or []), groups, rig.get("sky_env")))
        for n in (rig.get("notes") or []):
            say("  note: " + str(n))
        say("ExposureHost: kind=%s ev=%s wb=%s"
            % (host.kind, host.read_ev(), host.read_wb_kelvin()))
        if host.kind == "none":
            say("  WARNING: no exposure host — the E2 EV/WB reach experiment cannot run")
        if not rig.get("sun"):
            say("  WARNING: no sun classified — the sun solve cannot be tested")
        if not rig.get("dome"):
            say("  WARNING: no dome classified — the dome axis cannot be tested")
    except Exception as e:
        say("self-check unavailable (%s: %s)" % (type(e).__name__, e))

    # ---------------------------------------------------------------- verification
    # RENDER IT AND MEASURE IT. A scene nobody has looked at is a scene that might be
    # pointing at a wall, and the numbers below decide whether it can even host the
    # experiments: the tone fit bins samples by input LEVEL and refuses under
    # MIN_LEVELS, so a scene with no shadows and no highlights produces a sparse curve
    # no matter how correct the fitting code is. Rendered through the plugin's OWN
    # render path so the framing, size handling and colour management are the same
    # ones the harness will use.
    if os.environ.get("MXLT_SKIP_PREVIEW"):
        return 0
    try:
        from maxgaffer.core import metrics
        from maxgaffer.maxbridge import render as rd

        prev = os.path.join(OUT_DIR, "preview.png")
        cs = {}
        try:
            cs = rd.probe_colorspace()
        except Exception:
            cs = {}
        say("colorspace: %s" % cs)
        got = rd.render_frame(cam, prev, 960, 540)
        if not got:
            say("  WARNING: the verification render produced nothing")
            return 0
        # NO display-encode here, and that is a CORRECTION (2026-08-01). This script
        # used to reason "colour management is OCIO, therefore rt.save wrote a linear
        # buffer, therefore encode it" — and on this box that inference is FALSE: V-Ray
        # 7u3 under OCIO_Default saves an already display-referred plate, so the encode
        # was a second sRGB OETF on top of one. Measured damage: p5 lifted 0.047 -> 0.240
        # and log_key 0.075 -> 0.273, i.e. a washed-out plate reported as a brighter
        # scene. Whether a plate is linear is a per-configuration FACT that has to be
        # MEASURED, which is exactly what the plugin itself does at match start
        # (controller._verify_exposure_host's over-response branch: move EV, and if the
        # plate moves more than 3 stops it is linear). A preview script has no business
        # guessing what that measurement will say.
        st = metrics.compute_stats(got)
        if st:
            # percentiles are NESTED under "p" with STRING keys — st["p"]["5"], not
            # st["p5"]. The first version of this readout used the flat spelling, got
            # None on both ends, and confidently printed a dynamic range of 0.000 for a
            # scene whose real range is ~0.95.
            p = st.get("p") if isinstance(st.get("p"), dict) else {}
            p5 = float(p.get("5") or 0.0)
            p95 = float(p.get("95") or 0.0)
            say("PREVIEW STATS  key=%.4f p5=%.4f p95=%.4f hot_frac=%.4f"
                % (float(st.get("log_key") or 0), p5, p95,
                   float(st.get("hot_frac") or 0)))
            dr = p95 - p5
            say("  dynamic range p95-p5 = %.3f %s" % (
                dr, "(good spread for the tone fit)" if dr > 0.35
                else "(NARROW — the fit will bin into few levels)"))
            hf = float(st.get("hot_frac") or 0)
            say("  hot_frac = %.4f %s" % (
                hf, "(a real sun patch for the solve to rank)" if hf > 0.004
                else "(NO sun patch — sunsolve will skip; open the aperture or "
                     "raise the sun)"))
        say("PREVIEW %s" % got)
    except Exception as e:
        say("verification render unavailable (%s: %s)" % (type(e).__name__, e))
    return 0


rc = 1
try:
    rc = build()
except Exception:
    traceback.print_exc()
    say("BUILD RAISED:\n" + traceback.format_exc()[-1500:])
    rc = 3
finally:
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "build_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(log))
    except OSError:
        pass
    print("[build] exit rc=%d" % rc)
