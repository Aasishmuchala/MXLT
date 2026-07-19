"""Preflight — run anywhere ("python scripts/preflight.py [oc_key]") to see what's ready.

Off-Max it checks: core imports, the full test-suite floor (stdlib PNG stats), Pillow,
gateway reachability. Inside Max's listener it additionally checks pymxs, V-Ray, the
vrscene exporter, vantage.exe (vantage_console.exe only when the legacy "vantage_cli"
backend is configured — stock Vantage 3.x has no render CLI), and classifies the rig.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(label: str, fn):
    try:
        detail = fn()
        print(f"  [ok] {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  [!!] {label} — {e}")
        return False


def main():
    print("MaxGaffer preflight\n")

    def _core():
        from maxgaffer.core import critic, director, genome, metrics, rules, solver  # noqa
        return "genome/solver/critic/director/rules/metrics import"

    check("core modules", _core)

    def _pillow():
        from PIL import Image  # noqa

        return "reference JPEG ingestion: fast path"

    if not check("Pillow (optional)", _pillow):
        print("       → JPEG references will go through the Max-transcode fallback")

    def _stats_floor():
        import zlib  # noqa

        from maxgaffer.core import png_min  # noqa

        return "stdlib PNG stats floor present"

    check("stdlib stats floor", _stats_floor)

    key = sys.argv[1] if len(sys.argv) > 1 else ""
    if not key:
        try:
            from maxgaffer.maxbridge import config as cfgmod

            key = cfgmod.load().api_key
        except Exception:
            key = ""
    if key:
        def _gw():
            from maxgaffer.core.omega import ping

            return ping(key)

        check("Omega gateway", _gw)
    else:
        print("  [--] gateway: no oc_ key given/configured — skipped")

    # ---------------- inside Max only
    try:
        import pymxs  # noqa

        in_max = True
    except ImportError:
        in_max = False
        print("  [--] pymxs not present (running off-Max) — Max checks skipped")
    if in_max:
        def _vray():
            from pymxs import runtime as rt

            r = str(rt.classOf(rt.renderers.current))
            # class names carry underscores on some builds: V_Ray_GPU_7__update_2_hotfix_2
            if "vray" not in r.lower().replace("_", ""):
                raise RuntimeError(f"current renderer is {r} — set V-Ray")
            return r

        check("V-Ray active renderer", _vray)

        def _exporter():
            from pymxs import runtime as rt

            if not hasattr(rt, "vrayExportVRScene"):
                raise RuntimeError("vrayExportVRScene missing")
            return "vrayExportVRScene available"

        check("vrscene exporter", _exporter)

        def _vantage():
            from maxgaffer.maxbridge import config as cfgmod

            cfg = cfgmod.load()
            # stock Vantage 3.x HAS no render CLI (Chaos-confirmed) — the Developer
            # Edition console is only required when the artist explicitly chose the
            # vantage_cli backend; otherwise check the GUI exe used for the live link
            if cfg.final_render_backend == "vantage_cli":
                p = cfg.vantage_console
                if not os.path.exists(p):
                    raise RuntimeError(f"not found: {p} — the vantage_cli backend needs "
                                       "the Developer Edition console "
                                       "(config.vantage_console)")
                return p
            p = cfg.vantage_exe
            if not os.path.exists(p):
                raise RuntimeError(f"not found: {p} — set config.vantage_exe; the live "
                                   "link can still start Vantage via V-Ray's toolbar action")
            return p

        check("vantage.exe (handoff)", _vantage)

        # stock Vantage 3.x REMOVED its render CLI (SPEC §2) — the console is only
        # required on the legacy opt-in backend, same rule as onbox_spikes.py M/M2
        def _vantage_console():
            from maxgaffer.maxbridge import config as cfgmod

            p = cfgmod.load().vantage_console
            if not os.path.exists(p):
                raise RuntimeError(f"not found: {p} — backend 'vantage_cli' needs the "
                                   "Developer Edition console exe")
            return p

        from maxgaffer.maxbridge import config as cfgmod

        if cfgmod.load().final_render_backend == "vantage_cli":
            check("vantage_console.exe (Developer Edition CLI)", _vantage_console)
        else:
            print(f"  [--] vantage_console.exe not required — final_render_backend "
                  f"{cfgmod.load().final_render_backend!r} (stock Vantage 3.x has no CLI)")

        def _rig():
            from maxgaffer.maxbridge import scene

            rig = scene.classify_rig()
            return (f"sun={'yes' if rig['sun'] else 'NO'} "
                    f"dome={'yes' if rig['dome'] else 'NO'} "
                    f"groups={list(rig['groups'])} notes={rig['notes']}")

        check("scene rig", _rig)

        def _cams():
            from maxgaffer.maxbridge import scene

            cams = scene.list_cameras()
            if not cams:
                raise RuntimeError("no cameras in scene")
            return ", ".join(f"{c['name']} (yaw {c['yaw_deg']:.0f}°)" for c in cams[:6])

        check("cameras", _cams)

    print("\ndone.")


if __name__ == "__main__":
    main()
