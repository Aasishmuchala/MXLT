"""Preflight — run anywhere ("python scripts/preflight.py [oc_key]") to see what's ready.

Off-Max it checks: core imports, the full test-suite floor (stdlib PNG stats), Pillow,
gateway reachability. Inside Max's listener it additionally checks pymxs, V-Ray, the
vrscene exporter, vantage_console.exe (only for the vantage_cli finals backend), and
classifies the open scene's rig.
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
            # class names carry underscores on real boxes (V_Ray_GPU_7__update_2_hotfix_2)
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
            # stock Vantage 3.x HAS no render CLI (Chaos-confirmed) — the console is
            # only required when the artist explicitly chose the vantage_cli backend
            if cfg.final_render_backend != "vantage_cli":
                return ("not required — finals backend is "
                        f"'{cfg.final_render_backend}' (stock Vantage has no render CLI)")
            p = cfg.vantage_console
            if not os.path.exists(p):
                raise RuntimeError(f"not found: {p}")
            return p

        check("vantage_console.exe", _vantage)

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
