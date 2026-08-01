"""Probe: does Max have Pillow, and can render_frame be made to write 8-bit PNGs?

Two questions, both load-bearing for the zero-dependency stats floor:
  1. png_min's docstring asserts loop plates are "our own 8-bit RGB(A) PNGs written by
     Max". On this box render_frame produced a 16-BIT PNG, which png_min refuses — so
     display_encode_png silently returns None and, without Pillow, compute_stats returns
     None for every plate. Is Pillow present inside Max, i.e. is this fatal or fragile?
  2. Does rt.pngio.setType(#true24) make Max write the 8-bit form png_min decodes?
"""
import json, os, sys, traceback
from pymxs import runtime as rt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
OUT = r"E:\MXLT-test"
res = {"python": sys.version}

try:
    import PIL
    res["pillow"] = getattr(PIL, "__version__", "present")
except Exception as e:
    res["pillow"] = "ABSENT (%s)" % type(e).__name__

try:
    res["pngio_exists"] = rt.pngio is not None
    res["pngio_getType"] = str(rt.pngio.getType())
except Exception as e:
    res["pngio_exists"] = "err: %s" % e

from maxgaffer.core import png_min
from maxgaffer.maxbridge import render as rd

rt.loadMaxFile(os.path.join(OUT, "mxlt_hard_lighting.max"), quiet=True)
cam = rt.getNodeByName("MXLT_cam")

def _probe(label, setter):
    p = os.path.join(OUT, "probe_%s.png" % label)
    try:
        if setter:
            setter()
    except Exception as e:
        return {"set": "FAILED %s" % e}
    got = rd.render_frame(cam, p, 240, 135)
    row = {"rendered": bool(got)}
    if got and os.path.exists(p):
        with open(p, "rb") as f:
            head = f.read(33)
        import struct
        w, h, bd, ct, _cm, _fl, il = struct.unpack(">IIBBBBB", head[16:29])
        row.update(bitdepth=bd, colortype=ct, interlace=il, size=[w, h])
        row["png_min_reads"] = png_min.read_png_rgb(p, max_dim=64) is not None
    return row

res["default"] = _probe("default", None)
for name in ("true24", "true48", "gray8"):
    def _s(n=name):
        rt.pngio.setType(rt.Name(n))
    res["setType_" + name] = _probe(name, _s)

try:
    with open(os.path.join(OUT, "png_probe.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
except OSError:
    pass
print(json.dumps(res, indent=1, default=str))
