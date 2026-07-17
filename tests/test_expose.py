"""Software exposure — EV is a linear multiply, WB a chromatic gain, and the analytic
solver converges through it with no renderer applying exposure (the V-Ray GPU fix)."""
from maxgaffer.core import critic, expose, metrics, solver
from maxgaffer.core.genome import LightingState


def _mean_lum(pixels):
    return sum(0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels) / len(pixels)


GREY = [(120, 118, 116)] * 400


def test_identity_is_noop():
    out = expose.expose_pixels(GREY, ev=11.5, base_ev=11.5,
                               wb_kelvin=6500.0, base_wb=6500.0)
    assert out == GREY


def test_ev_plus_one_halves_linear_luminance():
    base = _mean_lum(GREY)
    darker = _mean_lum(expose.expose_pixels(GREY, 12.5, 11.5, 6500.0, 6500.0))
    brighter = _mean_lum(expose.expose_pixels(GREY, 10.5, 11.5, 6500.0, 6500.0))
    assert darker < base < brighter          # higher EV darker (V-Ray semantics)
    # +1 EV ≈ half the LINEAR light; check in linear space
    lb = metrics._srgb_to_linear(base / 255.0)
    ld = metrics._srgb_to_linear(darker / 255.0)
    assert 0.42 < ld / lb < 0.58             # ~0.5


def test_wb_higher_kelvin_renders_warmer():
    warm = expose.expose_pixels(GREY, 11.5, 11.5, 8500.0, 6500.0)
    cool = expose.expose_pixels(GREY, 11.5, 11.5, 4500.0, 6500.0)
    # warmer = more red, less blue relative to cooler
    r_w, b_w = warm[0][0], warm[0][2]
    r_c, b_c = cool[0][0], cool[0][2]
    assert r_w / max(1, b_w) > r_c / max(1, b_c)


def test_wb_gain_is_green_neutral():
    gr, gg, gb = expose.wb_gain(9000.0, 5000.0)
    assert gg == 1.0 and gr > gb            # warming gain boosts red over blue


def _make_raw(tmp_path):
    """A structured raw frame (bright, like an exposure-inert V-Ray GPU render)."""
    from PIL import Image
    im = Image.new("RGB", (120, 80))
    px = im.load()
    for y in range(80):
        for x in range(120):
            v = 150 + int(90 * (x / 119.0))       # gradient with headroom
            px[x, y] = (min(255, v + 20), min(255, v), min(255, v - 10))
    p = str(tmp_path / "raw.png")
    im.save(p)
    return p


def test_solver_converges_through_software_exposure(tmp_path):
    """The fix, end to end: the renderer NEVER applies exposure — software exposure of
    the SAME raw frame lets the analytic solver land EV exactly and drive WB in."""
    raw = _make_raw(tmp_path)
    base_ev, base_wb = 11.5, 5500.0
    tgt_ev, tgt_wb = 11.5, 5200.0

    def stats_at(ev, wb):
        dst = str(tmp_path / "exp.png")
        expose.expose_image_file(raw, dst, ev, base_ev, wb, base_wb)
        return metrics.compute_stats(dst)

    ref = stats_at(tgt_ev, tgt_wb)
    st = LightingState()
    st.set("exposure.ev", tgt_ev + 2.5)       # 2.5 stops off
    st.set("exposure.wb_kelvin", tgt_wb + 2000)
    first = critic.score(ref, stats_at(st.get("exposure.ev"),
                                       st.get("exposure.wb_kelvin"))).score
    for _ in range(10):
        cur = stats_at(st.get("exposure.ev"), st.get("exposure.wb_kelvin"))
        changes = solver.analytic_pass(st, ref, cur)
        if not changes:
            break
        for k, v in changes.items():
            st.set(k, v)
    final = critic.score(ref, stats_at(st.get("exposure.ev"),
                                       st.get("exposure.wb_kelvin"))).score
    assert abs(st.get("exposure.ev") - tgt_ev) <= 0.2      # EV lands
    assert abs(st.get("exposure.wb_kelvin") - tgt_wb) <= 600  # WB drives in
    assert final - first >= 40 and final >= 95              # big, real climb


def test_expose_image_file_identity_returns_src(tmp_path):
    raw = _make_raw(tmp_path)
    assert expose.expose_image_file(raw, str(tmp_path / "o.png"),
                                    11.5, 11.5, 6500.0, 6500.0) == raw
