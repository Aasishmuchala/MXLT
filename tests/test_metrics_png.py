"""The stdlib PNG floor must agree with Pillow — it is what runs inside Max."""

import math

import pytest

PIL = pytest.importorskip("PIL.Image")

from maxgaffer.core import metrics, png_min


def _make_png(tmp_path, name, painter, size=(96, 64), mode="RGB"):
    im = PIL.new(mode, size)
    px = im.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = painter(x, y)
    p = str(tmp_path / name)
    im.save(p, "PNG")
    return p


def test_png_min_matches_pillow_on_gradient(tmp_path):
    def grad(x, y):
        return (min(255, x * 3), min(255, y * 4), 128)

    path = _make_png(tmp_path, "grad.png", grad)
    rows = png_min.read_png_rgb(path, max_dim=96)
    assert rows is not None
    flat = [c for row in rows for px in row for c in px]
    with PIL.open(path) as im:
        rgb = im.convert("RGB")
        pixel_data = getattr(rgb, "get_flattened_data", rgb.getdata)
        ref = list(pixel_data())
    ref_flat = [c for px in ref for c in px]
    # same pixels (no subsampling at this size) — exact match after unfiltering
    assert flat == ref_flat


def test_png_min_rgba_and_grayscale(tmp_path):
    p_rgba = _make_png(tmp_path, "a.png", lambda x, y: (x % 256, y % 256, 40, 255), mode="RGBA")
    assert png_min.read_png_rgb(p_rgba) is not None
    p_gray = _make_png(tmp_path, "g.png", lambda x, y: (x + y) % 256, mode="L")
    rows = png_min.read_png_rgb(p_gray)
    assert rows is not None
    r, g, b = rows[1][1]
    assert r == g == b


def test_png_min_rejects_non_png(tmp_path):
    p = tmp_path / "fake.png"
    p.write_bytes(b"not a png at all")
    assert png_min.read_png_rgb(str(p)) is None
    assert png_min.read_png_rgb(str(tmp_path / "missing.png")) is None


def test_png_min_subsamples_large(tmp_path):
    path = _make_png(tmp_path, "big.png", lambda x, y: (100, 100, 100), size=(400, 200))
    rows = png_min.read_png_rgb(path, max_dim=100)
    assert rows is not None
    assert len(rows) <= 105 and len(rows[0]) <= 205  # subsampled well below source


def test_compute_stats_shape_and_sanity(tmp_path):
    # smooth luminance ramp → strictly increasing percentiles
    def ramp(x, y):
        v = min(255, int(x * 255 / 95))
        return (v, v, v)

    path = _make_png(tmp_path, "ramp.png", ramp)
    s = metrics.compute_stats(path)
    assert s is not None
    assert set(s) >= {"log_key", "p", "lab_mean", "lum_hist", "hue_hist", "contrast"}
    assert abs(sum(s["lum_hist"]) - 1.0) < 1e-6
    assert s["p"]["5"] < s["p"]["50"] < s["p"]["95"]
    assert 0.0 < s["log_key"] < 1.0

    # bimodal warm/cool split → mean b* must sit warm-of-neutral vs the cool half
    def split(x, y):
        return (60, 40, 20) if x < 48 else (180, 200, 230)

    s2 = metrics.compute_stats(_make_png(tmp_path, "split.png", split))
    cool = metrics.compute_stats(_make_png(tmp_path, "cool.png", lambda x, y: (180, 200, 230)))
    assert s2["lab_mean"][2] > cool["lab_mean"][2]   # adding warm pixels raises b*


def test_stats_agree_between_engines(tmp_path):
    """Pillow path vs stdlib path must produce close stats — the solver's numbers can't
    depend on which machine computed them."""
    def scene(x, y):
        return ((x * 2) % 256, (y * 3) % 256, ((x + y) * 2) % 256)

    path = _make_png(tmp_path, "scene.png", scene, size=(128, 96))
    via_pillow = metrics.compute_stats(path)

    real_pil = metrics._load_pixels

    def stdlib_only(p, max_dim=256):
        rows = png_min.read_png_rgb(p, max_dim=max_dim)
        if rows is None or not rows[0]:
            return None
        return [px for row in rows for px in row], len(rows[0]), len(rows)

    metrics._load_pixels = stdlib_only
    try:
        via_stdlib = metrics.compute_stats(path)
    finally:
        metrics._load_pixels = real_pil
    assert via_pillow and via_stdlib
    assert abs(math.log2(via_pillow["log_key"] / via_stdlib["log_key"])) < 0.1
    assert abs(via_pillow["p"]["50"] - via_stdlib["p"]["50"]) < 0.03
    for i in range(3):
        assert abs(via_pillow["lab_mean"][i] - via_stdlib["lab_mean"][i]) < 2.5


def test_hist_helpers():
    assert metrics.hist_emd([1, 0, 0], [0, 0, 1]) > metrics.hist_emd([1, 0, 0], [0, 1, 0])
    assert abs(metrics.cosine([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9
    assert metrics.cosine([1, 0], [0, 1]) == 0.0
    assert metrics.cosine([0, 0], [0, 0]) == 1.0  # two hue-less images are "the same"


def test_is_black_detects_a_dead_plate(tmp_path):
    """A dead render is not a dark render, and the difference has to be measured rather
    than inferred: on 2026-07-30 a whole TULA run was spent ranking 100% black frames
    (a dead Vantage live link had left V-Ray's distributed_rendering ON). A single lit
    pixel in 576 already lifts mean_rgb off zero, so this cannot fire on a moonlit shot."""
    black = str(tmp_path / "black.png")
    png_min.write_png_rgb(black, [[(0, 0, 0)] * 24 for _ in range(24)])
    stats = metrics.compute_stats(black)
    assert stats["mean_rgb"] == [0.0, 0.0, 0.0] and stats["p"]["95"] == 0.0
    assert metrics.is_black(stats) is True

    rows = [[(0, 0, 0)] * 24 for _ in range(24)]
    rows[12][12] = (255, 255, 255)                 # one lit pixel out of 576
    lit = str(tmp_path / "lit.png")
    png_min.write_png_rgb(lit, rows)
    assert metrics.is_black(metrics.compute_stats(lit)) is False

    # absence of evidence is not evidence: a missing or partial stats dict is not black
    assert metrics.is_black(None) is False
    assert metrics.is_black({}) is False
    assert metrics.is_black({"mean_rgb": "nonsense"}) is False
