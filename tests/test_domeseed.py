"""Dome seed: the RGBE codec must round-trip, the pano geometry must put light where the
reference puts it, and the scenario board must offer real (distinct, in-bounds) choices."""

import pytest

PIL = pytest.importorskip("PIL.Image")

from maxgaffer.core import domeseed, hdr_min, scenarios
from maxgaffer.core.genome import LightingState, spec_for


# --------------------------------------------------------------------------- helpers
def _full_rig_state() -> LightingState:
    st = LightingState()
    st.set("sun.enabled", 1)
    st.set("sun.azimuth_deg", 120.0)
    st.set("sun.altitude_deg", 35.0)
    st.set("sun.intensity", 1.0)
    st.set("sun.size", 3.0)
    st.set("sun.turbidity", 3.0)
    st.set("dome.enabled", 1)
    st.set("dome.rotation_deg", 0.0)
    st.set("dome.intensity", 1.0)
    st.set("exposure.ev", 12.0)
    st.set("exposure.wb_kelvin", 6500.0)
    st.groups["practicals"] = 1.0
    return st


def _make_ref(tmp_path, name, painter, size=(96, 64)):
    im = PIL.new("RGB", size)
    px = im.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = painter(x, y)
    p = str(tmp_path / name)
    im.save(p, "PNG")
    return p


# --------------------------------------------------------------------------- hdr_min
def test_hdr_roundtrip_gradient_and_hdr_values(tmp_path):
    rows = [[(x / 31.0, y / 15.0, 500.0 if (x, y) == (5, 5) else 0.25)
             for x in range(32)] for y in range(16)]
    rows[0][0] = (0.0, 0.0, 0.0)                       # exact black must stay exact
    p = str(tmp_path / "t.hdr")
    assert hdr_min.write_hdr(p, rows)
    back = hdr_min.read_hdr(p)
    assert back is not None and len(back) == 16 and len(back[0]) == 32
    assert back[0][0] == (0.0, 0.0, 0.0)
    for y in range(16):
        for x in range(32):
            for a, b in zip(rows[y][x], back[y][x]):
                # shared-exponent quantization: error bounded by the pixel's max channel
                assert abs(a - b) <= max(rows[y][x]) / 100.0 + 1e-4


def test_hdr_rle_and_flat_paths(tmp_path):
    flat_rows = [[(0.5, 0.25, 0.125)] * 4 for _ in range(3)]     # width 4 → flat encoding
    p1 = str(tmp_path / "flat.hdr")
    assert hdr_min.write_hdr(p1, flat_rows)
    assert hdr_min.read_hdr(p1) is not None

    # constant scanline (pure runs) + per-pixel noise (pure literals), both in RLE range
    const = [[(0.7, 0.7, 0.7)] * 64 for _ in range(4)]
    noisy = [[((x * 37 % 255) / 254.0, (x * 91 % 255) / 254.0, (x * 53 % 255) / 254.0)
              for x in range(64)] for _ in range(4)]
    for name, rows in (("const.hdr", const), ("noisy.hdr", noisy)):
        p = str(tmp_path / name)
        assert hdr_min.write_hdr(p, rows)
        back = hdr_min.read_hdr(p)
        assert back is not None
        for y in range(4):
            for x in range(64):
                for a, b in zip(rows[y][x], back[y][x]):
                    assert abs(a - b) <= max(rows[y][x]) / 100.0 + 1e-4


def test_hdr_rejects_garbage(tmp_path):
    bad = tmp_path / "bad.hdr"
    bad.write_bytes(b"definitely not radiance")
    assert hdr_min.read_hdr(str(bad)) is None
    assert hdr_min.read_hdr(str(tmp_path / "missing.hdr")) is None
    assert not hdr_min.write_hdr(str(tmp_path / "x.hdr"), [])
    assert not hdr_min.write_hdr(str(tmp_path / "y.hdr"), [[(1, 1, 1)], [(1, 1, 1), (2, 2, 2)]])


# --------------------------------------------------------------------------- pano geometry
def test_pano_puts_reference_left_on_the_left(tmp_path):
    # red left half, blue right half; camera yaw 0 → world azimuth just LEFT of north
    # must be red-dominant, just right blue-dominant
    pixels = [(220, 30, 30) if x < 48 else (30, 30, 220)
              for y in range(64) for x in range(96)]
    pano = domeseed.synthesize_pano(pixels, 96, 64, cam_yaw_deg=0.0,
                                    out_w=64, out_h=32, fov_deg=90.0)
    horizon = 32 * 90 // 180                            # alt ≈ 0 row
    left = pano[horizon][int((350.0 / 360.0) * 64)]     # az 350 = 10° left of yaw
    right = pano[horizon][int((10.0 / 360.0) * 64)]     # az 10 = 10° right
    assert left[0] > left[2], "left of camera should carry the reference's left (red)"
    assert right[2] > right[0], "right of camera should carry the reference's right (blue)"


def test_pano_continuous_at_back_seam_and_wrap(tmp_path):
    # smooth horizontal gradient reference — any fold/wrap discontinuity shows as a jump
    pixels = [(int(x * 255 / 95), 128, int(255 - x * 255 / 95))
              for y in range(64) for x in range(96)]
    pano = domeseed.synthesize_pano(pixels, 96, 64, cam_yaw_deg=0.0,
                                    out_w=128, out_h=32, fov_deg=90.0)
    y = 32 * 90 // 180
    max_step = 0.0
    for x in range(128):                                # includes the 127→0 wrap pair
        a, b = pano[y][x], pano[y][(x + 1) % 128]
        max_step = max(max_step, max(abs(a[i] - b[i]) for i in range(3)))
    # adjacent columns are 2.8° apart; a seam tear would jump a large fraction of the
    # gradient's full range — continuity keeps steps to the per-column slope
    assert max_step < 0.12, f"tonal tear across pano columns: {max_step:.3f}"


def test_pano_sky_converges_at_zenith(tmp_path):
    pixels = [(240, 240, 240) if x < 48 else (10, 10, 10)
              for y in range(64) for x in range(96)]   # violently split reference
    pano = domeseed.synthesize_pano(pixels, 96, 64, cam_yaw_deg=0.0,
                                    out_w=64, out_h=32, fov_deg=90.0)
    top = [domeseed._lum(p) for p in pano[0]]
    assert max(top) - min(top) < 0.06, "zenith must be (near-)constant across azimuth"


def test_inject_sun_lands_where_told():
    pano = [[(0.1, 0.1, 0.1)] * 64 for _ in range(32)]
    meta = domeseed.inject_sun(pano, az_deg=90.0, alt_deg=30.0,
                               strength=200.0, size_deg=4.0)
    assert meta["pixels"] > 0
    by, bx, best = 0, 0, -1.0
    for y in range(32):
        for x in range(64):
            v = domeseed._lum(pano[y][x])
            if v > best:
                by, bx, best = y, x, v
    assert abs(bx - 15.5) <= 2.0, f"sun column off: {bx} (expected ≈15.5 for az 90)"
    assert abs(by - 10.2) <= 2.0, f"sun row off: {by} (expected ≈10.2 for alt 30)"
    assert best > 50.0, "the disc must be HDR-bright, not a paint smudge"


def test_ingest_external_pano_orientation():
    # external pano convention: u=0.5 = camera forward; red left half, blue right half
    pixels = [(220, 30, 30) if x < 48 else (30, 30, 220)
              for y in range(32) for x in range(96)]
    pano = domeseed.ingest_pano(pixels, 96, 32, cam_yaw_deg=90.0, out_w=64, out_h=16)
    y = 8
    left = pano[y][int((80.0 / 360.0) * 64)]            # world az 80 = 10° left of forward
    right = pano[y][int((100.0 / 360.0) * 64)]
    assert left[0] > left[2] and right[2] > right[0]


# --------------------------------------------------------------------------- build_seed
def test_build_seed_sunny_end_to_end(tmp_path):
    ref = _make_ref(tmp_path, "sunny.png",
                    lambda x, y: (230 - y * 2, 200 - y * 2, 150))
    out = str(tmp_path / "seed.hdr")
    sem = {"sky": "clear", "time_of_day": "afternoon", "sun_active": True,
           "sun_bearing_deg": -45.0, "sun_altitude_band": "mid"}
    meta = domeseed.build_seed(out, ref_path=ref, semantics=sem, cam_yaw_deg=30.0,
                               out_w=64, out_h=32)
    assert meta is not None and meta["source"] == "reference"
    assert meta["reflection_quality"] == "sharp" and meta["blur_passes"] == 0
    assert meta["sun"] is not None
    # bearing −45 from yaw 30 → world azimuth 345
    assert abs(meta["sun"]["azimuth_deg"] - 345.0) < 1e-6
    rows = hdr_min.read_hdr(out)
    assert rows is not None
    peak = max(domeseed._lum(p) for r in rows for p in r)
    assert peak > 20.0, "a sunny seed must carry an HDR sun"

    # determinism: same inputs → byte-identical file
    out2 = str(tmp_path / "seed2.hdr")
    domeseed.build_seed(out2, ref_path=ref, semantics=sem, cam_yaw_deg=30.0,
                        out_w=64, out_h=32)
    with open(out, "rb") as first, open(out2, "rb") as second:
        assert first.read() == second.read()


def test_build_seed_lighting_blur_is_explicit_opt_in(tmp_path):
    ref = _make_ref(tmp_path, "checker.png",
                    lambda x, y: (255, 20, 20) if (x // 4) % 2 else (20, 20, 255))
    sharp_path = str(tmp_path / "sharp.hdr")
    blur_path = str(tmp_path / "blur.hdr")
    sharp = domeseed.build_seed(sharp_path, ref_path=ref, out_w=64, out_h=32)
    blurred = domeseed.build_seed(blur_path, ref_path=ref, out_w=64, out_h=32,
                                  blur_passes=2)
    assert sharp["reflection_quality"] == "sharp"
    assert blurred["reflection_quality"] == "lighting_blur"
    with open(sharp_path, "rb") as sharp_file, open(blur_path, "rb") as blur_file:
        assert sharp_file.read() != blur_file.read()


def test_build_seed_overcast_has_no_disc(tmp_path):
    ref = _make_ref(tmp_path, "flat.png", lambda x, y: (170, 175, 185))
    out = str(tmp_path / "ovc.hdr")
    meta = domeseed.build_seed(out, ref_path=ref, cam_yaw_deg=0.0,
                               semantics={"sky": "overcast", "sun_active": False,
                                          "time_of_day": "overcast_day"},
                               out_w=64, out_h=32)
    assert meta is not None and meta["sun"] is None and meta["overcast_lift"]
    rows = hdr_min.read_hdr(out)
    peak = max(domeseed._lum(p) for r in rows for p in r)
    assert peak < 5.0, "overcast seed must not contain a sun disc"


def test_build_seed_explicit_sun_overrides_semantics(tmp_path):
    ref = _make_ref(tmp_path, "r.png", lambda x, y: (200, 180, 150))
    out = str(tmp_path / "explicit.hdr")
    meta = domeseed.build_seed(out, ref_path=ref, cam_yaw_deg=0.0,
                               semantics={"sun_active": True, "sky": "clear",
                                          "sun_bearing_deg": 90.0},
                               sun_az_deg=200.0, sun_alt_deg=12.0,
                               out_w=64, out_h=32)
    assert meta["sun"]["azimuth_deg"] == 200.0
    assert meta["sun"]["altitude_deg"] == 12.0


def test_hdr_nan_channel_does_not_crash(tmp_path):
    """max(1.0, nan) keeps 1.0 — a NaN in a NON-max channel used to reach int(nan) and
    raise ValueError mid-write. Sanitize is per-channel now."""
    nan, inf = float("nan"), float("inf")
    assert hdr_min.float_to_rgbe(nan, 1.0, inf)[1] > 0    # finite channel survives
    assert hdr_min.float_to_rgbe(nan, nan, nan) == (0, 0, 0, 0)
    p = str(tmp_path / "nan.hdr")
    assert hdr_min.write_hdr(p, [[(nan, 1.0, inf), (0.5, 0.5, 0.5)] * 8])
    back = hdr_min.read_hdr(p)
    assert back is not None
    r, g, b = back[0][0]
    assert r == 0.0 and b == 0.0 and abs(g - 1.0) < 0.02


def test_seed_filename_token_changes_path():
    """Max caches bitmaps by path — a re-seed with changed inputs must change the file
    NAME, or the dome renders the stale pano."""
    a = domeseed.seed_filename("Cam A", "11112222")
    b = domeseed.seed_filename("Cam A", "33334444")
    assert a != b
    assert a.startswith("seed_Cam_A_") and a.endswith(".hdr")
    assert domeseed.seed_filename("Cam A") == "seed_Cam_A.hdr"   # token optional


def test_snap_fov_divides_180():
    for fov in (90.0, 100.0, 120.0, 60.0, 75.0):
        snapped = domeseed.snap_fov(fov)
        assert abs((180.0 / snapped) - round(180.0 / snapped)) < 1e-9


# --------------------------------------------------------------------------- scenarios
def test_scenarios_distinct_and_in_bounds():
    cur = _full_rig_state()
    board = scenarios.build_scenarios(None, cur, camera_yaw_deg=45.0)
    keys = [b["key"] for b in board]
    assert "as_analyzed" not in keys, "no reference → no 'as analyzed' slot"
    assert len(board) >= 3
    for i, b in enumerate(board):
        for k in b["state"].keys():
            assert spec_for(k) is not None
        for j in range(i + 1, len(board)):
            assert b["state"].diff(board[j]["state"]), \
                f"{b['key']} and {board[j]['key']} collapsed to the same rig"


def test_scenarios_as_analyzed_dedupes_its_twin():
    cur = _full_rig_state()
    sem = {"sky": "overcast", "sun_active": False, "time_of_day": "overcast_day",
           "sun_altitude_band": "na", "light_quality": "soft",
           "atmosphere": "light_haze", "wb_kelvin_estimate": 6800.0,
           "practicals_on": False, "contrast_character": "airy"}
    board = scenarios.build_scenarios(sem, cur, camera_yaw_deg=0.0)
    keys = [b["key"] for b in board]
    assert keys[0] == "as_analyzed"
    assert "overcast_soft" not in keys, "identical variant must dedupe against the analysis"


def test_scenarios_practicals_only_with_groups():
    with_groups = _full_rig_state()
    board = scenarios.build_scenarios(None, with_groups, 0.0)
    assert "practicals_dusk" in [b["key"] for b in board]

    no_groups = _full_rig_state()
    no_groups.groups.clear()
    board2 = scenarios.build_scenarios(None, no_groups, 0.0)
    assert "practicals_dusk" not in [b["key"] for b in board2]


def test_scenarios_respect_locks():
    cur = _full_rig_state()
    locked = {"sun.altitude_deg"}
    board = scenarios.build_scenarios(None, cur, 0.0, locks=locked)
    for b in board:
        assert b["state"].get("sun.altitude_deg") == cur.get("sun.altitude_deg"), \
            f"{b['key']} moved a locked parameter"


def test_scenarios_empty_rig_returns_empty_board():
    """No writable rig params → rules can't move anything → every card would be a no-op.
    The board must say so (empty), not render one meaningless candidate."""
    assert scenarios.build_scenarios(None, LightingState(), 0.0) == []


def test_seed_delete_guard_protects_other_cameras():
    """Two cameras share ONE dome: B's pre_seed snapshot can legitimately point at A's
    seed file. A's re-seed must not delete a file B still needs for Restore."""
    from maxgaffer.core.session import Session
    from maxgaffer.maxbridge.controller import _seed_still_referenced

    s = Session()
    a = s.entry("CamA")
    b = s.entry("CamB")
    a.seed_hdri = "/runs/seed_CamA_11112222.hdr"
    b.pre_seed = {"file": "/runs/seed_CamA_11112222.hdr", "rotation": 40.0}
    assert _seed_still_referenced(s, a.seed_hdri, exclude=a)      # B holds it — keep
    b.pre_seed = {}
    b.seed_hdri = "/runs/seed_CamB_33334444.hdr"
    assert not _seed_still_referenced(s, a.seed_hdri, exclude=a)  # nobody else — free


def test_api_as_state_accepts_both_forms_and_rejects_junk():
    """A LightingState fed to from_dict would 'succeed' as an EMPTY state (its .get
    method answers 'values' with 0.0) and silently wipe the camera on adopt."""
    from maxgaffer.api import _as_state

    st = _full_rig_state()
    assert _as_state(st) is st
    round_tripped = _as_state(st.to_dict())
    assert not st.diff(round_tripped)                    # dict form → same rig
    with pytest.raises(TypeError):
        _as_state(["not", "a", "state"])
