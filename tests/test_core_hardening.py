"""Round-3 stress hardening regressions — every test here reproduces a bug found by
hostile fuzzing/review (2026-07-18) and fails against the pre-fix code.

Themes: malformed input must degrade (None / default / skip), never raise; NaN/∞ must
never become a move, a score, or a stored value; and a cancelled/failed run must never
overwrite an accepted state.
"""

from __future__ import annotations

import json
import math
import os
import struct
import zlib

import pytest

from maxgaffer.core import (consensus, critic, director, domeseed, expose, feedback,
                            genome, hdr_min, omega, parse, planner, png_min, rules,
                            scenarios, scenedigest, session, solver)
from maxgaffer.core.genome import LightingState


# --------------------------------------------------------------------------- helpers
def _png_bytes(width: int, height: int, truncate_ihdr: bool = False) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    if truncate_ihdr:
        ihdr = ihdr[:5]
    def chunk(ctype: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + ctype + payload + b"\x00\x00\x00\x00"
    raw = b"".join(b"\x00" + b"\x80\x80\x80" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def _state(**values) -> LightingState:
    return LightingState(values=dict(values), groups={})


# --------------------------------------------------------------------------- png_min
class TestPngMinHardening:
    def test_truncated_ihdr_returns_none_not_struct_error(self, tmp_path):
        p = tmp_path / "bad.png"
        p.write_bytes(_png_bytes(4, 4, truncate_ihdr=True))
        assert png_min.read_png_rgb(str(p)) is None

    def test_max_dim_zero_does_not_divide_by_zero(self, tmp_path):
        p = tmp_path / "ok.png"
        p.write_bytes(_png_bytes(4, 4))
        assert png_min.read_png_rgb(str(p), max_dim=0) is not None

    def test_valid_png_still_decodes(self, tmp_path):
        p = tmp_path / "ok.png"
        p.write_bytes(_png_bytes(8, 8))
        rows = png_min.read_png_rgb(str(p))
        assert rows and rows[0][0] == (128, 128, 128)


# --------------------------------------------------------------------------- genome
class TestGenomeFinite:
    def test_inf_in_wrap_param_clamps_not_raises(self):
        assert genome.clamp("sun.azimuth_deg", float("inf")) == 0.0
        assert genome.clamp("sun.azimuth_deg", float("nan")) == 0.0

    def test_nan_non_wrap_lands_on_lower_bound(self):
        assert genome.clamp("sun.altitude_deg", float("nan")) == -4.0

    def test_llm_nan_literal_is_rejected_not_full_stepped(self):
        st = _state(**{"sun.altitude_deg": 30.0})
        new, accepted, rejected = genome.apply_changes(st, {"sun.altitude_deg": float("nan")})
        assert accepted == {} and rejected and new.get("sun.altitude_deg") == 30.0

    def test_llm_inf_literal_is_rejected(self):
        st = _state(**{"sun.azimuth_deg": 100.0})
        new, accepted, rejected = genome.apply_changes(st, {"sun.azimuth_deg": float("inf")})
        assert accepted == {} and new.get("sun.azimuth_deg") == 100.0

    def test_limit_step_nan_proposal_does_not_move(self):
        assert genome.limit_step("sun.altitude_deg", 30.0, float("nan")) == 30.0

    def test_from_dict_skips_junk_values(self):
        st = LightingState.from_dict({"values": {"sun.intensity": "abc",
                                                 "sun.altitude_deg": 25.0},
                                      "groups": {"g1": "lots", "g2": 2.0}})
        assert "sun.intensity" not in st.values
        assert st.values["sun.altitude_deg"] == 25.0
        assert "g1" not in st.groups and st.groups["g2"] == 2.0

    def test_from_dict_survives_non_dict_shapes(self):
        assert LightingState.from_dict([1, 2, 3]).values == {}
        assert LightingState.from_dict({"values": [1, 2]}).values == {}

    def test_nan_never_stored_so_diff_stays_honest(self):
        st = LightingState.from_dict({"values": {"sun.azimuth_deg": float("nan")}})
        other = _state(**{"sun.azimuth_deg": 250.0})
        assert st.diff(other)  # a difference MUST be visible


# --------------------------------------------------------------------------- session
class TestSessionHardening:
    def test_load_top_level_list(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text("[1,2,3]")
        assert session.Session.load(str(p)).cameras == {}

    def test_load_junk_state_values(self, tmp_path):
        p = tmp_path / "s.json"
        json.dump({"cameras": {"c": {"state": {"values": {"sun.intensity": "abc"}}}}},
                  p.open("w"))
        s = session.Session.load(str(p))
        assert s.cameras["c"].state is not None  # entry survives, junk value skipped

    def test_load_inf_wrap_value(self, tmp_path):
        p = tmp_path / "s.json"
        p.write_text('{"cameras":{"c":{"state":{"values":{"sun.azimuth_deg":1e999}}}}}')
        s = session.Session.load(str(p))     # was ValueError: math domain error
        # union semantics: a non-finite value is DROPPED per-key (with a warning),
        # never coerced into an invented 0° heading
        assert "sun.azimuth_deg" not in s.cameras["c"].state.values

    def test_string_locks_and_notes_do_not_become_char_sets(self, tmp_path):
        p = tmp_path / "s.json"
        json.dump({"cameras": {"c": {"locks": "sun.intensity", "notes": "too bright"}}},
                  p.open("w"))
        e = session.Session.load(str(p)).cameras["c"]
        assert e.locks == set() and e.notes == []

    def test_preset_loads_junk_value_returns_state_without_crash(self):
        st = session.preset_loads(json.dumps(
            {"maxgaffer_preset": 1, "state": {"values": {"sun.size": "huge"}}}))
        assert st is not None and "sun.size" not in st.values

    def test_save_unserializable_settings_cleans_tmp(self, tmp_path):
        p = str(tmp_path / "s.json")
        s = session.Session(p)
        s.settings["bad"] = object()
        assert s.save() is False
        assert not os.path.exists(p + ".tmp")


# --------------------------------------------------------------------------- solver
class TestSolverHardening:
    def test_malformed_lab_mean_returns_none_not_index_error(self):
        assert solver.solve_wb({"lab_mean": [50, 0]}, {"lab_mean": [50, 0, 5]},
                               6500.0) is None

    def test_none_log_key_returns_none_not_type_error(self):
        assert solver.solve_ev({"log_key": None}, {"log_key": 0.2}, 12.0) is None

    def test_nan_log_key_returns_none_not_blind_slam(self):
        assert solver.solve_ev({"log_key": float("nan")}, {"log_key": 0.2}, 12.0) is None

    def test_nan_b_star_returns_none_not_max_step(self):
        assert solver.solve_wb({"lab_mean": [50, 0, float("nan")]},
                               {"lab_mean": [50, 0, 5]}, 6500.0) is None

    def test_malformed_highlight_on_one_side_uses_full_means_on_both(self):
        # highlight-vs-full is apples-to-oranges; both sides must fall back together.
        # db = 20 - 2 = 18 → delta capped at WB_MAX_STEP 1500 → 6500 + 1500
        out = solver.solve_wb({"lab_mean": [50, 0, 20], "lab_mean_hi": [80, 0]},
                              {"lab_mean": [50, 0, 2], "lab_mean_hi": [80, 0, 3]}, 6500.0)
        assert out == 8000.0

    def test_pinned_at_bound_emits_none_not_noop_write(self):
        assert solver.solve_wb({"lab_mean": [50, 0, 40]}, {"lab_mean": [50, 0, 2]},
                               15000.0) is None
        assert solver.solve_ev({"log_key": 0.9}, {"log_key": 0.1}, -4.0) is None

    def test_sanity_sign_conventions_untouched(self):
        # darker render than ref → EV must go DOWN (V-Ray: higher EV = darker)
        assert solver.solve_ev({"log_key": 0.5}, {"log_key": 0.1}, 12.0) < 12.0
        # warmer ref → kelvin UP
        assert solver.solve_wb({"lab_mean": [50, 0, 20]}, {"lab_mean": [50, 0, 2]},
                               6500.0) > 6500.0


# --------------------------------------------------------------------------- critic
class TestCriticWeights:
    def _stats(self):
        return {"log_key": 0.4, "p": {"5": 0.1, "95": 0.9}, "lum_hist": [0.5] * 10,
                "lab_mean": [50, 1, 2], "hue_hist": [0.2] * 12, "grid": [0.1] * 9}

    def test_nan_weight_does_not_poison_score(self):
        v = critic.score(self._stats(), self._stats(), {"key": float("nan")})
        assert math.isfinite(v.score) and 0.0 <= v.score <= 100.0

    def test_negative_and_string_weights_ignored(self):
        v = critic.score(self._stats(), self._stats(),
                         {"key": -5.0, "color": "lots"})
        assert math.isfinite(v.score)


# --------------------------------------------------------------------------- parse
class TestParseHardening:
    def test_nan_literal_in_deltas_dropped(self):
        out = parse.validate_deltas('{"changes":[{"param":"sun.altitude_deg","value":NaN}]}')
        assert out["changes"] == {}

    def test_infinity_literal_in_deltas_dropped(self):
        out = parse.validate_deltas(
            '{"changes":[{"param":"sun.altitude_deg","value":-Infinity}]}')
        assert out["changes"] == {}

    def test_dict_form_changes_accepted(self):
        out = parse.validate_deltas('{"changes": {"sun.altitude_deg": 10}}')
        assert out["changes"] == {"sun.altitude_deg": 10.0}

    def test_nan_semantic_number_lands_on_default(self):
        out = parse.validate_analysis('{"sun_bearing_deg": NaN, "confidence": Infinity}')
        assert out["sun_bearing_deg"] == 0.0 and out["confidence"] == 0.5

    def test_end_to_end_nan_reply_moves_nothing(self):
        st = _state(**{"sun.altitude_deg": 30.0, "sun.azimuth_deg": 100.0})
        prop = parse.validate_deltas('{"changes":[{"param":"sun.altitude_deg","value":NaN},'
                                     '{"param":"sun.azimuth_deg","value":Infinity}]}')
        new, accepted, _ = genome.apply_changes(st, prop["changes"])
        assert accepted == {} and new.to_dict() == st.to_dict()


# --------------------------------------------------------------------------- omega
class TestOmegaHardening:
    def test_null_payload_degrades_to_omega_error(self):
        with pytest.raises(omega.OmegaError):
            omega.call("k", "s", [], post=lambda *a: (200, "null"))

    def test_list_payload_degrades_to_omega_error(self):
        with pytest.raises(omega.OmegaError):
            omega.call("k", "s", [], post=lambda *a: (200, '["hello"]'))

    def test_extract_text_shapes(self):
        assert omega.extract_text(None) == ""
        assert omega.extract_text({"content": "notalist"}) == ""
        assert omega.extract_text({"content": [{"type": "text", "text": "hi"}]}) == "hi"


# --------------------------------------------------------------------------- rules
class TestRulesHardening:
    def test_junk_cached_semantics_do_not_crash(self):
        st = _state(**{"sun.azimuth_deg": 0.0, "exposure.wb_kelvin": 6500.0})
        out, _ = rules.initial_state(
            {"time_of_day": "afternoon", "sky": "clear", "sun_active": True,
             "sun_bearing_deg": None, "sun_altitude_band": "mid",
             "wb_kelvin_estimate": "warm"}, st, 0.0)
        assert out.get("exposure.wb_kelvin") == 6500.0

    def test_nan_camera_yaw_falls_back_to_zero(self):
        st = _state(**{"sun.azimuth_deg": 10.0})
        out, _ = rules.initial_state({"sun_active": True, "sun_bearing_deg": 30.0},
                                     st, float("nan"))
        assert math.isfinite(out.get("sun.azimuth_deg"))


# --------------------------------------------------------------------------- feedback
class TestFeedbackLocks:
    def test_note_nudges_honor_locks(self):
        st = _state(**{"exposure.ev": 12.0})
        deltas = feedback.nudges_from_note("too bright", st.keys(), [])
        new, applied = feedback.apply_note_deltas(st, deltas,
                                                  locks={"exposure.ev"})
        assert applied == {} and new.get("exposure.ev") == 12.0

    def test_practicals_wildcard_skips_mg_groups(self):
        deltas = feedback.nudges_from_note("practicals too bright", [],
                                           ["MG_fill", "lamps"])
        assert "group.MG_fill" not in deltas and deltas.get("group.lamps") == -0.5


# --------------------------------------------------------------------------- consensus
class TestConsensusTies:
    def test_bool_tie_broken_by_most_confident(self):
        out = consensus.consolidate_analyses([
            {"practicals_on": True, "confidence": 0.99, "time_of_day": "midday"},
            {"practicals_on": False, "confidence": 0.01, "time_of_day": "midday"},
        ])
        assert out["practicals_on"] is True


# --------------------------------------------------------------------------- planner
class TestPlannerFinite:
    _CAT = {"exposure": {"ev", "temperature"}, "node:VRaySun001": {"turbidity"}}

    def test_nan_plan_value_rejected(self):
        ops, rejected, _ = planner.validate_plan(
            '{"ops":[{"op":"set","target":"exposure","prop":"ev","value":NaN}]}', self._CAT)
        assert ops == [] and rejected

    def test_inf_color_channel_rejected(self):
        ops, rejected, _ = planner.validate_plan(
            '{"ops":[{"op":"set","target":"exposure","prop":"ev","value":[Infinity,1,2]}]}',
            self._CAT)
        assert ops == [] and rejected

    def test_nan_placement_rejected(self):
        ops, rejected, _ = planner.validate_plan(
            '{"ops":[{"op":"create_light","light_type":"VRayLight_plane",'
            '"placement":{"bearing_deg":NaN,"distance":250,"height":120}}]}', self._CAT)
        assert ops == [] and rejected

    def test_valid_plan_still_passes(self):
        ops, _, _ = planner.validate_plan(
            '{"ops":[{"op":"set","target":"exposure","prop":"ev","value":11.5}]}',
            self._CAT)
        assert ops and ops[0]["value"] == 11.5


# --------------------------------------------------------------------------- scenarios/digest
class TestBoundaries:
    def test_scenarios_max_count_zero(self):
        st = _state(**{"sun.azimuth_deg": 0.0})
        assert scenarios.build_scenarios(None, st, 0.0, max_count=0) == []

    def test_digest_tiny_max_chars(self):
        raw = {"renderer": {"class": "V-Ray", "props": {"a": 1}}}
        out = scenedigest.to_text(raw, max_chars=10)
        assert len(out) <= 10


# --------------------------------------------------------------------------- expose
class TestExposeHardening:
    def test_nan_ev_is_identity_not_value_error(self):
        px = [(128, 128, 128)]
        assert expose.expose_pixels(px, float("nan"), 12.0, 6500.0, 6500.0) == px

    def test_failed_write_leaves_no_tmp(self, tmp_path):
        pytest.importorskip("PIL")
        src = tmp_path / "a.png"
        from PIL import Image
        Image.new("RGB", (4, 4), (128, 128, 128)).save(src)
        dst = tmp_path / "destdir"
        dst.mkdir()
        assert expose.expose_image_file(str(src), str(dst), 13.0, 11.0, 6500.0,
                                        6500.0) is None
        assert not (tmp_path / "destdir.tmp").exists()

    def test_jpg_destination_gets_jpeg_bytes(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image
        src = tmp_path / "a.png"
        Image.new("RGB", (4, 4), (128, 128, 128)).save(src)
        dst = tmp_path / "out.jpg"
        assert expose.expose_image_file(str(src), str(dst), 13.0, 11.0, 6500.0,
                                        6500.0) == str(dst)
        assert dst.read_bytes()[:2] == b"\xff\xd8"      # JPEG magic, not PNG


# --------------------------------------------------------------------------- hdr_min
class TestHdrMinHardening:
    def test_huge_values_saturate_not_raise(self, tmp_path):
        p = str(tmp_path / "x.hdr")
        assert hdr_min.write_hdr(p, [[(1e39, 0.5, 0.5)] * 16]) is True
        rows = hdr_min.read_hdr(p)
        assert rows and all(math.isfinite(c) for px in rows[0] for c in px)

    def test_failed_write_leaves_no_partial_file(self, tmp_path):
        # non-3-tuples raise mid-write — the truncated .hdr must be cleaned up
        p2 = str(tmp_path / "y.hdr")
        assert hdr_min.write_hdr(p2, [[(1.0, 2.0)] * 4]) is False
        assert not os.path.exists(p2)

    def test_old_style_rle_returns_none(self, tmp_path):
        p = tmp_path / "old.hdr"
        p.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 32\n"
                      + bytes([1, 1, 1, 1]) * 32)
        assert hdr_min.read_hdr(str(p)) is None


# --------------------------------------------------------------------------- domeseed
class TestDomeseedHardening:
    def test_nan_yaw_does_not_crash(self):
        rows = domeseed.synthesize_pano([(128, 128, 128)] * 64, 8, 8, float("nan"))
        assert rows and all(math.isfinite(c) for r in rows for p in r for c in p)

    def test_degenerate_inputs_return_empty(self):
        assert domeseed.synthesize_pano([], 0, 0, 0.0) == []
        assert domeseed.ingest_pano([(1, 2, 3)], 4, 4, 0.0) == []

    def test_normalize_key_nan_guard(self):
        rows, s = domeseed.normalize_key([[(float("nan"), 0.1, 0.1)] * 8] * 4)
        assert s == 1.0

    def test_sun_altitude_clamped_to_90(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image
        ref = tmp_path / "ref.png"
        Image.new("RGB", (16, 16), (140, 130, 120)).save(ref)
        meta = domeseed.build_seed(str(tmp_path / "s.hdr"), ref_path=str(ref),
                                   semantics={"sky": "clear", "sun_active": True,
                                              "time_of_day": "midday"},
                                   sun_az_deg=10.0, sun_alt_deg=95.0)
        assert meta and meta["sun"]["altitude_deg"] == 90.0
        assert meta["sun"]["pixels"] > 0      # the disc actually exists near the zenith

    def test_build_seed_nan_yaw_end_to_end(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image
        ref = tmp_path / "ref.png"
        Image.new("RGB", (16, 16), (140, 130, 120)).save(ref)
        meta = domeseed.build_seed(str(tmp_path / "s.hdr"), ref_path=str(ref),
                                   cam_yaw_deg=float("nan"))
        assert meta and math.isfinite(meta["cam_yaw_deg"])


# --------------------------------------------------------------------------- director
class _FakeHooks:
    def __init__(self, render_ok=True):
        self.applied = []
        self.render_ok = render_ok

    def make(self):
        from maxgaffer.core.director import Hooks
        return Hooks(
            apply=lambda st: self.applied.append(st.copy()),
            render=lambda tag: ("frame.png" if self.render_ok else None),
            stats=lambda path: {"log_key": 0.3},
            llm_deltas=lambda ctx: '{"changes": []}',
        )


class TestDirectorAbort:
    def test_cancel_before_first_render_applies_nothing(self):
        fake = _FakeHooks()
        hooks = fake.make()
        hooks.should_cancel = lambda: True
        st = _state(**{"sun.altitude_deg": 30.0, "exposure.ev": 12.0,
                       "exposure.wb_kelvin": 6500.0})
        res = director.run_match(st, {"log_key": 0.5}, {}, hooks,
                                 director.MatchConfig(max_iterations=3))
        assert res.stop_reason == "cancelled" and res.best_score is None
        assert fake.applied == []          # the scene was never touched

    def test_first_render_failure_returns_unmeasured(self):
        fake = _FakeHooks(render_ok=False)
        hooks = fake.make()
        st = _state(**{"sun.altitude_deg": 30.0})
        res = director.run_match(st, {"log_key": 0.5}, {}, hooks,
                                 director.MatchConfig(max_iterations=3))
        assert res.stop_reason == "render_failed" and res.best_score is None
        assert len(fake.applied) == 1      # the one pre-render apply, nothing after

    def test_run_polish_returns_five_values(self):
        fake = _FakeHooks()
        hooks = fake.make()
        st = _state(**{"exposure.ev": 12.0})
        out = director.run_polish(st, 50.0, {"log_key": 0.3}, hooks,
                                  director.MatchConfig(polish_rounds=1,
                                                       polish_max_probes=2))
        assert len(out) == 5

    def test_polish_escapes_compensation_ridge(self, monkeypatch):
        """A rotated valley (tonal↔geometry compensation) is invisible to single-axis
        moves — only the diagonal escape can ride it. Without the escape this stalls
        at 98.0 forever (the v0.9.7 deep-match regression)."""
        cur = {"st": None}

        def landscape(ref, c, weights=None):
            ev, wb = c["ev"], c["wb"]
            # valley floor along (wb-6500)/1000 == ev-12, summit at ev=12/wb=6500
            s = 100.0 - 40.0 * ((wb - 6500.0) / 1000.0 - (ev - 12.0)) ** 2 \
                - 2.0 * (ev - 12.0) ** 2
            return critic.Verdict(score=s)

        monkeypatch.setattr(director.critic, "score", landscape)
        hooks = director.Hooks(
            apply=lambda st: cur.__setitem__("st", st.copy()),
            render=lambda tag: "x.png",
            stats=lambda path: {"ev": cur["st"].get("exposure.ev"),
                                "wb": cur["st"].get("exposure.wb_kelvin")},
            llm_deltas=lambda ctx: "{}",
        )
        start = _state(**{"exposure.ev": 11.0, "exposure.wb_kelvin": 5500.0})
        best, score, probes, converged, proven = director.run_polish(
            start, 98.0, {}, hooks, director.MatchConfig(polish_rounds=6))
        assert score > 99.0      # single-axis alone provably stalls at 98.0
        assert best.get("exposure.ev") > 11.5 and best.get("exposure.wb_kelvin") > 6000.0


class TestSymmetricHighlightWB:
    """v0.9.7 regression: per-image highlight-clip exclusion mismatched the WB populations
    on same-scene matches and stalled deep match below the reachable 99."""

    def test_inclusive_mean_used_when_both_sides_clip_heavily(self):
        ref = {"lab_mean": [50, 0, 10], "lab_mean_hi": [80, 0, 2],
               "lab_mean_hi_full": [80, 0, 12], "hi_clip_frac": 0.5}
        cur = {"lab_mean": [50, 0, 0], "lab_mean_hi": [80, 0, 0],
               "lab_mean_hi_full": [80, 0, 2], "hi_clip_frac": 0.4}
        # inclusive: db = 12 - 2 = 10 → +900K; exclusive would read 2 - 0 = 2 → +180K
        assert solver.solve_wb(ref, cur, 6500.0) == 6500.0 + 900.0

    def test_exclusive_mean_kept_for_cross_scene(self):
        ref = {"lab_mean": [50, 0, 10], "lab_mean_hi": [80, 0, 2],
               "lab_mean_hi_full": [80, 0, 12], "hi_clip_frac": 0.0}
        cur = {"lab_mean": [50, 0, 0], "lab_mean_hi": [80, 0, 0],
               "lab_mean_hi_full": [80, 0, 2], "hi_clip_frac": 0.4}
        assert solver.solve_wb(ref, cur, 6500.0) == 6500.0 + 180.0

    def test_metrics_stores_both_highlight_means(self, tmp_path):
        pytest.importorskip("PIL")
        from PIL import Image
        im = Image.new("RGB", (16, 16), (140, 130, 120))
        for y in range(4):                       # blown top quartile (clipped = no chroma)
            for x in range(16):
                im.putpixel((x, y), (255, 255, 255))
        p = tmp_path / "img.png"
        im.save(p)
        from maxgaffer.core import metrics
        s = metrics.compute_stats(str(p))
        assert s["hi_clip_frac"] > 0.15
        assert len(s["lab_mean_hi"]) == 3 and len(s["lab_mean_hi_full"]) == 3
        assert s["lab_mean_hi"] != s["lab_mean_hi_full"]
