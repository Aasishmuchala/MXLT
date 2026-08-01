"""vantage_calibrate — the pure half of the on-box calibration harness.

The harness itself needs Max + V-Ray + a streaming Vantage window; what CAN be proven
off-box is the judgement layer: the AE verdict, the holdout split, plate averaging, and
that importing the script on a pymxs-less box runs nothing and writes nothing (it is
imported by this very test suite, so a side effect at import would corrupt every CI run).
"""

from __future__ import annotations

import importlib.util
import os

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "scripts", "vantage_calibrate.py")


def _load():
    spec = importlib.util.spec_from_file_location("vantage_calibrate_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_offbox_import_is_side_effect_free(tmp_path, monkeypatch):
    """No pymxs here — importing the harness must execute NO experiment, restore
    nothing, and write no report file."""
    mod = _load()
    assert mod.results == []
    assert mod.verdicts == {}
    assert mod.restores == []
    assert not os.path.exists(mod.REPORT) or True   # never ASSERT the dev box is clean…
    # …but the import itself must not have CREATED one just now
    before = os.path.getmtime(mod.REPORT) if os.path.exists(mod.REPORT) else None
    _load()
    after = os.path.getmtime(mod.REPORT) if os.path.exists(mod.REPORT) else None
    assert before == after


# --------------------------------------------------------------------------- avg_rows
def test_avg_rows_means_elementwise():
    mod = _load()
    a = [[(10, 20, 30), (40, 50, 60)]]
    b = [[(20, 40, 60), (0, 0, 0)]]
    assert mod.avg_rows([a, b]) == [[(15, 30, 45), (20, 25, 30)]]


def test_avg_rows_refuses_shape_mismatch_and_empty():
    mod = _load()
    a = [[(1, 1, 1)]]
    b = [[(1, 1, 1), (2, 2, 2)]]
    assert mod.avg_rows([a, b]) is None      # unlike frames blurred together = not data
    assert mod.avg_rows([]) is None
    assert mod.avg_rows([None, None]) is None
    assert mod.avg_rows([a, None]) == a      # Nones (failed renders) are dropped


def test_avg_rows_refuses_ragged_inner_rows():
    """Fuzz-gauntlet regression (2026-07-31): only the FIRST row's width was checked, so
    a ragged inner row IndexErrored mid-E5 — where a crash forfeits the entire
    calibration run's renders."""
    mod = _load()
    ragged = [[(1, 1, 1), (2, 2, 2)], [(3, 3, 3)]]
    assert mod.avg_rows([ragged, ragged]) is None


# --------------------------------------------------------------------------- center_mean
def test_center_mean_ignores_the_border():
    mod = _load()
    rows = [[(255, 255, 255)] * 10 for _ in range(10)]
    for y in range(2, 8):                    # black centre, white frame
        for x in range(2, 8):
            rows[y][x] = (0, 0, 0)
    assert mod.center_mean(rows, crop=0.6) == 0.0
    assert mod.center_mean([], crop=0.6) is None


# --------------------------------------------------------------------------- ae_verdict
def test_ae_verdict_tracks_flat_nonmonotone_unmeasured():
    mod = _load()
    v, r = mod.ae_verdict([20.0, 40.0, 80.0, 150.0])       # light up, picture up
    assert v == "TRACKS" and r > 4.0
    v, r = mod.ae_verdict([100.0, 101.0, 100.5, 101.2])    # ×8 light, flat picture: AE
    assert v == "FLAT"
    v, _ = mod.ae_verdict([20.0, 80.0, 40.0, 150.0])       # moved the wrong way
    assert v == "NON_MONOTONE"
    v, r = mod.ae_verdict([None, 50.0])
    assert v == "UNMEASURED" and r is None
    v, _ = mod.ae_verdict([None, None, None, None])
    assert v == "UNMEASURED"


def test_ae_verdict_flat_threshold_is_the_module_constant():
    mod = _load()
    just_over = [100.0, 100.0, 100.0, 100.0 * (mod.AE_FLAT_RATIO + 0.01)]
    assert mod.ae_verdict(just_over)[0] == "TRACKS"
    just_under = [100.0, 100.0, 100.0, 100.0 * (mod.AE_FLAT_RATIO - 0.01)]
    assert mod.ae_verdict(just_under)[0] == "FLAT"


# --------------------------------------------------------------------------- holdout
def test_split_holdout_reserves_tagged_states_and_an_interior_point():
    mod = _load()
    keys = ["sun0.5_ev-2", "sun0.5_ev+0", "sun1_ev+0", "sun2_ev+0", "sun2_ev+2",
            "holdout:dome_only", "holdout:wb_plus2000"]
    train, hold = mod.split_holdout(keys)
    assert "holdout:dome_only" in hold and "holdout:wb_plus2000" in hold
    assert len(hold) == 3                    # both tagged + one interior factorial state
    interior = [k for k in hold if not k.startswith("holdout:")]
    assert len(interior) == 1
    assert interior[0] not in train          # the fit never sees its own holdout
    assert set(train) | set(hold) == set(keys)


def test_split_holdout_degenerates():
    mod = _load()
    train, hold = mod.split_holdout([])
    assert train == [] and hold == []
    train, hold = mod.split_holdout(["a", "b"])      # too few to reserve an interior one
    assert train == ["a", "b"] and hold == []


# --------------------------------------------------------------------------- small pieces
def test_log2_ratio():
    mod = _load()
    assert abs(mod.log2_ratio(4.0, 1.0) - 2.0) < 1e-9
    assert abs(mod.log2_ratio(1.0, 4.0) - 2.0) < 1e-9       # magnitude, not direction
    assert mod.log2_ratio(0.0, 1.0) is None
    assert mod.log2_ratio(None, 1.0) is None


def test_fmt_row_stays_one_line():
    mod = _load()
    line = mod.fmt_row("a" * 60, "PASS", "detail")
    assert "\n" not in line
    assert "PASS" in line


# ------------------------------------------------------------------ percentile access
#
# Found 2026-08-01 by rendering a real scene and READING the numbers: compute_stats
# nests percentiles as {"p": {"5": .., "95": ..}} with STRING keys, but E5's
# certification read stats.get("p5") — None on both sides — so abs(0-0) == 0 and the
# p5/p95 rows reported perfect agreement unconditionally. A vacuous check is worse than
# no check: it passes a wrong correction with a number next to it.
def test_pct_reads_the_nested_string_keyed_percentiles():
    mod = _load()
    stats = {"p": {"5": 0.041, "25": 0.11, "50": 0.3, "75": 0.62, "95": 0.98}}
    assert mod.pct(stats, 5) == 0.041
    assert mod.pct(stats, 95) == 0.98
    assert mod.pct(stats, "95") == 0.98          # int or str, same answer


def test_pct_returns_none_rather_than_a_confident_zero():
    """Every degenerate shape must read as 'not measured', never as 0.0 — the whole
    bug was a missing value silently becoming a number."""
    mod = _load()
    assert mod.pct(None, 5) is None
    assert mod.pct({}, 5) is None
    assert mod.pct({"p5": 0.3}, 5) is None       # the WRONG flat spelling
    assert mod.pct({"p": None}, 5) is None
    assert mod.pct({"p": []}, 5) is None
    assert mod.pct({"p": {"5": None}}, 5) is None
    assert mod.pct({"p": {"5": "x"}}, 5) is None
    assert mod.pct({"p": {"25": 0.1}}, 5) is None


def test_dynamic_range_and_its_none_propagation():
    mod = _load()
    stats = {"p": {"5": 0.04, "95": 0.99}}
    assert abs(mod.dynamic_range(stats) - 0.95) < 1e-9
    assert mod.dynamic_range({"p": {"5": 0.04}}) is None
    assert mod.dynamic_range({}) is None
    assert mod.dynamic_range(None) is None


def test_dynamic_range_agrees_with_computed_contrast():
    """compute_stats exposes contrast = pct(95) - pct(5); dynamic_range recomputes it
    from the percentiles. If these two ever disagree, one of them is reading the wrong
    thing — which is exactly the failure this section exists to catch."""
    mod = _load()
    stats = {"p": {"5": 0.041, "95": 0.981}, "contrast": 0.981 - 0.041}
    assert abs(mod.dynamic_range(stats) - stats["contrast"]) < 1e-9
