# The Vantage measurement surface — design record (2026-07-31)

**Goal:** widen the Chaos Vantage window grab from its ordinal-only licence (the sun
solve's ranking) into a trustworthy measurement surface, so probes that cost ~60 s in
V-Ray on TULA-class scenes can come from the real-time renderer that — decisive fact,
confirmed by the artist — **renders the actual deliverable**. V-Ray probes measure an
image nobody ever sees; the grab is the only instrument pointed at the product.

**Method:** the design was adversarially stress-tested before implementation (five
independent attack lenses: rendering physics, statistics, operations, product
architecture, vendor-doc evidence). What follows is the design *as amended by that
stress test* — the naïve version of almost every piece below was refuted first.

## Verdicts that shaped the build

| # | Naïve assumption | Verdict | Consequence in code |
|---|---|---|---|
| A1 | Vantage's transfer is a static per-pixel 1D curve | **Not assumable** — auto-exposure exists ("ignores set camera exposure", Chaos docs 125272932), is not queryable, and temporal accumulation makes pixels a function of (state, time) | AE gate (E1) + per-state curve **dispersion** detector + convergence-grade settle |
| A2 | "Vantage exposure cannot be written from Max" (old vgrab docstring) | **Refuted by vendor docs** — Physical Camera ISO/f/shutter/EV/WB all stream (support table 124621427). Real hazards: AE + Vantage-local overrides (3.3 changelog 908558346) | Docstring corrected; EV/WB reach measured per box (E2), never assumed either way |
| A6 | Fit from pixel-registered pairs | **Refuted** — unverified framing model, different PSFs, noise asymmetry ⇒ contrast-flattened curve ⇒ `hot_frac` biased low | **Quantile/CDF matching** is the estimator (registration-free); registered `pixel_pairs` survive only as the spatial-operator *detector* |
| A9 | In-sample residual is the trust signal | **Inverted** — a single-axis sweep absorbs AE into the curve; small in-sample residual is consistent with a broken transfer | **Held-out residual** (states the fit never saw, on axes it never swept) is the only number that gates arming |
| A8 | ~1000× speedup | **~10–50×** — measurement-grade grabs are convergence-bound (est. 1.5–6 s/state on heavy scenes) | `CONVERGE_LIMIT_S` mode; dwell experiment (E3) measures the real knee per box |
| A4 | The live link streams every genome axis | Docs say core axes stream, but Vantage 3.3 **preserves locally-edited properties** — an axis can go silently probe-blind per session | Axis census (E4): every probed axis must visibly move the settle signature |
| A7 | Match in Vantage space directly | Shim direction (grab → V-Ray space) is right for phase 1 — and "phase C" collapses to scoring probes against `g(reference)` later | `delivery_gap()` reports how far the matched VFB sits from the client's Vantage frame |

Kill-switch that survives every failure above: the shipped **ordinal-only** sun solve is
valid under any monotone per-state transfer, auto-exposure included. The floor never
drops below what is on main today.

## What was built

| Piece | File | Role |
|---|---|---|
| The fit | `maxgaffer/core/vtone.py` | Pure, stdlib. `quantile_pairs` / `pixel_pairs` / `merge_samples`; `VTone.fit` (binned medians → PAV isotonic → 256-LUT/channel, flat extrapolation); `residual_on` (held-out MAE), `curve_dispersion` (AE detector), `delivery_gap`; JSON round-trip with provenance. Refuses (< 8 populated levels → `None`) rather than asserting. |
| Persistence | `maxgaffer/maxbridge/config.py` | Sidecar `%LOCALAPPDATA%/MaxGaffer/vantage_tone.json` (never inside config.json — the Options-menu overwrite class of bug stays impossible). Loud when present-but-unreadable. |
| The grab | `maxgaffer/maxbridge/vgrab.py` | `arm_tone`/`disarm_tone`; correction applied after the black test, before the write. A failing correction **refuses the grab** (refuse over lying). `converged=True` mode: poll past first motion until two consecutive signatures agree (`CONVERGE_EPSILON`) — measurement-grade stationarity, bounded by `CONVERGE_LIMIT_S`. Unarmed behaviour is byte-identical to before (tested). |
| The fifth refusal | `maxgaffer/maxbridge/controller.py` `_arm_tone_fit` | Gates, cheapest-lie-first: malformed → no held-out residual → residual > `RESIDUAL_LIMIT` → dispersion > `DISPERSION_LIMIT` (the AE signature) → Vantage build stamp changed. Every refusal names itself; passing arms tone-corrected grabs and logs the delivery gap. Runs start disarmed (no cross-run leaks). |
| The instrument | `scripts/vantage_calibrate.py` | On-box, one command, restore-in-finally, report always written. E0 environment (refuses render legs on V-Ray GPU — checklist #14) → E1 AE gate → E2 EV/WB reach → E3 dwell → E4 axis census → E5 paired fit (K-averaged V-Ray plates × converged grabs, per-state fits → dispersion, pooled train fit → **held-out** residual, metric-space certification on `log_key`/`p5`/`p95`/`hot_frac`, fit saved with provenance). |

Tests: `test_vtone.py` (23 — including the AE simulation that MUST fail loudly),
`test_vgrab.py` (+7), `test_controller_fixes.py` (+5 gate tests, hermetic against a real
fit on the dev box), `test_vantage_calibrate.py` (10, pure helpers + import-side-effect
freedom). Suite: **4022 passed / 37 skipped**.

## Box day (the part only the artist can do)

1. Open the real scene, start the live link, leave Vantage stock and visible.
2. MAXScript listener: `python.ExecuteFile @"<repo>\scripts\vantage_calibrate.py"`
3. Read the table. `E1 FLAT` → disable Vantage auto-exposure (viewport bar), re-run.
   `E5 FIT_OK` → the next `probe_backend: "vantage"` match logs
   "TONE-CORRECTED … held-out residual N/255".
4. Send back `scripts/vantage_calibrate_report.json` — `RESIDUAL_LIMIT` (6.0) and
   `DISPERSION_LIMIT` (4.0) in `core/vtone.py` are PROVISIONAL until real paired plates
   set them, and E3's `stationary_s` calibrates `CONVERGE_LIMIT_S`.

## Deliberately not built (scope, per the stress test)

* No stage beyond the sun solve consumes corrected grabs yet — the render_hook
  quarantine stands until the metric-space certification passes on a real box and a
  positive delivery signal exists for small-delta probes (polish deltas can land under
  `SETTLE_DELTA`, and a picture-moved heuristic cannot certify those).
* No `g(reference)` deliverable-aware scoring mode yet — it needs a trusted g first.
* No auto-raise of an occluded Vantage window (single-monitor boxes fall back to V-Ray,
  correctly and expensively).
