# Path to 10 — consolidated stress-test findings (2026-07-18)

> **STATUS — EXECUTED (commit ca18eff, same day).** Every Tier 1 item (1–10), every
> Tier 2 item (11–15), and Tier 3 items 16 (dedupe + png_min), 19, 20 and 21 are
> closed, covered by 20 new regressions (`tests/test_path_to_10.py`, suite 189
> green) and re-verified live on the box: the recovery experiment now runs with
> `software_exposure` OFF and the new runtime check auto-detects the inert V-Ray
> GPU host ("+2 EV moved the render only 0.00 stops"), flips the flag, and the
> loop converges — EV err 0.00, score 30.6→98.3. Remaining as future speed work
> (explicitly outside the 10 bar): numpy fast path, EXR loop renders, and the
> skip-render-for-EV/WB-probes optimization (do it together with EXR).

Sources: 3 independent adversarial code reviews (exposure/controller integration ·
director core engine · integration/UX/perf), a 10-probe live battery in a real Max
2026 + V-Ray GPU session, and measured perf benches. Live battery: **10/10 PASS**
(locks, cancellation, clamps/wrap, restore round-trip, 20 rapid applies @74 ms, flag
interplay, unicode camera names, sidecar persistence, sunless rig, perf). Suite: 169
green. The gaps below are what separates the current build (9/10) from a 10.

Confidence = reviewer's stated confidence. Line numbers as of commit d31727e.

## Tier 1 — correctness (the 10 is blocked until these are closed)

1. **Software exposure is wired into only 1 of 5 render call-sites.** `run_match`'s
   `render_hook` exposes; `render_finals_vray` (the actual deliverable!, C90),
   `refine()`'s ensemble probe (director's-note EV/WB edits invisible to branch
   scoring, C88), `probe_score`/plan-effect (C82) and the BOARD (`run_scenarios` —
   candidates differing only in WB score identically, C90) all render raw.
   Fix: one shared `_render_exposed(cam, path, w, h, state)` helper used by all five
   sites; retires the `_sw_state` side-channel. ~2–3 h.
2. **Auto-detect the inert exposure host** instead of a hand-set config flag: cheap
   runtime 2-probe EV-delta check (measured log2 ratio vs expected 2^Δ), auto-set
   `software_exposure=True` with a log line; extend `onbox_spikes.ev_direction` to
   assert magnitude, not just sign. ~1–2 h.
3. **`critic.score({}, {}) == 100.0`** — five of six components default missing data
   to their perfect-match value; a present-but-empty stats dict can end a match as
   `target_reached` on a false 100 (C85). Gate each component on its source keys
   (the `direction` component already does this). ~1 h.
4. **LLM deltas aren't capability-gated** — a hallucinated `dome.*`/`group.*` key on
   a rig that lacks it is fabricated at the spec's lower bound, persists into
   `state_table`/session/run.json, and legitimizes itself to the model on later
   iterations (C82). Gate `apply_changes` on `key in state.values` /
   known groups, matching rules/solver/feedback. ~1 h.
5. **No exception safety in `run_match`/`run_polish`** — any hook/critic raise skips
   the keep-best epilogue and leaves the last exploratory state applied; concretely
   triggerable via a wrong-length `lab_mean` (C80). `try/finally` re-apply best +
   harden `critic.score` against malformed fields. ~1–2 h.
6. **`best_render` stale in metrics-unavailable (LLM-visual) runs** — UI shows the
   iteration-0 *before* thumbnail as the final result; run.json matches (C92).
   Reassign in the epilogue. ~30 min.
7. **`apply.py` silently drops orphaned `dome.*`/`group.*` state** (saved state vs a
   since-deleted dome/renamed layer) — the parallel `sun.*` branch warns, these
   don't (C85). Mirror the warning. ~30 min.
8. **Render-mode match with a missing reference file raises** `RuntimeError` instead
   of the graceful "reference file not found — re-bind" log that no-render mode gets
   (live-probe finding). ~30 min.
9. **Unsaved-scene session collision** — two different never-saved scenes share the
   cached Session (`scene_path()==""`), so camera states from scene A can silently
   apply in scene B via `apply_on_select` (C85). Cache-bust on file-reset callbacks
   or refuse per-camera state for unsaved scenes. ~1–2 h.
10. **Clipped highlights bias the WB solve** — software-exposed 8-bit frames clip to
    neutral white, dragging the highlight-quartile b\* toward 0 and overshooting
    kelvin in bright exteriors (C80). Exclude per-channel-saturated pixels from
    `lab_mean_hi` (or expose a clip fraction and down-weight). ~1–2 h.

## Tier 2 — product/UX

11. **Settings UI gaps**: `software_exposure` (load-bearing, currently config.json
    hand-edit only), `final_render_backend`, `vantage_exe` (the exposed
    `vantage_console` field gates a backend that has no UI toggle) (C85). ~1 h.
12. **Group dimmers silently no-op on disabled lights** — classify/read/apply never
    touch `LIGHT_ON`; a slider move on an all-off group changes nothing with no
    warning; an authored-0 baseline can never be raised (C82). Surface enabled
    state + warn. Small–medium.
13. **`ceiling_converged` overclaims** — the diminishing-returns early-exit logs
    "no fine move improves — this IS the ceiling" without ever testing fine steps.
    Split floor-exhausted vs plateau, soften the message. Small.
14. **Deep match ≈ 138 renders ≈ ~20 min/camera** at defaults with no ETA or interim
    checkpoint surfaced to an interactive user. Show budget/progress; consider
    lowering interactive polish caps. Small.
15. **`stall_patience` counts slump iterations** — one marginal gain + one slump can
    stop a run as "stalled" before slump-revert's own 2-strike logic engages.
    Decouple the counters. Small.

## Tier 3 — performance & polish

16. **`compute_stats` hot loop**: dedupe the double `_srgb_to_linear` (6→3 pow calls
    per pixel, <1 h); fix `png_min` subsample step (480 px // 256 == 1 → the
    stdlib floor processes every pixel, <1 h); optional numpy fast path (the
    docstring already advertises one that doesn't exist, C90; few h). Measured now:
    stats ~166 ms, software-expose ~197 ms @480×270 — small next to a ~9 s render,
    so this tier is about deep-match totals (up to ~138 calls/run) and the finals
    path.
17. **Skip re-render for EV/WB-only polish probes** (reuse the last raw buffer +
    software exposure). Only valid when software exposure is on AND the host is
    confirmed inert; pair with EXR loop renders (8-bit clipping otherwise biases
    what a re-render would reveal). Medium — do after #1/#2 and ideally with #18.
18. **EXR/float loop renders** for HDR headroom (kills the clip-bias class at the
    root and makes #17 exact). Medium.
19. **`_ProgressRelay` connects to a lambda** — Qt can't marshal to the main thread;
    vantage_cli progress would touch widgets from the worker thread (C80). Bind to
    a method. ~15 min.
20. **Docs drift**: README/api docstring reference nonexistent
    `render_cameras_vantage` (real: `render_cameras`, V-Ray default) (C95); SPEC
    says 114 tests in one place, 77 in another; README checklist table stops at #16
    while spikes/BOX-DAY go to #19; `no_renders`/`software_exposure` undocumented.
    ~1 h.
21. Misc tiny: zero-pixel guard in `compute_stats`; atomic write in
    `expose_image_file`; `os.makedirs` inside `render_frame`'s try; align the
    expose identity-epsilon with the solver deadband; glob the Max-language startup
    folder in install.bat instead of hardcoding `ENU`.

## What the stress test verified as SOUND (no action)

Slump-revert re-measures before acting; sweep's partial-measurement guard; genome
clamp/wrap including antipodes and repeated-delta walks; rules capability gating;
feedback note parser (empty/foreign/contradictory/intensifier scoping); dock
threading contract for all standard flows; render size save/restore on all paths;
ref-stats caching; locks end-to-end; cancellation; restore round-trip; sidecar
persistence for saved scenes; sunless-rig degradation; the software-exposure sign
convention (algebraically an exact one-shot solve).

**Bottom line: Tier 1 + Tier 2 + the docs item ≈ 2–3 focused days to a defensible
10.** Tier 3 is the difference between a 10 and a fast 10.
