# MaxGaffer — Specification (v2, post stress-test)

**Owner:** Aasish Muchala · **Type:** internal Sthyra pipeline tool (not sold) ·
**Target:** 3ds Max 2026 (Py 3.11, PySide6) + V-Ray 7 + Chaos Vantage 3.x ·
**Sibling of:** MaxDirector (same Omega gateway, same hexagon, key borrowed automatically) ·
**Status:** v0.9.7 SOFTWARE EXPOSURE 2026-07-18 · renderer-independent EV/WB (auto-detected inert hosts) + no-render mode + stress-audit hardening · tests: run `pytest tests -q` for the live count (170+) · live benchmark: Phase C 99.14 ASSERTED (deterministic legs from basin) · Phase B full live pipeline 95.95 (EV landed exactly, ceiling proven) · on-box recovery: EV err 0.00, score 30.6→98.3 ·
awaiting on-box bring-up (tasks/plan.md P0).

## 1. Summary
Pick a camera from the shot board, bind a lighting reference image, press MATCH LIGHTING.
MaxGaffer analyzes the reference, first-guesses a rig from gaffer craft tables, then
iterates the scene's sun, HDRI dome, exposure, white balance and practical-light groups
until the render's LIGHT matches the reference — while the Vantage live link mirrors every
step in real time. Every camera keeps its own reference + matched state; "render all
matched cameras through vantage_console" is one button.

## 2. Locked decisions (v2 — stress-test survivors are law)

| Decision | Choice |
|---|---|
| Refine loop | director's notes → craft-table instant nudges (LOCAL intensifiers) + 3-lens ensemble (exposure/geometry/mood agents, every branch rendered + scored, winner → deep match w/ note pinned above the reference analysis); notes persist per camera; reference swappable mid-conversation |
| Deep match | annealed steps/deadbands + LLM-free adaptive coordinate line search (EV/WB axes included — anti-compensation-drift) to ≥99 when reachable, else a PROVEN ceiling (2 diminishing-return rounds or step-floor exhaustion) reported as content-gap |
| Split of powers | **math owns EV/WB** (histogram solver, highlight-quartile WB) · **LLM owns semantics** (3-sample ANALYZE consensus; ≤4 bounded changes/iter, trajectory-aware) · **sweep owns coarse sun azimuth** (LLM multiple-choice × direction-metric cross-check) · **critic owns accept/revert** (incl. direction component) · **human owns taste** (locks, sliders, restore, plan preview + measured plan effect) |
| Genome as gate | every proposal validated: unknown → dropped, locked → refused, bounds → clamped, per-iter steps → limited; 180° wrap resolves clockwise (deterministic antipode) |
| Trust model | **snapshot-first**: pre-match state auto-saved per camera + Restore button · one undo record per apply · matches are explorations, never commitments |
| Baselines | practical-group authored multipliers keyed by **light NAME**, persisted in session, **adopt-once, never re-captured** (kills the 0-poisoning failure); `forget_baseline` = explicit re-author hook |
| Analytic leash | EV/WB solver bounded to **±4 EV / ±3000 K total per run**; ≥2 leash hits → explicit albedo-mismatch diagnosis in the log ("lock EV, set by eye") |
| Contamination guard | analytic EV move ≥1.5 stops in an iteration → LLM intensity/group proposals for that iteration dropped (geometry survives); drops recorded in the audit trail (extend, never overwrite) |
| Stats | tone + color mood + DIRECTION (mean-centered 3×3 luminance grid — cross-scene-safe) — no SSIM; exposure key **center-weighted 60/40**; WB chroma from the **highlight quartile** (white-patch assumption, albedo-trap counter) |
| LLM wire | Omega `/v1/messages`, opus-4.8 vision, **no tools** (schema-in-prompt), base64 image blocks, retries w/ backoff, **one strict retry** on non-JSON analyze reply |
| Threading | pymxs on Max's MAIN thread, always · gateway/sidecar/vantage-batch on workers via injectable `io` runner · worker progress → Qt signal relay · camera board locked while a run is live |
| Dependencies | **stdlib floor** (urllib client + built-in PNG reader — loop stats can never fail); Pillow optional upgrade; no torch, no OpenCV, no required pip package |
| Loop renders | in-Max V-Ray at `loop_width` (480 px), VFB off, size save/restored — **sampler settings are never touched** (render setups belong to the artist) |
| Final renders | DEFAULT: per-camera V-Ray production renders in Max (stock **Vantage 3.x removed its render CLI** — Chaos-confirmed, Dev Edition only). Vantage-quality path: per-camera `vrayExportVRScene` (verified `startFrame:/endFrame:`) → Vantage in-app Batch Render queue via one-click export+launch. Legacy CLI kept behind `final_render_backend:"vantage_cli"` |
| Dome rotation | HDRI texmap horizontal-rotation spinner first; fallback **world-Z rotation at pivot via explicit matrix composition** W·T(−p)·Rz·T(p) (never `rt.rotate` — context-dependent coordsys) |
| Exposure hosts | scene V-Ray exposure control (auto-created via verified `vrayCreateVRayExposureControl()` when absent) → native Physical camera **direct Target-EV** (`exposure_gain_type=1` + `exposure_value`, doc-verified) → legacy VRayPhysical via ISO math (`shutter_speed` is 1/s vs native `shutter_length_seconds` — units handled per-property) → none (auto-lock) |
| WB conventions | spinner-kelvin ≡ swatch = illuminant color (`kelvin_to_rgb(K)` directly, no mired mirror); higher K → warmer render; on-box item #6 is the one visual sign check |
| Persistence | `<scene>.maxgaffer.json` sidecar (cameras, states, locks, semantics cache, pre-match, baselines, settings); unsaved scene → in-memory + loud warning |
| Property access | candidates-tuples everywhere, per-parameter fault isolation, gaps recorded in `rig["notes"]` and surfaced in the log |

## 3. Architecture (hexagon, test-enforced)

```
maxgaffer/core/       PURE python, zero pymxs/Qt (guard test) — runs + tests on macOS
  genome.py             parameter table: bounds, steps, wrap, log-scale, analytic, locks
  metrics.py            center-weighted key, percentiles, LAB, hue/lum histograms, EMD
  png_min.py            stdlib PNG reader (8-bit RGB(A)/gray, all 5 filters) — the floor
  solver.py             EV (log2 key ratio, deadband 0.15, step 2.5) + WB (LAB b*, 90K/unit)
  critic.py             0-100 tonal verdict: key .22 envelope .18 hist .20 color .25 hue .15
  rules.py              semantics → first guess (altitude/turbidity/size/practicals tables)
  director.py           the loop: keep-best, slump-revert, stall, leash, contamination guard
  session.py            per-camera entries + adopt-once baselines + pre-match snapshots
  prompts.py / parse.py ANALYZE · DELTAS · SWEEP, schema-in-prompt + strict shape validation
  omega.py / colortemp.py  gateway client (ported verbatim) · kelvin↔RGB
maxgaffer/maxbridge/  the ONLY pymxs importer
  scene.py              cameras+yaw, rig classify (sun/dome/layer-groups), sun angles, dome rot
  exposure.py           host abstraction (EC → physical cam → none)
  apply.py              state↔scene, name-keyed baselines, one undo per apply
  render.py             loop frames + Max-bitmap reference transcode (universal ingest)
  vantage.py            live-link probe (globals→actionMan scan→manual msg), export, CLI batch
  controller.py         session/rig/stats/LLM glue + run_match wiring + vantage two-phase
  config.py             %LOCALAPPDATA%/MaxGaffer/config.json (+ MaxDirector key borrow)
  draft.py              opt-in draft sampler: crash-safe snapshot → apply → restore/recover
maxgaffer/api.py      public API (MaxDirector LightMatch stage): match_camera / batch / vantage
maxgaffer/ui/dock.py  PySide6 dock: camera board · reference · match/batch · rig · vantage
maxgaffer/sidecar/    optional Pillow stats/b64 CLI for a system python
scripts/              install.bat · preflight.py
tests/                pure-core pytest suite (170+; `pytest tests -q` is the source of
                      truth), incl. stress rounds, no-render + software-exposure regressions
```

## 4. The match pipeline (with guard placement)

```
⓪ PLAN     (default ON) full digest (getPropNames on renderer/env/exposure/lights/cams)
           → ►LLM change plan → digest-grounded validation → preview/auto → execute
           (one undo, MG_ lights layer, before/after capture)      [scene-wide agent]
① SELECT   camera board → set viewport cam; apply saved per-camera state (toggle)
② BIND     reference image per camera (session-persisted; new ref invalidates analysis)
③ SNAPSHOT pre-match state saved                                    [restore-anytime]
④ ANALYZE  ►LLM  ref → semantics JSON (cached; 1 strict retry)
⑤ GUESS    rules tables: altitude band→deg, bearing+cam_yaw→azimuth, turbidity, size,
           WB estimate, practicals on/off — unlocked params only, rig-present params only
⑥ SWEEP    (default ON) 8 azimuth probes → LLM multiple-choice → azimuth   [skip: heavy scene]
⑦ ITERATE  ≤N: apply → render 480px → stats → critic score
           → keep-best / slump-revert(2) / stall(2) / target stop        [critic owns]
           → analytic EV/WB (deadbands, per-iter clamps, RUN LEASH ±4EV/±3000K)
           → ►LLM deltas (sees ref+render+state table+history)
             → genome validation → CONTAMINATION GUARD if |ΔEV|≥1.5     [drop intensity moves]
⑧ LAND     best state re-applied · per-camera state+score+locks saved · leash diagnosis if hit
⑧·5 REPORT "scene changed" popup: plan values before→after · lights placed · loop diffs
⑨ HAND     rig sliders (live-apply → Vantage mirrors) · locks · Restore pre-match
⑩ SHIP     all matched cameras → apply each state → vrscene each → vantage_console batch
           (exports main-thread; renders on worker; progress via Qt relay)
```

## 5. Data model

Genome (`values` + dynamic `group.<layer>` factors):
`sun.enabled/azimuth_deg/altitude_deg/intensity/size/turbidity · dome.enabled/rotation_deg/
intensity · exposure.ev/wb_kelvin (ANALYTIC) · group.* (log-step 1.0, bounds 0–10)`.

Session sidecar: `{version, cameras: {name: {reference, state, score, matched_at, locks,
semantics, pre_match}}, baselines: {light_name: authored_multiplier}, settings}`.

## 6. Reliability model (honest)

Deterministic (genome, solver math, critic, apply/read-back, session): **~95%** — tested.
Bounded LLM (semantics, deltas, sweep pick): **~85-90%** — junk tolerated, worst case is a
wasted iteration, never a corrupted scene. Ill-posed (cross-scene exposure under albedo
mismatch): **bounded, not solved** — leash + center-weighting + diagnosis; the human locks
EV and eyes it. Property-name drift on the box: **degrades per-parameter with named
warnings**, fixed once via the checklist.

**Expected outcome per run: 70–85% of the visual match** (sun geometry, exposure, WB,
balance land; the last 15% is taste). The tool kills the blank-page hour, not the grade.

## 7. Failure modes → designed responses

| Failure | Response |
|---|---|
| Albedo trap (white ref, dark scene) | center-weighted key · run leash · 2-hit diagnosis → lock EV |
| LLM counteracts a big EV fix | contamination guard drops intensity moves that iteration |
| Baseline poisoning after a 0-dim | adopt-once name-keyed baselines — structurally impossible now |
| Match ruins a good rig | pre-match snapshot + Restore button |
| Vantage batch freezes Max | two-phase: main-thread exports, worker renders, Qt relay progress |
| Camera switch mid-run | board guarded while busy |
| Sun in a Daylight assembly | rig note warns (controller-locked transform); checklist #11 |
| No sun / no dome / no exposure host | params absent from genome; rules skip; UI auto-locks |
| Gateway down / junk replies | typed errors surfaced; analyze retries once; loop survives junk |
| Unsaved scene | in-memory session + loud warning |

## 8. Non-goals
Selling/distribution · volumetrics & aerial
perspective matching · multi-dome · exact HDRI *content* matching (rotation/intensity only)
· sampler management · closing the taste gap without a human.

## 9. Success criteria
- **C0 (bring-up):** preflight all-green in Max listener; 14-item checklist done; WB
  direction (#6) visually confirmed.
- **C1 (first blood):** a real Sthyra scene + real reference reaches ≥75 critic score or
  visual acceptance in ≤2 runs; restore + per-camera recall verified.
- **C2 (shot board):** a multi-camera scene batch-renders overnight through
  vantage_console, every shot under its own matched light, zero manual steps after launch.

## 10. v0.9 addendum — dome seed + scenario board (amends §8)

§8's "no HDRI content matching" is REVISED: v0.9 *generates* dome content from the
reference (it still never edits a user's own HDRI files). Locked decisions:

- **Seed is deterministic and local.** `core/domeseed.py` synthesizes the pano
  (mirror-fold at a 180-divisor FOV for back-seam continuity, sky/ground extension
  constant at the poles, wrap-aware blur, mean-luminance normalization) and
  `core/hdr_min.py` writes real Radiance RGBE (new-style RLE in the 8..32767 width
  band — strict loaders assume it). No model, no cloud, byte-reproducible.
- **World-oriented pano; rotation stays genome-owned.** Column azimuth 0 = north; the
  dome's rotation is zeroed at bind and remains a live loop parameter (any V-Ray u-origin
  constant is measured once at Spike E and absorbed either way).
- **Generative panos enter through the same gate.** `build_seed(pano_path=…)` ingests an
  external equirect (DiffusionLight-class estimation) via identical orient/inject/write —
  the upgrade seam is a file path, never a bundled ML dependency.
- **Restore stays honest.** The dome's pre-seed texture/rotation snapshots once into the
  session (`pre_seed`); Restore re-binds it (texmap disabled when there was none) and
  clears the snapshot.
- **Board candidates speak semantics, not genome.** Variants are ANALYZE-vocabulary
  overrides through `rules.initial_state` — one semantics→state mapping to maintain, and
  an on-box craft fix corrects first guess and board together. The board probe-renders,
  scores via the critic only when a reference is bound, re-applies the found state on
  exit, and ADOPT is an ordinary `record_match`.
