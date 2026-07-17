# MaxGaffer — Execution Plan (v2, post stress-test)

Everything off-Max is DONE (core + bridge + UI + 69 tests + round-2 fixes). What remains is
on-box bring-up → first live match → production trial → v0.2 backlog. Each phase has
acceptance criteria; nothing advances on vibes.

---

## P0 — On-box bring-up (NOW AUTOMATED: one command, ~10 min)

Run `scripts\install.bat`, restart Max, open a THROWAWAY copy of a real scene, then:
`python.ExecuteFile @"C:\<repo>\scripts\onbox_spikes.py"` — it executes Spikes A–D below
as measured checks (probe renders for the sign conventions) and writes a PASS/FAIL report.
Fix any FAIL's named candidates tuple, re-run, commit. Manual leftovers: live-link menu
label if the probe says MANUAL (#9), VRAM watch on a heavy scene (#14).

The LLM leg needs nothing on the box — verified live 2026-07-16 (`live_gateway_smoke.py`
4/4, `sim_match.py` Phase A asserted 98.8). The manual spike walkthrough below remains as
the reference for diagnosing any FAIL.

### Spike A — property-name audit (~20 min) · checklist #1 #2 #3 #4 #5
Listener:
```maxscript
showProperties (getNodeByName "VRaySun001")        -- intensity_multiplier? size_multiplier? turbidity?
(getNodeByName "VRayLightDome").type               -- dome enum: expect 1
showProperties dome.texmap                          -- horizontalRotation?
showProperties SceneExposureControl.exposureControl -- .ev? .temperature? WB mode enum ints?
showProperties (getNodeByName "PhysCamera001")      -- ISO / f_number / shutter names
```
Patch targets: `scene.SUN_* / DOME_TEX_ROT / _dome_type_value` · `exposure.EC_* / CAM_* /
_nudge_wb_mode_*`. **Accept:** preflight "scene rig" line shows sun=yes dome=yes and the
expected groups; no ⚠ property warnings when moving each RIG slider once.

### Spike B — sign conventions (~15 min) · checklist #6
1. RIG slider `exposure.wb_kelvin` +2000 → loop render must be **warmer**. If inverted,
   report before touching code — the flip lives in ONE place (`solver.solve_wb` sign).
2. `exposure.ev` +2 → darker. 3. `sun.altitude_deg` 6° → long shadows; azimuth slider
   swings shadow direction; `dome.rotation_deg` spins the HDRI (watch in Vantage).
**Accept:** all four directions visually correct.

### Spike C — render / export / Vantage CLI (~30 min) · checklist #7 #8 #10
1. MATCH with 1 iteration, sweep off → run folder has `iter00.png` at 480×270, render
   setup untouched after.
2. `vrayExportVRScene "C:\tmp\t.vrscene"` → file exists; then with kwargs.
3. Hand-run one `vantage_console` command from `vantage.vantage_command` → PNG lands.
**Accept:** "Final render (selected)" produces a Vantage still end-to-end from the UI.

### Spike D — live link, daylight, contention (~25 min) · checklist #9 #11 #13 #14 (+#12)
1. **Start live link** button: if "no entry point found", start via V-Ray menu, note the
   exact menu label → pin it in `vantage.LIVE_LINK_GLOBALS`/action scan.
2. Daylight-assembly scene: confirm the rig-note warning appears; sun move either works or
   warns (never silently fails).
3. Toggle `sun.enabled` 0 with a VRaySky env: sky must not black out (else rules switch to
   intensity-dimming — one-line change in `core/rules.py`, flagged in SPEC).
4. Heavy scene + live link + GPU loop render together: watch VRAM. If starved → matching
   sessions run V-Ray CPU or link closed during MATCH (workflow note, no code).
5. Optional: point Settings → system python at a Pillow venv; `metrics_cli some.jpg` prints
   stats.
### Spike E — dome seed + scenario board (~10 min) · checklist #17 #18 (v0.9)
`onbox_spikes.py` now runs the core automatically: **Q/#17** seeds from a probe render,
round-trips the .hdr, binds it, renders under it, and restores the artist's dome in a
finally; **R** builds the scenario candidates from the live rig. Manual leftovers below.
0. **#19 COLOR PIPELINE (do this FIRST)**: render one probe via MaxGaffer and open the
   saved PNG next to the VFB. If Max's color management (OCIO/ACES) means the saved file
   is NOT display-transformed (darker/flatter than the VFB), every histogram the solver
   sees is biased — set color management to apply the view transform on save, or note
   the transform here (a fixed correction in the stats loader is a one-liner once
   measured). Full reasoning: docs/STRESS.md §1.
1. Bind any reference → RIG → **Seed dome**: `seed_<cam>.hdr` lands in the camera's run
   folder, the dome renders it (VRayBitmap reads our stdlib RGBE — verify no black dome /
   no gamma double-up; the file is linear, so the texmap should NOT apply sRGB de-gamma).
2. **u-origin offset**: seed a scene whose reference has an obvious sun; check the pano's
   sun disc lights from the expected compass direction. If V-Ray's spherical u=0 is not
   north, note the constant offset here — the fix is one baked rotation in
   `controller.seed_dome` (rotation currently zeroed) and the loop absorbs it either way.
3. **Restore pre-match** after a seed: previous HDRI file + rotation return (a dome that
   had NO texture gets its texmap disabled instead — confirm it doesn't render black).
   Then seed TWO cameras and switch between them: the dome must follow each camera's own
   seed (`_rebind_seed` on select/finals/vrscene-export), and a RE-seed with a swapped
   reference must visibly change the dome (fingerprinted filename defeats the bitmap
   cache — if it still shows stale, note it and we force a bitmap reload call here).
4. **BOARD** on the same camera: 5-6 candidates render, scores shown when a reference is
   bound, ADOPT applies + saves, "Keep current light" leaves the scene untouched.
**→ CHECKPOINT 0: preflight all-green inside Max. Commit "on-box verified" with the diff.**

## P1 — First live match (MVP smoke, ~1 hr)
Small-but-real interior. Bind a reference close in albedo (fair first fight). Sweep ON,
5 iterations, target 82. Record: wall time/iteration, final score, leash hits, LLM
rejections, and the money shot — reference vs `iterNN.png` side-by-side.
Then: select another camera (state recall applies), Restore pre-match (rig returns), reopen
scene (session + baselines survive reload).
**Accept (=C1):** ≥75 score or visual acceptance in ≤2 runs · recall + restore + reload all
correct · one full-res Vantage still of the matched camera.
**Then the hostile case:** deliberately mismatched-albedo reference → confirm the leash
diagnosis fires and lock-EV workflow feels usable (this is the tool's honest edge — feel it).

## P2 — Shot-board production trial (overnight)
TULA-style multi-camera scene: reference per camera (mixed times of day), match each
(note per-shot wall time), then **Render ALL matched cameras** to an output folder overnight.
**Accept (=C2):** every shot renders under its own light, zero manual steps after launch;
per-shot cost known → decide default iterations/sweep for heavy scenes.

## P3 — v0.2 backlog (post-C2, ranked)

| # | Item | Why / note |
|---|---|---|
| 1 | ✅ **shipped v0.2** — sweep `altitude_hint` consumed (verify at P1) | free accuracy |
| 2 | ✅ **shipped v0.2** — per-iteration thumbnails inline in the log (verify at P1) | trust + faster human abort |
| 3 | ✅ **shipped v0.2** — A/B flip button pre-match ↔ matched (verify at P1) | the director's favorite toggle |
| 4 | ✅ **shipped v0.3** — opt-in draft sampler w/ crash-safe on-disk snapshot + launch recovery (verify props at Spike, chk #15) | never touches GI/lights |
| 5 | ✅ **shipped v0.2** — run-dir auto-prune, `keep_runs`=10 | disk hygiene |
| 6 | ✅ **shipped v0.3** — photometric + standard lights in groups; light-target filter (chk #16) | LIGHT_MULT covers both conventions |
| 7 | ✅ **shipped v0.3** — EXR/HDR/TIFF refs ingest Max-first (transcode) for stats AND LLM blocks | Max bitmap I/O is the reader |
| 8 | ✅ **shipped v0.3** — `overcast_sun_mode: "dim"` keeps sun at 0.05×/size 12 (flip after Spike D #13) | VRaySky-coupling escape hatch |
| 9 | ✅ **shipped v0.3** — Match ALL (refs): unattended sequential queue, per-camera fault isolation, confirm dialog w/ render estimate | overnight matching |
| 10 | ✅ **shipped v0.3** — `maxgaffer/api.py`: match_camera / match_all_cameras / apply_camera_state / render_cameras_vantage | MaxDirector's LightMatch stage |

## Risks (top 5)
1. **WB direction inverted on box** (#6) — one visual check, one-line flip. Do it FIRST. (WB enum + kelvin prop now doc-verified on native Physical; visual check remains for the render's response.)
2. **Live-link** (#9) — doc-verified: it's the V-Ray toolbar action "Initiate a Live-Link to Chaos Vantage" (port 20701, TOGGLE). actionMan scan targets that exact label; manual click stays the fallback.
3. **Vantage CLI removed in 3.x (doc/support-verified)** — finals ship via the V-Ray backend by default; vrscene export + in-app batch queue is the Vantage-quality path. 4. **Loop render cost on heavy interiors** — mitigations exist (res, iterations, sweep off,
   CPU mode); measure at P1/P2 before adding machinery (#4 backlog).
4. **Albedo trap in daily use** — leash + diagnosis shipped; if it fires constantly on real
   pairs, revisit weights (`config.critic_weights`) with logged run data.
5. **V-Ray build drift** (property names) — candidates + checklist; one-time cost per build.

## Definition of done — v1.0
C0 + C1 + C2 checked · WB/EV/azimuth/dome directions verified · README checklist all ✓ ·
backlog #1–#3 shipped · memory + SPEC updated with measured per-iteration costs.
