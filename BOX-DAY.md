# Box day — the exact sequence

Everything off-box is proven (171 tests · 4 audit rounds · live sim 18.99→98.38 on the
real gateway). The box session is measurement, not hope.

## Bring-up (~15 min)

0. Get the code on the box: download `maxgaffer-0.9.6.zip` from
   https://github.com/Aasishmuchala/MXLT/releases/tag/v0.9.6 and unzip
   (or `git clone https://github.com/Aasishmuchala/MXLT`).
1. Double-click `scripts\install.bat` → restart Max 2026.
2. Customize → Customize User Interface → category **MaxGaffer** → drag the action to a
   toolbar → click it. The oc_ key auto-borrows from MaxDirector; otherwise Settings →
   paste key → **Test gateway** (expect: `gateway reachable … 'OK'`).
3. Open a **THROWAWAY COPY** of a real scene (VRaySun + dome VRayLight + ≥1 camera).
4. MAXScript listener:
   `python.ExecuteFile @"C:\<repo>\scripts\onbox_spikes.py"`

## Read the report

Printed table + `%LOCALAPPDATA%\MaxGaffer\spike_report.txt`.

- **S/#19 color mode — read this line FIRST.** If it says OCIO/ACES, hold one probe PNG
  next to the VFB before trusting any score. Different → tell me the mode string; the
  correction is a one-liner once measured (docs/STRESS.md §1).
- Any **FAIL** has a named fix point (candidates tuple / config field) — send the report
  back as-is, nothing to debug on your side.
- **G/#6 WB direction** and **H EV direction** are the two sign checks — if either says
  INVERTED, stop and report (each is a one-line flip, done blind it doubles the error).

## Manual leftovers (~10 min)

u-origin of the seeded dome (#18) · two-camera seed-follow on switch · re-seed with a
swapped reference visibly changes the dome · BOARD → ADOPT feel · VRAM with the live
link up on the heavy scene (#14).

## Then P1 — first real match (~1 hr)

Bind a reference that's FAIR (similar albedo family), Standard mode, sweep ON. Watch the
log; A/B flip at the end; Restore; reopen the scene (states must survive).

**Send back:** `spike_report.txt` · the run folder's `run.json` · one reference-vs-
`iterNN.png` pair. That's everything needed to calibrate or fix remotely.
