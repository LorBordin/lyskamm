# CLAUDE.md — Lyskamm

> Minimal stub (Phase 0). The full version is written in Phase 4 of `PLAN.md`.

Lyskamm is Lorenzo's amateur rocket: fully 3D-printed, made of many parts joined with
screws — **no glue on structural parts**. Motors (thrust curves only, motor design out of scope):
TSP F35 (primary, `scripts/motor_data/TSP_F35.eng`), EXP G115 (later, `scripts/motor_data/EXP_G115.eng`).

## Session protocol
- **Start of every session:** read `CLAUDE.md` and `PLAN.md` (Status section first).
- **End of every session:** update `PLAN.md` → **Status** (current phase, next step, open
  questions) and append an entry to **Session log** (date, what was done, decisions, what is left).
- Open questions are kept in `PLAN.md` Status and discussed when the related step comes up.

## Working rules
- CAD is developed **with** Lorenzo, not autonomously. **Never modify or save a `.FCStd` file**
  unless the change was planned and Lorenzo approved it.
- For each CAD change:
  1. Say what will change: file, feature/constraint, old → new value, expected effect.
  2. Wait for Lorenzo's OK.
  3. Apply the change, recompute, describe the result.
  4. Wait for Lorenzo to check it in FreeCAD before moving on.
  5. Commit to git (one commit per approved, verified change).
- Reading and inspecting files (including parsing `Document.xml` with Python, or opening a
  document in FreeCAD **without saving**) is always allowed.

## Setup
- FreeCAD 1.1.3, MCP server `freecad` (FreeCAD must be open). Existing files were saved with 1.0.
- Assembly `assemblies/lyskamm.FCStd` uses the **A2plus** workbench (imported parts are static
  copies — update from source after part edits).
- Printer: Bambu Lab P1S, 0.4 mm hardened-steel nozzle. PETG-CF for fins, ASA-CF for the rest.
- Print settings: all printed parts ASA-CF, 15% infill; fins PETG-CF. Settings live in the `.3mf`
  files (Bambu Studio); `scripts/print_settings.py` extracts them.
- Mass estimates for new parts: use Bambu Studio's slicer estimate (no volume-based estimates).
  Measured masses: `docs/inventory/masses.md`.
- Fasteners: M2 + heat-set inserts (insert hole Ø2.6, screw clearance Ø2.2; Ø3.1 holes in v1 are a mistake); M3 (clearance Ø3.5) to secure the aluminium bar. Sliding fit clearance ~0.15 mm. Units: mm.
