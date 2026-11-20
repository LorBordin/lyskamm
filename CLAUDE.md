# CLAUDE.md — Lyskamm

Lyskamm is Lorenzo's amateur rocket: fully 3D-printed, made of many parts joined with screws — **no glue on
structural parts**. Every part can be replaced or upgraded on its own. Single stage, Ø60 mm, ≈ 860 mm long
(measured 871 mm), 4 upper + 4 lower fins.

- Motors (thrust curves only; motor design is out of scope): **TSP F35** (Ø29, primary, needs an adapter in the
  Ø35.2 motor tube) `scripts/motor_data/TSP_F35.eng`; **EXP G115** (Ø35, later, the motor tube is sized for it)
  `scripts/motor_data/EXP_G115.eng`.
- Recovery: parachute Ø1 m with spill hole, deployed at apogee by the altimeter (no motor delay charge).
- Launch rail/rod: 2 m.

## Session protocol
- `PLAN.md` is the **roadmap** toward the first F35 flight (Phase 5): stages, projects (S structural, D deployment +
  e-bay, L launch system, M motor adapter, I integration, F flight readiness), dependencies, backlog per project.
- **Start of every session:** read `CLAUDE.md`, `PLAN.md` (Status first), then the active project's docs in
  `docs/projects/<id>_<name>/` (`design.md`, `plan.md`).
- **End of every session:** update `PLAN.md` → **Status** (current stage, next step, open questions) and append an
  entry to **Session log** (date, what was done, decisions, what is left); update the active project's `plan.md`.
- Open questions live in `PLAN.md` Status and are discussed when the related project comes up.
- A project gets its design doc + plan **when it starts**, written with Lorenzo (requirements, options, then steps).
- Setup history (Phases 0–4, old questions Q1–Q17, sessions 1–9): `docs/archive/setup_plan.md`. "Old Qn" and
  "setup step n" below refer to it.

## Design constraints
- **No glue on structural parts.** Joints are screwed.
- **Fasteners:** M2 + heat-set inserts (insert hole **Ø2.6**, depth 5; screw clearance **Ø2.2**, countersink 1.2).
  Ø3.1 insert holes in v1 are a mistake (backlog). M3 (clearance Ø3.5, head Ø7) secures the aluminium bar.
- **Joints between sections:** 8 radial M2 screws (every 45°), halfway along the 7.5 mm coupling; visible coupler
  ring 5 mm.
- **Sliding fit:** 0.2 mm clearance on Ø (`fit_slide_d`; e.g. 56 ↔ 55.8, 37 ↔ 37.2).
- **Body tube:** Ø60 outer, 2 mm wall (Ø56 inner); fin sections have a 1 mm outer shell (Ø58 inner).
- **Carbon rods (structural):** 4 carbon tubes Ø5/3 (bought, 7 g each; CAD 450 mm, real rods cut slightly shorter
  and fit exactly) run lengthwise through the aft airframe — at r = 23.4279 (`rod_r`), 45° off the fins, in a
  continuous channel from the top of `parachute_fuselage_seat_v2` to the bottom of `rear_transition` (449.5 mm).
  - **Why:** bending stiffness. The airframe assembles without them, but every joint is only a ≈7.5 mm coupling with
    M2 screws ≈3.7 mm from the part end, and the rigid printed material (ASA-CF) can crack there. The rods spread
    the stress over the whole (inner) structure instead of concentrating it at the joints. The load path will be
    analysed further (project S).
  - **Retention:** every part on the path has 4 through holes for the rods; the two ends (top of
    `parachute_fuselage_seat_v2`, bottom of `rear_transition`) have **closed** holes. Rods + closed end holes +
    fits + M2 screws hold the stack together.
  - Parts on the path: `parachute_fuselage_seat_v2`, `low_chute_ret`, `mid_fuselage`, `upper_retainer`,
    `lower_fuselage_v1`, `rear_transition`. Any redesign of these parts must keep the channel continuous and aligned
    (holes at 45°, same `rod_r`, `rod_hole_d`), the end holes closed, and the total channel length.
- **Aluminium bars (e-bay):** 4 × Ø5/3 × 100 mm (commercial stock) between the e-bay closures, secured with M3.
- **Printer:** Bambu Lab P1S, 0.4 mm hardened-steel nozzle. **ASA-CF** for all parts, **PETG-CF** for the fins,
  15 % infill. Settings live in the `.3mf` files (Bambu Studio); `scripts/print_settings.py` extracts them.
- **Units:** mm (and g for masses).

## Sources of truth
| What | Source |
|---|---|
| Shape, lengths, Ø of printed parts | CAD (`.FCStd`), confirmed by measurement. Small CAD vs measured differences are expected (polish, measurement error). |
| Shared dimensions | `parameters.FCStd` (spreadsheet `Params`); description per alias in `docs/proposal/parameters.md` |
| Fin dimensions | `lyskamm.ork` (CAD fins are enlarged on purpose to compensate the slicing loss on the airfoil) |
| Masses | weighed: `docs/inventory/masses.md`. New parts: **Bambu Studio slicer estimate** (never volume × density) |
| Section lengths (measured) | `docs/inventory/section_lengths.md` |
| Flight performance / stability | `lyskamm.ork` + `scripts/openrocket/Sim.java`; results in `docs/analysis/f35_simulation.md` |

## Repo map
```
CLAUDE.md, PLAN.md (roadmap), README.md
parameters.FCStd            master parameters (spreadsheet Params) — must not move (parts link ../../parameters.FCStd)
lyskamm.ork                 OpenRocket 24.12 model
assemblies/
  lyskamm_asm.FCStd         built-in Assembly (current): 30 App::Links, nose tip at origin, axis −Z
  lyskamm.FCStd             A2plus assembly — legacy, positions wrong in several places; do not use for positions
components/<part>/          <part>.FCStd + .step + .3mf next to each other (exports stay next to the source)
  nose_cone/  e-bay/  parachute_fuselage_seat/  lower_chute_retainer/  mid_fuselage/  upper_fins/
  carbon_rod/  upper_motor_retainer/  lower_fuselage_w_motor_mount/  lower_fins/  rear_transition/  logo/
launch_pad/                 early concept only — now part of project L (launch system)
docs/inventory/             read-only inventory (generated by scripts) + masses, section lengths, file status
docs/proposal/              approved Phase 2 proposals: parameters, conventions, repo_structure, ork_vs_cad
docs/analysis/              simulation results
docs/projects/<id>_<name>/  per-project design.md + plan.md (created when a project starts)
docs/archive/setup_plan.md  setup plan, Phases 0–4 (history, old Qn numbers)
docs/superpowers/           specs/ and plans/ from design sessions
scripts/                    see "Scripts" below
```
- Active vs legacy status of every file: `docs/inventory/file_status.md`. Legacy files are **deleted**, not archived
  (git keeps them: `git log --all -- <path>`).
- The English file/folder renames of `docs/proposal/repo_structure.md` group B (e.g. `e-bay` → `ebay`,
  `lower_fuselage_v1` → `lower_fuselage`, `pistone` → `piston`) are **not applied yet**. Renaming a part file
  breaks the relative links in `lyskamm_asm.FCStd` — update them in the same step.

## Naming conventions (full text: `docs/proposal/conventions.md`)
- English everywhere (files, objects, constraints, docs, commits); `snake_case`, ASCII; no version suffixes in new
  names (git keeps history). `upper` = toward the nose, `lower` = toward the tail.
- Inside a part only **Labels** change, never internal Names. Body = part name. Features `<Type>_<purpose>`
  (`Pad_body`, `Pocket_m2_inserts`, `PolarPattern_m2_inserts`); sketches `SK_<purpose>` matching their feature;
  datums `DP_`/`DL_`; Rocket-WB features keep their type name (`NoseCone`, `Fin`, `Transition`).
- Driving constraints are named. A constraint bound to a parameter has **the same name as the alias** and the
  expression `parameters#Params.<alias>`. Part-local dims get a descriptive name (`slot_w`, `boss_outer_r`).
- **v1 deviations stay literals** (Ø3.1 inserts, 5.15/5.25/5.75 rod/bar holes, 7.2/7.25 bosses, 3.2 M3, nose Ø2.1
  holes) so that parametrization never changes geometry; they are fixed in the Phase 5 redesigns.
- Target: every sketch fully constrained (DoF 0). Remaining open sketches (connector/deployment parts) are listed
  in `PLAN.md` under project D.

## Working rules
- CAD is developed **with** Lorenzo, not autonomously. **Never modify or save a `.FCStd` file** unless the change was
  planned and Lorenzo approved it. Lorenzo may authorise a batch ("stop me only if there are problems"); that
  authorisation covers only the batch it was given for.
- Reading and inspecting files (parsing `Document.xml` with Python, opening a document in FreeCAD **without
  saving**, working on copies in the scratchpad) is always allowed.
- **CAD change protocol** (every change):
  1. Say what will change: file, feature/constraint, old → new value, expected effect.
  2. Wait for Lorenzo's OK.
  3. Apply the change, recompute, run the checks below, describe the result.
  4. Wait for Lorenzo to check it in FreeCAD before moving on.
  5. Commit to git (one commit per approved, verified change; mention vol/CoM deltas in the message).
- **Checks after every CAD change** (not only "no Invalid + same volume"):
  - full forced recompute (`touch()` every object, then recompute; check the number of recomputed objects);
  - no Invalid features; volume and CoM vs the pre-change recompute (tolerance **1e-6 relative** when geometry
    must not change; anything above → ask Lorenzo);
  - sketch DoF, no redundant / conflicting / malformed / partially redundant constraints;
  - no sketch external geometry flagged `Missing` (in `Document.xml`: `ExternalGeo` with `flags & 8`);
  - **rebuild time from the first sketch** (a slow feature freezes the GUI when a sketch is closed; reference:
    mid/lower fuselage and rear_transition ≈ 6–8 s);
  - if the part is in the assembly: the assembly re-solves from a **fresh open** with unchanged placements and no
    new interferences.
- Small CAD vs measured differences are expected; don't flag them.

## Workflows

### FreeCAD + MCP
- FreeCAD **1.1.3** (AppImage `~/Applications/FreeCAD.AppImage`); files were originally saved with 1.0. Workbenches
  installed: Rocket WB (nose cone, transition, fins), A2plus (legacy assembly only), built-in Assembly.
- MCP server `freecad` (`.mcp.json`, `uvx freecad-mcp`): **FreeCAD must be open** and its RPC server running. If
  FreeCAD restarts, Lorenzo must start the RPC server again. Addon 0.1.31 vs server 0.1.32 warns on every call
  (harmless so far; old Q14, housekeeping).
- Read-only tools (`get_objects`, `get_object`, `get_view`, `list_documents`, `execute_code` that only reads) are
  fine any time. Anything that changes a document (`edit_object`, `create_object`, `delete_object`, writing code)
  follows the CAD change protocol. **Never save** without approval.
- **Save through the open GUI** (MCP `execute_code` → `doc.save()`). Saving from headless `freecadcmd` drops all GUI
  data (`GuiDocument.xml`: visibility, colours, thumbnail); if it happens, restore the GUI entries from the previous
  git version of the file (object names unchanged) before Lorenzo checks it.
- Headless: MCP `execute_code_headless` cannot find `freecadcmd` (old Q9). Run it from the AppImage in the shell
  (`~/Applications/FreeCAD.AppImage freecadcmd <script>`, or `/tmp/.mount_FreeCA*/usr/bin/freecadcmd` while the GUI
  is running), **on copies** in the scratchpad. Headless recompute needs `touch()` on all objects.

### `parameters.FCStd`
- Spreadsheet `Params` at the repo root; columns A alias · B value/formula · C unit · D description / used in.
  Aliases (body, fits, fasteners, rods, motor mount, outer shape) are documented in `docs/proposal/parameters.md`.
- Parts reference it by cross-document expressions `parameters#Params.<alias>`, stored as the relative path
  `../../parameters.FCStd` (from `components/<folder>/`); opening a part or the assembly loads it automatically.
- Set cells **with units** (`"0.3 mm"`); a bare number breaks expressions like `body_id - fit_slide_d` (unit
  mismatch). Address cells by alias, never by row number (rows have shifted before).
- A change propagates to every bound part: recompute and check all affected parts and the assembly (see setup step 3.99 in
  `docs/archive/setup_plan.md` for a reference run: `fit_slide_d` 0.2 → 0.3 changes exactly the 6 sliding-fit parts).

### Assembly `assemblies/lyskamm_asm.FCStd` (built-in Assembly WB)
- 30 `App::Link`s to each part's `Body` (relative paths), live: part edits show up after recompute, no "update from
  source". Frame: **nose tip at origin, rocket axis along −Z** (−z = OpenRocket's "from nose tip"); `nose_cone`
  grounded; `Fixed` joints on part edges (circle nearest the joint point), with offsets; no LCS added to parts.
- Built by `scripts/asm_build.py` (run in the GUI); flips/rolls solved from geometry by
  `scripts/asm_stack_solver.py` (hole patterns, rod channels, connector, keys, zero interference) — **not** copied
  from A2plus. Fins at 0/90/180/270°, carbon rods at 45°, e-bay rods at 0/180°.
- Mass/CG: each link has `Mass_g` (weighed); `scripts/assembly_cg.py` combines it with the CAD CoM (uniform density).
  Parts without CAD (parachute, cord, electronics, motor) are not included.
- Deferred, not in the assembly yet: `pistone`, `sede_presa_elettrica_sup`, `presa_elettrica`, logo inserts.
- Always run the **interference check**, not only the hole check: the 8 × 45° screw pattern does not fix roll.
- Known slivers (below print tolerance, old Q16): upper fin T-foot vs `mid_fuselage` socket 0.04 mm, lower alu rod
  vs `mid_closure` bore 0.13 mm³. (The carbon rod vs `rear_transition` 0.5 mm overlap is CAD only: real rods are
  cut slightly shorter.)
- Joints created from Python need `JointObject.ViewProviderJoint` on their view objects; use the GUI tools or add it.
- FreeCAD 1.1.3 solver bug: `Solve failed: vector::_M_range_check …` with certain joint sets (both lower alu rods on
  `lower_closure`). Workaround used: re-host the joints (rods on `mid_closure`); if it returns, split into
  sub-assemblies (e-bay / aft).

### OpenRocket and simulation
- `lyskamm.ork` (OpenRocket 24.12, runs with its bundled Java 17; system Java 25 makes the GUI quit silently):
  real materials, weighed mass overrides, measured section lengths, fins from the design, motor tube 37/35.2 × 93,
  parachute deploying at apogee. `docs/proposal/ork_vs_cad.md` lists the CAD ↔ `.ork` mapping.
- Sync with `parameters.FCStd` is manual: `Params` holds CAD design values, the `.ork` holds measured lengths.
  When a Params outer-shape value changes, update the `.ork` by hand and re-run the simulation.
- `scripts/openrocket/Sim.java` — headless F35/G115 check (rail length, adapter mass, extra mass sweep, delays);
  usage in the file header. Stability needs ≈ +200 g forward (electronics, deployment) — see the analysis doc.
- `scripts/rocket_simulator.py` — older standalone simulator (Italian); known issues (motor casing mass,
  hard-coded params); to be retired in project I. **Use `Sim.java`** (Lorenzo, 2026-11-20).

### Bambu Studio
- `.3mf` files next to the part sources hold the print settings; `scripts/print_settings.py` (`--json`) lists them.
  Some `.3mf` names differ from their content — see `docs/inventory/file_status.md`.
- Mass of a new part = Bambu Studio's slicer estimate with the project settings (ASA-CF / PETG-CF, 15 %).

### Scripts (`scripts/`)
| Script | What | Runs with |
|---|---|---|
| `cad_inventory.py` | inventory from `Document.xml` → `docs/inventory/summary.md`, `parts/*.md` | python3 (stdlib) |
| `cad_inventory_fc.py` | sketch DoF, geometry, recompute preview → `sketch_dof.md`, `geometry.md`, `fc_data.json` | freecadcmd, on copies |
| `shared_dims.py` | shared-dimension table (`MEANINGS`, `EXTRA_ACTIVE`) → `shared_dims.md` | python3 |
| `ork_inventory.py` | `.ork` → `docs/inventory/openrocket.md` | python3 |
| `print_settings.py` | Bambu Studio settings from `.3mf` | python3 |
| `asm_stack_solver.py`, `asm_build.py` | solve and build `lyskamm_asm.FCStd` | FreeCAD GUI |
| `assembly_cg.py` | assembly mass/CG | FreeCAD GUI or freecadcmd |
| `fc_migration_fix_test.py` | Phase 3.0 repair test (historical) | freecadcmd, on copies |
| `openrocket/Sim.java` | flight simulation | java + OpenRocket jar |
| `rocket_simulator.py` | old standalone simulator | python3 (`requirements.txt`) |

## CAD gotchas (learned in Phase 3)
- **Partial documents:** a part opened through the assembly links can be loaded partially (`doc.Partial`). Edits
  apply in memory but recompute is silently skipped (`Please reload partial document`) and saves are refused. Check
  `doc.Partial` before editing; `doc.restore()` reloads it fully.
- **Forced recompute:** `touch()` every object, recompute, and check how many objects were recomputed; a plain
  recompute after load may do nothing.
- **Toponaming after sketch edits:** chamfer/fillet edge lists and assembly joint references can be remapped to the
  wrong edges (e.g. a joint moved from the Ø60 circle to an ellipse → 18 parts jumped 352.5 mm). Re-select by
  geometry (edges lying on the old edges; for joints, the circle nearest the joint point) and verify the assembly
  from a **fresh open** (or compare link placements with the saved file), not only in the live GUI session.
- **"Transform body" patterns** (`TransformMode=1`, no `Originals`) copy and fuse the whole body — very slow, froze
  the GUI on sketch close in `mid_fuselage`. Pattern only the feature (`TransformMode=0`, `Originals=[...]`).
- **Replacing external links** by constraints: use **exact** angles/distances (45°, 1.55 …); copying the sketch's
  current values locks in the old sloppiness (µm slivers, 3e-4° errors).
- **Closing DoF:** probe what is free (temporarily lock each point x/y / radius, see which drops DoF), then lock it
  **at the current value** or tie construction geometry to existing points; check for constraints that become
  partially redundant.
- **1.0 → 1.1 migration:** sketch supports and dress-up edge names may be renumbered; Rocket WB `Fin.TtwOffset` is
  measured from the other end (set 0). Small kernel drift (≈ 0.01 mm³) between 1.0-stored and 1.1 shapes is normal.
- Shapes printed by v1 have hand-drawn points (e.g. boss corners r29.62 vs 29.59); they are kept as they are until a
  Phase 5 redesign.
