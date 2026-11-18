# Lyskamm — Project Plan

Fully 3D-printed, screw-assembled (no structural glue) rocket. Primary motor TSP F35 (29 mm),
later EXP G115 (35 mm). This plan is executed together with Lorenzo over several sessions.

**Session protocol:** at the start of every session read `CLAUDE.md` and `PLAN.md`;
at the end update **Status** and append to **Session log**.

**CAD change protocol (applies to every step marked 🔧):**
1. Propose: file, feature/constraint, old → new value, expected effect.
2. ⏸ **Review with Lorenzo** — wait for OK.
3. Apply via FreeCAD MCP; recompute; report result (volume/bbox before vs after, errors).
4. ⏸ **Lorenzo checks in FreeCAD** — wait for confirmation.
5. Commit (one commit per approved, verified change).

Read-only work (parsing `Document.xml`, opening docs in FreeCAD **without saving**) needs no approval.

---

## Findings from the initial read-only inspection (2026-11-16)

**OpenRocket (`lyskamm.ork`, OR 24.12)** — single stage "Sustainer", Ø60 mm (r=30), wall 2 mm, total length ≈ 864 mm:
nose cone 125 (power 0.4) · 5 · 105 · 5 · 105 · 5 · 241 · 5 · 105 (+4 freeform "front fins", 3 mm) · 5 · 113 (+4 trapezoidal "rear fins": root 60, tip 30, sweep 20, span 50, 3 mm) · transition 45 (r 30→21).
The 5 mm tubes look like the screw-joint rings between printed sections.
- Materials are placeholders (Cardboard, Balsa, Polystyrene) — no PETG-CF/ASA-CF.
- No inner tube / motor mount, no motor, no mass overrides, no recovery device, **no simulations**.

**FreeCAD (27 files, all saved with FreeCAD 1.0, user now on 1.1.3)**
- 257 sketches: **0 named constraints**, 254/257 with default labels (`Sketch001`…), **0 spreadsheets**, only 1 expression in the whole repo (`Constraints[2]=42.1`).
- Recurring raw values (candidates for shared parameters): 3.75 (×48), 6.5 (×30), 7.5 (×14), 23.4279 (×14), 9.6, 7.6, 6.15, 17, 34.5, 40 …
- 6 empty sketches (e.g. `lower_closure*.FCStd: Sketch001`).
- `fuselage_w_logo_v2.FCStd` is dead geometry (`Part::Feature` only, from STEP import).
- Sketch DoF (under-constraint) is not stored in the file → needs a FreeCAD solve (Phase 1).

**Assembly (`assemblies/lyskamm.FCStd`)** — **A2plus** workbench, 31 imported parts (static copies; must be "updated from source" after part edits). Parts it uses ("active"):
`lower_fuselage_v1`, `lower_fins`×4, `upper_retainer`, `mid_fuselage`, `upper_fin`×4, `rear_transition`, `carbon_rod`×4, `low_chute_ret`, `parachute_fuselage_seat_v2`, `sede_presa_elettrica_sup`, `e-bay/lower_closure_v2`, `alluminium_rod`×4, `mid_closure`, `e-bay/upper_closure`, `nose_cone`, `e-bay/fuselage`×4.
**Not in the assembly** (legacy/variants?): `fuselage_w_logo`, `fuselage_w_logo_v2`, `lower_closure` (v1), `presa_elettrica`, `parachute_fuselage_seat` (v1), `pistone`, `parachute_fuselage_seat/upper_closure`, `components/logo.FCStd`, `logo/logo*.FCStd`, `components/wall.FCStd`, `launch_pad/Unnamed.FCStd`.

**Repo hygiene**
- `.gitattributes` has `*.fcstd` (lowercase) → `.FCStd` files are not LFS-tracked; `git-lfs` is not installed, so `.3mf`/`.step` are plain blobs too.
- Mixed IT/EN names (`pistone`, `sede_presa_elettrica_sup`, `presa_elettrica`), typo `alluminium`, `upper_motor_retainer/upper_retainer` (extension-less file), untracked `init` (prompt text) and `.mcp.json`.
- `scripts/rocket_simulator.py`: hardcoded `rocket_mass_empty=0.910`, `cd=0.9`, Ø0.060; launch mass = empty + **propellant** mass only (motor casing missing).

---

## Phase 0 — Bootstrap
- [x] 0.1 Save this plan as `PLAN.md`. Files: `PLAN.md`
- [x] 0.2 Minimal `CLAUDE.md` stub: working rules + session protocol + "see PLAN.md" (full version in Phase 4). Files: `CLAUDE.md`
- [x] 0.3 Commit `PLAN.md` + `CLAUDE.md`.

## Phase 1 — Inventory (read-only)
- [x] 1.1 Write `scripts/cad_inventory.py` (stdlib only; parses `Document.xml` of every `.FCStd`): bodies, features (type, label, key props), sketches (geometry/constraint counts, driving dimensions, named?, label default?), spreadsheets/aliases, expressions. Output → `docs/inventory/<part>.md` + `docs/inventory/summary.md`. Files: `scripts/cad_inventory.py`, `docs/inventory/`
- [x] 1.2 *(done with `scripts/cad_inventory_fc.py`, run by `freecadcmd` from the AppImage on temp copies → `docs/inventory/sketch_dof.md`)* Sketch DoF check: open each file in FreeCAD via MCP (headless where possible), `solve()` each sketch, record DoF / malformed / conflicting; **close without saving**. Append to inventory. Flag: unconstrained (DoF>0), empty, default-labelled.
- [x] 1.3 *(→ `docs/inventory/geometry.md`, `fc_data.json`; mass estimate deferred to 1.6 — needs infill/density)* Measured geometry per active part: bounding box, volume (→ mass estimate at ASA-CF/PETG-CF density), key Ø/lengths (outer Ø, coupler Ø, wall, screw-hole PCD/Ø). Files: `docs/inventory/geometry.md`
- [x] 1.4 `.ork` inventory: components, lengths, radii, fins, positions → `docs/inventory/openrocket.md` (script: `scripts/ork_inventory.py`).
- [x] 1.5 *(→ `scripts/shared_dims.py`; "meaning" column left blank for review)* Shared-dimension analysis: cluster equal values across parts at interfaces (body Ø, coupler fit, M2 insert holes, rod holes, fin slots) → table "value · where used · likely meaning". Files: `docs/inventory/shared_dims.md`
- [x] 1.6 *(done — known total ≈ 575.5 g without screws/electronics; no volume-based estimate, use Bambu Studio for new parts; template: `docs/inventory/masses.md`, print params pre-filled from 3MF by `scripts/print_settings.py`)* Real masses: Lorenzo weighs the printed v1 parts (+ hardware) → `docs/inventory/masses.md`.
- [x] 1.7 *(done 2026-11-17 → `docs/inventory/file_status.md`, `MEANINGS` in `scripts/shared_dims.py`)* **Review with Lorenzo**: confirm active vs legacy files, meaning of shared dims. Commit inventory.

## Phase 2 — Proposal (documents only, no CAD changes)
- [x] 2.1 *(done 2026-11-17 → `docs/proposal/parameters.md`, reviewed with Lorenzo; mechanism (a); final approval in 2.5)* Master parameters: list of aliases (e.g. `body_od`, `body_wall`, `coupler_od`, `fit_slide_d=0.2`, `m2_insert_hole`, `m2_clear`, `rod_d`, `fin_t`, `motor_d`…), units, source part, ork counterpart. Decide mechanism: (a) single `parameters.FCStd` spreadsheet referenced by cross-document expressions vs (b) per-part sheet synced by script. Recommendation: (a). Files: `docs/proposal/parameters.md`
- [x] 2.2 *(approved 2026-11-17 → `docs/proposal/conventions.md`; English everywhere)* Naming conventions: files (`snake_case`, English, no version suffix — git holds versions), bodies, features (`Pad_body`, `Pocket_m2_holes`…), sketches (`SK_<purpose>`), constraints (`<alias-like name>`). Files: `docs/proposal/conventions.md`
- [x] 2.3 *(approved 2026-11-17 → `docs/proposal/repo_structure.md`; LFS dropped — A1 done; legacy files deleted, not archived)* Repo structure proposal: per-part folders, `exports/` for STEP/3MF, `archive/` for legacy, `parameters.FCStd` location, LFS fix (`*.FCStd`, install git-lfs or drop LFS). File moves/renames listed as a checklist (known so far: `components/logo.FCStd` → `components/logo/`; misnamed `parachute_fuselage_seat_v2.3mf` actually holds `lower_closure_v2`, and `parachute_fuselage_seat.3mf` holds the v2 seat — see `docs/inventory/file_status.md`) (each a separate, approved step; A2plus `sourceFile` paths must be updated). Files: `docs/proposal/repo_structure.md`
- [x] 2.4 *(done 2026-11-17 → `docs/proposal/ork_vs_cad.md`; fixes applied to `lyskamm.ork`; lengths wait for `docs/inventory/section_lengths.md`)* `.ork` ↔ CAD mismatch table: lengths, Ø, fin geometry & thickness (ork 3 mm), transition, materials, missing motor mount/motor/recovery/mass overrides, motor Ø 29 vs 35 mm. Files: `docs/proposal/ork_vs_cad.md`
- [x] 2.5 *(approved 2026-11-17)* ⏸ **Review with Lorenzo**: approve parameter list, conventions, structure, which .ork fixes to apply. Commit.

## Phase 3 — Parametrization (part by part, geometry must not change)
Pre-step:
- [ ] 3.0 FreeCAD 1.0 → 1.1.3 migration check: open each active part in 1.1.3, recompute, compare volume/bbox against Phase 1 baseline. 🔧 Save only with approval (one commit "resave in 1.1.3" per part if needed). Check A2plus works in 1.1.3 (else plan migration to built-in Assembly — separate decision).
  - **3.0 diagnosis (2026-11-17, copies only; test script `scripts/fc_migration_fix_test.py`).** All active parts load; Rocket WB + A2plus OK. 6 parts have features that fail on recompute; every fix below was tested on copies and gives a valid model:

    | Part | Cause | Fix | Result vs stored |
    |---|---|---|---|
    | `lower_fins`, `upper_fin` | Rocket WB now measures the through-the-wall tab offset from the other end → tab moves 3.5 / 5 mm aft, `Sketch` loses `Face10` | `Fin.TtwOffset` 3.5 → 0 / 5 → 0 | **identical** (= printed tab 0–60 / 0–90 mm) |
    | `e-bay/fuselage`, `fuselage_w_logo` | `Chamfer` (24 edges, 1.2 mm) has stale 1.0 edge names | clear and re-select the same 24 edges | **identical** |
    | `mid_fuselage` | `Sketch011` support `Face37` renumbered | support → `PolarPattern004.Face36`; `Chamfer` (32 edges) re-selected on the rebuilt edges (48 pieces) | valid; vol −2.45 mm³ (0.004 %), CoM 0.003 mm |
    | `lower_fuselage_v1` | chain: `Sketch014`/`Sketch015` supports renumbered; `Chamfer002` (44), `Chamfer` (13), `Chamfer004` (8) stale edges; `Sketch020` support renumbered + constraint 16 redundant in the 1.1 solver | supports → `Face1`, `Face1`, `Chamfer003.Face26`; re-select chamfer edges; delete redundant constraint 16 (PointOnObject, duplicate of 15) | valid; vol −1.93 mm³ (0.003 %), CoM 0.0002 mm |

    Progress: ✅ fins, ✅ e-bay fuselages (2026-11-17), ✅ mid_fuselage, ✅ lower_fuselage_v1 (2026-11-18) — all checked by Lorenzo. **3.0 done.**
    The ≈2 mm³ differences are FreeCAD 1.0 → 1.1 kernel drift (upstream features already differ by up to 0.7 mm³ before any fix). Proposed tolerance for 3.0 only: volume ≤ 0.01 %, CoM ≤ 0.01 mm, plus Lorenzo's visual check. Apply one part at a time (save = resave in 1.1.3), each approved, checked in FreeCAD, committed.
- [ ] 3.1 🔧 Create `parameters.FCStd` with the approved aliases. ⏸ Review with Lorenzo → check in FreeCAD → commit.

Per part, in this order (interfaces outward from the body tube): `e-bay/fuselage` → `mid_fuselage` → `lower_fuselage_v1` → `rear_transition` → `upper_retainer` → `low_chute_ret` → `parachute_fuselage_seat_v2` → `sede_presa_elettrica_sup` → `e-bay/lower_closure_v2` → `mid_closure` → `e-bay/upper_closure` → `nose_cone` → `upper_fin` → `lower_fins` → `carbon_rod` → `alluminium_rod`.
For each part (one checklist block per part, added to this file when the part starts):
- [ ] a. Record baseline: volume, bbox, center of mass (read-only).
- [ ] b. 🔧 Rename sketches/features per convention. ⏸ Review → apply → ⏸ Lorenzo checks → commit.
- [ ] c. 🔧 Name driving constraints and bind them to aliases (`=parameters#Params.body_od`), small batches (≤ ~10 constraints per change). Close DoF where needed. ⏸ Review → apply → ⏸ Lorenzo checks → commit.
- [ ] d. Verify geometry unchanged: volume/bbox/CoM equal to baseline (tolerance 1e-6).
- [ ] e. 🔧 Update the part in the A2plus assembly, solve, check interferences/fits with neighbours. ⏸ Lorenzo checks → commit.
- [ ] f. Re-export STEP/3MF only if Lorenzo asks.
- [ ] 3.99 Smoke test: change `fit_slide` 0.15 → 0.20 in a scratch copy (not saved), confirm all dependent parts update; discard.

## Phase 4 — CLAUDE.md (full)
- [ ] 4.1 Expand `CLAUDE.md`: project goal & constraints (no structural glue, M2 + heat-set inserts, ~0.15 mm sliding fit, P1S 0.4 mm hardened nozzle, PETG-CF fins / ASA-CF rest, mm), conventions (from 2.2), repo map, working rules, CAD change protocol.
- [ ] 4.2 Workflow sections: Bambu Studio (mass estimates of new parts from the slicer), FreeCAD MCP (FreeCAD 1.1.3 must be open; read-only vs write tools; never save without approval; A2plus update), OpenRocket (where params come from, how to sync with `parameters.FCStd`), simulator, how to use PLAN.md (Status + Session log).
- [ ] 4.3 ⏸ Review with Lorenzo → commit.

## Phase 5 — Redesign backlog (list only; plan in detail later)
- [ ] Electronics bay (e-bay: closures, aluminium rods, access, wiring, connector `presa_elettrica`).
- [ ] Launch rail / pad interface on the rocket (rail buttons or guides). The pad itself: see *Launch pad* section.
- [ ] Motor mount: current tube is Ø35.2 inner / Ø37 outer (`lower_fuselage_v1`, `rear_transition`; `upper_retainer` bore 37.2), sized for EXP G115 (Ø35). TSP F35 (Ø29) needs an **adapter** (Lorenzo, 2026-11-17). Motor retention.
- [ ] Recovery: chute size, ejection/deployment (`pistone`, chute retainers), shock cord attachment.
- [ ] OpenRocket model completion: real materials, motor mount, motor, masses/overrides, recovery, simulations (F35, G115), stability margin.
- [ ] Simulator fix: include motor casing mass; read params from `parameters.FCStd`/`.ork`.
- [ ] Lower motor retainer: design it; fixed with an M2 screw + heat-set insert → `rear_transition` Pocket002 Ø3.1 becomes Ø2.6.
- [ ] Fix oversized M2 insert holes Ø3.1 → Ø2.6 (`low_chute_ret` Pocket004–007, `upper_retainer` `sede_viti_*`) — v1 mistake (Lorenzo, 2026-11-17). Not in Phase 3 (geometry must not change there).
- [ ] Final mass/CG: known only after the e-bay electronics and the deployment system are designed; then weigh + balance with the F35 and re-run `scripts/openrocket/Sim.java`. Planned extra ≈ 200 g.
- [ ] **Stability with F35 and G115** (G115: Lorenzo will add nose weight if needed) (see `docs/analysis/f35_simulation.md`): if the measured CG is too far aft, options are nose weight, moving mass forward, or larger/further-aft fins. G115 needs its own check (heavier motor → less stable).
- [ ] Launch rail/rod **2 m** (Lorenzo's plan from the start; gives ≈ 13 m/s rail exit with the F35 at ≈ 0.9 kg). Goes into the launch-pad design.
- [ ] Fin attachment strength / flutter check with PETG-CF.
- [ ] Logo parts: all kept (flat logo, projected logo, `logo_red` = printed undersized PLA inserts); clean up names/location in Phase 2.3.
- [ ] Assembly is incomplete: small/inner flying parts (`pistone`, `fuselage_w_logo`, connectors, screw eyes, …) are not in `assemblies/lyskamm.FCStd` — add them (needed for fit checks and CG).
- [ ] Update `.ork` and `rocket_simulator.py` with weighed masses (`docs/inventory/masses.md`) and measured length 871 mm.
- [ ] [others — Lorenzo to fill in]

## Launch pad (separate project, deferred)
- `launch_pad/Unnamed.FCStd` is an early concept only. How to build the pad is still to be decided; ignore it in Phases 1–4 (no inventory review, no parametrization). Plan it here when Lorenzo is ready.

---

## Status
- **Current phase:** 3 — 3.0 (FreeCAD 1.1.3 migration fixes) done 2026-11-18.
- **Next step:** 3.1 create `parameters.FCStd` (spreadsheet `Params`, aliases from `docs/proposal/parameters.md`) — propose, wait for OK.
- **Open questions:**
  1. ~~Which `.FCStd` matches `fuselage_w_logo.3mf`~~ — resolved: `e-bay/fuselage_w_logo.FCStd` (added to `EXTRA_ACTIVE`). Active/legacy status of all non-assembly files → `docs/inventory/file_status.md`.
  2. Other known v1 issues besides e-bay and pad interface?
  3. ~~Master parameters mechanism~~ — resolved: (a) single `parameters.FCStd` (spreadsheet `Params`) with cross-document expressions.
  4. Keep A2plus, or migrate the assembly to FreeCAD's built-in Assembly workbench (later, separate decision)?
  5. Git LFS: install and fix `.gitattributes`, or drop LFS? Finding: LFS never actually used (no pointers in history, `*.fcstd` pattern wrong case, git-lfs not installed); recommendation in `repo_structure.md`: drop it.
  6. ~~`.ork` vs CAD source of truth~~ — resolved: lengths/body shape = CAD (confirmed by physical measurement); fin dimensions = `.ork` (CAD fins enlarged on purpose for slicing loss); masses = weighed.
  7. Delete or ignore the untracked `init` file; commit `.mcp.json`?
  8. ~~Rocket WB + A2plus missing~~ — installed by Lorenzo (2026-11-17), verified headless on copies: all Python objects load; `nose_cone` and `rear_transition` recompute identically; assembly loads (32 parts, no errors). ⚠ `lower_fins`/`upper_fin`: the Rocket `Fin` now regenerates +0.75 % volume and `Sketch` (attached to `Face10` of the fin) becomes Invalid → fix in Phase 3.0.
  9. The MCP `execute_code_headless` cannot find `freecadcmd` (FreeCAD is an AppImage). Configure `--freecadcmd` in `.mcp.json`, or keep running the AppImage's `freecadcmd` via shell?
  10. ~~Shared dims~~ — resolved (`MEANINGS` in `scripts/shared_dims.py`).
  11. Features already failing in FreeCAD 1.1.3 recompute (copies only): `lower_fins`/`upper_fin` Sketch (on `Face10`, after Rocket WB install); `e-bay/fuselage` Chamfer, `lower_fuselage_v1` Sketch014, `mid_fuselage` Sketch011, `wall` Sketch005, `launch_pad` Sketch001 … — fix during Phase 3.0?
  12. ~~Mass estimate~~ — resolved: weighed masses only; Bambu Studio for new parts.

## Session log
### 2026-11-16 — Session 1
- **Done:** read-only inspection of repo, `.ork`, all 27 `.FCStd` (Document.xml), A2plus assembly, simulator. Wrote PLAN.md.
- **Decisions:** Phase 0 bootstrap added so CLAUDE.md exists from session 2; parametrization order = interfaces outward from body tube; one commit per approved CAD change.
- **Left:** Phase 1 (inventory). Open questions saved in Status, to be discussed at the related steps.

### 2026-11-16 — Session 2 (Phase 1)
- **Done:** 1.1 `scripts/cad_inventory.py` → `docs/inventory/summary.md` + `parts/*.md` (29 files). 1.2/1.3 `scripts/cad_inventory_fc.py` (headless FreeCAD 1.1.3 on temp copies, nothing saved) → `sketch_dof.md`, `geometry.md`. 1.4 `scripts/ork_inventory.py` → `openrocket.md`. 1.5 `scripts/shared_dims.py` → `shared_dims.md`.
- **Findings:** 87/257 sketches with issues (DoF>0, 6 empty, 2 solve failures with redundant constraints); 0 named dimensions, 0 spreadsheets; 16 active parts; Rocket WB + A2plus missing in FreeCAD 1.1.3; several features already Invalid on 1.1.3 recompute; `.ork` lacks motor mount, motor, recovery, masses, rail buttons, simulations; nose cone 120 (CAD) vs 125 (.ork); fin geometry differs (see `parts/*fins*.md` vs `openrocket.md`).
- **Decisions:** baseline geometry = shape *stored* in the files (saved by FreeCAD 1.0); added `__pycache__/` to `.gitignore`.
- **Left:** 1.6 weigh parts, 1.7 review + commit inventory; open questions 8–12 added.
- **Later same session:** 1.6 done. `scripts/print_settings.py` extracts Bambu Studio settings from `.3mf`. Lorenzo filled `masses.md`: all printed parts ASA-CF 15% (fins PETG-CF); masses as weighed incl. inserts/screw eyes/connectors; known total ≈ 575.5 g (no screws, electronics, cord, battery); length 871 mm (est.). `fuselage_w_logo` is the 4th fuselage and `pistone` flies — both missing from the assembly (added to backlog; `pistone` added to `EXTRA_ACTIVE` in `shared_dims.py`). Decision: no volume-based mass estimate; use Bambu Studio for new parts. Q12 resolved.
- **Session end:** paused after 1.6 at Lorenzo's request; 1.7 in a new session. Checkpoint commit of scripts + `docs/inventory/` (pre-review; `.mcp.json` and `init` left untracked).

### 2026-11-17 — Session 3 (Phase 1.7 review)
- **Done:** matched the printed `fuselage_w_logo.3mf` to `e-bay/fuselage_w_logo.FCStd` by mesh volume → added to `EXTRA_ACTIVE`. Reviewed non-assembly files with Lorenzo → `docs/inventory/file_status.md`. Found `logo/logo_red.FCStd` = the printed PLA logo inserts (≈0.5 mm undersized) and that `parachute_fuselage_seat.3mf` holds the v2 seat while `parachute_fuselage_seat_v2.3mf` holds `lower_closure_v2`. Shared-dim meanings stored in `MEANINGS` (`scripts/shared_dims.py`), table regenerated.
- **Decisions:** carbon tubes and aluminium bar are 5/3 mm tubes; tube centres at r = 23.4279; M2 holes 2.2, insert holes 3.1; motor tube 35.2/37 sized for EXP G115, TSP F35 will need an adapter; launch pad split into its own deferred section; `logo.FCStd` to move into `logo/` (Phase 2.3).
- **Later same session:** Lorenzo confirmed Ø2.6 = M2 insert hole (Ø3.1 holes are a mistake → backlog), Ø3.5 = M3 clearance (screw securing the aluminium bar in `lower_closure_v2`), 6.15×9.6 = rounded slot for `presa_elettrica` (deployment system, to be redesigned later), Ø7/7.2 bosses, `parachute_fuselage_seat/upper_closure` legacy. **1.7 closed.** Fastener sizes added to CLAUDE.md.
- **Later same session:** `rear_transition` Pocket002 = screw hole for the lower motor retainer; `upper_closure` Pocket002 (Ø3.2) = M3 for the aluminium bar (other end). Open question 3 → option (a). 2.1 drafted: `docs/proposal/parameters.md` (aliases for body, fits, fasteners, tubes, motor mount; CAD vs `.ork` outer-shape table; v1 deviations kept as literals/separate aliases so Phase 3 does not change geometry).
- **2.1 review:** sliding fit = 0.2 on Ø (CLAUDE.md updated); lower motor retainer will use M2 + insert; aluminium bar 100 = commercial stock (may change with e-bay redesign); 8 screws per joint every 45° (v1: 2 holes × 4), coupler ring 5, coupling 7.5, screws at 7.5/2; parachute bay length open until deployment redesign.
- **2.2:** Lorenzo chose English for all names. Draft `docs/proposal/conventions.md`: rules, proposed English file names, feature/sketch/constraint patterns, Italian→English glossary.
- **2.2 approved.** 2.3 drafted: target layout (subsystem folders `ebay/`, `parachute_bay/`, `logo/`; `parameters.FCStd` at root; exports next to source; `archive/`), LFS findings, move checklist split into A (safe now) and B (assembly parts, blocked by A2plus). Found: `e-bay/lower_closure.3mf` = current print of `lower_closure_v2`; `low_chute_ret_test.3mf` = v1 seat test; extensionless `upper_retainer` = older STEP export.
- **2.3 review:** legacy files will be **deleted** (no `archive/`); exports stay next to source; LFS explained to Lorenzo, recommendation drop (awaiting OK); Lorenzo will install A2plus + Rocket WB (unblocks group B renames and Phase 3).
- **Later:** LFS dropped (`.gitattributes` → binary markers; commit 051414f). Lorenzo installed A2plus + Rocket WB → verified (see open question 8). 2.4 drafted: `docs/proposal/ork_vs_cad.md`. Findings: fin count is 4+4 in CAD (Rocket `FinCount=3` unused — corrected in `parameters.md`); lower fins overhang ≈19 mm past the lower fuselage onto the transition vs 5 mm forward in `.ork` (CP!); `.ork` has no mass overrides; assembly has overlapping outer surfaces (parachute bay vs mid fuselage ≈15 mm) so it is not reliable for positions; lengths disagree (nominal 860.5 / assembly 875.5 / measured 871 / `.ork` 864).
- **2.4 review:** fin dims in `.ork` are the design (CAD fins enlarged for slicer loss on the airfoil); the 19 mm fin overhang was an assembly error — real lower fins end 5 mm before the fuselage end, as in `.ork`; parachute Ø1 m with spill hole. Applied all agreed `.ork` fixes in one go (nose 120, transition aft Ø40 / wall 3, rear fins airfoil, component names, weighed mass overrides, parachute, piston, carbon rods, motor tube 37/35.2×93). Verified in OpenRocket 23.09 headless: loads, dry mass 575.5 g, CG 518 mm. Created `docs/inventory/section_lengths.md` for Lorenzo to measure.
- **Later:** Lorenzo measured sections (nose 125, e-bay tubes 104, mid 104.5, lower 112, transition 44.5, rings 5; spill hole ≈65 mm) and installed OpenRocket 24.12 (works with its bundled Java 17; system Java 25 makes the GUI quit silently). Applied lengths to `.ork` (nose = cone 120 + 5 mm base tube), fixed a material group; loads clean in 24.12. F35 simulated headless (`scripts/openrocket/Sim.java`): with weighed parts only, stability ≈ 0.1 cal ⚠; ≥ 1 cal needs ≈ 200 g more forward (electronics etc.) or CG with motor ≤ ≈ 511 mm. Apogee 150–316 m, descent 4.4–5.4 m/s, rod-exit 9–12 m/s (1 m rod). G115 not automatically safe (heavier tail). Results: `docs/analysis/f35_simulation.md`.
- **Later:** Lorenzo: plan is +200 g extra (final mass after electronics/deployment design), 2 m rail always planned, G115 CG to be tuned later with nose weight; fins measured = `.ork`; motor aft end 3 mm inside the transition → `.ork` motor overhang 41.5 mm. Re-ran F35 (2 m rail, +150/200/250 g, adapter 0/25 g): with +200 g stability 1.1–1.3 cal, rail exit ≈13 m/s, apogee ≈190–205 m, 6 s delay best. `Sim.java` now takes rail length and adapter mass.
- **Later:** parachute bay measured 241 (= `.ork`; section total 860 confirmed). No motor delay on the F35: the altimeter opens the parachute → `.ork` parachute set to deploy at apogee; `Sim.java` defaults to a plugged motor. Re-run: deploy speed ≈5 m/s in all configs.
- **2.5 approved.** Group A of `repo_structure.md` executed, one commit each (A2–A13): legacy files deleted (`wall`, `e-bay/lower_closure` v1, `parachute_fuselage_seat` v1, `parachute_fuselage_seat/upper_closure`, `low_chute_ret_test.3mf`, extensionless `upper_retainer`, `.keep`); parachute seat `.3mf` names fixed; logo files moved/renamed (`logo_flat`, `logo_projected`, `logo_inserts`). Checked first that no file links to them (only the assembly links parts, none in group A). Inventory regenerated (with add-ons installed, Rocket parts now recompute; fins still fail as known).
- **Left:** Phase 3.
- **3.0 diagnosis done** (copies only): causes and tested fixes for the 6 failing parts in the 3.0 table; `.mcp.json` committed.
