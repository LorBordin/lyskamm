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
- [ ] 1.7 ⏸ **Review with Lorenzo**: confirm active vs legacy files, meaning of shared dims. Commit inventory.

## Phase 2 — Proposal (documents only, no CAD changes)
- [ ] 2.1 Master parameters: list of aliases (e.g. `body_od`, `body_wall`, `coupler_od`, `fit_slide=0.15`, `m2_insert_hole`, `m2_clear`, `rod_d`, `fin_t`, `motor_d`…), units, source part, ork counterpart. Decide mechanism: (a) single `parameters.FCStd` spreadsheet referenced by cross-document expressions vs (b) per-part sheet synced by script. Recommendation: (a). Files: `docs/proposal/parameters.md`
- [ ] 2.2 Naming conventions: files (`snake_case`, English, no version suffix — git holds versions), bodies, features (`Pad_body`, `Pocket_m2_holes`…), sketches (`SK_<purpose>`), constraints (`<alias-like name>`). Files: `docs/proposal/conventions.md`
- [ ] 2.3 Repo structure proposal: per-part folders, `exports/` for STEP/3MF, `archive/` for legacy, `parameters.FCStd` location, LFS fix (`*.FCStd`, install git-lfs or drop LFS). File moves/renames listed as a checklist (each a separate, approved step; A2plus `sourceFile` paths must be updated). Files: `docs/proposal/repo_structure.md`
- [ ] 2.4 `.ork` ↔ CAD mismatch table: lengths, Ø, fin geometry & thickness (ork 3 mm), transition, materials, missing motor mount/motor/recovery/mass overrides, motor Ø 29 vs 35 mm. Files: `docs/proposal/ork_vs_cad.md`
- [ ] 2.5 ⏸ **Review with Lorenzo**: approve parameter list, conventions, structure, which .ork fixes to apply. Commit.

## Phase 3 — Parametrization (part by part, geometry must not change)
Pre-step:
- [ ] 3.0 FreeCAD 1.0 → 1.1.3 migration check: open each active part in 1.1.3, recompute, compare volume/bbox against Phase 1 baseline. 🔧 Save only with approval (one commit "resave in 1.1.3" per part if needed). Check A2plus works in 1.1.3 (else plan migration to built-in Assembly — separate decision).
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
- [ ] Launch rail / pad interface (rail buttons or guides; `launch_pad/Unnamed.FCStd`).
- [ ] Motor mount compatible with both F35 (Ø29) and G115 (Ø35) — adapter or two mounts; motor retention.
- [ ] Recovery: chute size, ejection/deployment (`pistone`, chute retainers), shock cord attachment.
- [ ] OpenRocket model completion: real materials, motor mount, motor, masses/overrides, recovery, simulations (F35, G115), stability margin.
- [ ] Simulator fix: include motor casing mass; read params from `parameters.FCStd`/`.ork`.
- [ ] Fin attachment strength / flutter check with PETG-CF.
- [ ] Logo parts: keep, archive or integrate.
- [ ] Assembly is incomplete: small/inner flying parts (`pistone`, `fuselage_w_logo`, connectors, screw eyes, …) are not in `assemblies/lyskamm.FCStd` — add them (needed for fit checks and CG).
- [ ] Update `.ork` and `rocket_simulator.py` with weighed masses (`docs/inventory/masses.md`) and measured length 871 mm.
- [ ] [others — Lorenzo to fill in]

---

## Status
- **Current phase:** 1 — Inventory (1.1–1.6 done, outputs in `docs/inventory/`, not yet committed)
- **Next step (new session):** 1.7 review of the inventory with Lorenzo — start by verifying which `.FCStd` matches `fuselage_w_logo.3mf` (open question 1), then add it to `EXTRA_ACTIVE` in `scripts/shared_dims.py`, regenerate `docs/inventory/`, and go through shared-dimension meanings and legacy files.
- **Pause point:** session 2 ended after 1.6 (2026-11-16). Phase 1 outputs committed as a pre-review checkpoint.
- **Open questions:**
  1. Which files are legacy vs still needed (see "In assembly = no" in `docs/inventory/summary.md`)? Known so far: `fuselage_w_logo` (4th fuselage) and `pistone` fly although not in the assembly. Answered: the printed logo fuselage is exactly `components/e-bay/fuselage_w_logo.3mf` (ASA-CF fuselage + small PLA logo inserts, PLA only for colour, non-structural). Its objects are named `fuselage_w_logo.step` + `logo_*.step` → in 1.7 verify which `.FCStd` matches (likely `fuselage_w_logo.FCStd`, by export name; check by comparing mesh volume/bbox) before adding it to `EXTRA_ACTIVE`.
  2. Other known v1 issues besides e-bay and pad interface?
  3. Master parameters mechanism: single `parameters.FCStd` with cross-document expressions (recommended) or per-part spreadsheets?
  4. Keep A2plus, or migrate the assembly to FreeCAD's built-in Assembly workbench (later, separate decision)?
  5. Git LFS: install and fix `.gitattributes`, or drop LFS?
  6. `.ork` vs CAD: which is the source of truth for outer dimensions?
  7. Delete or ignore the untracked `init` file; commit `.mcp.json`?
  8. **Rocket workbench and A2plus are not installed in FreeCAD 1.1.3** (only FreeCADMCP is). Without Rocket, `nose_cone`, `lower_fins`, `upper_fin`, `rear_transition` cannot recompute their base feature; without A2plus the assembly cannot be solved/updated. Install both, or replace those features (blocks Phase 3)?
  9. The MCP `execute_code_headless` cannot find `freecadcmd` (FreeCAD is an AppImage). Configure `--freecadcmd` in `.mcp.json`, or keep running the AppImage's `freecadcmd` via shell?
  10. Meaning of shared dimensions (`docs/inventory/shared_dims.md`), e.g. 3.75, 23.428, 7.5, 6.15/9.6, and the fits 55.8↔56 (Δ0.2), 41.9↔42.1, 37↔37.2.
  11. Features already failing in FreeCAD 1.1.3 recompute (copies only): `e-bay/fuselage` Chamfer, `lower_fuselage_v1` Sketch014, `mid_fuselage` Sketch011, `wall` Sketch005, `launch_pad` Sketch001 … — fix during Phase 3.0?
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
