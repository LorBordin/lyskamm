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
- [x] 3.0 FreeCAD 1.0 → 1.1.3 migration check: open each active part in 1.1.3, recompute, compare volume/bbox against Phase 1 baseline. 🔧 Save only with approval (one commit "resave in 1.1.3" per part if needed). Check A2plus works in 1.1.3 (else plan migration to built-in Assembly — separate decision).
  - **3.0 diagnosis (2026-11-17, copies only; test script `scripts/fc_migration_fix_test.py`).** All active parts load; Rocket WB + A2plus OK. 6 parts have features that fail on recompute; every fix below was tested on copies and gives a valid model:

    | Part | Cause | Fix | Result vs stored |
    |---|---|---|---|
    | `lower_fins`, `upper_fin` | Rocket WB now measures the through-the-wall tab offset from the other end → tab moves 3.5 / 5 mm aft, `Sketch` loses `Face10` | `Fin.TtwOffset` 3.5 → 0 / 5 → 0 | **identical** (= printed tab 0–60 / 0–90 mm) |
    | `e-bay/fuselage`, `fuselage_w_logo` | `Chamfer` (24 edges, 1.2 mm) has stale 1.0 edge names | clear and re-select the same 24 edges | **identical** |
    | `mid_fuselage` | `Sketch011` support `Face37` renumbered | support → `PolarPattern004.Face36`; `Chamfer` (32 edges) re-selected on the rebuilt edges (48 pieces) | valid; vol −2.45 mm³ (0.004 %), CoM 0.003 mm |
    | `lower_fuselage_v1` | chain: `Sketch014`/`Sketch015` supports renumbered; `Chamfer002` (44), `Chamfer` (13), `Chamfer004` (8) stale edges; `Sketch020` support renumbered + constraint 16 redundant in the 1.1 solver | supports → `Face1`, `Face1`, `Chamfer003.Face26`; re-select chamfer edges; delete redundant constraint 16 (PointOnObject, duplicate of 15) | valid; vol −1.93 mm³ (0.003 %), CoM 0.0002 mm |

    Progress: ✅ fins, ✅ e-bay fuselages (2026-11-17), ✅ mid_fuselage, ✅ lower_fuselage_v1 (2026-11-18) — all checked by Lorenzo. **3.0 done.**
    **Follow-up 2026-11-18 — sketch close hang in `mid_fuselage`:** closing any sketch *before* `PolarPattern007` froze FreeCAD 1.1.3 (force quit). Cause: `PolarPattern007` was "Transform body" (`TransformMode=1`, no `Originals`) → 4 copies of the whole fuselage fused + refined (≈48 s headless, hangs in GUI); present since the first commit. Fix (approved, checked): `TransformMode=0`, `Originals=[Pad005]` → body identical (Δvol 0.002 mm³, boolean diff 0), full rebuild 5.5 s. `e-bay/upper_closure` `PolarPattern` is also "Transform body" (small part, no hang seen; revisit in its Phase 3 rework).
    **Known, not fixed (geometry unaffected):** sketches with external geometry flagged `Missing` (links lost in the 1.0 → 1.1 move; frozen copies still drive constraints): `mid_fuselage` Sketch005/006/014/015, `lower_fuselage_v1` Sketch008/011/013/014/015/017/019; `lower_fuselage_v1` Sketch021 redundant constraint #30. **Cleaned up 2026-11-18 (session 6)** — see the Session log; Finder below still valid for other parts. Finder: in `Document.xml`, `ExternalGeo` geometries with `flags & 8` (Missing).
    The ≈2 mm³ differences are FreeCAD 1.0 → 1.1 kernel drift (upstream features already differ by up to 0.7 mm³ before any fix). Proposed tolerance for 3.0 only: volume ≤ 0.01 %, CoM ≤ 0.01 mm, plus Lorenzo's visual check. Apply one part at a time (save = resave in 1.1.3), each approved, checked in FreeCAD, committed.
- [x] 3.1 *(done 2026-11-18: repo root, spreadsheet `Params`, 39 aliases; parts link it as `../../parameters.FCStd` from `components/<subsystem>/`, tested on scratch copies)* 🔧 Create `parameters.FCStd` with the approved aliases. ⏸ Review with Lorenzo → check in FreeCAD → commit.

Per part, in this order (interfaces outward from the body tube): `e-bay/fuselage` → `mid_fuselage` → `lower_fuselage_v1` → `rear_transition` → `upper_retainer` → `low_chute_ret` → `parachute_fuselage_seat_v2` → `sede_presa_elettrica_sup` → `e-bay/lower_closure_v2` → `mid_closure` → `e-bay/upper_closure` → `nose_cone` → `upper_fin` → `lower_fins` → `carbon_rod` → `alluminium_rod`.
For each part (one checklist block per part, added to this file when the part starts):
- [ ] a. Record baseline: volume, bbox, center of mass (read-only).
- [ ] b. 🔧 Rename sketches/features per convention. ⏸ Review → apply → ⏸ Lorenzo checks → commit.
- [ ] c. 🔧 Name driving constraints and bind them to aliases (`=parameters#Params.body_od`), small batches (≤ ~10 constraints per change). Close DoF where needed. ⏸ Review → apply → ⏸ Lorenzo checks → commit.
- [x] d. Verify geometry unchanged: volume/bbox/CoM equal to baseline (tolerance 1e-6 **relative**). *(closed 2026-11-18 for all parts)*
- [x] e. 🔧 ~~Update the part in the A2plus assembly~~ → *replaced (Q4):* add the part to the new built-in Assembly, check interferences/fits with neighbours. ⏸ Lorenzo checks → commit. *(done 2026-11-19: `assemblies/lyskamm_asm.FCStd`, 30 parts, checked by Lorenzo; small internal parts deferred)*
- [ ] f. Re-export STEP/3MF only if Lorenzo asks.
#### Per-part progress (a–e)
Done 2026-11-18 (Lorenzo chose a batch run: labels + bindings for all parts, asking only on unclear points). Every part:
saved from the GUI, one commit each; checked against a plain 1.1.3 recompute of the untouched file (vol/CoM within
~1e-9 relative, no Invalid features, every bound value unchanged); final fresh headless reopen of all parts:
`parameters.FCStd` loads via `../../parameters.FCStd`, no Invalid features. Parts never resaved in 3.0 show
1.0 → 1.1.3 drift of ≤ 0.01 mm³ (nose_cone, closures, upper_retainer, rear_transition) — recorded in their commits.
`e-bay/fuselage` baseline: vol 18786.308720 mm³, CoM (0.000966, 18.662062, 52.500357).

| Part | a | b labels | c bindings | d | e assembly | Notes |
|---|---|---|---|---|---|---|
| e-bay/fuselage | ✅ | ✅ | ✅ 19 | ✅ | ✅ | SK_shell DoF 4 → 0 |
| e-bay/fuselage_w_logo | ✅ | ✅ shell only | ✅ 19 | ✅ | ✅ | logo bodies keep labels (logo is local) |
| mid_fuselage | ✅ | ✅ | ✅ 26 | ✅ | ✅ | |
| lower_fuselage_v1 | ✅ | ✅ | ✅ 21 | ✅ | ✅ | Sketch021 redundant #30 deleted (sketch was Invalid on plain recompute); DoF 12 → 13. **New baseline after `Missing` cleanup (session 6): vol 63875.289949 mm³** (−7.6e-7 relative, µm slivers removed) |
| rear_transition | ✅ | ✅ | ✅ 21 | ✅ | ✅ | `Pocket_fin_slot_fillers_trim` removes no material |
| upper_retainer | ✅ | ✅ | ✅ 21 | ✅ | ✅ | |
| low_chute_ret | ✅ | ✅ | ✅ 24 | ✅ | ✅ | |
| parachute_fuselage_seat_v2 | ✅ | ✅ | ✅ 13 | ✅ | ✅ | |
| sede_presa_elettrica_sup | ✅ | ✅ | ✅ 3 | ✅ | ⏸ deferred | deployment system, mostly local |
| pistone | ✅ | ✅ (Body → `piston`) | ✅ 2 | ✅ | ⏸ deferred | not in assembly |
| e-bay/lower_closure_v2 | ✅ | ✅ | ✅ 19 | ✅ | ✅ | `Pocket_bar_slots` = aluminium bar slots |
| e-bay/mid_closure | ✅ | ✅ | ✅ 22 | ✅ | ✅ | |
| e-bay/upper_closure | ✅ | ✅ | ✅ 21 | ✅ | ✅ | key pattern: Transform body → `Pad003` only (bool diff 0) |
| e-bay/presa_elettrica | ✅ | ✅ | — | ✅ | ⏸ deferred | local dims only |
| nose_cone | ✅ | ✅ | ✅ 10 | ✅ | ✅ | |
| upper_fin / lower_fins | ✅ | ✅ | ✅ 2 each | ✅ | ✅ | Fin root/TTW thickness = `fin_t` |
| carbon_rod / alluminium_rod | ✅ | ✅ | ✅ 3 each | ✅ | ✅ | |

Rules applied: v1 deviations stay literals (Ø3.1 inserts, 5.15/5.25/5.75 rod/bar holes, 7.2/7.25 bosses, 3.2 M3,
nose Ø2.1 holes + 1.0 chamfer); local dims (connector, deployment, 41.x Ø) stay literals; repeated dims: one
named constraint per sketch, duplicates bound to the same alias unnamed.
**Still open in Phase 3:** close DoF in the sketches listed below (only trivial ones were closed); ~~clean up `Missing` external geometry~~ (done session 6); step e is **replaced** by building a new assembly in the built-in Assembly workbench (Q4, resolved) — no A2plus update.
Sketches with DoF > 0 (2026-11-18): fuselage_w_logo logo Sketch010/012/014/016 (1); mid_fuselage SK_rod_bosses 5,
SK_fin_slots 2, SK_m2_holes_upper_67_5 1, SK_end_padding_lower 1; lower_fuselage_v1 SK_motor_centering 1, SK_fin_slot 2,
SK_fin_slot_lower 1, SK_transition_lugs 1, SK_transition_fit 3, SK_transition_fit_trim 1,
SK_shell_reinforcement_rods 11, SK_shell_reinforcement_fins 13 (SK_reinforcement_lower was 7, closed to 0 in session 6); rear_transition SK_rod_bosses 3, SK_motor_tube_ribs 2,
SK_fin_slot_fillers 2, SK_fin_slot_fillers_trim 1; upper_retainer SK_rod_bosses 4, SK_fin_slot_fillers 1; low_chute_ret
SK_connector_slot 7, SK_screw_eye_cable_holes 6; parachute_fuselage_seat_v2 SK_rod_bosses 8, SK_cable_passage 2,
SK_connector_seat 8, SK_connector_seat_stop 9, SK_connector_slot 7, SK_connector_pins 2; sede_presa_elettrica_sup SK_seat 5,
SK_connector_recess 3, SK_extraction_notches 1; pistone SK_body 15, SK_cavity 2; lower_closure_v2 SK_connector_slot 7,
SK_connector_pins 2, SK_keys 1, SK_key_trim 3, SK_m2_inserts_22_5/67_5 2; mid_closure SK_keys 2, SK_key_trim 1,
SK_bar_seat_lower 2; upper_closure SK_keys 2, SK_key_trim 1.

- [x] 3.99 *(done 2026-11-18, scratch copies in the session scratchpad, repo untouched)* Smoke test: `fit_slide_d` 0.2 → 0.3 mm. Exactly the sliding-fit parts changed (full forced recompute, no Invalid): `lower_closure_v2` −300.8, `mid_closure` −115.8, `upper_closure` −121.5, `low_chute_ret` −128.5, `rear_transition` −26.6, `upper_retainer` −125.4 mm³; all other parts unchanged. Note: set sheet cells **with units** (`"0.3 mm"`), a bare number breaks `body_id - fit_slide_d` (unit mismatch). Headless recompute needs `touch()` on all objects.

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
- **Current phase:** 3 — 3.0, 3.1 done; per-part labels + bindings (b–d) done for all 18 parts and checked by Lorenzo 2026-11-18.
- **Next step:** (1) ~~scratch trial~~ ✅, ~~new assembly~~ ✅ `assemblies/lyskamm_asm.FCStd` (checked by Lorenzo 2026-11-19; small internal parts — pistone, sede_presa_elettrica_sup, presa_elettrica, logo inserts — deferred, Lorenzo: "there'll be time to adjust"); (2) DoF closing, one part at a time (start with sketches of 1–2 DoF); (3) build the new assembly (step e replaced), incl. the missing parts. Done: 3.d, 3.99, `Missing` cleanup (both parts checked by Lorenzo).
- **Open questions:**
  1. ~~Which `.FCStd` matches `fuselage_w_logo.3mf`~~ — resolved: `e-bay/fuselage_w_logo.FCStd` (added to `EXTRA_ACTIVE`). Active/legacy status of all non-assembly files → `docs/inventory/file_status.md`.
  2. Other known v1 issues besides e-bay and pad interface?
  3. ~~Master parameters mechanism~~ — resolved: (a) single `parameters.FCStd` (spreadsheet `Params`) with cross-document expressions.
  4. ~~Keep A2plus or migrate?~~ — **resolved 2026-11-18 (Lorenzo): migrate to the built-in Assembly workbench**, built from scratch (live links, no "update from source"), after a scratch trial with 3–4 neighbouring parts (e.g. `mid_fuselage`, `lower_fuselage_v1`, `rear_transition`: links, cross-document `parameters.FCStd`, fit checks, CG). Keep the A2plus assembly untouched until the new one is trusted.
  5. Git LFS: install and fix `.gitattributes`, or drop LFS? Finding: LFS never actually used (no pointers in history, `*.fcstd` pattern wrong case, git-lfs not installed); recommendation in `repo_structure.md`: drop it.
  6. ~~`.ork` vs CAD source of truth~~ — resolved: lengths/body shape = CAD (confirmed by physical measurement); fin dimensions = `.ork` (CAD fins enlarged on purpose for slicing loss); masses = weighed.
  7. ~~Delete or ignore the untracked `init` file; commit `.mcp.json`?~~ — resolved: `init` deleted by Lorenzo (2026-11-18), `.mcp.json` committed.
  8. ~~Rocket WB + A2plus missing~~ — installed by Lorenzo (2026-11-17), verified headless on copies: all Python objects load; `nose_cone` and `rear_transition` recompute identically; assembly loads (32 parts, no errors). ⚠ `lower_fins`/`upper_fin`: the Rocket `Fin` now regenerates +0.75 % volume and `Sketch` (attached to `Face10` of the fin) becomes Invalid → fix in Phase 3.0.
  9. The MCP `execute_code_headless` cannot find `freecadcmd` (FreeCAD is an AppImage). Configure `--freecadcmd` in `.mcp.json`, or keep running the AppImage's `freecadcmd` via shell?
  10. ~~Shared dims~~ — resolved (`MEANINGS` in `scripts/shared_dims.py`).
  11. Features already failing in FreeCAD 1.1.3 recompute (copies only): `lower_fins`/`upper_fin` Sketch (on `Face10`, after Rocket WB install); `e-bay/fuselage` Chamfer, `lower_fuselage_v1` Sketch014, `mid_fuselage` Sketch011, `wall` Sketch005, `launch_pad` Sketch001 … — fix during Phase 3.0?
  12. ~~Mass estimate~~ — resolved: weighed masses only; Bambu Studio for new parts.
  13. ~~New assembly layout~~ — **resolved 2026-11-19 (Lorenzo approved the proposals):** nose tip at origin, axis −Z, `nose_cone` grounded; file `assemblies/lyskamm_asm.FCStd`; joints on part edges (no LCS added to parts); `Mass_g` on each link + `scripts/assembly_cg.py`. Original options (from the 2026-11-19 trial): (a) global frame — trial used rocket axis = +Z toward the nose, origin = `lower_fuselage_v1` origin (grounded); proposal for the real one: nose tip at origin, axis along −Z (positions = OpenRocket's "from nose tip"), ground `nose_cone`; (b) file name/location (`assemblies/lyskamm_asm.FCStd`?); (c) joints on circular edges (TNP-tracked OK in the trial) vs datum/LCS per part (more robust, but adds features to every part); (d) masses as a `Mass_g` property on each link + a CG script (`scripts/assembly_cg.py`).
  15. `fuselage_w_logo` orientation: two mirror-image fits (logo toward the nose or toward the tail); assembly uses flip=1 (`asm_build.py`) — Lorenzo to confirm by looking at the logo.
  16. Carbon rod channel (seat top → `rear_transition` bottom) is 449.5 mm, rod CAD is 450 mm → 0.5 mm overlap with `rear_transition` (real rods cut shorter? or one hole open?). Also: upper fin T-foot vs `mid_fuselage` internal socket 0.04 mm overlap (7.19 mm³/fin); lower alu rod vs `mid_closure` Ø7.25 bore 0.13 mm³. All below print tolerance.
  17. Pistone launch position — deferred with the other small internal parts (Lorenzo 2026-11-19).
  14. MCP addon 0.1.31 (protocol 1) vs server 0.1.32 (protocol 3): update the addon in FreeCAD's `Mod` folder (warning on every call; everything worked so far).

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

### 2026-11-18 — Session 4 (Phase 3.0 fixes)
- **Done:** 3.0 fixes applied through the FreeCAD GUI (MCP), each approved, checked by Lorenzo, committed: fins (`e7eec70`), e-bay fuselages (`7391efc`), `mid_fuselage` (`8f27561`), `lower_fuselage_v1` (`979b501`). Timestamp-only re-saves of already committed files discarded.
- **Bug found:** opening/closing a sketch in `mid_fuselage` froze FreeCAD. Not the migration fixes: `PolarPattern007` patterned the whole body (≈48 s per rebuild). Changed to pattern only `Pad005` → identical solid, rebuild 5.5 s; Lorenzo confirmed sketches close normally.
- **Learned:** checks after a CAD change must include rebuild time from the first sketch and the sketches' external geometry (`Missing` flags), not only Invalid features + volume. FreeCAD restarts kill the MCP RPC server (Lorenzo restarts it).
- **Left:** 3.1 `parameters.FCStd` (next session); `Missing` external geometry cleanup per part in Phase 3.

### 2026-11-18 — Session 5 (Phase 3.1 + per-part parametrization)
- **Done:** 3.1 `parameters.FCStd` (repo root, `Params`, now 44 aliases incl. `key_size`, `key_notch_size`, `m2_csk`, `fin_shell_wall`, `fin_shell_id`); cross-document link tested. `e-bay/fuselage` done step by step with Lorenzo (baseline, labels, bindings). Then, at Lorenzo's request, batch run over the other 17 parts (labels + Params bindings, one commit per part, ask only on unclear points). Lorenzo explained the unclear features (end paddings, transition fit, fin-slot fillers, connector seat stop, extraction notches, screw-eye/cable holes). Q7 resolved (`init` deleted).
- **Decisions:** Params holds CAD design values; small CAD-vs-measured differences are expected (polish, measurement error). Fin chord/height stay local to the fin parts. v1 deviations stay literals. Key notches on the e-bay shells = alignment V for the closure keys (7.5 key / 7.6 notch).
- **Issues:** during the propagation test I edited Params cells by stale row numbers (restored before saving); literal expressions on fin flange dims (removed). Rebuild of the whole part takes ≈5–7 s for mid/lower fuselage and rear_transition (as before).
- **Later same session:** Lorenzo checked all parts (OK); Pocket009 = aluminium bar slots → labelled.
- **Left:** DoF closing; Missing external geometry; A2plus update (e); 3.99.

### 2026-11-18 — Session 6
- **Done:** 3.99 smoke test (see 3.99). No CAD files changed.
- **Left:** `Missing` external geometry cleanup; DoF closing; step e (A2plus; decide Q4 first).
- **`Missing` external geometry cleanup (Lorenzo: do A/B/C autonomously, option 1 for C, check at the end).** Analysis of all 16 flagged links: 7 unused (deleted), 2 with an exact edge match (re-linked), the rest replaced by in-sketch geometry. All via the GUI, `mid_fuselage` commit `04c1fee`; `lower_fuselage_v1` commit below.
  - `mid_fuselage`: SK_end_padding_upper −4 → `Chamfer001.Edge101`; SK_rod_bosses r27 reference → construction circle Ø54 (`ref_circle_d`); SK_fin_slots slot walls → constraints on the sketch's own lines (parallel to the 45° axis, `slot_half_w` 1.55). Volume/CoM identical, rebuild 5.5 s, DoF unchanged.
  - `lower_fuselage_v1`: SK_fin_can → construction circle bound to `fin_shell_id`; SK_shell_lower tangents → `body_od` / `fin_shell_id` diameters; SK_transition_lugs → `DistanceX -3.1`, `DistanceY 20`, `Vertical`; SK_transition_fit_trim → `Pocket005.Edge1`; SK_reinforcement_lower (8 slot-wall links): angle ±45°/±135° + distance 1.55 per wall, Ø`fin_shell_id`, coincident joints; DoF 7 → 0.
  - **Side effect in `lower_fuselage_v1`:** the old sketch had its slot-wall lines off the exact 45° by ≈3e-4° and 3 µm; making them exact removed µm-size sliver faces/edges, so volume 63875.338302 → 63875.289949 (−0.048 mm³, 7.6e-7 relative; CoM +5e-5 mm). `Chamfer_fin_can` had to be re-selected (edge numbering changed; same 8 r29 arcs at z = 5). Rebuild 7.2 s (was 7.1 s).
  - Method note: replacing an external link by `Angle`/`Distance` constraints must use **exact** angles; copying the sketch's current angle locks in the old sloppiness.
- **Decision (end of session):** Lorenzo chose to migrate to the built-in Assembly workbench (Q4 resolved); build from scratch, scratch trial first, A2plus assembly kept untouched meanwhile. Step e of Phase 3 is replaced accordingly.
- **Left:** scratch trial of built-in Assembly; DoF closing; new assembly. Lorenzo checked both parts (OK); 3.d closed.

### 2026-11-19 — Session 7 (built-in Assembly scratch trial)
- **Done:** scratch trial of the built-in Assembly workbench (Q4) — copies only in the session scratchpad (`asm_trial/`, same folder layout as the repo); **repo `.FCStd` files untouched**. Parts: `rear_transition`, `lower_fuselage_v1`, `upper_retainer`, `mid_fuselage` (`upper_retainer` added because `mid_fuselage` and `lower_fuselage_v1` don't touch).
- **Setup:** one `Assembly`, `App::Link` to each part's `Body`; `lower_fuselage_v1` grounded; 3 `Fixed` joints on the coaxial r30 circles of the shoulder faces (lower z−20 ↔ rear x0; lower z93 ↔ retainer z0; retainer z5 ↔ mid z97.5).
- **Results (all OK):**
  - Stack: total 268 mm = 45 + 113 + 5 + 105 (= section lengths). Spigots sit in their sockets.
  - Holes: all M2 radial holes coincide (z and angle) at the 3 joints.
  - Interference (pairwise `common`, 4 s): 0 mm³ for all pairs. Fin slots: lower 45°, upper 45° (in line), `rear_transition` fillers 45°; retainer keys / rod bosses 0°/90°.
  - Live links: `fit_slide_d` 0.2 → 0.3 in the scratch `parameters.FCStd` → parts update in the assembly (retainer −125.4, rear −26.6 mm³, as in 3.99), placements unchanged, solve 0.02 s. Joint edge refs renumbered by the recompute (`Edge630→634`, `Edge247→293`) and the toponaming tracked them to the right circles. Restored to 0.2.
  - Save/reopen: links stored relative (`../components/...`); reopening the assembly auto-loads the 4 parts **and** `parameters.FCStd` (through the parts) in 3.2 s; no Invalid; solve 0.02 s.
  - CG: weighed masses as a `Mass_g` property on each link × CAD CoM → 148.5 g, CG 132.4 mm from the tail end (4 parts only; uniform-density CoM per part).
- **Gotchas found:**
  - The 8 × 45° M2 pattern does **not** fix the roll: `mid_fuselage` at 0° had all holes aligned but the retainer keys hit the wall (175 mm³ interference). Fixed with the joint offset yaw 45°. → Always run the interference check, not only the hole check.
  - `Fixed` joints on circles keep the closest direction: 2 joints needed a flip (GUI "Reverse" / `flipOnePart`).
  - Joints created from Python need `JointObject.ViewProviderJoint` on their view objects (else an error on close). Use the GUI tools or add them.
  - The old A2plus assembly puts `lower_fuselage_v1` ≈20.8 mm off: the `rear_transition` spigot sits outside the lower fuselage. One more reason not to copy positions from it.
- **Left:** Lorenzo checks the scratch assembly in the GUI (`trial_assembly` open); decide Q13 (frame/origin, file name, joint references, CG method); then build the real assembly part by part (step e). DoF closing still open.
- **New assembly built (same session, Lorenzo approved Q13):** `assemblies/lyskamm_asm.FCStd`, built-in Assembly, 30 `App::Link`s to part `Body`s (all airframe sections, 4+4 fins, 4 carbon rods, 4 aluminium rods), nose tip at origin, axis −Z, `nose_cone` grounded, 29 `Fixed` joints on part edges with offsets; `Mass_g` on every link. Scripts: `scripts/asm_stack_solver.py` (orientation/roll solver), `scripts/asm_build.py` (build), `scripts/assembly_cg.py` (mass/CG). No part file modified.
  - **Placement derived from geometry, not from A2plus** (A2plus is wrong in several places): per joint both flips × all rolls that align the radial M2 holes, keep zero interference; ties broken by: aluminium rods (100 mm, exact fit uc ↔ mc), lower-closure connector slot ↔ seat cable channel/connector seat (270°), carbon-rod channel seat → rear_transition (continuous, 4 × 90°), retainer keys, rear fin-slot fillers. Result: fins upper and lower in line at 0/90/180/270°, carbon rods at 45°, e-bay rods at 0/180°.
  - `mid_closure` is reversed vs A2plus (rod holes prove it). `fuselage_w_logo` is modelled with its axis at local z = 30 (offset handled in the link placement). The 2026-11-19 trial had `lower_fuselage_v1` 45° off (holes OK, rods/fins not) — the trial is superseded.
  - Checks: all-pairs interference — only the Q16 slivers; stack length 868 mm (tip → tail); save/reopen 4.6 s, relative paths, `parameters.FCStd` auto-loaded, solve 0.02 s, no Invalid. Mass 504.5 g, CG 533.8 mm from nose tip (CAD parts only).
  - **FreeCAD 1.1.3 solver bug:** with the 2 lower aluminium rods fixed to `lower_closure` the solve failed (`Solve failed: vector::_M_range_check: __n (which is 448) >= this->size()`), each joint alone was fine. Re-hosted both rods on `mid_closure` (physically correct too: they screw onto its threaded rod) → solves. If it comes back when adding parts: split into sub-assemblies (e-bay / aft).
- **Checked by Lorenzo ("perfect"); step e marked done.** Small internal parts deferred. Left: DoF closing; Q15 (logo direction), Q16 (small CAD overlaps) whenever convenient.
