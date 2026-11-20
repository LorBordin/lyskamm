# Lyskamm — Roadmap (Phase 5)

**Goal:** first flight on the **TSP F35**, no deadline. The airframe is **designed and checked for the EXP G115 plus
a margin** (margin from Lorenzo in the structural session).
**Anti-goal:** no "extraordinary changes" to the airframe — structural work changes the design only when unavoidable.

- **Session protocol:** read `CLAUDE.md`, then **Status** below, then the active project's docs in
  `docs/projects/<id>_<name>/` (`design.md`, `plan.md`). At the end: update Status + Session log here (and the
  project's plan).
- Each project gets its design doc + plan **when it starts**, written with Lorenzo. CAD changes follow the CAD change
  protocol in `CLAUDE.md`.
- Setup history (Phases 0–4, questions Q1–Q17, sessions 1–9): `docs/archive/setup_plan.md`.
  Roadmap design: `docs/superpowers/specs/2026-11-20-phase5-roadmap-design.md`.

---

## Stages

| Stage | Project | Scope | Depends on | Status | Docs |
|---|---|---|---|---|---|
| 0 | **Loads** | `Sim.java` for F35 and G115: peak acceleration, max speed, rail-exit speed; estimate of the parachute opening shock. Input to every other project. | — | not started | — |
| 1 | **S — Structural analysis, pass 1** (flight loads) | Joints (7.5 mm couplings, M2 ≈3.7 mm from the part end, rigid ASA-CF can crack), carbon-rod load path (bending stiffness), fin attachment + flutter (PETG-CF), motor thrust path. Design load: G115 + margin. | 0 | not started | — |
| 1 | **D — Deployment + e-bay** | Review of the current setup (piston, chute retainers, connector, shock cord); black-powder charge (Lorenzo's idea, to discuss); electronics choice (altimeter/pyro, battery, switch); e-bay structure; ground tests. | 0 | not started | — |
| 1 | **L — Launch system** | Rail buttons or not; design and build of the rocket side; pad + 2 m rod/rail (absorbs the old deferred launch-pad project, `launch_pad/Unnamed.FCStd` = early concept). | 0 | not started | — |
| 1 | **M — Motor adapter** (minor, anytime) | F35 adapter in the Ø35.2 tube; lower motor retainer (M2 + insert); motor retention. | — | not started | — |
| 2 | **S — pass 2** (ejection loads) | Bulkheads, piston, joint screws under the ejection pressure. | D (charge sized) | not started | — |
| 3 | **I — Integration** (minor) | Weigh everything; update `lyskamm.ork` (masses, 871 mm); final CG; simulations F35 + G115; stability margin; **mark CG and CP on the fuselage**. | S, D, L, M | not started | — |
| 4 | **F — First flight readiness** | Ground ejection test, pre-flight checklist → first F35 flight. | all | not started | — |

## Backlog items by project
Carried over from the old Phase 5 backlog (`docs/archive/setup_plan.md`); each is detailed when its project starts.

**S — Structural**
- Joint load path: 7.5 mm couplings, 8 radial M2 screws ≈3.7 mm from the part end in rigid ASA-CF can crack; the
  4 carbon rods (Ø5/3, `parachute_fuselage_seat_v2` → `rear_transition`) spread the stress (bending stiffness).
- Fin attachment strength / flutter check with PETG-CF.
- Oversized M2 insert holes Ø3.1 → Ø2.6 (`low_chute_ret` Pocket004–007, `upper_retainer` `sede_viti_*`) — v1 mistake;
  applied in whichever redesign touches the part.

**D — Deployment + e-bay**
- E-bay: closures, aluminium bars, access, wiring, connector `presa_elettrica`.
- Recovery: parachute Ø1 m with spill hole, deployed at apogee by the altimeter (no motor delay charge);
  ejection/deployment (`pistone`, chute retainers), shock cord attachment.
- Open-DoF sketches in the connector/deployment parts (close during the redesign): `low_chute_ret`
  SK_connector_slot 7, SK_screw_eye_cable_holes 6; `parachute_fuselage_seat_v2` SK_rod_bosses 8, SK_connector_seat 8,
  SK_connector_seat_stop 9, SK_connector_slot 7; `lower_closure_v2` SK_connector_slot 7; `sede_presa_elettrica_sup`
  SK_seat 5; `pistone` SK_body 15.
- Small internal parts missing from `lyskamm_asm`: `pistone` (launch position, old Q17), `sede_presa_elettrica_sup`,
  `presa_elettrica`, screw eyes, logo inserts — needed for fit checks and CG.

**L — Launch system**
- Rail buttons or guides on the rocket (or none).
- Launch rail/rod **2 m** (≈13 m/s rail exit with the F35 at ≈0.9 kg); pad design and build.

**M — Motor adapter**
- Motor tube Ø35.2 inner / Ø37 outer (`lower_fuselage_v1`, `rear_transition`; `upper_retainer` bore 37.2), sized for
  the G115 (Ø35); TSP F35 (Ø29) needs an adapter. Motor retention.
- Lower motor retainer: fixed with M2 + heat-set insert → `rear_transition` Pocket002 Ø3.1 becomes Ø2.6.

**I — Integration**
- Final mass/CG: weigh after e-bay, deployment, launch interface and adapter are built; planned extra ≈ 200 g.
- `lyskamm.ork`: weighed masses, measured length 871 mm, motor mount, recovery, adapter; simulations F35 + G115
  with `scripts/openrocket/Sim.java`.
- Stability: with weighed parts only ≈0.1 cal ⚠; +200 g forward gives ≈1.1–1.3 cal with the F35
  (`docs/analysis/f35_simulation.md`). G115 needs its own check (heavier motor → CG aft; Lorenzo will add nose weight
  if needed). If too far aft: nose weight, move mass forward, or larger/further-aft fins.
- Mark CG and CP on the fuselage.
- Retire `scripts/rocket_simulator.py` (Lorenzo prefers `Sim.java`).

**Housekeeping** (when convenient)
- English file/folder renames, `docs/proposal/repo_structure.md` group B (`e-bay` → `ebay`, `pistone` → `piston`, …);
  update the relative links in `lyskamm_asm.FCStd` in the same step.
- Old Q14: MCP addon 0.1.31 vs server 0.1.32 — update the addon in FreeCAD's `Mod` folder.
- Old Q9: `execute_code_headless` can't find `freecadcmd` — workaround `~/Applications/FreeCAD.AppImage freecadcmd`
  works; optionally configure `--freecadcmd` in `.mcp.json`.
- Old Q15: `fuselage_w_logo` orientation (assembly uses flip=1) — Lorenzo to confirm by looking at the logo.
- Old Q16 remainder: upper fin T-foot vs `mid_fuselage` socket 0.04 mm (7.19 mm³/fin); lower alu rod vs
  `mid_closure` bore 0.13 mm³ — below print tolerance.
- Hand-drawn v1 points (e.g. boss corners r29.62 vs 29.59) — symmetric clean-up when a part is redesigned.

---

## Status
- **Current stage:** setup finished (2026-11-20). Next: **stage 0 — Loads**.
- **Next step:** run `Sim.java` for F35 and G115 and collect peak acceleration, max speed, rail-exit speed; estimate
  the opening shock. Then choose which stage-1 project to start (S, D, L or M).
- **Open questions:**
  1. G115 design margin (Lorenzo, in the S session).
  2. Deployment method: black-powder charge? (D session.)
  3. Electronics: altimeter/pyro, battery, switch (D session).
  4. Rail buttons or not (L session).
  5. Other known v1 issues besides the ones listed (old Q2)?

## Session log
### 2026-11-20 — Session 9 (cont.): roadmap
- **Done:** Phase 4 closed (see archive). Discussed Phase 5 with Lorenzo: 3 main projects (structural analysis,
  deployment + e-bay, launch system) + minor items (motor adapter, final CG/CP marking, final stability and
  simulation). Design approved (option A: roadmap + one doc per project). Old `PLAN.md` archived to
  `docs/archive/setup_plan.md`; this roadmap created; `CLAUDE.md` session protocol updated (old 4.4).
- **Decisions (Lorenzo):** goal = first F35 flight, no deadline; airframe designed for the G115 + margin;
  deployment likely black powder (to discuss); no extraordinary changes unless unavoidable.
- **Left:** stage 0.
