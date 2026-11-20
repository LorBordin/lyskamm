# Phase 5 roadmap — design (short)

Date: 2026-11-20. Status: draft for Lorenzo's approval.

## Intent
The project setup (Phases 0–4) is finished. What is left is real design work, organised around a clear goal:
**first flight on the TSP F35**, no deadline, with the airframe **designed and checked for the EXP G115 plus a
margin** (Lorenzo gives the margin in the structural session).

## Goals / non-goals
- Goals: one short roadmap that shows the stages toward the first F35 flight, the dependencies between the
  projects, and where each project stands; each project planned in detail only when it starts.
- Non-goals: detailing projects now (deployment, electronics, launch system are not discussed yet); any CAD change.
- Anti-goal: "extraordinary changes" to the airframe — structural work changes the design only when unavoidable.

## Approaches (Lorenzo picked A)
- **A — roadmap + one doc per project** (picked): `PLAN.md` becomes a short roadmap; each project gets its own
  design doc + plan when it starts. Matches one-topic-per-session work; no guessed details.
- B — one big detailed `PLAN.md` now: mostly guesses, heavy rewrites later.
- C — separate plans, no roadmap: loses the cross-dependencies (CG waits on everything, structure waits on the
  ejection loads).

## The roadmap (stages)
0. **Loads** — `Sim.java` for F35 and G115: peak acceleration, max speed, rail-exit speed; estimate of the
   parachute opening shock. Input to every other project.
1. **Parallel tracks**
   - **S — Structural analysis, pass 1 (flight loads):** joints (7.5 mm couplings, M2 ≈3.7 mm from the part end),
     carbon-rod load path, fin attachment + flutter (PETG-CF), motor thrust path. Design load: G115 + margin.
   - **D — Deployment + e-bay:** current setup review, black-powder charge (Lorenzo's idea, to discuss), piston,
     retainers, shock cord, electronics choice (altimeter/pyro, battery, switch), e-bay structure, ground tests.
   - **L — Launch system:** rail buttons or not; design and build of the rocket side; pad + 2 m rod/rail
     (absorbs the deferred launch-pad project).
   - **M — Motor adapter (minor, anytime):** F35 adapter in the Ø35.2 tube, lower motor retainer (M2 + insert),
     motor retention.
2. **S pass 2 (ejection loads)** once D has sized the charge.
3. **I — Integration (minor):** weigh everything, update `lyskamm.ork` (masses, 871 mm), final CG, simulations
   F35 + G115, stability margin, mark CG and CP on the fuselage.
4. **F — First flight readiness:** ground ejection test, pre-flight checklist → first F35 flight.

## Where the current backlog items go
| Backlog item | Goes to |
|---|---|
| Ø3.1 → Ø2.6 insert holes (`low_chute_ret`, `upper_retainer`) | S (joint strength), applied in the redesign that touches the part |
| Joint/carbon-rod load path, fin strength/flutter | S |
| E-bay, recovery, open-DoF connector/deployment sketches, small parts missing from the assembly | D |
| Rail buttons, 2 m rod/rail, launch pad | L |
| F35 adapter, lower motor retainer (+ `rear_transition` Pocket002 Ø3.1 → Ø2.6) | M |
| `.ork` completion/update, final mass/CG, stability F35/G115, `rocket_simulator.py` retirement | I |
| English file renames (repo_structure group B), Q14/Q15 and the 2 small CAD overlaps of Q16 | "Housekeeping" list in the roadmap, done when convenient |

## Files (my calls)
- `PLAN.md` → rewritten as the roadmap: goal, stages, per-project status line, open questions, session log (new).
- Old content (Phases 0–4, Status, Session log 1–9) → `docs/archive/setup_plan.md`, unchanged.
- Each project, when it starts: `docs/projects/<id>_<name>/design.md` + `plan.md` (e.g. `S_structural/`).
- `CLAUDE.md` session protocol (step 4.4): read `PLAN.md` (roadmap) + the active project's docs; update the
  roadmap Status/log at the end. References to "PLAN.md Phase 3 / Qn" in `CLAUDE.md` point to the archive.

## Left to the builder
Wording of the roadmap; keeping cross-references to old Qn numbers valid (archive keeps the numbering).
