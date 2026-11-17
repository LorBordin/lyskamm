# OpenRocket vs CAD — mismatch table (Phase 2.4)

Status: **draft for review** (2026-11-17). Sources: `docs/inventory/openrocket.md` (`lyskamm.ork`), part files,
`assemblies/lyskamm.FCStd` (positions), `docs/inventory/masses.md` (weighed). Nothing is changed here; the
`.ork` fixes chosen in 2.5 are applied later as separate steps.

**Proposed rule (open question 6): CAD is the source of truth for geometry** (it is what was printed), weighed
masses are the source of truth for mass, and the `.ork` is updated to match both. Later, `.ork` values that
should follow CAD come from `parameters.FCStd`.

## 1. Outer geometry

| # | Item | CAD (as printed) | `.ork` | Proposed fix in `.ork` |
|---|---|---|---|---|
| G1 | Nose length | 120 (+ 7.5 shoulder inside the upper closure) | 125 | → 120 |
| G2 | Nose shape | power series, k = 0.4, Ø60 | same ✔ | — |
| G3 | Nose construction | printed shell, 5 walls, 0 % infill | hollow, thickness 2, polystyrene | keep shell; mass override (§3) |
| G4 | Coupler rings (visible) | 5 × 5 mm: upper / mid / lower closure, `low_chute_ret`, `upper_retainer` | 5 × 5 mm tubes ✔ | — |
| G5 | E-bay tubes | 2 × 105 (each = 2 half-shells `fuselage`) | 105 + 105 ✔ | — |
| G6 | Parachute bay | 242.5 | 241 | → 242.5 (still open: may change with the deployment redesign) |
| G7 | Mid fuselage | 105 | 105 ✔ | — |
| G8 | Lower fuselage | 113 | 113 ✔ | — |
| G9 | Transition length / shape | 45, power k = 0.5 | 45, k = 0.5 ✔ | — |
| G10 | Transition aft Ø | 40 | 42 | → 40 |
| G11 | Transition wall | 3 | 2 | → 3 |
| G12 | Body wall | 2 | 2 ✔ | — |
| G13 | Total length | nominal stack 860.5 · assembly 875.5 · **measured 871** | 864 | see §4 |

## 2. Fins

| # | Item | CAD | `.ork` | Proposed fix |
|---|---|---|---|---|
| F1 | Count | 4 + 4 (assembly has `lower_fins_001…004`, `upper_fin_001…004`) | 4 + 4 ✔ | — (the CAD `FinCount = 3` is an unused property, `FinSet = false`) |
| F2 | Lower fins: root / tip / sweep / height | 63.5 / 32 / 21 / 50 | 60 / 30 / 20 / 50 | → CAD values |
| F3 | Lower fins: section | airfoil, 3 mm | **square**, 3 mm | → airfoil |
| F4 | Lower fins: axial position | trailing edge ≈ 19 mm **aft** of the lower-fuselage end (fins reach onto the transition) | trailing edge 5 mm **forward** of the tube end | → match CAD. ⚠ Big effect on CP / stability. To verify on the real rocket. |
| F5 | Upper fins: root / tip / sweep / height | 95 / 41 / 50 / 50 (trapezoid) | 90 / 40 / 50 / 50 (freeform) | → CAD values (trapezoid) |
| F6 | Upper fins: position | on the mid fuselage, trailing edge ≈ 2.5 forward of its aft end | trailing edge 7.5 forward of the tube end | → match CAD (5 mm difference) |
| F7 | Fin material | PETG-CF, 6 walls, 25 % infill | balsa | mass override (§3); material only affects mass |

## 3. Mass

The `.ork` has **no mass components or overrides**: its mass comes from cardboard/balsa/polystyrene densities and
is meaningless. Proposal: **override the mass of each component with the weighed value** (`masses.md`).

| # | Item | Weighed (g) | Notes |
|---|---|---|---|
| M1 | Nose cone | 20 | |
| M2 | E-bay: 4 fuselage halves + 3 closures + 4 aluminium bars | 56 + 8 + 10.3 + 28 + 14.7 = 117 | `lower_closure_v2` 28 |
| M3 | Parachute bay (+ connector seat + connector) | 103 | |
| M4 | Piston (with screw eye) | 28 | not in the assembly |
| M5 | `low_chute_ret` | 8 | |
| M6 | Mid fuselage | 53 | |
| M7 | Upper fins | 4 × 11.75 = 47 | |
| M8 | `upper_retainer` | 13 | |
| M9 | Lower fuselage | 53.5 | |
| M10 | Lower fins | 4 × 8.25 = 33 | |
| M11 | Rear transition | 29 | |
| M12 | Carbon rods | 4 × 7 = 28 | |
| M13 | Parachute | 43 | |
| — | **Known total** | ≈ 575.5 (without screws, electronics, cord, battery) | `rocket_simulator.py` assumes 910 g |
| M14 | Missing | screws/inserts (80 + 80), electronics, battery, shock cord, deployment system | needed for a real CG |

Best single check: weigh the ready-to-fly rocket without motor and find its balance point (CG) — then the `.ork`
can be calibrated with one override on the whole rocket if the per-part values do not add up.

## 4. Length / positions

The numbers do not agree: nominal section sum **860.5**, assembly **875.5**, measured **871**, `.ork` **864**.
The assembly is not reliable for positions: its outer surfaces overlap (the parachute bay overlaps the mid
fuselage by ≈ 15 mm; the two halves of the aft e-bay tube are offset by 0.5 mm). Proposal: measure the
visible length of each section on the real rocket (nose → tail), use those for the `.ork`, and fix the assembly
in Phase 3.

## 5. Missing from the `.ork`

| # | Item | Data available | Proposal |
|---|---|---|---|
| X1 | Motor mount tube | Ø35.2 / Ø37, length 93 (`tubo_motore` in `lower_fuselage_v1`) | add inner tube |
| X2 | Motors | `scripts/motor_data/TSP_F35.eng` (Ø29), `EXP_G115.eng` (Ø35) | import both `.eng`; F35 needs the adapter (Phase 5) |
| X3 | Recovery | parachute 43 g, size ? | add parachute (size/Cd from Lorenzo) |
| X4 | Mass components | electronics, battery, cord, screws | add when weighed |
| X5 | Rail buttons / launch lug | none on the rocket yet | add when designed (Phase 5) |
| X6 | Simulations | — | F35 and G115 flights; check stability margin ≥ 1 cal |
| X7 | Materials | — | not needed if every component has a mass override |

## Questions for review (2.5)

1. Source of truth: CAD for geometry, weighed masses for mass, `.ork` follows? (open question 6)
2. Lower fins: on the real rocket, do they stick out ~19 mm past the end of the lower fuselage (onto the transition)?
3. Can you measure each section's visible length on the real rocket (nose, rings, tubes, transition)?
4. Parachute: diameter / model?
5. Apply all `.ork` fixes above in one go (after 2.5), or one by one?
