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

## Review answers (2026-11-17)

1. **Source of truth depends on the quantity.** Lengths and body shape: CAD (to be confirmed by the physical
   measurement, `docs/inventory/section_lengths.md`). **Fin dimensions: `.ork` is the design value.** The CAD fins
   are drawn larger on purpose, because Bambu Studio loses material when slicing the thin airfoil profile, so the
   printed fins come out close to the `.ork` size. F2 and F5 are therefore *not* mismatches.
2. **The 19 mm fin overhang (F4) came from the assembly, which is wrong.** On the real rocket the lower fins end
   5 mm before the end of the lower fuselage, as in the `.ork`. No change. Same for F6: `.ork` kept.
3. Lorenzo will measure the sections → `docs/inventory/section_lengths.md`.
4. Parachute: Ø 1 m with a spill hole (hole size to be measured).
5. Apply all `.ork` fixes in one go and report.

## Applied to `lyskamm.ork` (2026-11-17)

Checked by loading the file in OpenRocket 23.09 (headless): loads with only the version warnings the
original already had; dry mass without motor **575.5 g** (was 226 g), CG **518 mm** from the tip.

| Change | Before | After |
|---|---|---|
| Nose length (G1) | 125 | 120 |
| Transition aft Ø (G10) | 42 | 40 |
| Transition wall (G11) | 2 | 3 |
| Rear fins cross-section (F3) | square | airfoil |
| Component names | 10 × "Body Tube" | Upper closure ring, E-bay tube (upper), … Rear transition |
| Mass overrides (M1–M12) | none (cardboard/balsa densities) | weighed values on every component; aluminium bars added to the e-bay tubes (2 × 3.67 g each) |
| Parachute (X3) | — | Ø1000, 43 g, Cd auto (0.8), in the parachute bay, 20 mm from its top |
| Piston | — | mass 28 g at the aft end of the parachute bay ⚠ position guessed |
| Carbon rods | — | one 28 g mass, 450 long, centred ≈ 604 mm from the tip ⚠ position approximate |
| Motor tube (X1) | — | inner tube Ø37/35.2 × 93 at the aft end of the lower fuselage, motor mount, mass 0 (included in the weighed lower fuselage) |

**Not done yet:**
- Section lengths: waiting for the measurement (total is now 859 because only the nose changed).
- Parachute bay length 241 vs CAD 242.5: waiting for the measurement.
- Motors (X2): OpenRocket needs the `.eng` files in its motor library; then pick G115 in the GUI. The F35 needs the
  adapter first.
- Simulations (X6): after the motor is set.
- Spill hole: OpenRocket has no spill-hole field; it is approximated with the drag coefficient (Cd) once the hole size is known.
- Note: OpenRocket installed here is **23.09**, but the file was saved by **24.12**. It loads, with a warning. Saving
  from 23.09 would downgrade the file format: better to install 24.12.
