# F35 flight check (OpenRocket 24.12)

Date: 2026-11-17. Model: `lyskamm.ork` after the 2.4 fixes and the measured lengths (860 mm). Motor:
`scripts/motor_data/TSP_F35.eng` (67.5 Ns, 118 g). Script: `scripts/openrocket/Sim.java` (headless OpenRocket 24.12,
default launch conditions, 1 m launch rod).

The real ready-to-fly mass is not known yet: the model has only the weighed parts (575.5 g without motor, no
electronics, battery, screws, cord, F35 adapter). So the flights are repeated with an **extra mass in the e-bay**
(0 / 100 / 200 / 335 g; 335 g = the 910 g that `rocket_simulator.py` assumes).

| Extra in e-bay (g) | Lift-off mass (g) | Stability at rod exit (cal) | Rod-exit speed (m/s) | Apogee (m) | Max speed (m/s) | Best delay (s) | Descent (m/s) |
|---|---|---|---|---|---|---|---|
| 0 | 694 | **0.13 ⚠** | 11.6 | 316 | 77 | 6.5 | 4.4 |
| 100 | 794 | **0.7 ⚠** | 10.6 | 254 | 65 | 5.9 | 4.7 |
| 200 | 894 | 1.1–1.3 | 10.0 | 204 | 56 | 5.2 | 4.9 |
| 335 | 1029 | 1.7 | 9.0 | 152 | 46 | 4.4 | 5.3 |

Speed at parachute deployment for the F35 delays (4 / 6 / 8 s): the delay closest to *Best delay* gives 5–11 m/s; the
wrong one gives 17–35 m/s (OpenRocket warns "deployment at high speed").

## Findings

1. **Stability is the critical point.** With only the weighed parts, the CG with the motor is ≈ 563 mm from the tip
   and the CP ≈ 571 mm: margin ≈ 0.1 cal, **unstable in practice** (target ≥ 1–1.5 cal). It reaches ≥ 1 cal only if
   ≈ 200 g more mass sits in the e-bay (electronics, battery…). Everything that adds mass **at the tail** lowers it:
   the F35 adapter, a motor sitting further aft than modelled, screws in the lower parts.
   - Rule of thumb: for 1 cal, the CG **with the motor installed** must be ≤ ≈ 511 mm from the nose tip; for
     1.5 cal ≤ ≈ 481 mm.
2. **The G115 is not automatically safe if the F35 is.** It weighs 284 g (vs 118 g) and is longer, so it moves the
   CG further aft → less stability. It must be checked on its own.
3. Rod-exit speed 9–12 m/s with a 1 m rod is low (common guideline ≥ 13–15 m/s). A longer rod/rail (1.5–2 m) helps;
   relevant for the launch-pad design.
4. Thrust-to-weight: F35 average ≈ 35 N vs 6.8–10 N weight → 3.5–5 : 1. Acceptable but on the low side when heavy.
5. Descent rate 4.4–5.4 m/s under the 1 m chute: fine. The 65 mm spill hole removes < 0.5 % of the canopy area,
   so its effect on drag is negligible (Cd kept at 0.8).
6. Delay: choose by the real mass — ≈ 6 s if the rocket is ≈ 0.7–0.8 kg, ≈ 4 s if ≈ 1 kg.

## What decides it — measurements needed

- **Ready-to-fly mass and balance point (CG) with the F35 (+ adapter) installed** (`section_lengths.md` O3/O4). With
  these, the stability margin is known directly: CG ≤ ≈ 511 mm from the tip → OK.
- Motor position (O2): where the motor's aft end sits.
- Simple field check: the swing test (rocket swung on a string tied at the CG must fly nose-first).

## Model assumptions to keep in mind

Piston and carbon-rod positions are approximate; the motor is placed at the aft end of the lower fuselage with no
overhang; the F35 adapter is not modelled; the parachute bay length (241) and fin positions are not yet measured.
