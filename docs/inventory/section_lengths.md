# Section lengths — measurement sheet

Measure the **assembled rocket**, nose → tail. For each section, measure only the **visible outer length** (the part
you can see from outside). Couplings hidden inside the neighbouring part don't count. Write the result in
*Measured*. Leave `?` if you can't measure it.

*Guess* = nominal value from the CAD part (CAD is the reference for lengths). *`.ork` now* = value currently in
`lyskamm.ork`.

| # | Section (nose → tail) | Part file | Guess (mm) | `.ork` now (mm) | Measured (mm) | Notes |
|---|---|---|---|---|---|---|
| 1 | Nose cone (tip → base) | `nose_cone` | 120 | 120 | 125 | |
| 2 | Upper closure ring | `e-bay/upper_closure` | 5 | 5 | 5 | |
| 3 | E-bay tube, upper (2 half-shells) | `e-bay/fuselage` ×2 | 105 | 105 | 104 | |
| 4 | Mid closure ring | `e-bay/mid_closure` | 5 | 5 | 5 | |
| 5 | E-bay tube, lower (2 half-shells, one with logo) | `e-bay/fuselage` + `fuselage_w_logo` | 105 | 105 | 104 | |
| 6 | Lower closure ring | `e-bay/lower_closure_v2` | 5 | 5 | 5 | |
| 7 | Parachute bay | `parachute_fuselage_seat_v2` | 242.5 | 241 | 241 | visible length; CAD 242.5 is the full part |
| 8 | Lower chute retainer ring | `low_chute_ret` | 5 | 5 | 5 | |
| 9 | Mid fuselage (upper fins) | `mid_fuselage` | 105 | 105 |104.5 | |
| 10 | Upper motor retainer ring | `upper_retainer` | 5 | 5 | 5 | |
| 11 | Lower fuselage (lower fins) | `lower_fuselage_v1` | 113 | 113 | 112 | |
| 12 | Rear transition | `rear_transition` | 45 | 45 | 44.5 | |
| — | **Sum of the above** | | **860.5** | **860** (updated) | **860** | |
| — | **Total length, tip → tail, one measure** | | | | | earlier estimate 871 |

## Fin positions

| # | Measure | Guess (mm) | `.ork` now (mm) | Measured (mm) |
|---|---|---|---|---|
| F1 | Upper fins: trailing edge (aft end of fin root) → aft end of mid fuselage | 7.5 | 7.5 | 7.5 |
| F2 | Upper fins: root length along the body | 90 | 90 | 90 |
| F3 | Lower fins: trailing edge → aft end of lower fuselage | 5 | 5 | 5 |
| F4 | Lower fins: root length along the body | 60 | 60 | 60 |

## Other

| # | Measure | Guess | Measured |
|---|---|---|---|
| O1 | Parachute spill-hole diameter (mm) | ? | ≈ 65 |
| O2 | Motor (G115 or F35 + adapter): how far the motor's aft end is from the aft end of the rear transition (mm; + = sticks out) | 0 | −3 (3 mm inside the transition) |
| O3 | Ready-to-fly mass without motor (g) | ≈ 575 + screws, electronics, cord | |
| O4 | Balance point (CG) without motor, from nose tip (mm) | ≈ 518 (from `.ork`, incomplete masses) | |
