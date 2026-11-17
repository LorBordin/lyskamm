# Master parameters — proposal (Phase 2.1)

Status: **draft for review** (2026-11-17). Nothing in CAD changes until Phase 3, one approved step at a time.
Sources: `docs/inventory/shared_dims.md`, `geometry.md`, `openrocket.md`, `parts/*.md`, Lorenzo's answers in step 1.7.

## Mechanism (decided: option a)

- One file **`parameters.FCStd`** containing one spreadsheet, `Params`. Its location in the repo is decided in Phase 2.3; the working assumption is the repo root.
- Parts read values through cross-document expressions, e.g. `parameters#Params.body_od`.
  - FreeCAD loads `parameters.FCStd` automatically when a part that references it is opened. The relative path is stored, so the file must not be moved without updating the links.
  - `parameters.FCStd` itself is never edited by hand outside an approved change. Changing a value there changes every part, so it follows the CAD change protocol in `CLAUDE.md`.
- **What goes in the master:** only values shared by ≥ 2 parts or by CAD and OpenRocket (interfaces, fits, fasteners, outer shape). Dimensions used by a single part stay in that part, as **named sketch constraints**; naming rules come in Phase 2.2.
- **Derived values are formulas**, not numbers (e.g. `body_id = body_od - 2*body_wall`), so one change propagates correctly.
- **Phase 3 must not change geometry.** Where v1 parts use inconsistent values for the same purpose (e.g. 5.15 / 5.2 / 5.25 tube holes), each value keeps its own alias or stays a literal in the part, marked *deviation*. They get unified in Phase 5, not silently in Phase 3.
- Units: mm. Alias style: `snake_case`, `_d` = diameter, `_r` = radius, `_len` = axial length, `_t` = thickness.

## 1. Body tube

| Alias | Value | Formula | Used in | `.ork` | Notes |
|---|---|---|---|---|---|
| `body_od` | 60 | — | all body sections, closures, nose, transition fore Ø | Body Tube Ø 60 ✔ | |
| `body_wall` | 2 | — | fuselages | thickness 2 ✔ | |
| `body_id` | 56 | `body_od - 2*body_wall` | fuselages, nose, parachute seat, pistone | inner Ø 56 ✔ | |
| `fit_slide_d` | 0.2 | — | | — | Diametral sliding-fit clearance found in v1 (56 ↔ 55.8, 37 ↔ 37.2). ⚠ `CLAUDE.md` says ~0.15: which one is the rule? |
| `slide_od` | 55.8 | `body_id - fit_slide_d` | `lower_closure_v2`, `mid_closure`, `upper_closure`, `low_chute_ret`, `rear_transition` (fore shoulder), `upper_retainer` | — | Parts that slide into the tube. |
| `joint_len` | 7.5 | — | `mid_closure`, `upper_closure`, `low_chute_ret`, `mid_fuselage`, `lower_fuselage_v1`, `upper_retainer`, `nose_cone` | — | Overlap length of a joint (coupling / reinforcement ring). |

## 2. Fasteners

| Alias | Value | Formula | Used in | Notes |
|---|---|---|---|---|
| `m2_clear_d` | 2.2 | — | fuselages (`sede_viti_*`), `parachute_fuselage_seat_v2` | M2 screw clearance, outer part. |
| `m2_insert_d` | 2.6 | — | closures (`sede_viti_*`, `lower_closure_v2` Pocket007/008), `rear_transition` (`sede_vite_fusoliera_*`) | M2 heat-set insert hole. |
| `m2_insert_depth` | 5 | — | same as above | Pocket depth of the insert holes (inferred from v1). |
| `screw_axial` | 3.75 | — | 12 parts, every radial M2 joint | Distance of the radial screw holes from the part's end face. |
| `m3_clear_d` | 3.5 | — | `lower_closure_v2` `sede_vite_1`, `low_chute_ret` Pocket003, `pistone` Pocket001 | M3 clearance. *Deviation:* `upper_closure` Pocket002 = 3.2 (same function, other end of the aluminium bar). |
| `m3_head_d` | 7 | — | `lower_closure_v2` `sede_vite_2` (depth 3) | M3 head counterbore. |
| — | 3.1 | literal | `low_chute_ret` Pocket004–007, `upper_retainer` `sede_viti_*` | **v1 mistake** (oversized insert holes) → stays literal in Phase 3, fixed to `m2_insert_d` in Phase 5. |
| — | 3.1 | literal | `rear_transition` Pocket002 (through) | Screw for the lower motor retainer. ❓ Which screw size? |

## 3. Carbon tubes and aluminium bar

| Alias | Value | Formula | Used in | Notes |
|---|---|---|---|---|
| `rod_od` | 5 | — | `carbon_rod`, `alluminium_rod` | Both are tubes 5/3. |
| `rod_id` | 3 | — | same | |
| `rod_r` | 23.4279 | — | 11 parts | Radius from the rocket axis to the tube centres. Stored as 23.428 in some sketches — use the exact value. |
| `rod_hole_d` | 5.2 | `rod_od + 0.2` | `low_chute_ret`, `lower_fuselage_v1`, `parachute_fuselage_seat_v2`, `rear_transition` | *Deviations:* 5.15 (`mid_fuselage`, `upper_retainer`), 5.25 aluminium seats in closures (`sede_tondo_alluminio`). |
| `rod_boss_d` | 7 | — | `rear_transition`, `mid_fuselage`, `upper_retainer` | Boss around the tube. *Deviation:* 7.2 in `parachute_fuselage_seat_v2`, `sede_presa_elettrica_sup`. |
| `carbon_rod_len` | 450 | — | `carbon_rod` | Part-local unless an interface depends on it. |
| `alu_bar_len` | 100 | — | `alluminium_rod` | E-bay height between closures? ❓ |

## 4. Motor mount

| Alias | Value | Formula | Used in | `.ork` | Notes |
|---|---|---|---|---|---|
| `motor_d` | 35 | — | — | missing | EXP G115 (`EXP_G115.eng`). Current design target. |
| `motor_tube_id` | 35.2 | `motor_d + 0.2` | `lower_fuselage_v1` (`tubo_motore`), `rear_transition` | missing | |
| `motor_tube_od` | 37 | — | same | missing | Wall 0.9. |
| `motor_tube_fit_d` | 37.2 | `motor_tube_od + fit_slide_d` | `upper_retainer` | — | |
| `motor_d_f35` | 29 | — | — | — | TSP F35 (`TSP_F35.eng`): needs an adapter (Phase 5). Listed so the adapter can reference it. |

## 5. Outer shape (shared with OpenRocket)

Values that must agree between CAD and `.ork`. Mismatches are only listed here; Phase 2.4 decides which side is right.

| Alias | CAD | `.ork` | Notes |
|---|---|---|---|
| `nose_len` | 120 | 125 | ⚠ mismatch |
| `nose_shape_k` | 0.4 (power) | 0.4 (power) ✔ | |
| `fuselage_len` (e-bay, `fuselage`) | 105 | 105 ✔ | `fuselage_w_logo` too. |
| `mid_fuselage_len` | 105 | 105 ✔ | |
| `parachute_seat_len` | 242.5 | 241 | ⚠ small mismatch |
| `lower_fuselage_len` | 113 | 113 ✔ | |
| `closure_len` | — | 5 (×5 tubes) | `.ork` models closures as 5 mm tubes. |
| `transition_len` | 45 | 45 ✔ | |
| `transition_aft_d` | 40 | 42 | ⚠ mismatch |
| `transition_k` | 0.5 (power) | 0.5 ✔ | |
| `transition_t` | 3 | 2 | ⚠ mismatch |
| `fin_count` | 3 | 4 | ⚠ mismatch (both fin sets) |
| `fin_t` | 3 | 3 ✔ | Root and tip. |
| `lower_fin_root` / `tip` / `sweep` / `height` | 63.5 / 32 / 21 / 50 | 60 / 30 / 20 / 50 | ⚠ mismatch |
| `upper_fin_root` / `tip` / `sweep` / `height` | 95 / 41 / 50 / 50 | 90 / 40 / 50 / 50 (freeform) | ⚠ mismatch |
| total length | 871 (measured) | 864 | |

Note: nose, fins and transition are Rocket-workbench features. Linking their properties to `Params` only works once the Rocket workbench is installed (open question 8).

## 6. Not in the master (for now)

- Electrical connector slot `sede_presa_elettrica` (6.15 × 9.6, rounded) and `presa_elettrica`: part of the deployment system, which will be redesigned. Kept local.
- Logo geometry: local to `components/logo/`.
- Launch pad: deferred (own section in `PLAN.md`).

## Questions for review (2.5)

1. Sliding fit: v1 uses 0.2 on Ø everywhere. Should the rule in `CLAUDE.md` (~0.15) be changed to 0.2, or is 0.15 the target for new parts?
2. Screw size for the lower motor retainer (`rear_transition` Pocket002, Ø3.1)?
3. `alluminium_rod` length 100: is it set by the e-bay height (distance between closures)? If so it becomes a formula.
4. Anything missing from the list? (e.g. number of tubes / radial screws per joint, parachute bay length)
