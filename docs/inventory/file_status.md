# File status (active / legacy)

> **2026-11-17 — group A of `docs/proposal/repo_structure.md` done:** legacy files below were **deleted** (recover with `git log --all -- <path>`); logo files renamed. Paths in the table are the *old* ones.

Reviewed with Lorenzo in step 1.7 (2026-11-17). *Active* = part of the flying rocket (or needed to check a fit).
Only files whose status is not obvious from the assembly are listed. All 16 parts imported in
`assemblies/lyskamm.FCStd` are active.

## Not in the assembly

| File | Status | Notes |
|---|---|---|
| `components/e-bay/fuselage_w_logo.FCStd` | **active** (flies) | 4th fuselage, ASA-CF. Matches the printed `fuselage_w_logo.3mf` (volume 17545.3 vs 17545.2 mm³ in the mesh; bbox 60×105×30). Missing from the assembly. |
| `components/e-bay/fuselage_w_logo_v2.FCStd` | **check** | Imported STEP only (no feature tree): ASA-CF fuselage + PLA logo inserts together, used to check that the inserts fit their slots. |
| `components/e-bay/lower_closure.FCStd` | **deleted** (A3) | Replaced by `lower_closure_v2`. |
| `components/e-bay/presa_elettrica.FCStd` | **active** | PLA insert (electrical connector) for `lower_closure_v2`. |
| `components/parachute_fuselage_seat/parachute_fuselage_seat.FCStd` | **deleted** (A4) | Replaced by `_v2`. |
| `components/parachute_fuselage_seat/pistone.FCStd` | **active** (flies) | Missing from the assembly. |
| `components/parachute_fuselage_seat/upper_closure.FCStd` | **deleted** (A5) | Old widget used in `parachute_fuselage_seat` (v1). |
| `components/logo.FCStd` → `components/logo/logo_flat.FCStd` | **active** (reference) | Flat logo. Moved (A10). |
| `components/logo/logo.FCStd` → `logo_projected.FCStd` | **active** | Logo projected onto the curved fuselage surface (full size: main body 9.0×60.0). |
| `components/logo/logo_red.FCStd` → `logo_inserts.FCStd` | **active** | The **printed PLA inserts**: same logo, ≈0.5 mm smaller (main body 8.5×59.0). Its shapes match the `logo_*.step` objects in `fuselage_w_logo.3mf` exactly (450.4 / 111.0 / 106.0 / 112.3 mm³). |
| `components/wall.FCStd` | **deleted** (A2) | Test of a fuselage with thicker vertical ribs; never used. |
| `launch_pad/Unnamed.FCStd` | deferred | Early launch-pad concept. Ignored for now; has its own section in `PLAN.md`. |

## Print files (`.3mf`) with misleading names

| File | Actually contains |
|---|---|
| `components/parachute_fuselage_seat/parachute_fuselage_seat.3mf` | `parachute_fuselage_seat_v2` (mesh 133079.6 mm³ vs 133082.9 in the v2 `.FCStd`) — **this is the printed v2** → renamed to `parachute_fuselage_seat_v2.3mf` (A6) |
| `components/parachute_fuselage_seat/parachute_fuselage_seat_v2.3mf` | `lower_closure_v2.step` (60×60×40) — old duplicate → **deleted** (A7) |

## Other deleted files (group A)

- `components/lower_chute_retainer/low_chute_ret_test.3mf` — old test print of the v1 parachute seat (A8).
- `components/upper_motor_retainer/upper_retainer` (no extension) — older STEP export (A9).
- `components/.keep` (A13).
