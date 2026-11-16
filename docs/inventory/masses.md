# Masses and print parameters (step 1.6)

**How to fill in:** edit this file directly, or send me the values in chat and I'll fill them in.
- *Mass per piece*: weigh each printed v1 piece on its own, in grams, without screws or inserts. If identical pieces differ, write each value (e.g. `21.3 / 21.6 / 21.1 / 21.4`).
- *Print params (from 3MF)*: settings stored in the Bambu Studio `.3mf` files of the repo. **Several are ambiguous** (same part in more than one 3MF with different settings). Put ✅ in *Confirmed* if a candidate is what you printed, otherwise write what you really used.
- Leave `?` where you don't know. `Volume` is the CAD solid volume, for reference only.

Row order is only for convenience; it is not a verified stacking order.

**Confirmed by Lorenzo (2026-11-16):** all printed parts ASA-CF with 15% infill, except fins (PETG-CF); `upper_closure`/`mid_closure` 5 top layers; printed seat is `parachute_fuselage_seat_v2`; per-piece values for multi-piece parts are averages. Masses are **as weighed, including the hardware listed in Notes** (inserts, screw eyes, connectors). Small/inner parts (e.g. `pistone`, `fuselage_w_logo`) fly but are not in the A2plus assembly. No volume-based mass estimate: for new parts use Bambu Studio's slicer estimate.

## Printed and bought parts (active = in the assembly)

| # | Part | Qty in rocket | Volume (cm³) | Print params candidates from 3MF (filament · layer · walls · infill · pattern · top/bottom) | Confirmed / actual params | Mass per piece (g) | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `nose_cone` | 1 | 193.1 | ASA-CF · 0.16 · 5w · 0% · grid · 6/4 (`nose_cone.3mf`) | ok | 20 | |
| 2 | `upper_closure` | 1 | 14.9 | ⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`e-bay/upper_closure.3mf`)<br>(PLA · crosszag in `parachute_fuselage_seat/upper_closure.3mf` is a *different* part with the same name — ignore)<br>⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 3/3 (`upper_retainer.3mf`) | ✅ ASA-CF · 15% · 5 top layers | 8 | mass includes 16 heat-set inserts |
| 3 | `fuselage` | 3 | 18.8 | ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`fuselage.3mf`) | ok | 14 | 3 of the 4 e-bay fuselage segments |
| 3bis | `fuselage_w_logo` | 1 | 17.5 | ASA-CF · 0.2 · 2w · 15% · grid · 5/3 + logo inserts PLA (`fuselage_w_logo.3mf`) | ok | 14 (weighed together with the other 3) | 4th fuselage segment; not in the A2plus assembly |
| 4 | `mid_closure` | 1 | 15.3 | ⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`e-bay/upper_closure.3mf`)<br>⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 3/3 (`upper_retainer.3mf`) | ✅ ASA-CF · 15% · 5 top layers | 10.33 (avg) | includes threaded rod to screw the alluminium rods and 16 heat-set inserts |
| 5 | `lower_closure_v2` | 1 | 96.1 | ⚠ ASA-CF · 0.2 · 3w · 10% · gyroid · 5/3 (`lower_closure.3mf`)<br>⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`parachute_fuselage_seat_v2.3mf`) | ✅ ASA-CF · 15% (walls not confirmed) | 28 | mass includes screw eye, elettrical connector for deployment system and 8 heat-set inserts |
| 6 | `alluminium_rod` | 4 | 1.3 | bought part (not printed) | ok | 3.67 (avg) | |
| 7 | `parachute_fuselage_seat_v2` | 1 | 133.1 | ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`parachute_fuselage_seat.3mf`) | ok | 103 | mass includes `sede_presa_elettrica_sup.3mf` and lower electrical connector (bought piece)|
| 7bis | `pistone` | 1 | 28.4 | ASA-CF · 0.2 · 6w · 25% · gyroid · 5/3 (`pistone.3mf`) | ok | 28 | mass includes screw eye; not in the A2plus assembly |
| 8 | `sede_presa_elettrica_sup` | 1 | 3.8 | PLA · 0.2 · 2w · 15% · grid · 5/3 (`sede_presa_elettrica_sup.3mf`) | ASA-CF · 15% per Lorenzo (3MF says PLA) | 3.1 | Estimate from bambustudio |
| 9 | `low_chute_ret` | 1 | 37.2 | ⚠ PLA · 0.2 · 2w · 15% · crosszag · 5/3 (`low_chute_ret.3mf`)<br>⚠ ASA-CF · 0.2 · 2w · 15% · gyroid · 3/3 (`upper_retainer.3mf`) | ✅ ASA-CF · 15% | 8 | mass includes screw eye and 16 heat-set inserts |
| 10 | `mid_fuselage` | 1 | 67.6 | ASA-CF · 0.2 · 2w · 15% · crosszag · 5/3 (`mid_fuselage.3mf`) | ok | 53 | |
| 11 | `upper_fin` | 4 | 9.8 | PETG-CF · 0.2 · 6w · 25% · crosszag · 5/3 (`upper_fin.3mf`) | partial | 11.75 (avg of 4) | 2 of them are printed with with 100% infill (all layers); difference is small since almost all layers are walls/shells |
| 12 | `upper_retainer` | 1 | 22.4 | ASA-CF · 0.2 · 2w · 15% · gyroid · 3/3 (`upper_retainer.3mf`) | ok | 13 | mass includes 16 heat-set inserts |
| 13 | `lower_fuselage_v1` | 1 | 63.9 | ASA-CF · 0.2 · 2w · 15% · gyroid · 5/3 (`lower_fuselage.3mf`) | ok | 53.5 | |
| 14 | `lower_fin` | 4 | 6.8 | PETG-CF · 0.2 · 6w · 25% · gyroid · 5/3 (`lower_fin.3mf`) | ok | 8.25 (avg) | |
| 15 | `rear_transition` | 1 | 41.1 | ASA-CF · 0.2 · 2w · 15% · grid · 5/3 (`rear_transition.3mf`) | ok | 29 | mass includes 8 heat-set inserts |
| 16 | `carbon_rod` | 4 | 5.7 | bought part (not printed) | ok | 7 | |

## Hardware and other items (not in CAD)

| Item | Spec (size/length/model) | Qty | Mass per piece or total (g) | Notes |
|---|---|---|---|---|
| M2 screws | M2 × 4,6,8 | 80 | n.d. | |
| M2 heat-set inserts | M2 × 2.5, 3, 4 × 3.2 | 80 | n.d. | |
| Other screws/nuts | n.d. | n.d. | n.d. | |
| Parachute | ? | 1 | 43 | |
| Shock cord + attachments | ? | ? | ? | |
| Electronics (altimeter / flight computer) | ? | ? | ? | |
| Battery | ? | ? | ? | |
| Wiring / connectors / switch | ? | ? | ? | |
| Ejection / deployment system | ? | ? | ? | |
| Wadding / heat protection | ? | ? | ? | |
| Other | ? | ? | ? | |

## Whole-rocket checks (optional but very useful)

| Measurement | Value | Notes |
|---|---|---|
| Assembled rocket, ready to fly, **without motor** (g) | ? | `rocket_simulator.py` currently assumes 910 g |
| CG position without motor, from nose tip (mm) — balance point | ? | |
| Total length assembled (mm) | 871 | best estimate, measurement error likely; `.ork` says 864 |

## Unused / legacy prints (only if you have them and want them recorded)

Files not in the assembly: ~~`fuselage_w_logo`~~ (flies), `fuselage_w_logo_v2`, `lower_closure` (v1), `presa_elettrica`, `parachute_fuselage_seat` (v1), ~~`pistone`~~ (flies), `parachute_fuselage_seat/upper_closure`, `logo*`, `wall`, `launch_pad/Unnamed`. Is any of them actually on the flying rocket (e.g. `presa_elettrica`, `pistone`, logo inserts)? If yes, add it to the parts table above.

## Computed totals (provisional — generated from the tables above)

| Item | Qty | Mass/piece (g) | Subtotal (g) |
|---|---|---|---|
| nose_cone | 1 | 20 | 20.00 |
| upper_closure | 1 | 8 | 8.00 |
| fuselage | 4 | 14 | 56.00 |
| mid_closure | 1 | 10.33 | 10.33 |
| lower_closure_v2 | 1 | 28 | 28.00 |
| alluminium_rod | 4 | 3.67 | 14.68 |
| parachute_fuselage_seat (+ sede_presa_elettrica_sup + connector) | 1 | 103 | 103.00 |
| pistone | 1 | 28 | 28.00 |
| low_chute_ret | 1 | 8 | 8.00 |
| mid_fuselage | 1 | 53 | 53.00 |
| upper_fin | 4 | 11.75 | 47.00 |
| upper_retainer | 1 | 13 | 13.00 |
| lower_fuselage_v1 | 1 | 53.5 | 53.50 |
| lower_fin | 4 | 8.25 (avg) | 33.00 |
| rear_transition | 1 | 29 | 29.00 |
| carbon_rod | 4 | 7 | 28.00 |
| parachute | 1 | 43 | 43.00 |
| **Known total** | | | **575.5** |

- `sede_presa_elettrica_sup` (3.1 g) is **not** added: already included in `parachute_fuselage_seat` (see its note).
- All 80 M2 heat-set inserts are already included in part masses (16+16+8+16+16+8 = 80).
- **Not included** (n.d. or not implemented yet): 80 M2 screws, other screws/nuts, shock cord, electronics, battery, wiring, ejection system, wadding.
- `rocket_simulator.py` assumes 910 g empty → 334 g would be left for the items above.
