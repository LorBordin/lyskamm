# `components/parachute_fuselage_seat/pistone.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×1, Pocket×2, SketchObject×3
- Spreadsheets: 0 · Expressions: 0 · Sketches: 3
- Driving/reference dimensions in sketches: 14 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch001`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Pocket001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=40; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch001; Type=Length; Length=35; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket['Edge52', 'Edge51', 'Edge64', 'Edge62', 'Edge60', 'Edge54']; Radius=2; UseAllEdges=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch002; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 25 | 60 | **NO** | Ø=55.5, L=23.4279, Ang=45, Ø=7.4, Ø=7.4, Ø=7.4, Ø=7.4, Ø=56 |
| Sketch001 | Pad:Face15 / FlatFace | 9 | 24 | **NO** | Ø=13.4, Ø=13.4, Ø=13.4, Ø=13.4, Ø=49.5 |
| Sketch002 | Fillet:Face31 / FlatFace | 1 | 2 | yes | Ø=3.5 |
