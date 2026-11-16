# `launch_pad/Unnamed.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×3, PolarPattern×2, SketchObject×3
- Spreadsheets: 0 · Expressions: 0 · Sketches: 3
- Driving/reference dimensions in sketches: 19 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch001`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | Pad001 | Pad | Profile=Sketch001; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad001; Axis=Sketch001['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch002; Type=Length; Length=20; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pad002['Edge40']; Radius=2; UseAllEdges=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pad002, Fillet; Axis=Sketch002['N_Axis']; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 14 | 35 | **NO** | Ø=60, L=50, L=50, Ø=165, Ø=205, Ø=10, Ø=10, Ø=10, Ø=10, Ang=45, L=2 |
| Sketch001 | Pad:Face3 / FlatFace | 14 | 33 | **NO** | Ø=60, Ang=45, Ø=10, Ø=10, Ø=10, Ø=10, L=2 |
| Sketch002 | PolarPattern:Face2 / FlatFace | 1 | 2 | yes | Ø=7.5 |
