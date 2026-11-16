# `components/e-bay/presa_elettrica.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×1, Pocket×2, SketchObject×4
- Spreadsheets: 0 · Expressions: 0 · Sketches: 4
- Driving/reference dimensions in sketches: 6 (named: 0)

## Flags

- ⚠ empty sketch `Sketch002`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Pocket001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=14.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pad['Edge2', 'Edge5', 'Edge8', 'Edge1']; Radius=3; UseAllEdges=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch001; Type=Length; Length=0.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch003; Type=Length; Length=14; Length2=5; Offset=0; Midplane=false; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 4 | 10 | yes | L=9.5, Ly=6.05 |
| Sketch001 | Fillet:Face2 / FlatFace | 2 | 5 | yes | Ø=2, Ø=2, Lx=1.8 |
| Sketch002 | Pocket:Face5 / FlatFace | 0 | 0 | yes | — |
| Sketch003 | Pocket:Face14 / FlatFace | 14 | 34 | yes | Lx=2.8 |
