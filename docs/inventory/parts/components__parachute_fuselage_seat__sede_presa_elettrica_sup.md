# `components/parachute_fuselage_seat/sede_presa_elettrica_sup.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×3, Pad×1, Pocket×5, SketchObject×6
- Spreadsheets: 0 · Expressions: 1 · Sketches: 6
- Driving/reference dimensions in sketches: 30 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch001`
- ⚠ not fully constrained `Sketch005`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet002

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=20; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch002; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet001` | Fillet001 | Fillet | Base=Pocket001['Edge19', 'Edge5']; Radius=2; UseAllEdges=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch003; Type=Length; Length=14; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket002` | Pocket002 | Pocket | Profile=Sketch001; Type=Length; Length=1; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket002['Edge40', 'Edge38', 'Edge35', 'Edge36']; Radius=3; UseAllEdges=false |
| `Pocket003` | Pocket003 | Pocket | Profile=Sketch004; Type=Length; Length=1; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket004` | Pocket004 | Pocket | Profile=Sketch005; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet002` | Fillet002 | Fillet | Base=Pocket004['Edge50', 'Edge49', 'Edge44', 'Edge42']; Radius=1; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 10 | 26 | **NO** | Ø=56, Ø=42.1, L=23.4279, L=23.4279, Ø=7.2, Ø=7.2 |
| Sketch001 | Pad:Face8 / FlatFace | 11 | 29 | **NO** | Ø=41.66, Ø=56, L=20, Ang=-45, L=20, Ang=-45, L=20, Ang=-45, L=0.5, L=20, Ang=-45, L=9.6, Ang=-45, L=9.6, Ang=-45, L=6.15 |
| Sketch002 | — / FlatFace | 4 | 11 | yes | Ø=2, Ø=2, Ang=45, L=1.8, L=1.8 |
| Sketch003 | Fillet001:Face14 / FlatFace | 14 | 34 | yes | L=2.8 |
| Sketch004 | Fillet:Face31 / FlatFace | 8 | 16 | yes | — |
| Sketch005 | Pocket003:Face1 / FlatFace | 18 | 43 | **NO** | L=3, L=3 |

## Expressions

- Sketch.Constraints[2] = `42.1`
