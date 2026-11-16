# `components/wall.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Pad×4, Plane×1, Pocket×2, PolarPattern×2, SketchObject×7
- Spreadsheets: 0 · Expressions: 0 · Sketches: 7
- Driving/reference dimensions in sketches: 9 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch005`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=95; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch001:Edge1, Pad:Vertex4; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad001` | Pad001 | Pad | Profile=Sketch002; Type=Length; Length=1; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad001; Axis=Z_Axis; Angle=360; Occurrences=16; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch003; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad003` | Pad003 | Pad | Profile=Sketch004; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch005; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch006; Type=TwoLengths; Length=2; Length2=2; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket001; Axis=Z_Axis; Angle=360; Occurrences=2; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=60, Ø=58 |
| Sketch001 | Pad:Face4 / FlatFace | 1 | 3 | yes | — |
| Sketch002 | DatumPlane / FlatFace | 4 | 10 | yes | Lx=1 |
| Sketch003 | PolarPattern:Face67 / FlatFace | 2 | 4 | yes | Ø=56 |
| Sketch004 | Pad002:Face2 / FlatFace | 2 | 4 | yes | Ø=56 |
| Sketch005 | Pad003:Face4 / FlatFace | 4 | 9 | **NO** | — |
| Sketch006 | DatumPlane / FlatFace | 8 | 20 | yes | Lx=15.2, Lx=0.2, Lx=0.2, Lx=15.2 |
