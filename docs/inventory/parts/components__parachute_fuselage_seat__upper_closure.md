# `components/parachute_fuselage_seat/upper_closure.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×2, Pad×2, Plane×2, Pocket×6, PolarPattern×1, SketchObject×9
- Spreadsheets: 0 · Expressions: 0 · Sketches: 9
- Driving/reference dimensions in sketches: 22 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch001`
- ⚠ not fully constrained `Sketch004`
- ⚠ not fully constrained `Sketch008`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Pocket005

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=16.4; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch001; Type=Length; Length=12.6; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket['Edge12', 'Edge16', 'Edge11', 'Edge14']; Radius=3; UseAllEdges=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch002; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | Pad001 | Pad | Profile=Sketch003; Type=Length; Length=9.6; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket002` | Pocket002 | Pocket | Profile=Sketch004; Type=Length; Length=13.4; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet001` | Fillet001 | Fillet | Base=Pocket002['Edge29', 'Edge30', 'Edge32', 'Edge42']; Radius=3; UseAllEdges=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch005:Edge1, Sketch005:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch005:Edge2, Sketch005:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket003` | sede_viti_1 | Pocket | Profile=Sketch007; Type=Length; Length=8; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket004` | sede_viti_2 | Pocket | Profile=Sketch006; Type=Length; Length=8; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket003, Pocket004; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch008; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=55.8, Ø=41.7 |
| Sketch001 | Pad:Face4 / FlatFace | 8 | 18 | **NO** | L=0.5, Ly=6.15, Lx=9.6 |
| Sketch002 | Fillet:Face12 / FlatFace | 3 | 7 | yes | Ø=2, Ø=2, Lx=2.4 |
| Sketch003 | Pocket001:Face1 / FlatFace | 2 | 4 | yes | — |
| Sketch004 | Pad001:Face16 / FlatFace | 8 | 18 | **NO** | Ly=0.5, Ly=6.6, Lx=10.1 |
| Sketch005 | Fillet001:Face20 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch006 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch007 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch008 | PolarPattern:Face20 / FlatFace | 17 | 41 | **NO** | Ly=10, Ly=10, L=1, Lx=1 |
