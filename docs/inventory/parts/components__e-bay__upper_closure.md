# `components/e-bay/upper_closure.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×4, Plane×3, Pocket×7, PolarPattern×2, SketchObject×13
- Spreadsheets: 0 · Expressions: 0 · Sketches: 13
- Driving/reference dimensions in sketches: 23 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch004`
- ⚠ not fully constrained `Sketch005`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | Pad001 | Pad | Profile=Sketch001; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch002; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch003:Edge1, Pad002:Vertex1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad003` | Pad003 | Pad | Profile=Sketch004; Type=UpToLast; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Axis=Z_Axis; Angle=360; Occurrences=2; Reversed=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch005; Type=Length; Length=10; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch006; Type=Length; Length=2.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket002` | Pocket002 | Pocket | Profile=Sketch007; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket003` | Pocket003 | Pocket | Profile=Sketch008; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch009:Edge1, Sketch009:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch009:Edge2, Sketch009:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket004` | sede_viti_1 | Pocket | Profile=Sketch010; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket005` | sede_viti_2 | Pocket | Profile=Sketch011; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket004, Pocket005; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch012; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket006['Edge83', 'Edge74', 'Edge76', 'Edge81']; Radius=3; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 1 | 2 | yes | Ø=60 |
| Sketch001 | Pad:Face2 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch002 | Pad001:Face3 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch003 | Pad002:Face2 / FlatFace | 1 | 3 | yes | — |
| Sketch004 | DatumPlane / FlatFace | 3 | 6 | **NO** | L=7.5 |
| Sketch005 | PolarPattern:Face15 / FlatFace | 2 | 3 | **NO** | — |
| Sketch006 | Pocket:Face13 / FlatFace | 2 | 5 | yes | Ø=5.25, Ø=5.25, L=23.4279 |
| Sketch007 | Pocket001:Face15 / FlatFace | 2 | 4 | yes | Ø=3.2, Ø=3.2 |
| Sketch008 | Pocket002:Face8 / FlatFace | 2 | 4 | yes | Ø=7.25, Ø=7.25 |
| Sketch009 | Pocket003:Face8 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch010 | DatumPlane001 / FlatFace | 2 | 6 | yes | Ø=2.6, Ø=2.6, Ly=3.75, L=3.75 |
| Sketch011 | DatumPlane002 / FlatFace | 2 | 6 | yes | Ø=2.6, Ø=2.6, Ly=3.75, Ly=3.75 |
| Sketch012 | PolarPattern001:Face8 / FlatFace | 4 | 13 | yes | Ø=48.1 |
