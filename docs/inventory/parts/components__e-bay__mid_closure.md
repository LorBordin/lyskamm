# `components/e-bay/mid_closure.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×4, Plane×3, Pocket×6, PolarPattern×3, SketchObject×12
- Spreadsheets: 0 · Expressions: 0 · Sketches: 12
- Driving/reference dimensions in sketches: 22 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch004`
- ⚠ not fully constrained `Sketch005`
- ⚠ not fully constrained `Sketch006`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern002

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | external_ring | Pad | Profile=Sketch; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | lower_coupling | Pad | Profile=Sketch001; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | upper_coupling | Pad | Profile=Sketch002; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch003:Edge1, Pad002:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad003` | Pad003 | Pad | Profile=Sketch004; Type=Length; Length=4; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad003; Axis=Z_Axis; Angle=360; Occurrences=2; Reversed=false |
| `Pocket` | Pocket | Pocket | Profile=Sketch005; Type=Length; Length=20; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | sede_tondo_alluminio_inf | Pocket | Profile=Sketch006; Type=Length; Length=12.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket003` | sede_tondo_alluminio_sup | Pocket | Profile=Sketch008; Type=Length; Length=7.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket001, Pocket003; Axis=Sketch006['N_Axis']; Angle=360; Occurrences=2; Reversed=false |
| `Pocket004` | Pocket004 | Pocket | Profile=Sketch009; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket004['Edge93', 'Edge92', 'Edge91', 'Edge90']; Radius=3; UseAllEdges=false |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch010:Edge1, Sketch010:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch010:Edge2, Sketch010:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket005` | sede_viti_1 | Pocket | Profile=Sketch011; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket006` | sede_viti_2 | Pocket | Profile=Sketch012; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pocket005, Pocket006; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 1 | 2 | yes | Ø=60 |
| Sketch001 | Pad:Face2 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch002 | Pad001:Face3 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch003 | Pad002:Face3 / FlatFace | 1 | 3 | yes | — |
| Sketch004 | DatumPlane / FlatFace | 6 | 13 | **NO** | Lx=7.5, Lx=7.5 |
| Sketch005 | PolarPattern:Face30 / FlatFace | 2 | 3 | **NO** | — |
| Sketch006 | Pocket:Face19 / FlatFace | 10 | 27 | **NO** | Ø=5.75, L=23.4279 |
| Sketch008 | — / FlatFace | 1 | 2 | yes | Ø=5.25 |
| Sketch009 | PolarPattern001:Face22 / FlatFace | 4 | 12 | yes | Ø=47.8, Ø=7.25, Ø=7.25 |
| Sketch010 | Fillet:Face1 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch011 | DatumPlane001 / FlatFace | 2 | 6 | yes | Ø=2.6, Ø=2.6, Ly=3.75, Ly=3.75 |
| Sketch012 | DatumPlane002 / FlatFace | 2 | 6 | yes | Ø=2.6, Ø=2.6, L=3.75, Ly=3.75 |
