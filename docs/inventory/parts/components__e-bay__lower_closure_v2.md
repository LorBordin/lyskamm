# `components/e-bay/lower_closure_v2.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×4, Plane×3, Pocket×10, PolarPattern×3, SketchObject×17
- Spreadsheets: 0 · Expressions: 0 · Sketches: 17
- Driving/reference dimensions in sketches: 28 (named: 0)

## Flags

- ⚠ empty sketch `Sketch001`
- ⚠ not fully constrained `Sketch008`
- ⚠ not fully constrained `Sketch009`
- ⚠ not fully constrained `Sketch011`
- ⚠ not fully constrained `Sketch012`
- ⚠ not fully constrained `Sketch014`
- ⚠ not fully constrained `Sketch015`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Pocket009

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | pistone_inf | Pad | Profile=Sketch002; Type=Length; Length=25; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | chiusura_sup | Pad | Profile=Sketch003; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | sede_vite_1 | Pocket | Profile=Sketch004; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | sede_vite_2 | Pocket | Profile=Sketch005; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket002` | sede_tondo_alluminio | Pocket | Profile=Sketch006; Type=Length; Length=5.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket, Pocket001, Pocket002; Axis=Sketch004['N_Axis']; Angle=360; Occurrences=2; Reversed=false |
| `Pocket003` | sede_anelllo | Pocket | Profile=Sketch007; Type=Length; Length=23; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket004` | sede_presa_elettrica | Pocket | Profile=Sketch008; Type=Length; Length=14.6; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket004['Edge24', 'Edge22', 'Edge20', 'Edge19']; Radius=3; UseAllEdges=false |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch009; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch010:Edge1, Sketch010:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad003` | Pad003 | Pad | Profile=Sketch011; Type=UpToLast; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pad003; Axis=Z_Axis; Angle=360; Occurrences=2; Reversed=false |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch012; Type=Length; Length=10; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch013:Edge1, Sketch013:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch013:Edge2, Sketch013:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket007` | Pocket007 | Pocket | Profile=Sketch014; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket008` | Pocket008 | Pocket | Profile=Sketch015; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pocket007, Pocket008; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pocket009` | Pocket009 | Pocket | Profile=Sketch016; Type=ThroughAll; Length=8; Length2=5; Offset=0; Midplane=false; Reversed=true |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 1 | 2 | yes | Ø=60 |
| Sketch001 | Pad:Face3 / FlatFace | 0 | 0 | yes | — |
| Sketch002 | Pad:Face2 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch003 | Pad001:Face3 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch004 | Pad002:Face2 / FlatFace | 1 | 3 | yes | Ø=3.5, Ly=23.4279 |
| Sketch005 | — / FlatFace | 1 | 2 | yes | Ø=7 |
| Sketch006 | Pocket001:Face7 / FlatFace | 1 | 2 | yes | Ø=5.25 |
| Sketch007 | PolarPattern:Face6 / FlatFace | 1 | 2 | yes | Ø=3.9 |
| Sketch008 | Pocket003:Face2 / FlatFace | 10 | 22 | **NO** | Ø=41.9, Ø=55.8, Lx=0.5, Lx=6.15, Ly=9.6 |
| Sketch009 | Fillet:Face12 / FlatFace | 3 | 6 | **NO** | Ø=2, Ø=2, Ly=2.4 |
| Sketch010 | Pocket005:Face18 / FlatFace | 1 | 3 | yes | Ang=0 |
| Sketch011 | DatumPlane / FlatFace | 4 | 10 | **NO** | Ly=7.5, Lx=7.5 |
| Sketch012 | PolarPattern001:Face18 / FlatFace | 2 | 2 | **NO** | — |
| Sketch013 | Pocket006:Face18 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch014 | DatumPlane002 / FlatFace | 2 | 5 | **NO** | Ly=7.5, Ø=2.6, Ly=3.75 |
| Sketch015 | DatumPlane001 / FlatFace | 2 | 5 | **NO** | Ly=7.5, Ø=2.6, Ly=3.75 |
| Sketch016 | PolarPattern002:Face13 / FlatFace | 2 | 4 | yes | — |
