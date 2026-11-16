# `components/mid_fuselage/mid_fuselage.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Chamfer×2, Pad×7, Plane×4, Pocket×5, PolarPattern×9, SketchObject×14
- Spreadsheets: 0 · Expressions: 0 · Sketches: 14
- Driving/reference dimensions in sketches: 29 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch005`
- ⚠ not fully constrained `Sketch006`
- ⚠ not fully constrained `Sketch012`
- ⚠ not fully constrained `Sketch015`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern008

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | fusoliera | Pad | Profile=Sketch; Type=Length; Length=90; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | inserto_fins | Pad | Profile=Sketch002; Type=Length; Length=80; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad002` | rinforzo_inf | Pad | Profile=Sketch003; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad001; Axis=Sketch002['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pocket` | inserto_fins_2 | Pocket | Profile=Sketch006; Type=Length; Length=97.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket; Axis=Sketch006['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad003` | inserto_tondo | Pad | Profile=Sketch005; Type=Length; Length=90; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pad003; Axis=Sketch005['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad004` | rinforzo_sup | Pad | Profile=Sketch007; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch008:Edge2, Sketch008:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch008:Edge1, Sketch008:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket001` | sede_viti_inf_1 | Pocket | Profile=Sketch009; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket002` | sede_viti_inf_2 | Pocket | Profile=Sketch010; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern003` | PolarPattern003 | PolarPattern | Originals=Pocket001; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `PolarPattern004` | PolarPattern004 | PolarPattern | Originals=Pocket002; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch011:Edge1, Sketch011:Vertex1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch011:Edge2, Sketch011:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket003` | sede_viti_sup_1 | Pocket | Profile=Sketch012; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket004` | sede_viti_sup_2 | Pocket | Profile=Sketch013; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern005` | PolarPattern005 | PolarPattern | Originals=Pocket003; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `PolarPattern006` | PolarPattern006 | PolarPattern | Originals=Pocket004; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer` | Chamfer | Chamfer | Base=PolarPattern006['Edge98', 'Edge99', 'Edge93', 'Edge94', 'Edge95', 'Edge92', 'Edge96', 'Edge97', 'Edge193', 'Edge195', 'Edge194', 'Edge216', 'Edge215', 'Edge217', 'Edge220', 'Edge218', 'Edge219', 'Edge207', 'Edge208', 'Edge206', 'Edge211', 'Edge209', 'Edge210', 'Edge123', 'Edge121', 'Edge122', 'Edge125', 'Edge124', 'Edge126', 'Edge191', 'Edge190', 'Edge192']; ChamferType=EqualDistance; Size=1.2; Size2=1; Angle=45 |
| `Chamfer001` | Chamfer001 | Chamfer | Base=Chamfer['Edge198', 'Edge231', 'Edge165', 'Edge132', 'Edge230']; ChamferType=EqualDistance; Size=0.5; Size2=1; Angle=45 |
| `Pad005` | Pad005 | Pad | Profile=Sketch014; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern007` | PolarPattern007 | PolarPattern | Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pad006` | Pad006 | Pad | Profile=Sketch015; Type=Length; Length=12.5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern008` | PolarPattern008 | PolarPattern | Originals=Pad006; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=60, Ø=58 |
| Sketch002 | Pad:Face4 / FlatFace | 13 | 35 | yes | Ang=45, L=9.1, L=13, L=2.1, L=1.6, L=3.1, L=9 |
| Sketch003 | Pad001:Face15 / FlatFace | 2 | 4 | yes | Ø=56, Ø=60 |
| Sketch005 | PolarPattern:Face1 / FlatFace | 10 | 23 | **NO** | Ø=5.15, Ø=7, Ly=23.4279 |
| Sketch006 | PolarPattern:Face1 / FlatFace | 5 | 13 | **NO** | Ang=45 |
| Sketch007 | PolarPattern002:Face43 / FlatFace | 2 | 4 | yes | Ø=56 |
| Sketch008 | Pad004:Face79 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch009 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch010 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch011 | PolarPattern004:Face37 / FlatFace | 2 | 4 | yes | — |
| Sketch012 | DatumPlane002 / FlatFace | 1 | 2 | **NO** | Ø=2.2, Ly=3.75 |
| Sketch013 | DatumPlane003 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch014 | Chamfer001:Face149 / FlatFace | 4 | 11 | yes | Ø=60 |
| Sketch015 | PolarPattern007:Face59 / FlatFace | 5 | 14 | **NO** | Ø=60 |
