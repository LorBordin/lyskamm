# `components/lower_fuselage_w_motor_mount/lower_fuselage_v1.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Chamfer×4, Fillet×1, Pad×10, Plane×5, Pocket×9, PolarPattern×4, SketchObject×21
- Spreadsheets: 0 · Expressions: 0 · Sketches: 21
- Driving/reference dimensions in sketches: 38 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch002`
- ⚠ not fully constrained `Sketch009`
- ⚠ not fully constrained `Sketch012`
- ⚠ not fully constrained `Sketch013`
- ⚠ not fully constrained `Sketch014`
- ⚠ not fully constrained `Sketch015`
- ⚠ not fully constrained `Sketch019`
- ⚠ not fully constrained `Sketch020`
- ⚠ not fully constrained `Sketch021`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Chamfer004

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | fusoliera | Pad | Profile=Sketch; Type=Length; Length=93; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | rinforzo_sup | Pad | Profile=Sketch001; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad002` | centraggio_motore | Pad | Profile=Sketch002; Type=Length; Length=75.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | sede_tondo | Pocket | Profile=Sketch003; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad002, Pocket; Axis=Sketch002['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad003` | tubo_motore | Pad | Profile=Sketch004; Type=Length; Length=93; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch005:Edge1, Sketch005:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch005:Edge2, Sketch005:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket001` | sede_viti_sup_1 | Pocket | Profile=Sketch006; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket002` | sede_viti_sup_2 | Pocket | Profile=Sketch007; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pad004` | fin_can_1 | Pad | Profile=Sketch008; Type=Length; Length=45; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pocket003` | fin_can_2 | Pocket | Profile=Sketch009; Type=Length; Length=45; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pad006` | fusoliera_inf | Pad | Profile=Sketch011; Type=Length; Length=20; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket004` | fin_can_5 | Pocket | Profile=Sketch012; Type=Length; Length=20; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad007` | incastro_rear_transition_1 | Pad | Profile=Sketch013; Type=Length; Length=7; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pad007; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch014; Type=Length; Length=20; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch015; Type=Length; Length=20; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket006['Edge228', 'Edge234', 'Edge232', 'Edge230', 'Edge223', 'Edge226', 'Edge92', 'Edge153', 'Edge208', 'Edge207', 'Edge238', 'Edge220', 'Edge217', 'Edge244', 'Edge248', 'Edge214', 'Edge254', 'Edge211']; Radius=1; UseAllEdges=false |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch016:Edge2, Sketch016:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane004` | DatumPlane004 | Plane | AttachmentSupport=Sketch016:Edge1, Sketch016:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad008` | rinforzo_inferiore | Pad | Profile=Sketch019; Type=Length; Length=25; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pocket007` | sede_viti_inf_1 | Pocket | Profile=Sketch017; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket008` | sede_viti_inf_2 | Pocket | Profile=Sketch018; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern003` | PolarPattern003 | PolarPattern | Originals=Pocket001, Pocket002, Pocket007, Pocket008; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer002` | Chamfer002 | Chamfer | Base=PolarPattern003['Edge290', 'Edge291', 'Edge289', 'Edge284', 'Edge282', 'Edge283', 'Edge298', 'Edge297', 'Edge296', 'Edge304', 'Edge305', 'Edge303', 'Edge311', 'Edge312', 'Edge310', 'Edge267', 'Edge266', 'Edge265', 'Edge269', 'Edge270', 'Edge268', 'Edge277', 'Edge276', 'Edge275', 'Edge248', 'Edge249', 'Edge247', 'Edge245', 'Edge246', 'Edge264', 'Edge262', 'Edge263', 'Edge261', 'Edge259', 'Edge260', 'Edge258', 'Edge256', 'Edge257', 'Edge255', 'Edge253', 'Edge254', 'Edge252', 'Edge250', 'Edge251']; ChamferType=EqualDistance; Size=1.2; Size2=1; Angle=45 |
| `PolarPattern004` | PolarPattern004 | PolarPattern | Originals=Pad004, Pocket003, Pocket004; Axis=Sketch008['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer003` | Chamfer003 | Chamfer | Base=PolarPattern004['Edge240', 'Edge235', 'Edge231', 'Edge244', 'Edge249', 'Edge253', 'Edge258', 'Edge265']; ChamferType=EqualDistance; Size=0.95; Size2=1; Angle=45 |
| `Pad009` | rinforzo_fusoliera_1 | Pad | Profile=Sketch020; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad010` | rinforzo_fusoliera_2 | Pad | Profile=Sketch021; Type=Length; Length=41; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Chamfer` | Chamfer | Chamfer | Base=Pad010['Edge162', 'Edge154', 'Edge173', 'Edge174', 'Edge188', 'Edge196', 'Edge211', 'Edge225', 'Edge233', 'Edge249', 'Edge263', 'Edge271', 'Edge28']; ChamferType=EqualDistance; Size=1.95; Size2=1; Angle=45 |
| `Chamfer004` | Chamfer004 | Chamfer | Base=Chamfer['Edge580', 'Edge598', 'Edge616', 'Edge618', 'Edge600', 'Edge591', 'Edge573', 'Edge18']; ChamferType=EqualDistance; Size=0.95; Size2=1; Angle=45 |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=58, Ø=60 |
| Sketch001 | Pad:Face4 / FlatFace | 2 | 4 | yes | Ø=56 |
| Sketch002 | Pad001:Face4 / FlatFace | 4 | 10 | **NO** | Lx=7, L=11 |
| Sketch003 | — / FlatFace | 1 | 3 | yes | Ø=5.2, Ly=23.4279 |
| Sketch004 | PolarPattern:Face26 / FlatFace | 2 | 4 | yes | Ø=37, Ø=35.2 |
| disegno_di_costruzione | Pad003:Face20 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch006 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=2.2, L=3.75 |
| Sketch007 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch008 | Pocket002:Face4 / FlatFace | 5 | 14 | yes | L=9.1, L=4.55 |
| Sketch009 | Pad004:Face30 / FlatFace | 10 | 28 | **NO** | L=11.1, L=3.1, L=1.55, L=1.6, L=1.6, L=2.1, L=2.1 |
| Sketch011 | — / FlatFace | 2 | 4 | yes | — |
| Sketch012 | Pad006:Face16 / FlatFace | 4 | 10 | **NO** | L=3.1 |
| Sketch013 | DatumPlane002 / FlatFace | 9 | 26 | **NO** | Lx=4.55, Lx=13.37, Lx=4.55, Lx=13.37, Ly=5, Ly=0 |
| Sketch014 | PolarPattern002:Face2 / FlatFace | 2 | 2 | **NO** | Ø=54 |
| Sketch015 | Pocket005:Face2 / FlatFace | 2 | 3 | **NO** | — |
| disegno_di_costruzione_2 | — / FlatFace | 2 | 4 | yes | — |
| Sketch017 | DatumPlane003 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch018 | DatumPlane004 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch019 | — / FlatFace | 16 | 41 | **NO** | Ø=56 |
| Sketch020 | Chamfer003:Face7 / FlatFace | 16 | 37 | **NO** | — |
| Sketch021 | Pad009:Face8 / FlatFace | 16 | 33 | **NO** | — |
