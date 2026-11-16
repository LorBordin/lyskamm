# `components/parachute_fuselage_seat/parachute_fuselage_seat.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Chamfer×3, Fillet×2, Pad×8, Plane×5, Pocket×6, PolarPattern×5, SketchObject×16
- Spreadsheets: 0 · Expressions: 0 · Sketches: 16
- Driving/reference dimensions in sketches: 30 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch001`
- ⚠ not fully constrained `Sketch006`
- ⚠ not fully constrained `Sketch007`
- ⚠ not fully constrained `Sketch008`
- ⚠ not fully constrained `Sketch009`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern004

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | fusoliera | Pad | Profile=Sketch; Type=Length; Length=235; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | sede_tondi | Pad | Profile=Sketch001; Type=Length; Length=209; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pad002` | tubo_interno_paracadute | Pad | Profile=Sketch002; Type=Length; Length=242.5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad003` | accopiamento_inf | Pad | Profile=Sketch003; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad004` | chiusura_tondi_1 | Pad | Profile=Sketch004; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad005` | chiusura_tondi_2 | Pad | Profile=Sketch005; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pad005['Edge147']; Radius=1; UseAllEdges=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pad004, Pad005, Fillet; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pad006` | Pad006 | Pad | Profile=Sketch006; Type=Length; Length=26; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Pad006:Face17; MapMode=FlatFace; AttachmentOffset=pos('0', '0', '-9') |
| `Pad007` | Pad007 | Pad | Profile=Sketch007; Type=Length; Length=16; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pad007; Axis=Sketch007['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer` | Chamfer | Chamfer | Base=PolarPattern002['Edge182', 'Edge187', 'Edge192', 'Edge177']; ChamferType=EqualDistance; Size=8; Size2=1; Angle=45 |
| `Pocket` | sede_presa_elettrica | Pocket | Profile=Sketch008; Type=Length; Length=12.6; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet001` | Fillet001 | Fillet | Base=Pocket['Edge262', 'Edge267', 'Edge265', 'Edge263']; Radius=3; UseAllEdges=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch009; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch010:Edge1, Sketch010:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch010:Edge2, Sketch010:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket002` | sede_viti_inf_1 | Pocket | Profile=Sketch011; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket003` | sede_viti_inf_2 | Pocket | Profile=Sketch012; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Chamfer001` | Chamfer001 | Chamfer | Base=Pocket003['Edge190', 'Edge186', 'Edge185', 'Edge187', 'Edge189', 'Edge191']; ChamferType=EqualDistance; Size=1; Size2=1; Angle=45 |
| `PolarPattern003` | PolarPattern003 | PolarPattern | Originals=Pocket002, Pocket003, Chamfer001; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch013:Edge1, Sketch013:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane004` | DatumPlane004 | Plane | AttachmentSupport=Sketch013:Edge2, Sketch013:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket004` | sede_viti_sup_1 | Pocket | Profile=Sketch014; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch015; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Chamfer002` | Chamfer002 | Chamfer | Base=Pocket005['Edge37', 'Edge38', 'Edge39', 'Edge40', 'Edge42', 'Edge41']; ChamferType=EqualDistance; Size=1; Size2=1; Angle=45 |
| `PolarPattern004` | PolarPattern004 | PolarPattern | Originals=Pocket004, Pocket005, Chamfer002; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=60, Ø=58 |
| Sketch001 | Pad:Face4 / FlatFace | 12 | 24 | **NO** | L=23.4279, Ang=45, Ø=5.2, Ø=7 |
| Sketch002 | PolarPattern:Face23 / FlatFace | 2 | 4 | yes | — |
| Sketch003 | Pad002:Face4 / FlatFace | 2 | 4 | yes | Ø=60, Ø=56 |
| Sketch004 | Pad003:Face10 / FlatFace | 1 | 2 | yes | — |
| Sketch005 | Pad004:Face41 / FlatFace | 1 | 2 | yes | Ø=2.9 |
| Sketch006 | PolarPattern001:Face55 / FlatFace | 2 | 4 | **NO** | Ø=56 |
| Sketch007 | DatumPlane / FlatFace | 6 | 10 | **NO** | — |
| Sketch008 | Chamfer:Face58 / FlatFace | 8 | 17 | **NO** | Ly=6.15, Lx=9.6, Ly=0.5 |
| Sketch009 | Fillet001:Face16 / FlatFace | 3 | 6 | **NO** | Lx=2.4, Ø=2, Ø=2 |
| Sketch010 | Pocket001:Face53 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch011 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.1, Ly=3.75 |
| Sketch012 | DatumPlane002 / FlatFace | 1 | 3 | yes | Ø=2.1, Ly=3.75 |
| Sketch013 | Pad006:Face64 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch014 | DatumPlane003 / FlatFace | 1 | 3 | yes | Ø=2.1, L=3.75 |
| sede_viti_sup_2 | DatumPlane004 / FlatFace | 1 | 3 | yes | Ø=2.1, L=3.75 |
