# `components/upper_motor_retainer/upper_retainer.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×5, Plane×5, Pocket×5, PolarPattern×5, SketchObject×14
- Spreadsheets: 0 · Expressions: 0 · Sketches: 14
- Driving/reference dimensions in sketches: 30 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch009`
- ⚠ empty sketch `Sketch010`
- ⚠ empty sketch `Sketch011`
- ⚠ not fully constrained `Sketch013`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | blocco_motore | Pad | Profile=Sketch; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | congiunzione_inf | Pad | Profile=Sketch001; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch002:Edge2, Sketch002:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch002:Edge1, Sketch002:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket` | sede_viti_inferiori_1 | Pocket | Profile=Sketch003; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket001` | sede_viti_inferiori_2 | Pocket | Profile=Sketch004; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket, Pocket001; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pad002` | congiunzione_sup | Pad | Profile=Sketch005; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch006:Edge1, Sketch006:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch006:Edge2, Sketch006:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket002` | sede_viti_sup_1 | Pocket | Profile=Sketch007; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket003` | sede_viti_sup_2 | Pocket | Profile=Sketch008; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket003, Pocket002; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane004` | DatumPlane004 | Plane | AttachmentSupport=PolarPattern001:Face57; MapMode=FlatFace; AttachmentOffset=pos('0', '0', '5') |
| `Pad003` | sede_tondo_1 | Pad | Profile=Sketch009; Type=Length; Length=15; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pocket004` | sede_tondo_2 | Pocket | Profile=Sketch012; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pad003; Axis=Sketch009['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `PolarPattern003` | PolarPattern003 | PolarPattern | Originals=Pocket004; Axis=Sketch012['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad004` | Pad004 | Pad | Profile=Sketch013; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern004` | PolarPattern004 | PolarPattern | Originals=Pad004; Axis=Sketch013['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=PolarPattern004['Edge27', 'Edge12', 'Edge25', 'Edge42', 'Edge40', 'Edge57', 'Edge5', 'Edge46']; Radius=1; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=60, Ø=15 |
| Sketch001 | Pad:Face3 / FlatFace | 18 | 48 | yes | Ø=55.8, Ø=37.2, Ø=50, Lx=3, Ly=3, Lx=3, Ly=3 |
| Sketch002 | Pad001:Face6 / FlatFace | 2 | 6 | yes | Ang=45, Ang=22.5 |
| Sketch003 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=3.1, L=3.75 |
| Sketch004 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch005 | PolarPattern:Face3 / FlatFace | 18 | 48 | yes | Ø=55.8, Ø=50, L=3, L=3, Lx=3, L=3 |
| Sketch006 | Pad002:Face15 / FlatFace | 2 | 4 | yes | — |
| Sketch007 | DatumPlane003 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch008 | DatumPlane002 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch009 | DatumPlane004 / FlatFace | 9 | 22 | **NO** | Ang=45, Ø=7, L=23.4279 |
| Sketch010 | — / FlatFace | 0 | 0 | yes | — |
| Sketch011 | Pad003:Face62 / FlatFace | 0 | 0 | yes | — |
| Sketch012 | Pad003:Face15 / FlatFace | 1 | 2 | yes | Ø=5.15 |
| Sketch013 | PolarPattern003:Face15 / FlatFace | 4 | 11 | **NO** | Ly=3 |
