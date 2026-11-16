# `components/parachute_fuselage_seat/parachute_fuselage_seat_v2.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Chamfer×2, Fillet×2, Pad×7, Plane×2, Pocket×4, PolarPattern×2, SketchObject×12
- Spreadsheets: 0 · Expressions: 0 · Sketches: 12
- Driving/reference dimensions in sketches: 24 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch001`
- ⚠ not fully constrained `Sketch004`
- ⚠ not fully constrained `Sketch005`
- ⚠ not fully constrained `Sketch006`
- ⚠ not fully constrained `Sketch007`
- ⚠ not fully constrained `Sketch008`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Chamfer001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | fusoliera | Pad | Profile=Sketch; Type=Length; Length=242.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | sede_tondi | Pad | Profile=Sketch001; Type=Length; Length=209; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad002` | sede_tondi_1 | Pad | Profile=Sketch002; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad003` | sede_tondi_2 | Pad | Profile=Sketch003; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pad003['Edge24']; Radius=1; UseAllEdges=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad001, Pad002, Pad003, Fillet; Axis=Sketch001['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad004` | sezione_cavi | Pad | Profile=Sketch004; Type=Length; Length=200; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pad005` | Pad005 | Pad | Profile=Sketch005; Type=Length; Length=16; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Chamfer` | Chamfer | Chamfer | Base=Pad005['Edge83']; ChamferType=EqualDistance; Size=7; Size2=1; Angle=45 |
| `Pad006` | Pad006 | Pad | Profile=Sketch006; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Pocket` | sede_presa_elettrica | Pocket | Profile=Sketch007; Type=Length; Length=12.6; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet001` | Fillet001 | Fillet | Base=Pocket['Edge124', 'Edge125', 'Edge129', 'Edge127']; Radius=3; UseAllEdges=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch008; Type=Length; Length=9; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch009:Edge1, Sketch009:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch009:Edge2, Sketch009:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket002` | sede_viti_inf_1 | Pocket | Profile=Sketch010; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket003` | sede_viti_inf_2 | Pocket | Profile=Sketch011; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket003, Pocket002; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer001` | Chamfer001 | Chamfer | Base=PolarPattern001['Edge125', 'Edge124', 'Edge126', 'Edge130', 'Edge129', 'Edge128', 'Edge133', 'Edge134', 'Edge132', 'Edge138', 'Edge137', 'Edge136', 'Edge141', 'Edge142', 'Edge140', 'Edge145', 'Edge146', 'Edge144', 'Edge149', 'Edge150', 'Edge148', 'Edge153', 'Edge154', 'Edge152']; ChamferType=EqualDistance; Size=1.2; Size2=1; Angle=45 |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 2 | 4 | yes | Ø=60, Ø=56 |
| Sketch001 | Pad:Face3 / FlatFace | 11 | 24 | **NO** | L=23.4279, Ang=45, Ø=5.2, Ø=7.2 |
| Sketch002 | Pad001:Face7 / FlatFace | 1 | 2 | yes | — |
| Sketch003 | Pad002:Face10 / FlatFace | 1 | 2 | yes | Ø=2.9 |
| Sketch004 | PolarPattern:Face13 / FlatFace | 4 | 10 | **NO** | Ø=41.82 |
| Sketch005 | Pad004:Face12 / FlatFace | 6 | 9 | **NO** | — |
| Sketch006 | Chamfer:Face6 / FlatFace | 10 | 22 | **NO** | Ly=2, L=2 |
| Sketch007 | Pad006:Face29 / FlatFace | 9 | 21 | **NO** | Ø=41.66, Ly=0.5, Ly=6.15, Lx=9.6, R=28(ref) |
| Sketch008 | Fillet001:Face16 / FlatFace | 3 | 6 | **NO** | Ø=2, Ø=2, L=2.4 |
| Sketch009 | Pocket001:Face22 / FlatFace | 3 | 9 | yes | Ang=45, Ang=22.5 |
| Sketch010 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
| Sketch011 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.2, Ly=3.75 |
