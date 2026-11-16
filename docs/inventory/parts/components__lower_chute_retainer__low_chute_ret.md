# `components/lower_chute_retainer/low_chute_ret.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Fillet×1, Pad×3, Plane×4, Pocket×8, PolarPattern×2, SketchObject×13
- Spreadsheets: 0 · Expressions: 0 · Sketches: 13
- Driving/reference dimensions in sketches: 34 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch005`
- ⚠ not fully constrained `Sketch006`

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern001

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | Pad001 | Pad | Profile=Sketch001; Type=Length; Length=5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch002; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | tasca_superiore | Pocket | Profile=Sketch003; Type=Length; Length=7.5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | sede_tondi | Pocket | Profile=Sketch004; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket002` | sede_presa_elettrica | Pocket | Profile=Sketch005; Type=Length; Length=12.6; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket003` | Pocket003 | Pocket | Profile=Sketch006; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=Pocket003['Edge72', 'Edge71', 'Edge73', 'Edge74']; Radius=3; UseAllEdges=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch007:Edge2, Sketch007:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch007:Edge1, Sketch007:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket004` | Pocket004 | Pocket | Profile=Sketch008; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch009; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket004, Pocket005; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch010:Edge2, Sketch010:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch010:Edge1, Sketch010:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch011; Type=Length; Length=7; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket007` | Pocket007 | Pocket | Profile=Sketch012; Type=Length; Length=7; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket007, Pocket006; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch001 | Pad:Face3 / FlatFace | 1 | 2 | yes | Ø=60 |
| Sketch002 | Pad001:Face5 / FlatFace | 1 | 2 | yes | Ø=55.8 |
| Sketch003 | Pad002:Face7 / FlatFace | 1 | 3 | yes | Ø=41.9 |
| Sketch004 | Pocket:Face7 / FlatFace | 6 | 15 | yes | L=23.4279, Ang=45, Ø=5.2, Ø=5.2, Ø=5.2, Ø=5.2 |
| Sketch005 | Pocket001:Face9 / FlatFace | 9 | 21 | **NO** | L=6.15, L=9.6, Lx=0.5 |
| Sketch006 | — / FlatFace | 14 | 25 | **NO** | Ø=2, Ø=2, Ø=3.5, Ly=5, Ø=2, Ø=2, Ly=2.4 |
| Sketch007 | Fillet:Face22 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch008 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch009 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch010 | PolarPattern:Face1 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch011 | DatumPlane002 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
| Sketch012 | DatumPlane003 / FlatFace | 1 | 3 | yes | Ø=3.1, Ly=3.75 |
