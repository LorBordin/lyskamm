# `components/e-bay/fuselage.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Body×1, Chamfer×1, Pad×1, Plane×3, Pocket×3, PolarPattern×1, SketchObject×5
- Spreadsheets: 0 · Expressions: 0 · Sketches: 5
- Driving/reference dimensions in sketches: 17 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Chamfer

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=105; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Pad:Edge3, Pad:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket` | Pocket | Pocket | Profile=Sketch001; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch002:Edge2, Sketch002:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Sketch002:Edge1, Sketch002:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket001` | sede_viti_1 | Pocket | Profile=Sketch003; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket002` | sede_viti_2 | Pocket | Profile=Sketch004; Type=Length; Length=3; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket001, Pocket002; Axis=Z_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Chamfer` | Chamfer | Chamfer | Base=PolarPattern['Edge36', 'Edge37', 'Edge35', 'Edge33', 'Edge34', 'Edge32', 'Edge30', 'Edge31', 'Edge29', 'Edge24', 'Edge25', 'Edge23', 'Edge22', 'Edge21', 'Edge20', 'Edge19', 'Edge18', 'Edge17', 'Edge15', 'Edge16', 'Edge14', 'Edge27', 'Edge26', 'Edge28']; ChamferType=EqualDistance; Size=1.2; Size2=1; Angle=45 |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 4 | 8 | **NO** | Ø=60, Ø=56 |
| Sketch001 | DatumPlane / FlatFace | 6 | 16 | yes | Lx=7.6, Ly=7.6, Lx=7.6, Ly=7.6 |
| Sketch002 | Pocket:Face6 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch003 | DatumPlane001 / FlatFace | 2 | 6 | yes | Ø=2.2, Ø=2.2, Ly=3.75, Ly=3.75 |
| Sketch004 | DatumPlane002 / FlatFace | 2 | 6 | yes | Ø=2.2, Ø=2.2, Ly=3.75, Ly=3.75 |
