# `components/e-bay/fuselage_w_logo.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Mirroring×3, Body×8, Boolean×1, Chamfer×1, FeatureBase×3, Pad×5, Plane×7, Pocket×11, PolarPattern×1, SketchObject×18
- Spreadsheets: 0 · Expressions: 0 · Sketches: 18
- Driving/reference dimensions in sketches: 61 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch010`
- ⚠ not fully constrained `Sketch012`
- ⚠ not fully constrained `Sketch014`
- ⚠ not fully constrained `Sketch016`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Boolean
- **main_body** (`Body001`, Body) Tip=Pocket004
- **lower_right_wing** (`Body002`, Body) Tip=Pocket006
- **mid_right_wing** (`Body003`, Body) Tip=Pocket008
- **upper_right_wing** (`Body004`, Body) Tip=Pocket010
- **Body005** (`Body005`, Body) Tip=Clone
- **Body006** (`Body006`, Body) Tip=Clone001
- **Body007** (`Body007`, Body) Tip=Clone002

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
| `Clone` | Clone | FeatureBase | BaseFeature=Body002 |
| `Clone001` | Clone001 | FeatureBase | BaseFeature=Body003 |
| `Clone002` | Clone002 | FeatureBase | BaseFeature=Body004 |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch009:Edge1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane004` | DatumPlane004 | Plane | AttachmentSupport=Pad002:Edge3, Pad002:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane005` | DatumPlane005 | Plane | AttachmentSupport=Pad003:Edge3, Pad003:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane006` | DatumPlane006 | Plane | AttachmentSupport=Pad004:Edge3, Pad004:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad001` | Pad001 | Pad | Profile=Sketch005; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch006; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad003` | Pad003 | Pad | Profile=Sketch007; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad004` | Pad004 | Pad | Profile=Sketch008; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Part__Mirroring` | lower_left_wing | Mirroring | Source=Body005; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Part__Mirroring001` | mid_left_wing | Mirroring | Source=Body006; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Part__Mirroring002` | upper_left_wing | Mirroring | Source=Body007; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Pocket003` | Pocket003 | Pocket | Profile=Sketch010; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket004` | Pocket004 | Pocket | Profile=Sketch011; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch012; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch013; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket007` | Pocket007 | Pocket | Profile=Sketch014; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket008` | Pocket008 | Pocket | Profile=Sketch015; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket009` | Pocket009 | Pocket | Profile=Sketch016; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket010` | Pocket010 | Pocket | Profile=Sketch017; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Boolean` | Boolean | Boolean | Group=Body001, Body002, Body003, Body004, Part__Mirroring, Body005, Body005, Part__Mirroring, Part__Mirroring001, Body006, Body006, Part__Mirroring001, Part__Mirroring002, Body007, Body007, Part__Mirroring002; Type=Cut |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 4 | 8 | **NO** | Ø=60, Ø=56 |
| Sketch001 | DatumPlane / FlatFace | 6 | 16 | yes | Lx=7.6, Ly=7.6, Lx=7.6, Ly=7.6 |
| Sketch002 | Pocket:Face6 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch003 | DatumPlane001 / FlatFace | 2 | 6 | yes | Ø=2.2, Ø=2.2, Ly=3.75, Ly=3.75 |
| Sketch004 | DatumPlane002 / FlatFace | 2 | 6 | yes | Ø=2.2, Ø=2.2, Ly=3.75, Ly=3.75 |
| Sketch005 | XY_Plane001 / FlatFace | 11 | 24 | yes | Lx=9, Ly=40, Ly=16, Ly=4, Lx=4 |
| Sketch006 | XY_Plane002 / FlatFace | 4 | 12 | yes | Lx=6.5, Lx=8.5, Ly=12.5, L=16, Ly=10, Ly=20 |
| Sketch007 | XY_Plane003 / FlatFace | 4 | 12 | yes | Lx=17, Ly=7.5, Lx=6.5, Ly=1, Ly=34.5 |
| Sketch008 | XY_Plane004 / FlatFace | 7 | 22 | yes | Ly=6.33, Ly=9.495, Lx=6.5, Lx=17, Ly=18.5, Lx=6.5, Lx=23.5, Ly=8.5, Ly=2, Lx=15.5 |
| Sketch009 | Pad001:Face8 / FlatFace | 1 | 2 | yes | — |
| Sketch010 | DatumPlane003 / FlatFace | 2 | 4 | **NO** | Ø=60 |
| Sketch011 | DatumPlane003 / FlatFace | 2 | 5 | yes | Ø=59, Ly=1 |
| Sketch012 | DatumPlane004 / FlatFace | 2 | 4 | **NO** | Ø=60, Lx=6.5 |
| Sketch013 | DatumPlane004 / FlatFace | 3 | 8 | yes | Ø=59, Lx=6.5, Ly=1 |
| Sketch014 | DatumPlane005 / FlatFace | 2 | 4 | **NO** | Ø=60, Lx=6.5 |
| Sketch015 | DatumPlane005 / FlatFace | 2 | 6 | yes | Ø=59, Lx=6.5, Ly=1 |
| Sketch016 | DatumPlane006 / FlatFace | 2 | 4 | **NO** | Ø=60, Lx=6.5 |
| Sketch017 | DatumPlane006 / FlatFace | 2 | 6 | yes | Ø=59, Lx=6.5, Ly=1 |
