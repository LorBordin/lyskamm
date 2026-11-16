# `components/logo/logo_red.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: Mirroring×3, Body×7, FeatureBase×3, Pad×4, Plane×4, Pocket×8, SketchObject×13
- Spreadsheets: 0 · Expressions: 0 · Sketches: 13
- Driving/reference dimensions in sketches: 66 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch001`
- ⚠ not fully constrained `Sketch002`
- ⚠ not fully constrained `Sketch003`
- ⚠ not fully constrained `Sketch005`
- ⚠ not fully constrained `Sketch009`
- ⚠ not fully constrained `Sketch011`
- ⚠ not fully constrained `Sketch013`

## Bodies / parts

- **main_body** (`Body`, Body) Tip=Pocket001
- **lower_right_wing** (`Body001`, Body) Tip=Pocket009
- **mid_right_wing** (`Body002`, Body) Tip=Pocket005
- **upper_right_wing** (`Body003`, Body) Tip=Pocket007
- **Body004** (`Body004`, Body) Tip=Clone
- **Body005** (`Body005`, Body) Tip=Clone001
- **Body006** (`Body006`, Body) Tip=Clone002

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad001` | Pad001 | Pad | Profile=Sketch001; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad002` | Pad002 | Pad | Profile=Sketch002; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pad003` | Pad003 | Pad | Profile=Sketch003; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch004:Edge1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket` | Pocket | Pocket | Profile=Sketch005; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch006; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=Pad002:Edge11, Pad002:Vertex1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket004` | Pocket004 | Pocket | Profile=Sketch009; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch010; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Pad003:Edge3, Pad003:Vertex1; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Clone` | Clone | FeatureBase | BaseFeature=Body001 |
| `Part__Mirroring` | lower_left_wing | Mirroring | Source=Body004; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Clone001` | Clone001 | FeatureBase | BaseFeature=Body002 |
| `Part__Mirroring001` | mid_left_wing | Mirroring | Source=Body005; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Clone002` | Clone002 | FeatureBase | BaseFeature=Body003 |
| `Part__Mirroring002` | upper_left_wing | Mirroring | Source=Body006; Normal=('1', '0', '0'); Base=('0', '0', '0') |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch011; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Pocket007` | Pocket007 | Pocket | Profile=Sketch012; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane004` | DatumPlane004 | Plane | AttachmentSupport=Sketch001:Edge1, Sketch001:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket008` | Pocket008 | Pocket | Profile=Sketch013; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket009` | Pocket009 | Pocket | Profile=Sketch014; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=true |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | XY_Plane / FlatFace | 23 | 58 | **NO** | Lx=9, Ly=40, Ly=4, Lx=4, Ly=16, Lx=0.25, Ly=0.25, L=0.25, L=0.25, Lx=0.25 |
| Sketch001 | XY_Plane001 / FlatFace | 10 | 30 | **NO** | Lx=6.5, Lx=8.5, Ly=12.5, L=16, Ly=10, Ly=20, Lx=0.25, Lx=0.25, L=0.25, L=0.25 |
| Sketch002 | XY_Plane002 / FlatFace | 10 | 29 | **NO** | Lx=17, Ly=7.5, Lx=6.5, Ly=1, Ly=34.5, Lx=0.25, Lx=0.25, L=0.25, L=0.25 |
| Sketch003 | XY_Plane003 / FlatFace | 16 | 49 | **NO** | Ly=6.33, Ly=9.495, Lx=6.5, Lx=17, Ly=18.5, Lx=6.5, Lx=23.5, Ly=8.5, Ly=2, Lx=15.5, Ly=3.17, Ly=14.29, Ly=27, Lx=0.25, Lx=0.25, Lx=0.25, L=0.25, L=0.25, L=0.25 |
| Sketch004 | Pad:Face8 / FlatFace | 1 | 2 | yes | — |
| Sketch005 | DatumPlane / FlatFace | 2 | 4 | **NO** | Ø=60 |
| Sketch006 | DatumPlane / FlatFace | 2 | 5 | yes | Ø=59, Ly=1 |
| Sketch009 | DatumPlane002 / FlatFace | 2 | 4 | **NO** | Ø=60, Lx=6.5 |
| Sketch010 | DatumPlane002 / FlatFace | 2 | 6 | yes | Ø=59, Lx=6.5, Ly=1 |
| Sketch011 | DatumPlane003:Plane / FlatFace | 2 | 4 | **NO** | Ø=60, L=6.5 |
| Sketch012 | DatumPlane003 / FlatFace | 2 | 6 | yes | Ø=59, L=6.5, Ly=1 |
| Sketch013 | DatumPlane004 / FlatFace | 2 | 4 | **NO** | Ø=60, L=6.5 |
| Sketch014 | DatumPlane004 / FlatFace | 2 | 6 | yes | Ø=59, Lx=6.5, L=1 |
