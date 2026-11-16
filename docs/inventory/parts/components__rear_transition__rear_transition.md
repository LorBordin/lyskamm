# `components/rear_transition/rear_transition.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: FeaturePython×1, Body×2, FeatureBase×1, Fillet×3, Pad×5, Plane×4, Pocket×6, PolarPattern×9, Revolution×1, SketchObject×15
- Spreadsheets: 0 · Expressions: 0 · Sketches: 15
- Driving/reference dimensions in sketches: 22 (named: 0)

## Flags

- ⚠ not fully constrained `Sketch`
- ⚠ not fully constrained `Sketch012`
- ⚠ empty sketch `Sketch014`
- ⚠ not fully constrained `Sketch016`
- ⚠ not fully constrained `Sketch017`

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet002 BaseFeature=Transition
- **Body001** (`Body001`, Body) 

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Transition` | Transition | Rocket.FeatureTransition | AftAutoDiameter=false; AftCapBarWidth=3; AftCapStyle=solid; AftDiameter=40; AftShoulder=false; AftShoulderAutoDiameter=false; AftShoulderDiameter=10; AftShoulderLength=0.1; AftShoulderThickness=3; AngleOffset=0; AxialOffset=0; Clipped=false; Coefficient=0.5; CoreDiameter=10; ForeAutoDiameter=false; ForeCapBarWidth=3; ForeCapStyle=solid; ForeDiameter=60; ForeShoulder=true; ForeShoulderAutoDiameter=false; ForeShoulderDiameter=55.8; ForeShoulderLength=20; ForeShoulderThickness=2; Label=Transition; Length=45; LocationReference=Bottom of the parent component; RadialOffset=0; RadialReference=Surface of the parent component; Resolution=100; Thickness=3; TransitionStyle=hollow; TransitionType=power series |
| `BaseFeature` | BaseFeature | FeatureBase | BaseFeature=Transition |
| `Pad` | sede_tondo_ensterno | Pad | Profile=Sketch; Type=UpToShape; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Pocket` | sede_tondo_interno | Pocket | Profile=Sketch001; Type=Length; Length=30; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pad; Axis=Sketch['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `PolarPattern001` | PolarPattern001 | PolarPattern | Originals=Pocket; Axis=Sketch001['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Revolution` | base | Revolution | Profile=Sketch002; ReferenceAxis=Sketch002['H_Axis']; Angle=360; Midplane=false; Reversed=false |
| `Pad001` | sede_tondo_interno_1 | Pad | Profile=Sketch004; Type=Length; Length=10; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern002` | PolarPattern002 | PolarPattern | Originals=Pad001; Axis=Sketch004['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Fillet` | Fillet | Fillet | Base=PolarPattern002['Edge110', 'Edge112', 'Edge114', 'Edge116']; Radius=0.5; UseAllEdges=false |
| `Pocket002` | Pocket002 | Pocket | Profile=Sketch007; Type=ThroughAll; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern004` | PolarPattern004 | PolarPattern | Originals=Pocket002; Axis=Sketch007['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=PolarPattern004:Face7, Sketch005:Vertex4; MapMode=TangentPlane; AttachmentOffset=pos('0', '0', '0') |
| `Pocket003` | sede_vite_fusoliera_1 | Pocket | Profile=Sketch008; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern005` | PolarPattern005 | PolarPattern | Originals=Pocket003; Axis=X_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=PolarPattern005:Face7, Sketch005:Vertex2; MapMode=TangentPlane; AttachmentOffset=pos('0', '0', '0') |
| `Pocket004` | sede_vite_fusoliera_2 | Pocket | Profile=Sketch009; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern006` | PolarPattern006 | PolarPattern | Originals=Pocket004; Axis=X_Axis; Angle=360; Occurrences=4; Reversed=false |
| `DatumPlane002` | DatumPlane002 | Plane | AttachmentSupport=PolarPattern006:Face7, Sketch005:Vertex3; MapMode=TangentPlane; AttachmentOffset=pos('0', '0', '0') |
| `Pocket005` | Pocket005 | Pocket | Profile=Sketch010; Type=Length; Length=10; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern007` | PolarPattern007 | PolarPattern | Originals=Pocket005; Axis=X_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pad003` | Pad003 | Pad | Profile=Sketch012; Type=Length; Length=64; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `PolarPattern008` | PolarPattern008 | PolarPattern | Originals=Pad003; Axis=Sketch012['N_Axis']; Angle=360; Occurrences=4; Reversed=false |
| `Pad004` | Pad004 | Pad | Profile=Sketch013; Type=Length; Length=62; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `Fillet001` | Fillet001 | Fillet | Base=Pad004['Edge149', 'Edge154', 'Edge159', 'Edge164', 'Edge144']; Radius=1; UseAllEdges=false |
| `DatumPlane003` | DatumPlane003 | Plane | AttachmentSupport=Sketch015:Edge1, Sketch015:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pad005` | Pad005 | Pad | Profile=Sketch016; Type=Length; Length=3; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `PolarPattern009` | PolarPattern009 | PolarPattern | Originals=Pad005; Axis=X_Axis; Angle=360; Occurrences=4; Reversed=false |
| `Pocket006` | Pocket006 | Pocket | Profile=Sketch017; Type=Length; Length=5; Length2=5; Offset=0; Midplane=false; Reversed=false |
| `Fillet002` | Fillet002 | Fillet | Base=Pocket006['Edge402', 'Edge375', 'Edge404', 'Edge396', 'Edge103', 'Edge394', 'Edge91', 'Edge93']; Radius=1; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | BaseFeature:Face4 / FlatFace | 8 | 20 | **NO** | Ø=7, Ly=3.5, Ly=23.4279 |
| Sketch001 | Pad:Face12 / FlatFace | 1 | 2 | yes | Ø=5.2 |
| Sketch002 | XY_Plane / FlatFace | 4 | 11 | yes | Lx=3, Ly=14.55 |
| Sketch004 | Revolution:Face40 / FlatFace | 1 | 2 | yes | Ø=2.5 |
| Sketch005 | — / FlatFace | 3 | 9 | yes | Ang=135, Ang=45, Ang=22.5 |
| Sketch007 | Fillet:Face49 / FlatFace | 3 | 6 | yes | Ø=3.1 |
| Sketch008 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=2.6, L=3.75 |
| Sketch009 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.6, L=3.75 |
| Sketch010 | DatumPlane002 / FlatFace | 5 | 11 | yes | Lx=9.2, Ly=15.1 |
| Sketch012 | — / FlatFace | 4 | 10 | **NO** | L=3, Lx=1.5 |
| Sketch013 | PolarPattern008:Face59 / FlatFace | 2 | 4 | yes | Ø=35.2, Ø=37 |
| Sketch014 | Fillet001:Face9 / FlatFace | 0 | 0 | yes | — |
| Sketch015 | Fillet001:Face3 / FlatFace | 1 | 3 | yes | — |
| Sketch016 | DatumPlane003 / FlatFace | 4 | 9 | **NO** | Ly=3 |
| Sketch017 | PolarPattern009:Face113 / FlatFace | 2 | 3 | **NO** | — |
