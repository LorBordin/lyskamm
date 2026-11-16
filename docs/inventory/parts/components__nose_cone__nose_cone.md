# `components/nose_cone/nose_cone.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: FeaturePython×1, Body×1, Chamfer×1, FeatureBase×1, Pad×1, Plane×2, Pocket×2, PolarPattern×1, SketchObject×4
- Spreadsheets: 0 · Expressions: 0 · Sketches: 4
- Driving/reference dimensions in sketches: 8 (named: 0)

## Bodies / parts

- **Body** (`Body`, Body) Tip=PolarPattern BaseFeature=NoseCone

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `NoseCone` | NoseCone | Rocket.FeatureNoseCone | AngleOffset=0; AutoDiameter=false; AxialOffset=0; BluntedDiameter=5; CapBarWidth=3; CapStyle=solid; Coefficient=0.4; Diameter=60; Label=NoseCone; Length=120; LocationReference=Bottom of the parent component; NoseStyle=solid; NoseType=power series; OgiveDiameter=120; RadialOffset=0; RadialReference=Surface of the parent component; Resolution=100; Shoulder=false; ShoulderAutoDiameter=false; ShoulderDiameter=23.62; ShoulderLength=24.79; ShoulderThickness=1.57; Thickness=1.57 |
| `BaseFeature` | BaseFeature | FeatureBase | BaseFeature=NoseCone |
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=7.5; Length2=10; Offset=0; Midplane=false; Reversed=false |
| `DatumPlane` | DatumPlane | Plane | AttachmentSupport=Sketch001:Edge1, Sketch001:Vertex2; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `DatumPlane001` | DatumPlane001 | Plane | AttachmentSupport=Sketch001:Edge2, Sketch001:Vertex3; MapMode=NormalToEdge; AttachmentOffset=pos('0', '0', '0') |
| `Pocket` | Pocket | Pocket | Profile=Sketch002; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Pocket001` | Pocket001 | Pocket | Profile=Sketch003; Type=Length; Length=4; Length2=5; Offset=0; Midplane=false; Reversed=true |
| `Chamfer` | Chamfer | Chamfer | Base=Pocket001['Edge7', 'Edge6', 'Edge9', 'Edge8']; ChamferType=EqualDistance; Size=1; Size2=1; Angle=45 |
| `PolarPattern` | PolarPattern | PolarPattern | Originals=Pocket, Pocket001, Chamfer; Axis=X_Axis; Angle=360; Occurrences=4; Reversed=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | BaseFeature:Face2 / FlatFace | 2 | 4 | yes | Ø=56 |
| Sketch001 | Pad:Face3 / FlatFace | 3 | 9 | yes | Ang=45, Ang=45, Ang=22.5 |
| Sketch002 | DatumPlane / FlatFace | 1 | 3 | yes | Ø=2.1, L=3.75 |
| Sketch003 | DatumPlane001 / FlatFace | 1 | 3 | yes | Ø=2.1, L=3.75 |
