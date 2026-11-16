# `components/lower_fins/lower_fins.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: FeaturePython×1, Body×1, FeatureBase×1, Fillet×1, Pad×1, SketchObject×1
- Spreadsheets: 0 · Expressions: 0 · Sketches: 1
- Driving/reference dimensions in sketches: 4 (named: 0)

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet BaseFeature=Fin

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Fin` | Fin | Rocket.FeatureFin | AngleOffset=0; AutoDiameter=true; AutoHeight=false; AxialOffset=0; Cant=0; FinCount=3; FinSet=false; FinSpacing=120; FinType=trapezoid; Height=50; Label=Fin; Length=0; LocationReference=Bottom of the parent component; MinimumEdge=false; MinimumEdgeSize=0.2; RadialOffset=0; RadialReference=Surface of the parent component; RootChord=63.5; RootCrossSection=airfoil; RootLength1=20000000000000000; RootLength2=80000000000000000; RootPerCent=true; RootThickness=3; Span=81.28; SweepAngle=23.8; SweepLength=21; TipChord=32; TipCrossSection=Same as root; TipLength1=20000000000000000; TipLength2=80000000000000000; TipPerCent=true; TipSameThickness=true; TipThickness=3; Ttw=true; TtwAutoHeight=false; TtwHeight=11; TtwLength=60; TtwOffset=3.5; TtwThickness=3; TubeAutoOuterDiameter=true; TubeOuterDiameter=0; TubeThickness=1 |
| `BaseFeature` | BaseFeature | FeatureBase | BaseFeature=Fin |
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=60; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Fillet` | Fillet | Fillet | Base=Pad['Edge28', 'Edge35', 'Edge7', 'Edge38', 'Edge42', 'Edge4', 'Edge41', 'Edge1', 'Edge2', 'Edge36', 'Edge39']; Radius=0.5; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | BaseFeature:Face10 / FlatFace | 8 | 22 | yes | Ly=2, Lx=1.5, Lx=1.5, Ly=2 |
