# `components/upper_fins/upper_fin.FCStd`

- FreeCAD version: 1.0R39319 (Git)
- Object counts: FeaturePython×1, Body×1, Chamfer×1, FeatureBase×1, Fillet×1, Pad×1, SketchObject×1
- Spreadsheets: 0 · Expressions: 0 · Sketches: 1
- Driving/reference dimensions in sketches: 2 (named: 0)

## Bodies / parts

- **Body** (`Body`, Body) Tip=Fillet BaseFeature=Fin

## Features (document order)

| Name | Label | Type | Key properties |
|---|---|---|---|
| `Fin` | Fin | Rocket.FeatureFin | AngleOffset=0; AutoDiameter=true; AutoHeight=false; AxialOffset=0; Cant=0; FinCount=3; FinSet=false; FinSpacing=120; FinType=trapezoid; Height=50; Label=Fin; Length=0; LocationReference=Bottom of the parent component; MinimumEdge=false; MinimumEdgeSize=0.2; RadialOffset=0; RadialReference=Surface of the parent component; RootChord=95; RootCrossSection=airfoil; RootLength1=2000000000000000; RootLength2=8000000000000000; RootPerCent=true; RootThickness=3; Span=81.28; SweepAngle=45; SweepLength=50; TipChord=41; TipCrossSection=Same as root; TipLength1=2000000000000000; TipLength2=8000000000000000; TipPerCent=true; TipSameThickness=true; TipThickness=3; Ttw=true; TtwAutoHeight=false; TtwHeight=12; TtwLength=90; TtwOffset=5; TtwThickness=3; TubeAutoOuterDiameter=true; TubeOuterDiameter=0; TubeThickness=1 |
| `BaseFeature` | BaseFeature | FeatureBase | BaseFeature=Fin |
| `Pad` | Pad | Pad | Profile=Sketch; Type=Length; Length=90; Length2=10; Offset=0; Midplane=false; Reversed=true |
| `Chamfer` | Chamfer | Chamfer | Base=Pad['Edge42']; ChamferType=EqualDistance; Size=10; Size2=1; Angle=45 |
| `Fillet` | Fillet | Fillet | Base=Chamfer['Edge11', 'Edge8', 'Edge4', 'Edge1', 'Edge2', 'Edge6', 'Edge13', 'Edge23', 'Edge5']; Radius=0.5; UseAllEdges=false |

## Sketches

| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |
|---|---|---|---|---|---|
| Sketch | BaseFeature:Face10 / FlatFace | 8 | 20 | yes | Ly=2, Lx=1.5 |
