"""Build assemblies/lyskamm_asm.FCStd (built-in Assembly WB). Run inside the FreeCAD GUI.

Frame: nose tip at origin, rocket axis along -Z. Placements were solved from geometry
(stack_solver.py): hole patterns, carbon/aluminium rod channels, connector, keys, zero interference.
"""
import FreeCAD as App, FreeCADGui as Gui, Part, math
import JointObject, UtilsAssembly
from FreeCAD import Vector, Placement, Rotation, Matrix
import asm_stack_solver as ss

REPO = '/home/lorenzo/projects/lyskamm/'
C = REPO + 'components/'
OUT = REPO + "assemblies/lyskamm_asm.FCStd"


def fin_plc(theta, z_le, r0=30.0):
    t = math.radians(theta)
    X = Vector(0, 0, -1); Z = Vector(math.cos(t), math.sin(t), 0); Y = Z.cross(X)
    m = Matrix(X.x, Y.x, Z.x, 0, X.y, Y.y, Z.y, 0, X.z, Y.z, Z.z, 0, 0, 0, 0, 1)
    return Placement(Vector(0, 0, z_le) + Z * r0, Rotation(m))


def rod_plc(theta, z_top, r=23.43):
    """Rod modelled along +Z from 0: put its z=0 end at z_top-length later; here: axis -> -Z, top at z_top."""
    t = math.radians(theta)
    return Placement(Vector(r * math.cos(t), r * math.sin(t), z_top), Rotation(Vector(1, 0, 0), 180))


S = ss.place
# name: (file, body, placement, host, mass_g)
PARTS = [
    ('nose_cone',        'nose_cone/nose_cone',                         S('x', (0, 127.5), 0, 0, 0),              None,               20),
    ('upper_closure',    'e-bay/upper_closure',                         S('z', (0, 5), 1, 0, -127.5),             'nose_cone',        8),
    ('ebay_upper_a',     'e-bay/fuselage',                              S('z', (0, 105), 0, 0, -132.5),           'upper_closure',    14),
    ('ebay_upper_b',     'e-bay/fuselage',                              S('z', (0, 105), 0, 180, -132.5),         'upper_closure',    14),
    ('mid_closure',      'e-bay/mid_closure',                           S('z', (0, 5), 0, 0, -237.5),             'ebay_upper_a',     10.33),
    ('ebay_lower_a',     'e-bay/fuselage',                              S('z', (0, 105), 0, 0, -242.5),           'mid_closure',      14),
    ('ebay_lower_logo',  'e-bay/fuselage_w_logo',                       S('y', (-52.5, 52.5), 1, 0, -242.5).multiply(Placement(Vector(0, 0, -30), Rotation())), 'mid_closure', 14),
    ('lower_closure',    'e-bay/lower_closure_v2',                      S('z', (0, 5), 1, 270, -347.5),           'ebay_lower_a',     28),
    ('parachute_seat',   'parachute_fuselage_seat/parachute_fuselage_seat_v2', S('z', (0, 242.5), 1, 0, -352.5),  'lower_closure',    103),
    ('low_chute_ret',    'lower_chute_retainer/low_chute_ret',          S('z', (7.5, 12.5), 1, 270, -595),        'parachute_seat',   8),
    ('mid_fuselage',     'mid_fuselage/mid_fuselage',                   S('z', (-7.5, 97.5), 0, 45, -600),        'low_chute_ret',    53),
    ('upper_retainer',   'upper_motor_retainer/upper_retainer',         S('z', (0, 5), 1, 0, -705),               'mid_fuselage',     13),
    ('lower_fuselage',   'lower_fuselage_w_motor_mount/lower_fuselage_v1', S('z', (-20, 93), 1, 45, -710),        'upper_retainer',   53.5),
    ('rear_transition',  'rear_transition/rear_transition',             S('x', (0, 45), 0, 45, -823),             'lower_fuselage',   29),
]
for i, th in enumerate((0, 90, 180, 270)):
    PARTS.append(('upper_fin_%d' % (i + 1), 'upper_fins/upper_fin', fin_plc(th, -607.5), 'mid_fuselage', 11.75))
for i, th in enumerate((0, 90, 180, 270)):
    PARTS.append(('lower_fin_%d' % (i + 1), 'lower_fins/lower_fins', fin_plc(th, -758.0), 'lower_fuselage', 8.25))
for i, th in enumerate((45, 135, 225, 315)):
    PARTS.append(('carbon_rod_%d' % (i + 1), 'carbon_rod/carbon_rod', rod_plc(th, -383.5), 'parachute_seat', 7))
for i, th in enumerate((0, 180)):
    PARTS.append(('alu_rod_upper_%d' % (i + 1), 'e-bay/alluminium_rod', rod_plc(th, -137.5), 'upper_closure', 3.67))
for i, th in enumerate((0, 180)):
    PARTS.append(('alu_rod_lower_%d' % (i + 1), 'e-bay/alluminium_rod', rod_plc(th, -243.0), 'mid_closure', 3.67))  # on mid_closure: avoids a 1.1.3 solver bug, see PLAN


def open_doc(path):
    name = path.split('/')[-1]
    if name in App.listDocuments():
        return App.getDocument(name)
    return App.openDocument(C + path + '.FCStd', hidden=True)


def nearest_circle_edge(shape_world, link_plc, target):
    """Edge name (local numbering) of the circle/arc whose centre is nearest to target (world)."""
    best = None
    inv = link_plc.inverse()
    tl = inv.multVec(target)
    for i, e in enumerate(shape_world.Edges):
        try:
            c = e.Curve
        except Exception:
            continue
        if isinstance(c, Part.Circle):
            d = (c.Center - tl).Length - 1e-3 * c.Radius   # prefer larger radius on ties
            if best is None or d < best[0]:
                best = (d, 'Edge%d' % (i + 1))
    return best[1]


def nearest_vertex_ref(shape_local, link_plc, target):
    tl = link_plc.inverse().multVec(target)
    vi = min(range(len(shape_local.Vertexes)), key=lambda i: (shape_local.Vertexes[i].Point - tl).Length)
    v = shape_local.Vertexes[vi]
    for i, e in enumerate(shape_local.Edges):
        if any(x.isSame(v) for x in e.Vertexes):
            return ['Edge%d' % (i + 1), 'Vertex%d' % (vi + 1)]


def build():
    A = App.newDocument('lyskamm_asm')
    A.saveAs(OUT)
    asm = A.addObject('Assembly::AssemblyObject', 'Assembly'); asm.Type = 'Assembly'
    jg = asm.newObject('Assembly::JointGroup', 'Joints')
    links = {}
    for name, path, plc, host, mass in PARTS:
        d = open_doc(path)
        l = asm.newObject('App::Link', name)
        l.LinkedObject = d.getObject('Body')
        l.Placement = plc
        l.addProperty('App::PropertyFloat', 'Mass_g', 'Lyskamm', 'Weighed mass per piece (docs/inventory/masses.md)')
        l.Mass_g = mass
        links[name] = l
    A.recompute()
    return A, asm, jg, links


def add_joints(A, asm, jg, links):
    g = jg.newObject('App::FeaturePython', 'GroundedJoint')
    JointObject.GroundedJoint(g, links['nose_cone'])
    JointObject.ViewProviderGroundedJoint(g.ViewObject)
    desired = {n: Placement(l.Placement) for n, l in links.items()}
    for name, path, plc, host, mass in PARTS:
        if host is None:
            continue
        P, H = links[name], links[host]
        ps = P.LinkedObject.Shape   # local
        hs = H.LinkedObject.Shape
        # joint location: the part's point nearest the host along the axis -> use part bbox centre on axis
        wb = P.Shape.BoundBox
        if name.startswith(('upper_fin', 'lower_fin')):
            target = desired[name].multVec(Vector(0, 0, 0))          # fin root LE point
            ref2 = [P, nearest_vertex_ref(ps, desired[name], target)]
        elif name.startswith(('carbon_rod', 'alu_rod')):
            target = desired[name].multVec(Vector(0, 0, 0))          # rod top end centre
            e = nearest_circle_edge(ps, desired[name], target); ref2 = [P, [e, e]]
        else:
            target = Vector(0, 0, wb.ZMax)          # forward end = shoulder on the host
            e = nearest_circle_edge(ps, desired[name], target)
            ref2 = [P, [e, e]]
        e1 = nearest_circle_edge(hs, desired[host], target)
        ref1 = [H, [e1, e1]]
        j = jg.newObject('App::FeaturePython', 'Fixed_%s' % name)
        JointObject.Joint(j, 0)
        JointObject.ViewProviderJoint(j.ViewObject)
        jcs1 = UtilsAssembly.findPlacement(ref1, False)
        jcs2 = UtilsAssembly.findPlacement(ref2, False)
        # Fixed: Plc_H * JCS1 == Plc_P * JCS2 * Offset2
        j.Offset2 = jcs2.inverse().multiply(desired[name].inverse()).multiply(desired[host]).multiply(jcs1)
        j.Reference1 = ref1
        j.Reference2 = ref2
        j.Placement1 = jcs1
        j.Placement2 = jcs2.multiply(j.Offset2)
        j.Label = 'Fixed_%s_to_%s' % (name, host)
    for n, l in links.items():
        l.Placement = desired[n]
    A.recompute()
    r = asm.solve()
    A.recompute()
    worst = 0
    for n, l in links.items():
        dp = (l.Placement.Base - desired[n].Base).Length
        dr = abs(math.degrees(l.Placement.Rotation.multiply(desired[n].Rotation.inverted()).Angle))
        worst = max(worst, dp, dr)
        if dp > 1e-6 or dr > 1e-6:
            print('MOVED', n, dp, dr)
    print('solve', r, 'worst deviation', worst)
    return desired

