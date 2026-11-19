"""Find flip/roll of each airframe section from geometry (run inside FreeCAD).

Global frame: nose tip at origin, rocket axis along -Z (tail at negative z).
Each part: (doc, body, local axis, visible extent [a, b] along that axis).
flip=0: local a at the forward end; flip=1: local b at the forward end.
"""
import FreeCAD as App, Part, math
from FreeCAD import Vector, Placement, Rotation

R = '/home/lorenzo/projects/lyskamm/components/'

# local axis -> -Z (towards the tail)
BASE = {
    'x': Rotation(Vector(0, 1, 0), 90),    # x -> -z
    'y': Rotation(Vector(1, 0, 0), -90),   # y -> -z
    'z': Rotation(Vector(1, 0, 0), 180),   # z -> -z
}


def unit(ax):
    return {'x': Vector(1, 0, 0), 'y': Vector(0, 1, 0), 'z': Vector(0, 0, 1)}[ax]


def place(ax, ext, flip, roll, z_front):
    a, b = ext
    s = a if flip == 0 else b
    rot = BASE[ax]
    if flip:
        rot = Rotation(Vector(1, 0, 0), 180).multiply(rot)
    rot = Rotation(Vector(0, 0, 1), roll).multiply(rot)
    p = Placement(Vector(0, 0, z_front), rot)
    return p.multiply(Placement(unit(ax) * (-s), Rotation()))


def radial_holes(shape, rmax=1.7, zlo=-1e9, zhi=1e9):
    out = []
    for f in shape.Faces:
        s = f.Surface
        if isinstance(s, Part.Cylinder) and s.Radius < rmax and abs(s.Axis.z) < 1e-3:
            c = f.BoundBox.Center
            r = math.hypot(c.x, c.y)
            if r > 20 and zlo <= c.z <= zhi:
                out.append((round(c.z, 2), math.degrees(math.atan2(c.y, c.x)) % 360))
    # dedupe
    res = []
    for z, a in out:
        if not any(abs(z - z2) < 0.05 and min(abs(a - a2), 360 - abs(a - a2)) < 0.5 for z2, a2 in res):
            res.append((z, a))
    return res


def angdiff(a, b):
    d = (a - b) % 360
    return min(d, 360 - d)


def roll_candidates(nh, ph):
    """Rolls (deg) that rotate every hole of ph (part at roll 0) onto a hole of nh at the same z."""
    common_z = sorted({z for z, _ in ph} & {round(z, 2) for z, _ in nh})
    ph_c = [(z, a) for z, a in ph if z in common_z]
    if not ph_c:
        return None, 0
    z0, a0 = ph_c[0]
    cands = []
    for z, a in nh:
        if abs(z - z0) < 0.05:
            r = (a - a0) % 360
            ok = all(any(abs(z2 - z1) < 0.05 and angdiff((a1 + r) % 360, a2) < 0.3 for z2, a2 in nh) for z1, a1 in ph_c)
            if ok:
                cands.append(round(r, 2))
    return sorted(set(cands)), len(ph_c)


def try_part(neighbour_shapes, shape, ax, ext, z_front, rolls=None):
    """Return list of (flip, roll, n_matched_holes, max_interference)."""
    results = []
    for flip in (0, 1):
        p0 = place(ax, ext, flip, 0, z_front)
        s0 = shape.copy(); s0.Placement = p0.multiply(s0.Placement)
        bb = s0.BoundBox
        ph = radial_holes(s0)
        nh = []
        for ns in neighbour_shapes:
            nh += radial_holes(ns, zlo=bb.ZMin - 0.1, zhi=bb.ZMax + 0.1)
        cands, n = roll_candidates(nh, ph) if rolls is None else (rolls, -1)
        if cands is None:
            cands = [0.0]
        for r in cands:
            s = shape.copy(); s.Placement = place(ax, ext, flip, r, z_front).multiply(s.Placement)
            inter = max([s.common(ns).Volume for ns in neighbour_shapes if ns.BoundBox.intersect(s.BoundBox)] or [0])
            results.append((flip, r, n, round(inter, 3)))
    return results
