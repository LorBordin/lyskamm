"""Mass and centre of gravity of the Lyskamm assembly (FreeCAD built-in Assembly).

Run inside FreeCAD (GUI Python console, MCP execute_code) or with freecadcmd:
    exec(open('scripts/assembly_cg.py').read())
or
    freecadcmd scripts/assembly_cg.py

Each App::Link in the Assembly carries `Mass_g` (weighed mass per piece, docs/inventory/masses.md).
The CG of each piece is its CAD centre of mass (uniform density). Frame: nose tip at origin,
rocket axis along -Z, so "from nose tip" = -z.
Parts without a CAD model (parachute, cord, electronics, motor...) are not included.
"""
import os
import FreeCAD as App

ASM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals()
                        else os.getcwd(), '..', 'assemblies', 'lyskamm_asm.FCStd')


def get_doc():
    for d in App.listDocuments().values():
        if d.getObject('Assembly') and d.Name.startswith('lyskamm_asm'):
            return d
    return App.openDocument(os.path.normpath(ASM_FILE))


def assembly_cg(doc):
    asm = doc.getObject('Assembly')
    rows, total, moment = [], 0.0, App.Vector()
    missing = []
    for o in asm.Group:
        if o.TypeId != 'App::Link':
            continue
        m = getattr(o, 'Mass_g', None)
        if not m:
            missing.append(o.Label)
            continue
        c = o.Shape.CenterOfMass
        rows.append((o.Label, m, -c.z))
        total += m
        moment += c * m
    cg = moment * (1.0 / total)
    return rows, total, cg, missing


if __name__ in ('__main__', 'assembly_cg') or True:
    doc = get_doc()
    rows, total, cg, missing = assembly_cg(doc)
    for label, m, x in sorted(rows, key=lambda r: r[2]):
        print('%-22s %7.2f g   CG %7.1f mm from nose tip' % (label, m, x))
    print('-' * 56)
    print('TOTAL %.1f g   CG %.1f mm from nose tip   (radial offset %.2f mm)'
          % (total, -cg.z, (cg.x ** 2 + cg.y ** 2) ** 0.5))
    if missing:
        print('No Mass_g:', ', '.join(missing))
