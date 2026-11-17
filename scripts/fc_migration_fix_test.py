"""Phase 3.0 repair test: re-link features broken by the FreeCAD 1.0 -> 1.1.3 migration.

Run with the AppImage's freecadcmd. Works on TEMPORARY COPIES; nothing is saved. Used in the
2026-11-17 diagnosis (see PLAN.md, step 3.0). Strategy, applied once per failing feature, in
recompute order, using the references stored in the file (FreeCAD 1.0 topology):
  - Sketch with lost support face  -> face with the same centre/area in the rebuilt shape
    (+ delete constraints that the 1.1 solver reports as redundant);
  - Chamfer/Fillet with stale edge links -> all rebuilt edges lying on the stored edges.
Fins (Rocket WB): Fin.TtwOffset -> 0 reproduces the printed tab position.
"""
import FreeCAD, Part, shutil, tempfile, os, json
R='/home/lorenzo/projects/lyskamm/components/'
def facematch(old,name,new):
    e=old.getElement(name); return min(((f.CenterOfMass-e.CenterOfMass).Length+abs(f.Area-e.Area),'Face%d'%i) for i,f in enumerate(new.Faces,1))
def edges_on(oldshape,names,new):
    comp=Part.Compound([oldshape.getElement(n) for n in names]); sel=[]
    for i,e in enumerate(new.Edges,1):
        pts=[e.valueAt(e.FirstParameter+(e.LastParameter-e.FirstParameter)*t) for t in (0.25,0.5,0.75)]
        if all(comp.distToShape(Part.Vertex(p))[0]<1e-4 for p in pts): sel.append('Edge%d'%i)
    return sel
out={}
import re
for f in ['mid_fuselage/mid_fuselage.FCStd','lower_fuselage_w_motor_mount/lower_fuselage_v1.FCStd','e-bay/fuselage.FCStd','e-bay/fuselage_w_logo.FCStd']:
    log=[]; d=tempfile.mkdtemp(); p=os.path.join(d,os.path.basename(f)); shutil.copy(R+f,p)
    doc=FreeCAD.openDocument(p)
    stored={o.Name:o.Shape.copy() for o in doc.Objects if hasattr(o,'Shape') and not o.Shape.isNull()}
    body=doc.getObject('Body'); b0=body.Shape.copy()
    orig={}
    for o in doc.Objects:
        if o.TypeId.startswith('Sketcher') and o.AttachmentSupport: orig[o.Name]=[(ob,[n.split('.')[-1].lstrip('?') for n in subs]) for ob,subs in o.AttachmentSupport]
        if o.TypeId in ('PartDesign::Chamfer','PartDesign::Fillet'): orig[o.Name]=(o.Base[0],[n.split('.')[-1].lstrip('?') for n in o.Base[1]])
    done=set()
    doc.recompute()
    for it in range(10):
        bad=[o for o in doc.Objects if 'Invalid' in o.State]
        if not bad: break
        bad=[o for o in bad if o.Name not in done]
        if not bad: break
        o=bad[0]; done.add(o.Name)
        if o.TypeId.startswith('Sketcher'):
            new=[]
            for ob,subs in orig[o.Name]:
                ns=[facematch(stored[ob.Name],n,ob.Shape) for n in subs]
                log.append('%s support %s.%s -> %s (err %.4f)'%(o.Name,ob.Name,subs,[x[1] for x in ns],max(x[0] for x in ns)))
                new.append((ob,[x[1] for x in ns]))
            o.AttachmentSupport=new
            doc.recompute()
            m=o.getStatusString()
            if 'redundant' in m:
                nums=sorted((int(x) for x in re.findall(r'(?m)^\s*(\d+)\s*$',m)),reverse=True)
                for k in nums: log.append('%s: delete redundant constraint %d (%s)'%(o.Name,k,o.Constraints[k-1].Type)); o.delConstraint(k-1)
        elif o.TypeId in ('PartDesign::Chamfer','PartDesign::Fillet'):
            ob,names=orig[o.Name]
            sel=edges_on(stored[ob.Name],names,ob.Shape)
            log.append('%s: %d stored edges -> %d new edges lying on them'%(o.Name,len(names),len(sel)))
            o.Base=(ob,[]); doc.recompute(); o.Base=(ob,sel)
        else:
            log.append('UNHANDLED '+o.Name); break
        doc.recompute()
    bad=[o.Name+': '+o.getStatusString() for o in doc.Objects if 'Invalid' in o.State]
    b1=body.Shape
    log.append('RESULT invalid=%s | body vol %.3f -> %.3f (diff %.3f) | CoM shift %.5f | stored∩new %.3f'%(bad,b0.Volume,b1.Volume,b1.Volume-b0.Volume,(b0.CenterOfMass-b1.CenterOfMass).Length,b0.common(b1).Volume))
    for s in [o for o in doc.Objects if o.TypeId.startswith('Sketcher') and o.ExternalGeometry]:
        for ob,subs in s.ExternalGeometry:
            for n in subs:
                if '?' in n: log.append('stale external ref: %s -> %s.%s'%(s.Name,ob.Name,n))
    out[f]=log; FreeCAD.closeDocument(doc.Name)
print(json.dumps(out, indent=1))
