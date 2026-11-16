#!/usr/bin/env python3
"""Read-only inventory of FreeCAD .FCStd files (parses Document.xml, stdlib only).

Usage:
    python3 scripts/cad_inventory.py            # all *.FCStd in the repo -> docs/inventory/
    python3 scripts/cad_inventory.py FILE ...   # only the given files (printed to stdout)

Never writes to .FCStd files.
"""
import collections
import pathlib
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "inventory"
ASSEMBLY = REPO / "assemblies" / "lyskamm.FCStd"

# Sketcher::Constraint type enum (FreeCAD 1.x)
CONS_TYPES = {
    0: "None", 1: "Coincident", 2: "Horizontal", 3: "Vertical", 4: "Parallel",
    5: "Tangent", 6: "Distance", 7: "DistanceX", 8: "DistanceY", 9: "Angle",
    10: "Perpendicular", 11: "Radius", 12: "Equal", 13: "PointOnObject",
    14: "Symmetric", 15: "InternalAlignment", 16: "SnellsLaw", 17: "Block",
    18: "Diameter", 19: "Weight",
}
DIM_TYPES = {6, 7, 8, 9, 11, 18}
DIM_SHORT = {6: "L", 7: "Lx", 8: "Ly", 9: "Ang", 11: "R", 18: "Ø"}

# Feature properties worth listing, per type suffix
FEATURE_PROPS = {
    "Pad": ["Profile", "Type", "Length", "Length2", "Offset", "Midplane", "Reversed", "UpToFace"],
    "Pocket": ["Profile", "Type", "Length", "Length2", "Offset", "Midplane", "Reversed", "UpToFace"],
    "Revolution": ["Profile", "ReferenceAxis", "Angle", "Midplane", "Reversed"],
    "Groove": ["Profile", "ReferenceAxis", "Angle", "Midplane", "Reversed"],
    "Hole": ["Profile", "Diameter", "Depth", "Threaded", "ThreadSize"],
    "Fillet": ["Base", "Radius", "UseAllEdges"],
    "Chamfer": ["Base", "ChamferType", "Size", "Size2", "Angle"],
    "PolarPattern": ["Originals", "Axis", "Angle", "Occurrences", "Reversed"],
    "LinearPattern": ["Originals", "Direction", "Length", "Occurrences", "Reversed"],
    "Mirrored": ["Originals", "MirrorPlane"],
    "Boolean": ["Group", "Type"],
    "Plane": ["AttachmentSupport", "MapMode", "AttachmentOffset"],
    "FeatureBase": ["BaseFeature"],
    "Mirroring": ["Source", "Normal", "Base"],
    "Body": ["Tip", "BaseFeature"],
}
# Rocket workbench features: dump all simple props except these
ROCKET_SKIP = {"Comment", "Description", "Manufacturer", "PartNumber", "Label2", "Group",
               "Placement", "Shape", "ShapeMaterial", "Proxy", "ExpressionEngine", "Visibility",
               "Texture", "Profile"}
# Enumerations stored as bare indices in Document.xml
ENUMS = {
    ("Pad", "Type"): ["Length", "UpToLast", "UpToFirst", "UpToFace", "TwoLengths", "UpToShape"],
    ("Pocket", "Type"): ["Length", "ThroughAll", "UpToFirst", "UpToFace", "TwoLengths", "UpToShape"],
    ("Chamfer", "ChamferType"): ["EqualDistance", "TwoDistances", "DistanceAngle"],
    ("Boolean", "Type"): ["Fuse", "Cut", "Common"],
    ("*", "MapMode"): ["Deactivated", "Translate", "ObjectXY", "ObjectXZ", "ObjectYZ", "FlatFace",
                       "TangentPlane", "NormalToEdge", "FrenetNB", "FrenetTN", "FrenetTB", "Concentric",
                       "SRF", "ThreePointsPlane", "ThreePointsNormal", "Folding"],
}


def enum_name(ftype, prop, v):
    items = ENUMS.get((ftype, prop)) or ENUMS.get(("*", prop))
    try:
        return items[int(v)] if items else v
    except (ValueError, IndexError):
        return v


SKIP_TYPES = {"App::Origin", "App::Line", "App::Plane"}
DEFAULT_LABEL = re.compile(r"^(Sketch|Pad|Pocket|Fillet|Chamfer|PolarPattern|LinearPattern|"
                           r"DatumPlane|Revolution|Groove|Body|BaseFeature|Boolean|Mirrored|"
                           r"Part__Mirroring|Hole)\d*$")


def fmt_num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f"{f:.4f}".rstrip("0").rstrip(".") if f != int(f) else str(int(f))


def prop_value(p):
    """Best-effort readable value of a <Property> element."""
    ch = list(p)
    if not ch:
        return None
    c = ch[0]
    tag = c.tag
    if tag in ("Float", "Integer", "Bool", "String", "Link", "Uuid"):
        v = c.get("value")
        enum = p.find("CustomEnumList")
        if tag == "Integer" and enum is not None:
            items = [e.get("value") for e in enum.findall("Enum")]
            i = int(v)
            return items[i] if 0 <= i < len(items) else v
        return fmt_num(v) if tag in ("Float",) else v
    if tag == "LinkSub":
        subs = [s.get("value") for s in c.findall("Sub") if s.get("value")]
        return f"{c.get('value')}{subs}" if subs else c.get("value")
    if tag in ("LinkList", "LinkSubList"):
        out = []
        for l in c:
            v = l.get("value") or l.get("obj")
            sub = l.get("sub")
            out.append(f"{v}:{sub}" if sub else v)
        return ", ".join(x for x in out if x)
    if tag == "PropertyPlacement":
        pos = tuple(fmt_num(c.get(k)) for k in ("Px", "Py", "Pz"))
        ang = float(c.get("A", 0))
        axis = tuple(fmt_num(c.get(k)) for k in ("Ox", "Oy", "Oz"))
        return f"pos{pos}" + (f" rot {ang*57.29577951308232:.3f}° about {axis}" if abs(ang) > 1e-12 else "")
    if tag == "PropertyVector":
        return tuple(fmt_num(c.get(k)) for k in ("valueX", "valueY", "valueZ"))
    return None


def load(path):
    z = zipfile.ZipFile(path)
    root = ET.fromstring(z.read("Document.xml"))
    types = {o.get("name"): o.get("type") for o in root.find("Objects").findall("Object")}
    data = {}
    for o in root.find("ObjectData").findall("Object"):
        data[o.get("name")] = {p.get("name"): p for p in o.findall("Properties/Property")}
    return root, types, data


def label(P, name):
    return P["Label"].find("String").get("value") if "Label" in P else name


def expressions(P):
    ee = P.get("ExpressionEngine")
    if ee is None:
        return []
    return [(e.get("path"), e.get("expression")) for e in ee.iter("Expression")]


def analyze_sketch(P):
    cl = P["Constraints"].find("ConstraintList") if "Constraints" in P else None
    cons = cl.findall("Constrain") if cl is not None else []
    geo = P["Geometry"].find("GeometryList") if "Geometry" in P else None
    ngeo = int(geo.get("count")) if geo is not None else 0
    nconstr_geo = 0
    if geo is not None:
        for g in geo:
            gp = g.find("GeoExtensions")
            txt = ET.tostring(g, encoding="unicode")
            if 'construction="1"' in txt or 'Construction value="1"' in txt or "InternalGeometryType" in txt and 'geometryModeFlags="' in txt and 'geometryModeFlags="00000000000000000000000000000001"' in txt:
                nconstr_geo += 1
    dims = []
    for i, c in enumerate(cons):
        t = int(c.get("Type"))
        if t in DIM_TYPES:
            v = float(c.get("Value"))
            if t == 9:
                v = v * 57.29577951308232
            dims.append({
                "idx": i, "type": t, "name": c.get("Name") or "",
                "value": v, "driving": c.get("IsDriving", "1") == "1",
                "active": c.get("IsActive", "1") == "1",
            })
    fc = P.get("FullyConstrained")
    fully = fc.find("Bool").get("value") == "true" if fc is not None else None
    support = prop_value(P["AttachmentSupport"]) if "AttachmentSupport" in P else (
        prop_value(P["Support"]) if "Support" in P else None)
    mapmode = enum_name("Sketch", "MapMode", prop_value(P["MapMode"])) if "MapMode" in P else None
    return {"ngeo": ngeo, "ncons": len(cons), "dims": dims, "fully": fully,
            "support": support, "mapmode": mapmode,
            "ctypes": collections.Counter(CONS_TYPES.get(int(c.get("Type")), "?") for c in cons)}


def analyze(path):
    root, types, data = load(path)
    rel = path.relative_to(REPO)
    res = {"file": str(rel), "progver": root.get("ProgramVersion"), "objects": [],
           "sketches": [], "sheets": [], "exprs": [], "bodies": [], "flags": [],
           "a2p": [], "counts": collections.Counter(t for t in types.values() if t not in SKIP_TYPES)}
    for name, t in types.items():
        if t in SKIP_TYPES:
            continue
        P = data.get(name, {})
        lab = label(P, name)
        short = t.split("::")[-1]
        for path_, ex in expressions(P):
            res["exprs"].append((lab, path_, ex))
        if "sourceFile" in P:  # A2plus imported part
            res["a2p"].append((name, lab, P["sourceFile"].find("String").get("value")))
            continue
        if t == "Sketcher::SketchObject":
            s = analyze_sketch(P)
            s.update(name=name, label=lab)
            res["sketches"].append(s)
            continue
        if t == "Spreadsheet::Sheet":
            cells = []
            cp = P.get("cells")
            if cp is not None:
                for c in cp.iter("Cell"):
                    cells.append((c.get("address"), c.get("content"), c.get("alias")))
            res["sheets"].append((name, lab, cells))
            continue
        props = {}
        if t == "Part::FeaturePython" and "Proxy" in P and (P["Proxy"].find("Python") is not None) \
                and (P["Proxy"].find("Python").get("module") or "").startswith("Rocket."):
            short = P["Proxy"].find("Python").get("module")
            for k, p in P.items():
                if k in ROCKET_SKIP:
                    continue
                v = prop_value(p)
                if v not in (None, ""):
                    props[k] = v
        else:
            for k in FEATURE_PROPS.get(short, []):
                if k in P:
                    v = enum_name(short, k, prop_value(P[k]))
                    if v not in (None, ""):
                        props[k] = v
            if short.startswith("Feature") and "Proxy" in P:
                pr = P["Proxy"].find("Python")
                if pr is not None and pr.get("module"):
                    props["module"] = pr.get("module")
        if "Suppressed" in P and prop_value(P["Suppressed"]) == "true":
            props["Suppressed"] = "true"
        obj = {"name": name, "label": lab, "type": short, "props": props}
        if t in ("PartDesign::Body", "App::Part"):
            grp = P.get("Group")
            obj["members"] = prop_value(grp) if grp is not None else ""
            res["bodies"].append(obj)
        res["objects"].append(obj)
    # flags
    for s in res["sketches"]:
        if s["ngeo"] == 0:
            res["flags"].append(f"empty sketch `{s['label']}`")
        elif s["fully"] is False:
            res["flags"].append(f"not fully constrained `{s['label']}`")
    if not res["sketches"] and res["counts"].get("Part::Feature"):
        res["flags"].append("no parametric history (only Part::Feature solids)")
    return res


def md_part(r):
    L = [f"# `{r['file']}`", "", f"- FreeCAD version: {r['progver']}",
         f"- Object counts: " + ", ".join(f"{k.split('::')[-1]}×{v}" for k, v in sorted(r["counts"].items())),
         f"- Spreadsheets: {len(r['sheets'])} · Expressions: {len(r['exprs'])} · Sketches: {len(r['sketches'])}"]
    named = sum(1 for s in r["sketches"] for d in s["dims"] if d["name"])
    ndims = sum(len(s["dims"]) for s in r["sketches"])
    L.append(f"- Driving/reference dimensions in sketches: {ndims} (named: {named})")
    if r["flags"]:
        L += ["", "## Flags", ""] + [f"- ⚠ {f}" for f in r["flags"]]
    if r["bodies"]:
        L += ["", "## Bodies / parts", ""]
        for b in r["bodies"]:
            L.append(f"- **{b['label']}** (`{b['name']}`, {b['type']}) " +
                     " ".join(f"{k}={v}" for k, v in b["props"].items()))
    feats = [o for o in r["objects"] if o["type"] not in ("Body", "Part")]
    if feats:
        L += ["", "## Features (document order)", "", "| Name | Label | Type | Key properties |", "|---|---|---|---|"]
        for o in feats:
            pv = "; ".join(f"{k}={v}" for k, v in o["props"].items())
            L.append(f"| `{o['name']}` | {o['label']} | {o['type']} | {pv} |")
    if r["sketches"]:
        L += ["", "## Sketches", "", "| Sketch | Support / map mode | Geo | Cons | Fully constrained | Dimensions |",
              "|---|---|---|---|---|---|"]
        for s in r["sketches"]:
            dims = ", ".join(
                f"{DIM_SHORT[d['type']]}{('['+d['name']+']') if d['name'] else ''}={fmt_num(round(d['value'], 4))}"
                f"{'' if d['driving'] else '(ref)'}{'' if d['active'] else '(off)'}" for d in s["dims"]) or "—"
            fc = {True: "yes", False: "**NO**", None: "?"}[s["fully"]]
            L.append(f"| {s['label']} | {s['support'] or '—'} / {s['mapmode'] or '—'} | {s['ngeo']} | {s['ncons']} | {fc} | {dims} |")
    if r["sheets"]:
        L += ["", "## Spreadsheets", ""]
        for name, lab, cells in r["sheets"]:
            L.append(f"### {lab}")
            L += [f"- {a}: `{c}`" + (f" alias `{al}`" if al else "") for a, c, al in cells]
    if r["exprs"]:
        L += ["", "## Expressions", ""] + [f"- {o}.{p} = `{e}`" for o, p, e in r["exprs"]]
    if r["a2p"]:
        L += ["", "## A2plus imported parts", "", "| Object | Label | Source |", "|---|---|---|"]
        L += [f"| `{n}` | {l} | `{s}` |" for n, l, s in r["a2p"]]
    return "\n".join(L) + "\n"


def md_summary(results):
    a2p_sources = set()
    for r in results:
        for _, _, s in r["a2p"]:
            a2p_sources.add(str((ASSEMBLY.parent / s).resolve().relative_to(REPO)))
    L = ["# CAD inventory — summary", "",
         "Generated by `scripts/cad_inventory.py` (read-only parse of `Document.xml`). Do not edit by hand.", "",
         "| File | In assembly | Sketches | Not fully constr. | Empty | Dims | Named dims | Sheets | Exprs | Flags |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    tot = collections.Counter()
    for r in results:
        sk = r["sketches"]
        nf = sum(1 for s in sk if s["fully"] is False)
        ne = sum(1 for s in sk if s["ngeo"] == 0)
        nd = sum(len(s["dims"]) for s in sk)
        nn = sum(1 for s in sk for d in s["dims"] if d["name"])
        ina = "assembly" if r["a2p"] else ("yes" if r["file"] in a2p_sources else "**no**")
        page = "parts/" + r["file"].replace("/", "__").replace(".FCStd", ".md")
        L.append(f"| [{r['file']}]({page}) | {ina} | {len(sk)} | {nf} | {ne} | {nd} | {nn} | {len(r['sheets'])} | {len(r['exprs'])} | {len(r['flags'])} |")
        tot.update(sketches=len(sk), nf=nf, ne=ne, nd=nd, nn=nn, sh=len(r["sheets"]), ex=len(r["exprs"]))
    L.append(f"| **total** | | {tot['sketches']} | {tot['nf']} | {tot['ne']} | {tot['nd']} | {tot['nn']} | {tot['sh']} | {tot['ex']} | |")
    # default labels
    dl = sum(1 for r in results for s in r["sketches"] if DEFAULT_LABEL.match(s["label"]))
    dlf = sum(1 for r in results for o in r["objects"] if DEFAULT_LABEL.match(o["label"]))
    nob = sum(1 for r in results for o in r["objects"])
    L += ["", f"- Sketches with default labels: {dl}/{tot['sketches']}",
          f"- Other objects with default labels: {dlf}/{nob}"]
    rocket = [(r["file"], o) for r in results for o in r["objects"] if o["type"].startswith("Rocket.")]
    if rocket:
        L += ["", "## Rocket workbench features", "",
              "These parts start from a Rocket-workbench feature (addon must be installed in FreeCAD).", ""]
        for f, o in rocket:
            L.append(f"- `{f}` → **{o['label']}** ({o['type']})")
    return "\n".join(L) + "\n"


def main(argv):
    if argv:
        for a in argv:
            print(md_part(analyze(pathlib.Path(a).resolve())))
        return
    files = sorted(p for p in REPO.rglob("*.FCStd") if ".git" not in p.parts)
    results = [analyze(p) for p in files]
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    for r in results:
        (OUT / "parts" / (r["file"].replace("/", "__").replace(".FCStd", ".md"))).write_text(md_part(r))
    (OUT / "summary.md").write_text(md_summary(results))
    print(f"wrote {len(results)} part pages + summary to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main(sys.argv[1:])
