#!/usr/bin/env python3
"""Shared-dimension analysis (read-only) -> docs/inventory/shared_dims.md

Inputs: Document.xml of each .FCStd (via cad_inventory.analyze), docs/inventory/fc_data.json
(cylindrical-face diameters, from cad_inventory_fc.py) and lyskamm.ork.
Lists every length value used in >= 2 *active* parts (= parts imported in the A2plus assembly).
"""
import collections
import json
import pathlib
import sys
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cad_inventory as ci  # noqa: E402

REPO = ci.REPO
OUT = REPO / "docs" / "inventory" / "shared_dims.md"


# Parts that fly but are not imported in the A2plus assembly (confirmed by Lorenzo)
EXTRA_ACTIVE = [
    "components/e-bay/fuselage_w_logo.FCStd",
    "components/parachute_fuselage_seat/pistone.FCStd",
]

# Meanings reviewed with Lorenzo in step 1.7 (2026-11-17). All confirmed unless marked open.
MEANINGS = {
    60: "Body tube outer Ø",
    56: "Body tube inner Ø",
    55.8: "Outer Ø of parts that slide into the 56 bore (sliding fit, −0.2 on Ø)",
    2: "Body wall thickness (60/56); other uses coincidental",
    5: "Outer Ø of carbon tubes and aluminium bar (5/3 tubes); many Pad/Pocket lengths are coincidental",
    3: "Inner Ø of carbon tubes / aluminium bar; also fin root thickness; other uses coincidental",
    23.428: "Radius from rocket axis to carbon-tube centres (23.4279)",
    3.1: "MISTAKE: oversized M2 insert holes (should be 2.6): low_chute_ret Pocket004–007, upper_retainer sede_viti_*. Not a mistake: rear_transition Pocket002 (through) = screw for the lower motor retainer; upper_closure Pocket002 Ø3.2 = M3 screw securing the aluminium bar (other end)",
    2.2: "M2 screw clearance hole Ø (outer part, e.g. fuselages)",
    2.6: "M2 heat-set insert hole Ø (correct value)",
    3.75: "Axial distance of radial M2 holes from the part's end face",
    7.5: "Overlap length of joints (coupling / reinforcement ring)",
    6.15: "Electrical-connector slot, short side (`sede_presa_elettrica`, rounded corners; for `presa_elettrica`; deployment system, will be redesigned)",
    9.6: "Electrical-connector slot, long side (`sede_presa_elettrica`, with 6.15)",
    7: "Outer Ø of round bosses around carbon tubes (`sede_tondo*`, `inserto_tondo`); also M3 counterbore in lower_closure_v2",
    7.2: "Outer Ø of round bosses around carbon tubes in parachute seat parts",
    5.15: "Hole for 5 mm tube (+0.15)",
    5.2: "Hole for 5 mm tube (+0.2)",
    5.25: "Seat for 5 mm aluminium bar in closures (`sede_tondo_alluminio`, +0.25)",
    35.2: "Motor tube inner Ø (EXP G115, 35 mm; TSP F35 29 mm needs an adapter)",
    37: "Motor tube outer Ø",
    3.5: "M3 clearance hole; in lower_closure_v2 (`sede_vite_1` + Ø7×3 counterbore) an M3 screw secures the aluminium bar to the closure. Also low_chute_ret Pocket003, pistone Pocket001",
}


def active_parts():
    res = ci.analyze(ci.ASSEMBLY)
    act = {str((ci.ASSEMBLY.parent / s).resolve().relative_to(REPO)) for _, _, s in res["a2p"]}
    return sorted(act | set(EXTRA_ACTIVE))


def ork_values():
    with zipfile.ZipFile(REPO / "lyskamm.ork") as z:
        root = ET.fromstring(z.read(next(n for n in z.namelist() if n.endswith(".ork"))))
    vals = collections.defaultdict(set)
    for c in root.iter():
        name = c.findtext("name")
        if name is None:
            continue
        for k in ("length", "thickness", "rootchord", "tipchord", "sweeplength", "height"):
            t = c.findtext(k)
            if t:
                vals[round(float(t.split()[-1]) * 1000, 3)].add(f"{name}.{k}")
        for k in ("radius", "aftradius", "foreradius"):
            t = c.findtext(k)
            if t:
                r = float(t.split()[-1]) * 1000
                vals[round(2 * r, 3)].add(f"{name}.Ø({k}×2)")
                th = c.findtext("thickness")
                if th and k in ("radius",):
                    vals[round(2 * r - 2 * float(th) * 1000, 3)].add(f"{name}.inner Ø")
    return vals


def main():
    act = active_parts()
    short = {p: pathlib.Path(p).stem for p in act}
    uses = collections.defaultdict(lambda: collections.defaultdict(list))  # value -> part -> [where]
    for p in act:
        r = ci.analyze(REPO / p)
        for s in r["sketches"]:
            for d in s["dims"]:
                if d["type"] == 9:  # angles handled separately
                    continue
                v = round(d["value"], 3)
                if v <= 0:
                    continue
                uses[v][p].append(f"{s['label']}.{ci.DIM_SHORT[d['type']]}")
        for o in r["objects"]:
            for k in ("Length", "Length2", "Radius", "Size", "Diameter", "Height", "RootChord", "TipChord",
                      "SweepLength", "Span", "Thickness", "RootThickness", "AftDiameter", "ForeDiameter"):
                if k in o["props"]:
                    try:
                        v = round(float(o["props"][k]), 3)
                    except ValueError:
                        continue
                    if v > 0 and not (k == "Length2" and o["type"] in ("Pad", "Pocket")):
                        uses[v][p].append(f"{o['label']}.{k}")
    fc = json.loads((REPO / "docs" / "inventory" / "fc_data.json").read_text())
    for rec in fc:
        if rec["file"] not in act:
            continue
        for b in rec["bodies"]:
            if b["stored"]:
                for dval in b["stored"]["cyl_diameters"]:
                    uses[round(dval, 3)][rec["file"]].append("cyl-face Ø")
    ork = ork_values()
    L = ["# Shared dimensions between active parts", "",
         "Generated by `scripts/shared_dims.py`. *Active* = imported in `assemblies/lyskamm.FCStd` + `EXTRA_ACTIVE` (flying parts missing from the assembly).",
         "A value is listed when it appears in ≥ 2 active parts (sketch dimension, feature length, or",
         "measured cylindrical-face diameter). Equal numbers are **candidates** only: the *Meaning* column",
         "was reviewed with Lorenzo in step 1.7 (`MEANINGS` in the script); '?' = still open.", "",
         f"Active parts ({len(act)}): " + ", ".join(f"`{short[p]}`" for p in act), "",
         "| Value (mm) | # parts | Parts → where used | Matches .ork | Meaning |",
         "|---|---|---|---|---|"]
    for v in sorted(uses, key=lambda x: (-len(uses[x]), x)):
        parts = uses[v]
        if len(parts) < 2:
            continue
        where = "<br>".join(
            f"`{short[p]}`: " + ", ".join(sorted(set(w))[:6]) + (" …" if len(set(w)) > 6 else "")
            for p, w in sorted(parts.items()))
        om = ", ".join(sorted(ork.get(v, [])))
        L.append(f"| {ci.fmt_num(v)} | {len(parts)} | {where} | {om} | {MEANINGS.get(v, '')} |")
    # near-equal values (possible inconsistencies, e.g. 55.8 vs 56)
    vals = sorted(v for v in uses if v >= 1)
    near = [(a, b) for a, b in zip(vals, vals[1:]) if 0 < b - a <= 0.3 and a >= 10]
    L += ["", "## Near-equal values (possible fit clearances or inconsistencies)", "",
          "Pairs of values within 0.3 mm of each other, ≥ 10 mm (interface fits).", ""]
    for a, b in near:
        pa = ", ".join(sorted(short[p] for p in uses[a]))
        pb = ", ".join(sorted(short[p] for p in uses[b]))
        L.append(f"- {ci.fmt_num(a)} ({pa}) ↔ {ci.fmt_num(b)} ({pb}) — Δ {b - a:.3f}")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
