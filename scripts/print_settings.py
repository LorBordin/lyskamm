#!/usr/bin/env python3
"""Read-only: extract Bambu Studio print settings from every .3mf in the repo.

Usage: python3 scripts/print_settings.py            -> prints a markdown table
       python3 scripts/print_settings.py --json     -> JSON (used by other scripts)
Global values come from Metadata/project_settings.config; per-object overrides
from Metadata/model_settings.config. Filament = the one assigned to each object's extruder.
"""
import json
import pathlib
import sys
import xml.etree.ElementTree as ET
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
KEYS = ["layer_height", "wall_loops", "sparse_infill_density", "sparse_infill_pattern",
        "top_shell_layers", "bottom_shell_layers"]


def read(path):
    z = zipfile.ZipFile(path)
    names = z.namelist()
    if "Metadata/project_settings.config" not in names:
        return None
    g = json.loads(z.read("Metadata/project_settings.config"))
    objs = []
    if "Metadata/model_settings.config" in names:
        root = ET.fromstring(z.read("Metadata/model_settings.config"))
        for o in root.findall("object"):
            md = {m.get("key"): m.get("value") for m in o.findall("metadata") if m.get("key")}
            objs.append(md)
    ftypes, fdens = g.get("filament_type", []), g.get("filament_density", [])
    out = []
    for md in objs or [{}]:
        ext = int(md.get("extruder", 1)) - 1
        row = {"file": str(path.relative_to(REPO)), "object": md.get("name", ""),
               "filament": ftypes[ext] if ext < len(ftypes) else "?",
               "density_g_cm3": fdens[ext] if ext < len(fdens) else "?",
               "profile": g.get("print_settings_id"), "printer": g.get("printer_model")}
        for k in KEYS:
            row[k] = md.get(k, g.get(k))
            if k in md:
                row[k] = f"{md[k]}*"  # * = per-object override
        out.append(row)
    return out


def main():
    rows = []
    for p in sorted(REPO.rglob("*.3mf")):
        if ".git" in p.parts:
            continue
        r = read(p)
        if r:
            rows += r
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
        return
    print("| 3MF | Object | Filament (g/cm³) | Layer | Walls | Infill | Pattern | Top/Bottom |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['file']}` | {r['object']} | {r['filament']} ({r['density_g_cm3']}) | {r['layer_height']} | "
              f"{r['wall_loops']} | {r['sparse_infill_density']} | {r['sparse_infill_pattern']} | "
              f"{r['top_shell_layers']}/{r['bottom_shell_layers']} |")


if __name__ == "__main__":
    main()
