# Lyskamm Rocket Project 🚀

**Lyskamm** is an amateur rocket that is entirely **3D-printed** and **assembled without glue**.
Each part is designed to be replaced or upgraded on its own; sections are joined by sliding fits and
radial M2 screws in heat-set inserts.

- Single stage, Ø60 mm, ≈ 870 mm long, 4 upper + 4 lower fins
- Motors: TSP F35 (Ø29, with adapter) — later EXP G115 (Ø35)
- 4 carbon rods (Ø5/3) run through the aft airframe for bending stiffness
- Parachute Ø1 m, deployed at apogee by the altimeter
- Printed on a Bambu Lab P1S: ASA-CF (airframe), PETG-CF (fins)

## Repository

```
lyskamm/
├── parameters.FCStd     # master parameters (spreadsheet Params), linked by every part
├── lyskamm.ork          # OpenRocket model (OpenRocket 24.12)
├── assemblies/
│   ├── lyskamm_asm.FCStd    # full rocket, FreeCAD built-in Assembly (current)
│   └── lyskamm.FCStd        # old A2plus assembly (legacy)
├── components/          # one folder per part or subsystem: .FCStd source + .step + .3mf print file
├── docs/
│   ├── inventory/       # CAD/OpenRocket inventory, weighed masses, measured lengths
│   ├── proposal/        # parameters, naming conventions, repo structure, OpenRocket vs CAD
│   └── analysis/        # flight simulations
├── scripts/             # inventory, assembly build/CG, print settings, OpenRocket simulation
└── launch_pad/          # early concept (separate project)
```

## Tools

- **FreeCAD 1.1.3** with the Rocket and built-in Assembly workbenches. Open a part or the assembly and
  `parameters.FCStd` is loaded automatically.
- **OpenRocket 24.12** for stability and flight performance; headless runs with `scripts/openrocket/Sim.java`
  (see the file header).
- **Bambu Studio** for slicing; `scripts/print_settings.py` lists the settings stored in the `.3mf` files.

Design rules, conventions and workflows are documented in `CLAUDE.md`; the project setup plan and its log are
in `PLAN.md`.

## License

This project is released for educational and research purposes.
A specific license will be defined in future updates.
