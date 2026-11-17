# Repo structure — proposal (Phase 2.3)

Status: **draft for review** (2026-11-17). Names follow `conventions.md` (approved). Nothing is moved by this
document; every move below is a separate, approved step.

## Target layout

```
lyskamm/
├── CLAUDE.md, PLAN.md, README.md
├── parameters.FCStd              # master parameters (spreadsheet Params), created in Phase 3
├── lyskamm.ork                   # OpenRocket model
├── assemblies/lyskamm.FCStd      # A2plus assembly
├── components/                   # one folder per part, or per subsystem when parts belong together
│   ├── nose_cone/                nose_cone
│   ├── ebay/                     ebay_fuselage, ebay_fuselage_logo, upper_closure, mid_closure,
│   │                             lower_closure, aluminium_rod, connector_insert
│   ├── parachute_bay/            parachute_bay, piston, upper_connector_seat
│   ├── lower_chute_retainer/     lower_chute_retainer
│   ├── mid_fuselage/             mid_fuselage
│   ├── upper_fins/               upper_fin
│   ├── carbon_rod/               carbon_rod
│   ├── upper_motor_retainer/     upper_motor_retainer
│   ├── lower_fuselage/           lower_fuselage
│   ├── lower_fins/               lower_fins
│   ├── rear_transition/          rear_transition
│   └── logo/                     logo_flat, logo_projected, logo_inserts (+ logo_*.step pieces)
├── launch_pad/                   # deferred, untouched
├── docs/  inventory/, proposal/
└── scripts/
```

Decisions in this layout:

- **Exports stay next to their source** (`<part>.FCStd`, `<part>.step`, `<part>.3mf` in the same folder), as now. A separate
  `exports/` folder would add moves and break `scripts/print_settings.py` paths for no real gain.
- **`parameters.FCStd` at the repo root.** Parts reference it with a relative path (`../../parameters.FCStd`
  from a part folder), so it must not move after Phase 3 starts.
- **Legacy files are deleted** (Lorenzo, 2026-11-17). Git keeps them: `git log --all -- <path>` finds them and
  `git checkout <commit> -- <path>` brings one back. `docs/inventory/file_status.md` records what was removed.
- Folders use the snake_case rule too: `e-bay` → `ebay`, `parachute_fuselage_seat` → `parachute_bay`,
  `lower_fuselage_w_motor_mount` → `lower_fuselage`.

## Git LFS (open question 5) — recommendation: **drop LFS**

Findings:
- `git-lfs` is not installed, and **no file in the history is an LFS pointer**: every `.FCStd`, `.step`, `.3mf`
  is stored as a normal git object.
- `.gitattributes` asks LFS for `*.3mf`, `*.step`, `*.fcstd` — but `*.fcstd` (lower case) does not even match
  `*.FCStd`. On a machine with LFS installed, the next commit of a `.3mf`/`.step` would silently switch it to LFS.
- Repo size: `.git` 93 MB, largest file 5.5 MB (GitHub's limit is 100 MB per file; LFS is not needed).

Proposal: replace `.gitattributes` with

```
*.FCStd binary
*.3mf   binary
*.step  -diff
```

(`binary` = no text diffs or merges; STEP is text but diffs are useless). If the repo grows a lot later, LFS can be
reintroduced with `git lfs migrate`.

## Move checklist

Each line is one approved step (one commit). After every step: regenerate `docs/inventory/` and fix paths in
`scripts/shared_dims.py` (`EXTRA_ACTIVE`) and docs.

### A. Safe now — files not referenced by the assembly — ✅ done 2026-11-17 (A1–A11, A13; A12 moves with group B)

| # | From | To | Why |
|---|---|---|---|
| A1 | `.gitattributes` | new content above | Drop LFS (if approved) |
| A2 | `components/wall.FCStd` | delete | Legacy test |
| A3 | `components/e-bay/lower_closure.{FCStd,step}` | delete | Replaced by v2 |
| A4 | `components/parachute_fuselage_seat/parachute_fuselage_seat.{FCStd,step}` | delete | Replaced by v2 |
| A5 | `components/parachute_fuselage_seat/upper_closure.{FCStd,step,3mf}` | delete | Old widget of v1 seat |
| A6 | `components/parachute_fuselage_seat/parachute_fuselage_seat.3mf` | `…/parachute_fuselage_seat_v2.3mf` (temporary name until B) | It is the print of **v2** |
| A7 | `components/parachute_fuselage_seat/parachute_fuselage_seat_v2.3mf` (holds `lower_closure_v2`, Dec 2025) | delete | Older duplicate; the current print of `lower_closure_v2` is `e-bay/lower_closure.3mf` (Jan 2026) |
| A8 | `components/lower_chute_retainer/low_chute_ret_test.3mf` (holds v1 `parachute_fuselage_seat`) | delete | Old test print |
| A9 | `components/upper_motor_retainer/upper_retainer` (no extension) | delete | Older STEP export (2025-10-03) of `upper_retainer`; `.step` is newer (2025-10-15) |
| A10 | `components/logo.FCStd` | `components/logo/logo_flat.FCStd` | Lorenzo's request |
| A11 | `components/logo/logo.FCStd`, `logo_red.{FCStd,step}` | `logo_projected.FCStd`, `logo_inserts.{FCStd,step}` | English names |
| A12 | `components/e-bay/fuselage_w_logo_v2.{FCStd,step}` | `components/ebay/ebay_fuselage_logo_fitcheck.*` (with B) | Fit-check file, keep |
| A13 | `components/.keep` | delete | Folder is no longer empty |

⚠ A6/A7 swap names around: done in one commit to avoid overwriting.

### B. Renames of assembly parts — **blocked by A2plus** (open question 8)

The assembly stores each part's path (`sourceFile`, e.g. `./../components/e-bay/fuselage.FCStd`). Renaming a
file breaks that link until the path is updated, which needs A2plus (not installed in FreeCAD 1.1.3). So these
wait until A2plus is installed (or the assembly is migrated, open question 4), and are best done together with
each part's Phase 3 rework:

| From | To |
|---|---|
| `components/e-bay/` (folder) | `components/ebay/` |
| `e-bay/fuselage` | `ebay/ebay_fuselage` |
| `e-bay/fuselage_w_logo` (not in assembly) | `ebay/ebay_fuselage_logo` |
| `e-bay/alluminium_rod` | `ebay/aluminium_rod` |
| `e-bay/lower_closure_v2` (+ `lower_closure.3mf`) | `ebay/lower_closure` |
| `e-bay/presa_elettrica` (not in assembly) | `ebay/connector_insert` |
| `parachute_fuselage_seat/` (folder) | `parachute_bay/` |
| `parachute_fuselage_seat_v2` | `parachute_bay/parachute_bay` |
| `pistone` (not in assembly) | `parachute_bay/piston` |
| `sede_presa_elettrica_sup` | `parachute_bay/upper_connector_seat` |
| `lower_chute_retainer/low_chute_ret` | `lower_chute_retainer/lower_chute_retainer` |
| `lower_fuselage_w_motor_mount/lower_fuselage_v1` | `lower_fuselage/lower_fuselage` (+ `.3mf` already named `lower_fuselage`) |
| `upper_motor_retainer/upper_retainer` | `upper_motor_retainer/upper_motor_retainer` |
| `lower_fins/lower_fin.{3mf,step}` | `lower_fins.{3mf,step}` (match the `.FCStd`) |

`.step` and `.3mf` files are renamed with their `.FCStd`. Note: a `.3mf` keeps the old object names inside
(e.g. `lower_closure_v2.step`) until it is re-exported — harmless.

## Review answers (2026-11-17)

1. LFS: explained; recommendation = drop. ❓ awaiting Lorenzo's OK.
2. Legacy files: **delete** (no `archive/`).
3. Exports next to their source: **OK**.
4. Group B: Lorenzo will install A2plus (and the Rocket workbench) in FreeCAD 1.1.3.
