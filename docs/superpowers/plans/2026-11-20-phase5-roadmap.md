# Phase 5 Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the finished setup plan with a short Phase 5 roadmap toward the first F35 flight, archive the old plan, and point `CLAUDE.md` at the new structure.

**Architecture:** Documentation only. Old `PLAN.md` moves unchanged (git mv) to `docs/archive/setup_plan.md`; a new `PLAN.md` holds the roadmap; `CLAUDE.md`/`README.md` references are updated. No `.FCStd` file is touched.

**Tech Stack:** Markdown, git.

**Spec:** `docs/superpowers/specs/2026-11-20-phase5-roadmap-design.md`

## Global Constraints
- Goal line verbatim: **first flight on the TSP F35**, no deadline; airframe **designed and checked for the EXP G115 plus a margin** (margin from Lorenzo in the structural session).
- Anti-goal: no "extraordinary changes" to the airframe unless unavoidable.
- Stage ids and names exactly: 0 Loads · 1 S / D / L / M (parallel) · 2 S pass 2 · 3 I Integration · 4 F First flight readiness.
- Project docs path (created only when a project starts): `docs/projects/<id>_<name>/design.md` + `plan.md`.
- Archive keeps the old text unchanged (old Qn numbers stay valid).
- English, mm, one commit at the end with the attribution line.

## Review Focus
1. Old references "PLAN Qn" / "PLAN.md Phase 3" / "PLAN step 3.0" must still resolve → they point to `docs/archive/setup_plan.md` (CLAUDE.md edited; scripts are historical, left as they are because the archive keeps the numbering — note it in the archive header).
2. Every Phase 5 backlog item of the old plan appears in the roadmap (spec mapping table) — none dropped.
3. Open questions still open in the old Status (2, 9, 14, 15, 16 remainder, 17) are carried into the roadmap.
4. `git mv` preserves history of the old plan (`git log --follow docs/archive/setup_plan.md` shows old commits).
5. CLAUDE.md session protocol no longer says "Phases 0–4 … update this section" (4.4 done).

---

### Task 1: Archive the setup plan

**Files:** Move: `PLAN.md` → `docs/archive/setup_plan.md`

- [ ] **Step 1:** `git mv PLAN.md docs/archive/setup_plan.md`; mark 4.4 `[x]` (done 2026-11-20); add one header line under the title: "Archived 2026-11-20 — setup (Phases 0–4) finished; Phase 5 continues in `PLAN.md` (roadmap). Qn numbers and step numbers referenced elsewhere point here."
- [ ] **Step 2: Verify:** `git status` shows a rename; `grep -c "### 20" docs/archive/setup_plan.md` = 9 sessions.

### Task 2: Write the roadmap `PLAN.md`

**Files:** Create: `PLAN.md`

Sections, in order:
- [ ] **Step 1:** Title, goal (verbatim from Global Constraints), anti-goal, session protocol pointer ("read Status first, then the active project's `docs/projects/...`"), link to archive and spec.
- [ ] **Step 2:** Stages 0–4 as a table/list with: id, scope (from the spec), depends on, status (`not started`), project folder (`—` until started).
- [ ] **Step 3:** "Backlog items by project" — the spec's mapping table, each old Phase 5 item listed under its project (S, D, L, M, I, Housekeeping), with the facts from the old backlog (Ø values, part names, 200 g, 2 m rail, stability ≈0.1 cal without extra mass).
- [ ] **Step 4:** Status (current: stage 0 next; next step: Loads with `Sim.java` F35 + G115) and open questions carried over: old Q2, Q9 (AppImage workaround found), Q14, Q15, Q16 remainder (fin T-foot 0.04 mm, alu rod 0.13 mm³), Q17, plus new: G115 margin (Lorenzo, S session), deployment method (black powder, D session), electronics choice (D).
- [ ] **Step 5:** Session log starting with "2026-11-20 — Session 9 (cont.)": roadmap created, design approved, option A.
- [ ] **Step 6: Verify** every item of the old Phase 5 backlog appears: `sed -n '/^## Phase 5/,/^## Launch pad/p' docs/archive/setup_plan.md` read side by side with the new file.

### Task 3: Update `CLAUDE.md` and `README.md`

**Files:** Modify: `CLAUDE.md` (session protocol lines 14–21, repo map line 61, line 93, lines 126/133/145/158), `README.md:40`, `docs/proposal/parameters.md:95`

- [ ] **Step 1:** Session protocol: start = read `CLAUDE.md`, `PLAN.md` Status, and the active project's `docs/projects/<id>_<name>/` docs; end = update `PLAN.md` Status + Session log (and the project's plan); `PLAN.md` = Phase 5 roadmap; setup history in `docs/archive/setup_plan.md`; new projects get design + plan docs when they start.
- [ ] **Step 2:** Repo map: add `docs/archive/` (setup plan), `docs/projects/` (per-project design/plan), `docs/superpowers/` (specs/plans). `PLAN.md Phase 3` / `PLAN Qn` → `setup_plan.md` (Phase 3 / Qn). Q16 line: drop the carbon-rod sliver (resolved). Launch-pad line: now part of project L.
- [ ] **Step 3:** `README.md:40`: "`PLAN.md` holds the roadmap; the setup history is in `docs/archive/setup_plan.md`". `parameters.md:95`: "Launch pad: project L in `PLAN.md`".
- [ ] **Step 4: Verify:** `grep -n "PLAN" CLAUDE.md README.md docs/proposal/parameters.md` — every hit either means the roadmap or names `setup_plan.md`.

### Task 4: Commit

- [ ] **Step 1:** `git add -A PLAN.md docs/archive docs/superpowers CLAUDE.md README.md docs/proposal/parameters.md`; `git commit` "Phase 5 roadmap: archive setup plan, new PLAN.md, CLAUDE.md session protocol (4.4)" + attribution.
- [ ] **Step 2: Verify:** `git status --short` empty; `git log --follow --oneline docs/archive/setup_plan.md | wc -l` > 1.
