# Platform snapshot — pre drawer-experiment v2

Recoverable snapshot of the full stable platform **before** starting the drawer
deployment-calibration experiment (v2). Nothing is deleted; this is a safety point.

## Repo state at snapshot
- **commit**: `ade8e77cee979f12ddfe9c55a7259409c0f540bc` (see `git_head.txt`)
- **branch**: `main`
- **working tree**: CLEAN (no dirty files; no `working_tree.patch` needed) — see `git_status.txt`
- **recovery tag**: `archive/pre_drawer_experiment_v2_20260702_204058`
- **snapshot time**: 2026-07-02 20:40 (local)
- **remote**: https://github.com/ysdjy/paper

## Why this snapshot
Next phase switches from "extend the general test platform" to "turn the current
stable scene + skill backend into a reproducible, leakage-free experiment system
for the paper's core drawer result". v2 work adds new versioned files and does not
overwrite v1. This snapshot lets us restore the full platform if any v2 change
regresses it.

## Contents
- `git_head.txt` — commit + branch
- `git_status.txt` — working-tree status at snapshot
- `file_hashes.json` — SHA256 + size of 108 key in-repo files (scene manifests,
  grasp poses, waypoints, skills/, drawer configs, adapters, deployment_calibration/)
- `external_assets.json` — 10 external USD assets (Cabinet_44853, CoffeeMachine,
  Microwave, Fridge, Dishwasher, Franka, Stand, L_Desk, Sektion) with abs path,
  size, mtime, SHA256. These live in the shared IsaacLab tree, NOT in git.
- `runtime_commands.md` — verified run commands for the full platform + skills.

## Recovery
- Full repo state: `git -C projects/paper checkout archive/pre_drawer_experiment_v2_20260702_204058`
  (or `git checkout ade8e77 -- <path>` for a single file).
- Verify any restored file against `file_hashes.json`.
- External USDs are unchanged shared binaries; verify against `external_assets.json`.

## Scope note
- No files were moved/deleted to create this snapshot. It records state only.
- v2 experiment files are added alongside v1 (v1 kept as legacy).
