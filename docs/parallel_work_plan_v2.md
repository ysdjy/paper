# Parallel work plan — v2

Two (optionally three) Claude instances work in parallel on the drawer deployment-calibration paper.
Isaac/GPU work and offline analysis are separated so they never block each other.

## Claude A — runtime / simulation (THIS instance)
Owns everything that needs Isaac Sim / GPU and the real data it produces.
- Isaac scene, skills, member-aware adapter, reset invariants, damping injection + runtime verification.
- Matched candidate bank (same candidate θ set across damping levels so groups are comparable).
- Real episode data generation (capability map, pilot, formal sessions) and GPU experiments.
- Branch: `experiment/runtime-selection-v2`, working dir: `projects/paper/`.
- Produces `deployment_calibration/data/<run_id>/episodes.jsonl` (+ trajectories, metadata).

## Claude B — offline evaluation / modeling
Owns everything that runs on the produced data WITHOUT Isaac (pure numpy/torch, CPU).
- Offline data validation, AUROC correction (proper CI / small-N handling), held-out D3 (predict hidden
  state on unseen sessions), session-split audit, history models, Oracle baselines, bootstrap CIs,
  regret computation, automated report generation.
- Branch: `experiment/offline-eval-v2`, working dir: `../paper_offline_v2` (separate git worktree).
- Consumes A's `episodes.jsonl`; NEVER launches Isaac; NEVER edits runtime code.

## Claude C — scientific audit (optional, later)
- Independent audit of scientific validity + paper results; reviews leakage, split integrity, metric
  definitions, claims vs evidence. Does NOT modify main code during this phase.

## Coordination rules
- File ownership: see `docs/file_ownership_v2.md`. Neither A nor B silently edits the other's tree.
- Contract (`deployment_calibration/contracts/` + `docs/paper_experiment_contract_v2.md`) is frozen; any
  change requires a `contract_change_request` doc first (see file_ownership_v2.md).
- Data flow: A commits `episodes.jsonl` under `deployment_calibration/data/<run_id>/`; B reads it read-only
  and writes results under `deployment_calibration/{offline_v2,models_v2,evaluation/offline_v2,tests/offline_v2}/`
  and `docs/offline_v2/`.
- Handoff spec (paths, semantics, model-legal vs oracle-only fields): `docs/runtime_offline_handoff_v2.md`.
- Current status: pilot complete (MODIFY, borderline-GO). No 432-episode formal run and no selection
  re-pilot started yet — awaiting user go-ahead.
