# File ownership — v2

Prevents two Claude instances from silently clobbering each other. Repo root = `projects/paper/`.

## Claude A (runtime / simulation) owns — writes here
- `scene/`
- `franka_v1_skill_lab/`
- `franka_skill_state_machine/`
- `deployment_calibration/adapters/`
- `deployment_calibration/data_generation/`
- `deployment_calibration/configs/`
- `deployment_calibration/data/`            (episodes.jsonl + trajectories + metadata — A produces)

## Claude B (offline evaluation / modeling) owns — writes here
- `deployment_calibration/offline_v2/`
- `deployment_calibration/models_v2/`
- `deployment_calibration/evaluation/offline_v2/`
- `deployment_calibration/tests/offline_v2/`
- `docs/offline_v2/`

B reads A's `deployment_calibration/data/` READ-ONLY. B does NOT edit A's directories.
A does NOT edit B's directories.

## Shared / read-mostly (existing v2 code)
- `deployment_calibration/evaluation/*_v2.py`, `deployment_calibration/baselines/*_v2.py`,
  `deployment_calibration/tests/*_v2.py` (top level): current shared v2 code. To avoid conflicts, B
  should ADD new offline work under the `offline_v2/` subdirs above rather than editing these in place;
  coordinate before changing a shared file.

## FROZEN — neither A nor B may silently modify
- `deployment_calibration/contracts/`   (episode_schema_v2.py, episode_schema.py)
- `docs/paper_experiment_contract_v2.md`
- `docs/paper_experiment_contract_v1.md`

### Changing the contract
Any change to the frozen contract requires FIRST creating a change-request doc:
`docs/contract_change_request_<YYYYMMDD_HHMMSS>.md` stating: what field/semantics change, why, impact on
existing data (`open_drawer_v2` runs), migration/versioning plan (bump CONTRACT_VERSION, do NOT rewrite
old data). Only after that doc is committed may the contract be edited — and the version string must bump.

## Branches / worktrees
- A: branch `experiment/runtime-selection-v2` in `projects/paper/`.
- B: branch `experiment/offline-eval-v2` in `../paper_offline_v2` (separate worktree, adjacent to paper/,
  NOT nested inside it).
- Shared coordination docs (`docs/parallel_work_plan_v2.md`, `docs/runtime_offline_handoff_v2.md`,
  `docs/file_ownership_v2.md`) live on both branches (branched from the same prep commit).
