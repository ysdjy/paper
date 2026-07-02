# Paper dependency freeze — v2

Goal: the **formal paper drawer run-chain** must not silently change because another project
edits shared Python config. This documents what is in-repo/frozen, what is external, what is
frozen, and which modules affect the formal result.

## 1. Fully in-repo (Python code, no external project)
- `franka_skill_state_machine/` — skills (open/close drawer, coffee, door, grasp, place), runtime,
  state_machine, executor. **All execution logic is in this repo.**
- `deployment_calibration/` — contracts, adapters, generators, baselines, evaluation (paper method).
- `scene/stackpkg/franka/` — **private copies** of `custom_drawer_config.py`, `microwave_door_config.py`,
  `stack_joint_policy_env_cfg.py`, `stack_joint_pos_env_cfg.py`. Verified to contain every symbol the
  run-chain imports (CABINET_USD_SCALE, DRAWER_TARGETS, FUNCTIONAL_DRAWERS, DEFAULT_TARGET,
  COFFEE_LEVER_JOINT/LINK, DOOR_*, HINGE_JOINT, *_SUCCESS_ANGLE, HANDLE_OFFSET_LOCAL, ...).
- `scene/paper_scene_v2/` — **frozen scene facts** (mechanism registry, handle poses, object states,
  external asset hashes, version). Captured live + hash-validated.

## 2. Frozen scene facts the paper depends on (single source)
The formal run-chain sources **mechanism identity** (member → articulation → joint → link → handle
local pose → baseline actuator damping) from:

    scene/paper_scene_v2/mechanism_registry.json   (5 mechanisms, member-aware)
    scene/paper_scene_v2/grasp_poses.json          (handle poses, single source)

These are in-repo, versioned (`version.json` records source_commit), and hash-checkable. A change in
the shared config does NOT silently change these — they are frozen snapshots. `validate_paper_scene_v2.py`
enforces single-source handle consistency; re-exporting + diffing detects drift.

Frozen mechanism table (baseline):

| drawer | member | joint | link | baseline damping (z_secret axis) |
|---|---|---|---|---|
| top_drawer | cabinet | joint_0 | link_0 | 3.0 |
| middle_drawer | cabinet | joint_2 | link_2 | 3.0 |
| bottom_drawer | cabinet | joint_1 | link_1 | 3.0 |
| sektion_top_drawer | sektion_cabinet | drawer_top_joint | drawer_handle_top | 3.0 |
| sektion_bottom_drawer | sektion_cabinet | drawer_bottom_joint | drawer_handle_bottom | 3.0 |

## 3. External USD (referenced, not in git; hashed)
Recorded with abs path + size + mtime + SHA256 in
`scene/paper_scene_v2/external_assets.json` and `archive/platform_snapshot_pre_drawer_v2/external_assets.json`:
- `simv2/USD/Cabinet_44853/configuration/cabinet_{base,physics}.usd` (desktop cabinet)
- `SapienAssetPipeline/usd_assets/IsaacProps/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`
- `Connection/.../FrankaEmika/panda_instanceable.usd` (robot)
- `SapienAssetPipeline/usd_assets/CoffeeMachine_103046/coffeemachine.usd` (revolute extension)

These are **stable binaries**; the other project edits cfg code, not these USDs. Hash mismatch → validation fails.

## 4. Residual shared-Python dependency (execution backend) + mitigation
The low-level skill execution still imports geometry constants from the shared tree:

    franka_skill_state_machine/runtime/drawer_target_config.py
        -> isaaclab_tasks...custom_drawer_config  (CABINET_USD_SCALE, DRAWER_TARGETS, ...)
    franka_skill_state_machine/runtime/microwave_door_config.py
        -> isaaclab_tasks...microwave_door_config (COFFEE_LEVER_JOINT/LINK, DOOR_*, ...)
    franka_skill_state_machine/runtime/target_registry.py
        -> isaaclab_tasks...stack_joint_pos_env_cfg

Why not repoint to `scene/stackpkg` now: the import path `stackpkg.franka.*` only resolves with
`projects/paper/scene` on `sys.path`; the GUI harness puts `projects/paper` (not `.../scene`) on the
path, so a naive repoint breaks the running platform (observed earlier this project). Repointing safely
requires threading `scene/` onto the path in every entrypoint — deferred to avoid destabilizing the
platform right before experiments.

**Mitigation (what actually protects the formal result):**
1. Paper mechanism identity is read from the **frozen** `paper_scene_v2`, not the shared config.
2. Stage-0 (`test_scene_consistency_v2`) cross-checks that the live `DRAWER_TARGETS` joint/link names
   still equal the frozen `mechanism_registry.json` — a shared-config edit that changes them fails loudly.
3. `git_commit` + `dirty_worktree` are stamped into every run's metadata; formal data uses clean commits.

## 5. Modules that affect the formal paper result
- `franka_skill_state_machine/skills/open_drawer_skill.py`, `close_drawer_skill.py` (execution)
- `franka_skill_state_machine/runtime/drawer_obs_adapter.py` (handle pose = single source)
- `deployment_calibration/adapters/articulated_drawer_v2.py` (member-aware driver — to be added)
- `scene/paper_scene_v2/*` (frozen mechanism identity + handle poses)
- external cabinet/sektion/robot USDs (geometry)

Not affecting the formal result (kept, not in run-chain): learned_drawer/ RL, perception, teleop, pi0.5,
coffee (disabled), doors, props.
