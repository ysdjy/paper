# Franka V1 Skill Lab

A unified workspace for the Franka manipulation project, built around **one
scene contract**, **one environment contract**, and **one joint-action control
framework**. Everything here — skills, teleop, pi0.5 training, perception —
consumes the same **Franka Skill Scene V1**.

> Status legend used throughout this project: **ready** (works now) ·
> **wrapper** (re-uses a proven legacy module) · **stub** (interface only,
> raises `NotImplementedError`) · **TODO** (planned, not started).

## The V1 contract (one scene to rule them all)

| field         | value |
|---------------|-------|
| scene name    | Franka Skill Scene V1 |
| base task id  | `Isaac-Stack-Cube-Franka-JointPolicy-v0` |
| control mode  | **joint action** (7 arm-joint targets + gripper; NOT IK-Rel/IK-Abs) |
| robot         | Franka |
| objects       | `cube_1 cube_2 cube_3 knife cabinet microwave coffee_machine` |
| skills        | `grasp place open_drawer close_drawer open_door close_door` |
| consumers     | `skill_runtime teleop_collection pi05_training perception_foundationpose` |

The source of truth is the python package [`scene/`](scene/) (`scene_contract.py`,
`scene_registry.py`, `v1_task_ids.py`) plus [`configs/scene_v1.yaml`](configs/scene_v1.yaml).

```
                 scene/  (V1 contract + registry)
                    │
   layout_editor/ ──┤  edit & save the V1 scene -> scene_v1_registry.json
                    │
        ┌───────────┼─────────────────────┬───────────────────────┐
        ▼           ▼                     ▼                       ▼
  skill_runtime  teleop_collection   pi05_training        perception_foundationpose
  (joint skills) (GELLO -> joint)    (pi0.5, HTTP server) (sim-GT / FoundationPose pose)
```

## Modules

| dir | what | status |
|-----|------|--------|
| [`scene/`](scene/) | V1 scene contract + active-scene registry | **ready** |
| [`layout_editor/`](layout_editor/) | edit/save the V1 scene, update registry | **ready** (needs GPU) |
| [`skill_runtime/`](skill_runtime/) | joint-action state-machine skill tester | **wrapper** over `franka_skill_state_machine` |
| [`teleop_collection/`](teleop_collection/) | GELLO read + joint-target safety + demo recording | mixed (**ready** mock / **stub** record) |
| [`pi05_training/`](pi05_training/) | pi0.5 fine-tune pipeline (HTTP policy server) | mixed (**ready** adapters / **wrapper** workflow) |
| [`perception_foundationpose/`](perception_foundationpose/) | object 6D pose (sim-GT now, FoundationPose later) | mixed (**ready** sim-GT / **stub** FP) |
| [`asset_pipeline/`](asset_pipeline/) | notes on SAPIEN -> USD asset conversion | **wrapper** (points at `SapienAssetPipeline/`) |

## Quick start (acceptance commands)

```bash
# 1. Validate the project layout + scene contract (no GPU)
python projects/franka_v1_skill_lab/scripts/check_project_layout.py

# 2. Offline smoke tests (no GPU, no hardware)
bash projects/franka_v1_skill_lab/scripts/smoke_foundationpose_sim_v1.sh
bash projects/franka_v1_skill_lab/scripts/smoke_pi05_v1.sh
bash projects/franka_v1_skill_lab/scripts/smoke_gello_readonly.sh

# 3. Layout editor (needs GPU + display)
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0

# 4. Joint-action skill UI (needs GPU + display)
./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
  --num_envs 1 --show_affordance_debug \
  --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
  --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
  --seed 1
```

See [`docs/commands.md`](docs/commands.md) for the full command list and
[`docs/migration_report.md`](docs/migration_report.md) for what was migrated,
what is a wrapper/stub, and known blockers.

## Design rules (do not violate)

1. The mainline is **joint action**. Never wire V1 back to IK-Abs / IK-Rel.
2. Heavy deps stay isolated: **OpenPI** (`.venv_openpi`), **FoundationPose**
   (its own conda env), **GELLO** (`.venv-gello`). IsaacLab talks to them over
   HTTP / ZMQ / files only.
3. Old directories are **not deleted**. Real implementations may still live in
   `franka_skill_state_machine/`, `SceneLayoutModule/`, `pi05_isaacsim_baseline/`,
   `franka_d435_foundationpose/`, `projects/gello_franka_teleop/`; V1 wraps them.
4. No long training runs, no large model downloads from this project.
