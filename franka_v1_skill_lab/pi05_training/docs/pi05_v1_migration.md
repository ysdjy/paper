# pi0.5 V1 migration notes

## What changed from the legacy pipeline
| aspect | legacy (`pi05_isaacsim_baseline`) | V1 |
|--------|-----------------------------------|----|
| task   | `Isaac-Stack-Cube-Franka-IK-Rel-v0` | `Isaac-Stack-Cube-Franka-JointPolicy-v0` |
| action | 7-D EE delta `[dx,dy,dz,drx,dry,drz,grip]` | 7 joint targets + gripper |
| obs    | joint+EE+images | joint_pos/vel, gripper, ee_pose(aux), object_poses, images |
| object poses | stub | from `perception_foundationpose` (sim-GT now) |

## What is reused (not duplicated)
- HTTP policy server + client + mock/openpi backends → `pi05_isaacsim_baseline/adapters/policy_server/`
- HDF5 → LeRobot builder → `pi05_isaacsim_baseline/adapters/data_conversion/`
- Closed-loop runner → `pi05_isaacsim_baseline/scripts/isaaclab/run_policy_in_isaaclab.py`
  (already supports `--env_kind joint` / detects JOINT tasks)

## What V1 adds
- `adapters/isaaclab_obs_adapter_v1.py` — joint observation builder
- `adapters/joint_action_adapter_v1.py` — policy joint action → safe Franka target
- `adapters/skill_demo_to_lerobot.py` — V1 field mapping (+ validator that
  rejects EE-delta action fields)
- `WORKFLOW/*.sh` — V1 task override around the legacy scripts

## Isolation (unchanged)
OpenPI/JAX/torch live in `.venv_openpi`. `env_isaaclab` only ever runs the
stdlib HTTP client. The mock backend lets the whole loop run with no GPU.

## Blockers / TODO
- Real LoRA training not run here (no GPU time, no model download by policy).
- OpenPI config must be registered for the joint action space before real train.
- Collection loop (`collect_teleop_demos_joint_v1.py`) is a stub.
