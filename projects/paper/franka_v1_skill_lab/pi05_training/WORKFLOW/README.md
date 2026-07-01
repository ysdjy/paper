# pi05_training/WORKFLOW/ — V1 pipeline front door

Numbered wrapper scripts mirroring `pi05_isaacsim_baseline/WORKFLOW/`, but with
the V1 joint-action task baked in via `pipeline.env`.

| script | does | status |
|--------|------|--------|
| `pipeline.env` | V1 config (task, ports, repo id, legacy paths) | ready |
| `1_collect.sh` | → `teleop_collection` joint collector (mock ready; live via isaaclab.sh) | ready (mock) |
| `2_convert.sh` | demo HDF5 → normalized + LeRobot (joint mapping) | ready (normalized; LeRobot best-effort) |
| `3_train.sh` | print the real OpenPI LoRA command (never auto-runs) | wrapper |
| `4_serve.sh` | start the policy server (mock by default) | wrapper |
| `5_eval.sh` | print the closed-loop eval command for the V1 task | wrapper |
| `stop.sh` | stop server / pipeline procs | wrapper |

Every script `source`s `pipeline.env`, `cd`s to the repo root, and is safe to
run for inspection (the heavy ones print the real command instead of launching).

```bash
source projects/franka_v1_skill_lab/pi05_training/WORKFLOW/pipeline.env
echo "$V1_TASK_ID $V1_CONTROL_MODE"   # Isaac-Stack-Cube-Franka-JointPolicy-v0 joint
```

**Why wrappers, not copies:** the real collect/convert/train/serve/eval logic is
hundreds of lines of OpenPI/LeRobot/IsaacLab glue that must run across two venvs.
Duplicating it would drift. The V1 layer only overrides the task + action space.
See `../docs/pi05_v1_migration.md`.
