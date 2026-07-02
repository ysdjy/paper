# Architecture — franka_v1_skill_lab

## One scene, four consumers
Everything hangs off the **Franka Skill Scene V1** contract (`scene/`). The
layout editor produces the active scene; four consumers read it.

```
                       ┌────────────────────────┐
                       │ scene/  (V1 contract)   │
                       │  scene_contract.py      │  task=Isaac-Stack-Cube-
                       │  scene_registry.py      │       Franka-JointPolicy-v0
                       │  v1_task_ids.py         │  control_mode=joint
                       └───────────┬────────────┘
                                   │ scene_v1_registry.json
                 layout_editor/ ───┤ (Save V1 → update registry)
                                   │
        ┌──────────────────┬───────┴────────┬─────────────────────────┐
        ▼                  ▼                ▼                         ▼
 skill_runtime/     teleop_collection/  pi05_training/        perception_foundationpose/
 joint state        GELLO q[0:7] →      obs/action adapters   sim-GT pose now,
 machine (wraps     joint target +      + HTTP policy server  FoundationPose later;
 franka_skill_      safety; record      (OpenPI isolated      unified PoseEntry feeds
 state_machine)     demos               in .venv_openpi)      pi0.5 + skill debug
```

## Control contract (the spine)
- **Action = joint.** Every consumer ultimately produces a 7-DoF Franka joint
  target (+ gripper), handed to the env via
  `SceneStateProvider.make_joint_action_from_q_des(q_des, gripper)`.
- grasp/place compute `q_des` with internal DLS IK; drawer skills do physical
  `ik_pull`; pi0.5 emits `q_des` directly; teleop maps GELLO `q` → `q_des`.
- This is why the base task is `JointPolicy`, not IK-Rel/IK-Abs.

## Environment isolation (hard rule)
| heavy dep | env | bridge to IsaacLab |
|-----------|-----|--------------------|
| OpenPI / JAX | `.venv_openpi` | HTTP (policy server) |
| FoundationPose | own conda env | ZMQ / files |
| GELLO / Dynamixel | `.venv-gello` | serial → q values |
| IsaacLab / IsaacSim | `env_isaaclab` | — |

`env_isaaclab` never imports OpenPI, FoundationPose, or RealSense. The V1
adapters that touch those are façades/stubs that delegate to the isolated
projects.

## Data flow for a pi0.5 rollout (target end state)
```
env (joint) ──obs──► isaaclab_obs_adapter_v1 ──┐
perception (PoseEntry) ─────────────────────────┤► Observation(JSON)
                                                 │      │ HTTP POST /infer
                                                 ▼      ▼
                              policy server (.venv_openpi)  → action(JSON)
                                                 │
                          joint_action_adapter_v1 (clip/limit)
                                                 │
              SceneStateProvider.make_joint_action_from_q_des ──► env.step
```

## Why wrappers instead of a big move
The skill runtime (~5k lines), the OpenPI pipeline, the FoundationPose bridge,
and the GELLO reader are each proven and/or tied to an isolated venv. Moving
them risks breaking working code and mixing heavy deps. V1 instead pins a
**contract** and wraps the implementations, so the mainline is unified while the
implementations stay where they run safely. Folding them in later only touches
the thin shims (`skill_runtime/_legacy.py`, the WORKFLOW wrappers, the façades).
