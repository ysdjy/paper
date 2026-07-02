# Module boundaries

Who may import / call whom. Arrows point "depends on".

```
scene/                ← (nobody; pure contract, no deps)
layout_editor/        → scene/, IsaacLab
skill_runtime/        → scene/, franka_skill_state_machine/ (legacy), IsaacLab
teleop_collection/    → scene/ ; (optional) gello_franka_teleop/ (.venv-gello) ; IsaacLab for drive
pi05_training/        → scene/, teleop_collection/ (demos), pi05_isaacsim_baseline/ (.venv_openpi via HTTP)
perception_foundationpose/ → scene/ ; franka_d435_foundationpose/ (own env) ; IsaacLab for sim-GT
```

## Rules
1. `scene/` imports nothing from sibling modules. It is the root of the DAG.
2. No module imports another module's *internal* runtime; they share the
   **contract** (`scene/`) and the **pose schema** (`perception_foundationpose/sim/pose_entry.py`).
3. Cross-venv calls go over a bridge only:
   - pi0.5 ↔ IsaacLab: HTTP (`/health`, `/infer`).
   - FoundationPose ↔ IsaacLab: ZMQ / files.
   - GELLO ↔ IsaacLab: serial read → q values (no shared imports).
4. `env_isaaclab` must stay importable without OpenPI / FoundationPose /
   pyrealsense2. The V1 façades enforce this (they raise rather than import heavy
   deps in mock/stub mode).
5. Legacy directories are read-only references for V1; V1 wraps, never edits them.

## Allowed shared types
- `franka_v1_skill_lab.scene.*` — task id, object list, registry.
- `franka_v1_skill_lab.perception_foundationpose.sim.pose_entry.{PoseEntry,PoseResult}`.
- `franka_v1_skill_lab.pi05_training.adapters.*` obs/action dataclasses.

These are pure-python and safe to import from any venv.
