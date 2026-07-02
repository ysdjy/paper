# perception_foundationpose/ — object 6D pose for the V1 scene

## 1. What this module does
Publishes 6D poses of the V1 objects in ONE unified format (`PoseEntry`), from
two interchangeable sources:
- **stage 1 (ready):** sim ground truth dressed up as FoundationPose output
  (`sim/sim_gt_pose_as_foundationpose.py`). Reads object poses from the live
  IsaacLab scene, confidence = 1.0.
- **stage 2 (stub):** real FoundationPose runner (`foundationpose/`) + D435 RGB-D
  capture (`d435/`). Interfaces defined; heavy CUDA deps isolated.

`foundationpose/foundationpose_object_adapter.py` is the single `poll() ->
PoseResult` entry point both pi0.5 and the skill runtime call.

## 2. Upstream dependencies
- Stage 1: an IsaacLab `SceneStateProvider` (from `skill_runtime`/eval).
- Stage 2: `franka_d435_foundationpose/` (its own conda env, ZMQ pose server) and
  the FoundationPose repo cloned separately. Never imported into `env_isaaclab`.
- `configs/foundationpose.yaml`, `scene/` (tracked-object list).

## 3. Downstream consumers
- `pi05_training` observation adapter — injects `object_poses` into the policy obs.
- `skill_runtime` — debug overlay of perceived vs. true poses.

## 4. Common commands
```bash
# offline contract smoke test (no Isaac, no GPU)
bash projects/franka_v1_skill_lab/scripts/smoke_foundationpose_sim_v1.sh

# synthetic PoseResult
python projects/franka_v1_skill_lab/perception_foundationpose/sim/sim_gt_pose_as_foundationpose.py --demo
```
```python
from franka_v1_skill_lab.perception_foundationpose.foundationpose.foundationpose_object_adapter import ObjectPoseAdapter
adapter = ObjectPoseAdapter(mode="sim_gt", provider=scene_state_provider)
result = adapter.poll()      # -> PoseResult; result.to_dict()["objects"]
```

## 5. Current status
- `sim/pose_entry.py`, `sim/sim_gt_pose_as_foundationpose.py`,
  `sim/test_sim_pose_adapter.py` — **ready**.
- `foundationpose/foundationpose_object_adapter.py` — **ready** for `sim_gt`,
  **stub** for `foundationpose` mode.
- `foundationpose/foundationpose_runner.py`, `d435/d435_observation_adapter.py`
  — **stub** (interfaces + mock; delegate to `franka_d435_foundationpose/`).

## 6. Troubleshooting
- *`ObjectPoseAdapter(mode="foundationpose").poll()` raises* — by design; FP is
  not wired up. Use `mode="sim_gt"`, or finish `foundationpose_runner.py`.
- *quat looks wrong* — IsaacLab is `[w,x,y,z]`; `PoseEntry.quat` is `[x,y,z,w]`
  (`_wxyz_to_xyzw` converts). Don't double-convert.
- *no poses returned* — the object name isn't in the live scene or not in the
  tracked list; check `configs/foundationpose.yaml: tracked_objects`.

See `docs/perception_contract.md`, `foundationpose/masks_and_meshes.md`,
`d435/d435_setup.md`.
