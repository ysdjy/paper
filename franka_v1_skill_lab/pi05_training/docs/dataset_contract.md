# V1 pi0.5 dataset contract

- **task_id:** `Isaac-Stack-Cube-Franka-JointPolicy-v0`
- **control_mode:** `joint`
- **action:** `joint_target` (7) + `gripper_command` (1). NOT EE delta.
- **observation:** `joint_pos`(7), `joint_vel`(7), `gripper_width`(1),
  `ee_pose`(7, aux), `object_poses` (PoseEntry list), optional `images`.
- **repo_id:** `franka_v1_joint_demos` (override in `configs/pi05_dataset.yaml`).

## Concrete tensors the converter emits

`skill_demo_to_lerobot.py` reads the teleop HDF5 and produces, per frame:

| field | dim | layout |
|-------|-----|--------|
| `observation.state` | 8 | `joint_pos[7]` + `gripper_width[1]` |
| `action` | 8 | `joint_target[7]` + `gripper_command[1]` |
| `wrist_image` | (256,256,3) | resized D435 RGB (only if images present) |
| `task` | str | per-episode `task_instruction` attr |

Outputs (relative to the V1 data root):
- **normalized (always):** `data/processed/normalized_dataset/<name>/` —
  `states.npy (N,8)`, `actions.npy (N,8)`, `episode_index.npy (N,)`,
  `episodes.jsonl`, `metadata.json`, `images/wrist/ep<ee>_step<ssssss>.png`.
- **LeRobot (best effort):** `data/lerobot/<name>/` — built only if `lerobot`
  imports; failure is reported, never fatal (normalized output still stands).
- **report:** `data/reports/convert_latest_report.json`.

Only `success=true` demos are exported (use `--include_failed` to override), so
failed/discarded episodes never reach the training set.

## Camera images (three shared-scene cameras) → LeRobot fields

The V1 scene has THREE cameras (see `sensors/README.md`). The LeRobot/pi0.5
`image` (main view) comes from the fixed front camera; `wrist_image` from the
LIBERO-aligned eye-in-hand camera; depth (optional) from the FoundationPose
camera. The observation `images` dict carries the logical keys produced by
`isaaclab_obs_adapter_v1.wrist_images_from_frames(front_frame, vla_frame, fp_frame)`:

| LeRobot field | logical key | shape / dtype | source camera |
|---------------|-------------|---------------|---------------|
| **`image`** | `images.front_rgb` (+ `images.image`) | `(256, 256, 3)` uint8 | **`vla_front_static`** (fixed front, static) |
| **`wrist_image`** | `images.wrist_rgb` (+ `images.eye_in_hand_rgb`, `images.robot0_eye_in_hand_rgb`) | `(H, W, 3)` uint8 | **`vla_libero_eye_in_hand`** (wrist, LIBERO-compatible) |
| (depth) | `images.wrist_depth` | `(480, 640)` float32 m | `foundationpose_d435_rgbd` — **optional** |

- LeRobot dataset stores `image`/`wrist_image` at `image_resize` = **256×256**
  (`pi05_isaacsim_baseline/configs/dataset_mapping_isaaclab_franka.yaml`); the
  converter resizes each camera to that. The front camera already renders 256×256.
- The teleop HDF5 keys are `obs/images/front_rgb` and `obs/images/wrist_rgb`
  (added to the converter's `image_key_candidates`).

- VLA RGB is `128×128` by default (`--vla_camera_resolution 128 128`), also
  `224×224`. The three RGB keys carry the SAME image (aliases) so a LIBERO/pi0.5
  loader can pick whichever name it expects.
- `wrist_depth` is present only when the sim ran with `--enable_cameras` and the
  FoundationPose camera rendered depth; otherwise the key is **absent** (never
  silently zero-filled).
- Frames come from `sensors.d435.WristCameraAdapter(env, camera_name).capture()`
  (live) or `WristCameraAdapter.mock_frame(camera_name)` (offline smoke).
- For pi0.5/VLA fine-tuning the policy consumes the eye-in-hand RGB; depth is
  reserved for FoundationPose / geometry-aware variants.

Source HDF5 schema: `teleop_collection/data_format/demo_hdf5_schema.md`.
LeRobot field mapping: `pi05_training/adapters/skill_demo_to_lerobot.py`.

A dataset is V1-valid iff: action maps to `actions/joint_target`, control_mode
is `joint`, and the object set is a subset of the V1 contract objects.
