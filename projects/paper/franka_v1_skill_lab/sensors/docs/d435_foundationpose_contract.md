# Wrist cameras contract (V1) — FoundationPose RGB-D + VLA RGB

The V1 scene fixes **two** wrist cameras (source of truth:
`sensors/d435/d435_config.py`). They are distinct logical sensors, not one camera
reused:

| camera | prim | consumer | streams | resolution |
|--------|------|----------|---------|------------|
| `foundationpose_d435_rgbd` | `panda_hand/foundationpose_d435_rgbd` | FoundationPose | RGB **+ depth** | 640×480 |
| `vla_libero_eye_in_hand` | `panda_hand/vla_libero_eye_in_hand` | pi0.5 / VLA | RGB (opt. depth) | 128×128 (also 224×224) |

The visible D435 housing (`panda_hand/D435_Body`) is **hidden by
default** (`--show_camera_body` to reveal); it has no collider/rigid body.

## FoundationPose camera (`foundationpose_d435_rgbd`)

Exports everything FoundationPose needs:

| field | source | status |
|-------|--------|--------|
| RGB image | camera `rgb` | ready (GPU) |
| Depth image | `distance_to_image_plane`, float32 m | ready (GPU) — **REQUIRED** |
| Camera intrinsics `K` | `adapter.intrinsics()` (live) / nominal D435 | ready |
| Camera-to-world pose | `adapter.camera_pose_world()` | ready (GPU) |
| Object mesh path | per-object map in the input builder | partial |
| Initial mask | sim `semantic_segmentation` / `null` | stub |

**Depth is mandatory.** If the renderer did not produce depth,
`WristCameraAdapter.capture()` raises
`Depth output is not available for foundationpose_d435_rgbd.` — never a silent
`None`.

### `foundationpose_input_builder.build_foundationpose_input(frame, objects)` output

```json
{
  "camera": {
    "name": "foundationpose_d435_rgbd",
    "frame_id": "foundationpose_d435_rgbd",
    "rgb_path": "... or null (in-memory)",
    "depth_path": "... or null (in-memory)",
    "has_rgb": true,
    "has_depth": true,
    "intrinsics": { "fx": 610.0, "fy": 610.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480 },
    "camera_pose_world": { "position": [x, y, z], "quat_wxyz": [w, x, y, z] }
  },
  "objects": [ { "name": "cube_1", "mesh_path": null, "mask_path": null, "pose_source": "sim_gt" } ]
}
```

### Live debug-output files (`test_d435_observation.py --camera foundationpose_d435_rgbd --save_debug_outputs`)
```
sensors/d435/debug_outputs/foundationpose_d435_rgbd_rgb.png
sensors/d435/debug_outputs/foundationpose_d435_rgbd_depth.npy
sensors/d435/debug_outputs/foundationpose_d435_rgbd_depth_visual.png
sensors/d435/debug_outputs/foundationpose_d435_rgbd_intrinsics.json
sensors/d435/debug_outputs/foundationpose_d435_rgbd_camera_pose_world.json
```

## VLA camera (`vla_libero_eye_in_hand`)

Pose derived from the **robosuite Panda `robot0_eye_in_hand`** camera that LIBERO
uses (`LIBERO_REFERENCE` in `d435_config.py`):

```
parent body: right_hand
parent local pos:  (0, 0, 0.1065)   parent local quat (wxyz): (0.924, 0, 0, -0.383)  # Rz(-45°)
camera local pos:  (0.05, 0, 0)     camera local quat (wxyz): (0, 0.707108, 0.707108, 0)
fovy: 75°   default size: 128×128   MuJoCo camera convention = OpenGL
```

`robosuite_eye_in_hand_to_isaac_offset()` composes `parent ∘ camera_local`, then
applies the `right_hand → panda_hand` alignment quaternion, and returns the offset
in **OpenGL** convention (so `CameraCfg.OffsetCfg(convention="opengl")` reproduces
the MuJoCo camera frame).

> **APPROXIMATE.** The `right_hand → panda_hand` alignment defaults to identity
> (`RIGHTHAND_TO_PANDAHAND_QUAT_WXYZ`, `LIBERO_POSE_IS_APPROXIMATE = True`).
> robosuite's `right_hand` +Z and Isaac's `panda_hand` +Z both point along the
> gripper approach, so the view direction is preserved; the in-plane roll may need
> a small calibration. Verify with the axis debug (it confirms the camera looks
> along the wrist approach, **not** straight down):
> ```bash
> python projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py   # prints axes
> ```

Image keys (RGB): `images.wrist_rgb` (project-unified), `images.eye_in_hand_rgb`,
`images.robot0_eye_in_hand_rgb` (LIBERO-compatible).

## Quaternion convention note
- D435/VLA `camera_pose_world.quat_wxyz` is **w-first** (Isaac).
- `PoseEntry.quat` (objects) is **w-last** `[x,y,z,w]` (ROS/FoundationPose).
