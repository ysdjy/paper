# sensors/ — V1 shared wrist cameras

This module owns the cameras that are part of the **shared V1 scene contract**,
so every consumer (skill runtime, teleop, pi0.5/VLA, FoundationPose) loads the
same instrumented scene through one `scene_v1_registry.json`.

V1 fixes **three** cameras (not one camera reused):

| camera | LeRobot field | consumer | streams | resolution | mount |
|--------|---------------|----------|---------|------------|-------|
| `vla_front_static` | `image` | pi0.5 / VLA | RGB | 256×256 | **fixed** (env root, does NOT move with arm) |
| `vla_libero_eye_in_hand` | `wrist_image` | pi0.5 / VLA | RGB (opt. depth) | 128×128 / 224×224 | wrist (`panda_hand`) |
| `foundationpose_d435_rgbd` | (depth) | FoundationPose | RGB **+ depth** | 640×480 | wrist (`panda_hand`) |

The front camera pose reuses Isaac Lab's proven `table_cam`
(`pos=(1.0,0,0.4)`, ros, looking back at the workspace) — the same view the pi0.5
data pipeline maps `front_rgb`/`image` to.

```
sensors/
├── d435/
│   ├── d435_config.py            ← PURE PYTHON source of truth: both CameraProfiles,
│   │                               prim paths, intrinsics, LIBERO conversion
│   ├── d435_scene_cfg.py         ← attach_wrist_cameras(env_cfg): both cams + hidden body
│   ├── d435_observation_adapter.py ← WristCameraAdapter(env, camera_name) -> frame dict
│   ├── d435_visual_debug.py      ← print_camera_debug / print_camera_axes
│   ├── test_d435_observation.py  ← offline contract test + LIVE capture (--save_debug_outputs)
│   ├── dump_d435_view.py         ← quick capture to PNG/npy
│   └── debug_outputs/            ← live capture lands here
└── docs/d435_foundationpose_contract.md
```

## One scene, many consumers
`attach_wrist_cameras(env_cfg)` is the ONE place the cameras are added. Every
Isaac entry calls it before `gym.make`, so the cameras live in the shared scene:

| consumer | how it gets the cameras |
|----------|-------------------------|
| `layout_editor` | `--enable_wrist_cameras` (+ `--show_camera_body` to reveal mesh, `--vla_camera_resolution`) |
| `skill_runtime` | `--enable_wrist_cameras --show_camera_debug` (patches `gym.make`) |
| `teleop_collection` | `--enable_wrist_d435` → records `images/wrist_rgb` (VLA) + `images/wrist_depth` (FP) |
| `pi05_training` | `wrist_images_from_frames(vla, fp)` → `images.wrist_rgb` / `…robot0_eye_in_hand_rgb` / `…wrist_depth` |
| `perception_foundationpose` | `foundationpose_input_builder` reads RGB/depth/K/pose from the FP camera |

## Status
- **ready (no GPU):** config, adapter contract + `mock_frame`, the offline half of
  `test_d435_observation.py`, registry/contract integration, FoundationPose input
  builder, LIBERO pose math + axis check.
- **ready (GPU + `--enable_cameras`):** live RGB / depth capture and `--save_debug_outputs`.
- **hidden by default:** the `D435_Body` mesh (visual only, no collider).
- **approximate:** the VLA `right_hand → panda_hand` alignment (identity default;
  see `sensors/docs/d435_foundationpose_contract.md`).
- **stub:** left/right IR (not provided — RGB-D only), masks (`null` placeholder),
  real-hardware capture (delegated to `franka_d435_foundationpose/`).
