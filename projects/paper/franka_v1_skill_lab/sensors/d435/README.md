# V1 cameras (front static + 2 wrist)

Three cameras. All numbers live in [`d435_config.py`](d435_config.py) — change
them there, nowhere else.

```
{ENV_REGEX_NS}/                            # env root (static)
    vla_front_static           # RGB 256x256, fixed front view  -> LeRobot `image`
{ENV_REGEX_NS}/Robot/panda_hand/           # wrist (moves with arm)
    D435_Body                  # visible housing — HIDDEN by default, no collider
    foundationpose_d435_rgbd   # RGB + depth, 640x480  (FoundationPose)
    vla_libero_eye_in_hand     # RGB (opt depth), 128x128 / 224x224  -> LeRobot `wrist_image`
```
(Each camera is mounted directly under an EXISTING parent — the spawner needs the
parent prim to exist, so no extra grouping Xform; names stay self-describing.)

## 0. Front static camera (`vla_front_static`) — LeRobot `image`
- **Fixed** under the env root (`{ENV_REGEX_NS}/vla_front_static`); does **not**
  move with `panda_hand`.
- Pose reuses Isaac Lab's `table_cam` (`stack_ik_rel_visuomotor_env_cfg.py`):
  `pos=(1.0, 0.0, 0.4)`, ros `rot=(0.35355,-0.61237,-0.61237,0.35355)` — third-
  person view looking back onto the Franka + cubes workspace (not top-down).
- 256×256 (matches the pi0.5 converter `image_resize`), fovx ≈ 47°.
- Maps to LeRobot/pi0.5 **`image`** (the main view). Keys: `images.front_rgb`,
  `images.image`.

## 1. Mount link
Both cameras hang off **`{ENV_REGEX_NS}/Robot/panda_hand`** (Franka wrist flange),
mounted directly on it. The `D435_Body` cuboid is **visual only — no collider,
no rigid body**, so Franka dynamics are unchanged. It is **invisible by default**;
pass `--show_camera_body` to reveal it.

## 2. foundationpose_d435_rgbd (RGB-D)
| item | value |
|------|-------|
| optical offset | pos `(0.13, 0.0, -0.15)`, rot_wxyz `(-0.70614, 0.03701, 0.03701, -0.70614)`, convention `ros` |
| optical axis | forward + slightly down onto the grasp workspace |
| color | 640×480 `rgb` uint8 |
| depth | 640×480 `distance_to_image_plane` float32 m — **REQUIRED** |
| intrinsics | live `data.intrinsic_matrices`, else nominal `fx=fy=610, cx=320, cy=240` |

Depth is mandatory: `capture()` raises
`Depth output is not available for foundationpose_d435_rgbd.` if the renderer
didn't produce it (never silent `None`).

## 3. vla_libero_eye_in_hand (RGB, LIBERO-aligned)
Pose from robosuite Panda `robot0_eye_in_hand` (the camera LIBERO uses):

```
parent body right_hand: pos (0,0,0.1065)  quat_wxyz (0.924,0,0,-0.383)
camera local:           pos (0.05,0,0)    quat_wxyz (0,0.707108,0.707108,0)
fovy 75°   default 128x128   MuJoCo/OpenGL camera convention
```

`robosuite_eye_in_hand_to_isaac_offset()` composes these and returns an Isaac
offset in **OpenGL** convention. The `right_hand → panda_hand` alignment is
**APPROXIMATE** (identity default, `LIBERO_POSE_IS_APPROXIMATE = True`). The view
axis points along the wrist approach (eye-in-hand) — **not straight down**;
confirm with:
```bash
python projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py   # prints camera axes
```

- resolution: `--vla_camera_resolution 128 128` (default) or `224 224`
- image keys: `images.wrist_rgb` (unified), `images.eye_in_hand_rgb`,
  `images.robot0_eye_in_hand_rgb` (LIBERO-compatible)

## 4. Read intrinsics / pose in code
```python
from franka_v1_skill_lab.sensors.d435 import WristCameraAdapter
fp  = WristCameraAdapter(env, "foundationpose_d435_rgbd")
vla = WristCameraAdapter(env, "vla_libero_eye_in_hand")
K     = fp.intrinsics()             # {fx, fy, cx, cy, width, height}
pose  = fp.camera_pose_world()      # {position, quat_wxyz}  (world)
frame = fp.capture()                # rgb/depth/intrinsics/pose/name/...
```

## 5. Frame contract
```python
{
  "name": "...", "consumer": "...",
  "rgb": HxWx3 uint8 | None, "depth": HxW float32 m | None, "mask": ... | None,
  "intrinsics": {fx,fy,cx,cy,width,height},
  "camera_pose_world": {"position":[x,y,z], "quat_wxyz":[w,x,y,z]},
  "image_keys": [...], "frame_id": "...", "timestamp": float,
}
```

## 6. Quick checks
```bash
# offline contract + camera axes (no GPU)
python projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py

# live FoundationPose RGB-D capture (GPU)
./isaaclab.sh -p projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py \
  --num_envs 1 --camera foundationpose_d435_rgbd --save_debug_outputs

# live VLA RGB capture (GPU)
./isaaclab.sh -p projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py \
  --num_envs 1 --camera vla_libero_eye_in_hand --vla_camera_resolution 128 128 --save_debug_outputs

# see them on the Franka in the layout editor (GPU)
./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --enable_wrist_cameras --hide_camera_body
```
