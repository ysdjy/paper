# V1 teleop demo HDF5 schema

Written by `recording/hdf5_writer.py` (driven by
`entries/collect_teleop_demos_joint_v1.py`) and consumed by
`pi05_training/adapters/skill_demo_to_lerobot.py` (→ normalized dataset → LeRobot).
**Joint action space** — the action is a 7-DoF arm joint target + gripper, never
an EE delta. Verified end-to-end against the live Isaac env.

One HDF5 file per collection *session*; each saved episode is a `data/demo_<i>`
group. Failed episodes (if `--save_failed`) go to a separate
`<out_dir>/failed/teleop_demos_*.hdf5`, never into the success file.

```
teleop_demos_<stamp>.hdf5
├── attrs: schema_version="franka-scene-v1"  task_id  control_mode="joint"
│          scene_registry  created_unix
└── data/                          attrs: num_demos
    └── demo_0/                    (demo_1, demo_2, … one per saved episode)
        ├── attrs: episode_id  task_instruction  skill_type  target_name
        │          success(bool)  num_steps(int)  seed(int)
        │          has_images  has_depth  has_objects
        ├── obs/
        │   ├── joint_pos        (T, 7)  float32   Franka arm joints
        │   ├── joint_vel        (T, 7)  float32
        │   ├── gripper_width    (T, 1)  float32
        │   ├── ee_pose          (T, 7)  float32   [x,y,z,qx,qy,qz,qw] env-local pos (aux)
        │   ├── objects/<name>   (T, 7)  float32   [x,y,z,qw,qx,qy,qz]  (--record_objects)
        │   │                                       names: cube_1/2/3, knife, cabinet
        │   └── images/                             (--enable_wrist_d435)
        │       ├── wrist_rgb    (T, H, W, 3) uint8   gzip
        │       └── wrist_depth  (T, H, W)    float32 gzip  (--record_depth)
        ├── actions/
        │   ├── joint_target     (T, 7)  float32   << THE ACTION (NOT EE delta)
        │   └── gripper_command  (T, 1)  float32   0=open .. 1=closed (continuous teleop value)
        ├── teleop/
        │   ├── raw_q            (T, 7)  float32   raw GELLO read
        │   └── filtered_q       (T, 7)  float32   after EMA + clamp + joint-limit safety
        └── timestamps           (T,)    float64   wall-clock seconds
```

Notes
- `obs/joint_pos` is read AFTER the action is applied, so `(joint_pos_t, action_t)`
  is the standard `(obs, action-just-applied)` pair pi0.5 trains on.
- The wrist D435 camera is attached via `sensors/d435` only when
  `--enable_wrist_d435` is set AND the app is launched with `--enable_cameras`
  (the collect entry sets that automatically). `--no_camera` forces it off.
- Object poses are sim ground truth. `--record_foundationpose_objects` is a
  planned stub that will swap sim-GT for FoundationPose estimates (Claude 2's
  perception module) — today it records sim-GT and logs a NOTE.

Mapping to the pi0.5 dataset lives in
`pi05_training/adapters/skill_demo_to_lerobot.py` (`V1_FIELD_MAPPING`):
`observation.state = joint_pos[7] + gripper_width[1]` and
`action = joint_target[7] + gripper_command[1]`.
