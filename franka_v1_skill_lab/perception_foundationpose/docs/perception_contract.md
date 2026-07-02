# V1 perception contract

Every pose source (sim-GT, FoundationPose, mock) emits a `PoseResult`:

```json
{ "source": "sim_gt|foundationpose|mock", "frame": "world",
  "objects": [ {
      "name": "cube_1",
      "position": [x, y, z],          // world, meters
      "quat": [x, y, z, w],           // world (xyzw, NOT wxyz)
      "confidence": 1.0,
      "pose_in_camera": [[..4x4..]] | null,
      "mesh_path": "..." | null,
      "mask_path": "..." | null
  } ] }
```

- Quaternion order is **xyzw** (ROS/FoundationPose). IsaacLab poses are wxyz and
  are converted on the way out.
- `confidence` is 1.0 for sim-GT (ground truth).
- Frame chain for real FP: `T_base_object = T_base_ee · T_ee_camera · T_camera_object`.
- Consumers (`pi05_training`, `skill_runtime`) depend ONLY on this contract, so
  swapping sim-GT ↔ FoundationPose requires no consumer change.
