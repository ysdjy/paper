# D435 setup (V1)

The RGB-D source for FoundationPose. Two backends, both already implemented in
`franka_d435_foundationpose/franka_d435_foundationpose/camera/`:
- **sim**: a rendered D435 mounted on the Franka EE (needs `--enable_cameras`).
  Depth = optical-axis distance in meters.
- **realsense**: physical Intel D435 via `pyrealsense2`. Raw depth is mm → ×0.001.

Intrinsics/extrinsics in `configs/foundationpose.yaml` (`camera:`) and the legacy
`franka_d435_foundationpose/configs/camera_d435.yaml` / `hand_eye.yaml`. The V1
`d435_observation_adapter.py` is a thin façade; capture is delegated, never
re-implemented (RealSense libs stay out of `env_isaaclab`).
