# teleop_collection/gello/ — GELLO source + joint safety

## 1. What this module does
- `mock_gello_source.py` — `GelloSource` interface with `MockGelloSource`
  (synthetic, hardware-free) and `RealGelloSource` (wrapper to the legacy
  reader, **stub**).
- `joint_teleop_safety.py` — `JointTeleopSafety`: EMA low-pass → per-step delta
  clamp → Franka joint-limit clamp → e-stop latch. Pure python (lists, no numpy).
- `gello_reader.py` — `make_gello_source(device)` factory (mock / keyboard /
  gello). **ready.**
- `gello_to_franka_mapper.py` — `GelloToFrankaMapper`: read → safe `q_des[7]` +
  binary gripper command (±1) + continuous gripper norm [0,1]. **ready.**
- `configs/` — V1 GELLO config (see `configs/teleop_gello.yaml` at project root).

## 2. Upstream dependencies
- Real hardware path: `projects/gello_franka_teleop/` (`.venv-gello`, Dynamixel).
- The mock path has **no** dependencies.

## 3. Downstream consumers
- `teleop_collection/entries/*` and ultimately `pi05_training` demos.

## 4. Common commands
```python
from franka_v1_skill_lab.teleop_collection.gello.joint_teleop_safety import JointTeleopSafety
from franka_v1_skill_lab.teleop_collection.gello.mock_gello_source import MockGelloSource
src = MockGelloSource(); saf = JointTeleopSafety()
q0,_ = src.read(); saf.reset(q0)
q_raw, grip = src.read(); q_safe = saf.step(q_raw)
```

## 5. Current status
**ready** — mock, safety, AND real hardware. `RealGelloSource` reads the leader
arm live via `gello_isaac_teleop.reader.ThreadedGelloReader` inside the Isaac
process (`dynamixel_sdk` is importable in `env_isaaclab`). Verified live on
`/dev/ttyUSB0` with the calibrated `configs/gello_franka.yaml` (per-joint
offsets/signs + gripper open/close). Gripper raw degrees → normalized [0,1]
(0=open, 1=closed).

## 6. Troubleshooting
- *Franka joint 4 out of range* — the mock biases joint 4 to ≈ −1.5 to stay in
  the `(-3.07, -0.07)` soft limit; for real GELLO set per-joint offsets.
- *Targets too sluggish* — raise `lowpass_alpha`; too jumpy — lower it or
  `max_step_rad`.
