#!/usr/bin/env bash
# Franka V1 — pi0.5 pipeline smoke test. STATUS: ready (no OpenPI, no Isaac, no GPU).
# Validates the V1 obs adapter, joint action adapter, and demo->LeRobot mapping.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_pi05] 1/3 obs + joint-action adapter self-test"
"$PY" - <<'PYEOF'
import sys
sys.path.insert(0, "projects")
from franka_v1_skill_lab.pi05_training.adapters.isaaclab_obs_adapter_v1 import build_observation
from franka_v1_skill_lab.pi05_training.adapters.joint_action_adapter_v1 import JointActionAdapter

obs = build_observation(
    joint_pos=[0.0]*7, joint_vel=[0.0]*7, gripper_width=0.04,
    ee_pose=[0.4,0.0,0.3,0,0,0,1], object_poses=[{"name":"cube_1"}],
)
d = obs.to_dict()
assert d["control_mode"] == "joint", d
assert len(d["joint_pos"]) == 7 and len(d["ee_pose"]) == 7

adapter = JointActionAdapter(max_step_rad=0.10)
adapter.reset([0.0]*7)
# a wild action must be clamped to within max_step_rad of the previous q
out = adapter.adapt({"joint_target":[5.0]*7, "gripper_command": 9.0}, current_q=[0.0]*7)
assert all(abs(v) <= 0.10 + 1e-6 for v in out["joint_target"]), out
assert 0.0 <= out["gripper_command"] <= 1.0, out
print("  obs/action adapters OK (joint action space, clamped)")

# two wrist cameras flow into the observation as logical keys:
#   VLA cam -> images.wrist_rgb / eye_in_hand_rgb / robot0_eye_in_hand_rgb
#   FP  cam -> images.wrist_depth
from franka_v1_skill_lab.pi05_training.adapters.isaaclab_obs_adapter_v1 import wrist_images_from_frames
from franka_v1_skill_lab.sensors.d435.d435_observation_adapter import WristCameraAdapter
vla = WristCameraAdapter.mock_frame("vla_libero_eye_in_hand")
fp  = WristCameraAdapter.mock_frame("foundationpose_d435_rgbd", with_depth=True)
imgs = wrist_images_from_frames(vla, fp)
for k in ("images.wrist_rgb", "images.robot0_eye_in_hand_rgb", "images.wrist_depth"):
    assert k in imgs, (k, list(imgs))
obs2 = build_observation(
    joint_pos=[0.0]*7, joint_vel=[0.0]*7, gripper_width=0.04,
    ee_pose=[0.4,0.0,0.3,0,0,0,1], images=imgs,
)
assert "images.wrist_rgb" in obs2.to_dict()["images"], obs2.to_dict()["images"].keys()
print("  two wrist cameras OK (VLA rgb keys + FP depth key)")
PYEOF

echo "[smoke_pi05] 2/3 demo -> LeRobot mapping (dry run)"
"$PY" projects/franka_v1_skill_lab/pi05_training/adapters/skill_demo_to_lerobot.py --dry_run >/dev/null
echo "  mapping is V1-valid (joint action, not EE delta)"

echo "[smoke_pi05] 3/3 pipeline.env loads"
# shellcheck disable=SC1091
source projects/franka_v1_skill_lab/pi05_training/WORKFLOW/pipeline.env
[ "$V1_TASK_ID" = "Isaac-Stack-Cube-Franka-JointPolicy-v0" ] || { echo "  bad V1_TASK_ID: $V1_TASK_ID"; exit 1; }
[ "$V1_CONTROL_MODE" = "joint" ] || { echo "  bad control mode"; exit 1; }
echo "  pipeline.env OK (task=$V1_TASK_ID mode=$V1_CONTROL_MODE)"

echo "[smoke_pi05] OK"
