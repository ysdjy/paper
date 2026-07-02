#!/usr/bin/env bash
# Franka V1 — skill UI smoke test. STATUS: ready (compile check); GUI run needs GPU.
# Byte-compiles the V1 entry + scene contract and prints the real launch command.
# It does NOT launch Isaac Sim here (that needs a GPU/display).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_skill_ui] byte-compile V1 entry + wrappers + scene contract"
"$PY" -m py_compile \
  projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
  projects/franka_v1_skill_lab/scene/scene_contract.py \
  projects/franka_v1_skill_lab/scene/scene_registry.py
echo "  compile OK"

echo "[smoke_skill_ui] scene registry resolves (incl. both wrist cameras):"
"$PY" projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py --expect_sensor foundationpose_d435_rgbd | sed 's/^/  /'

cat <<'CMD'

[smoke_skill_ui] To run the joint-action skill UI (needs GPU + display):

  ./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
    --num_envs 1 --show_affordance_debug \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
    --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
    --seed 1

[smoke_skill_ui] Same, but loading the two shared-scene wrist cameras + printing pose:

  ./isaaclab.sh -p projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py \
    --num_envs 1 --show_affordance_debug --show_camera_debug --enable_wrist_cameras \
    --grasp_backend joint_ik --place_backend joint_ik --drawer_backend ik_pull \
    --scene_registry projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json \
    --seed 1
  (--show_camera_debug auto-enables camera rendering; D435 body stays hidden unless --show_camera_body.)

CMD
echo "[smoke_skill_ui] OK"
