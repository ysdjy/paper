#!/usr/bin/env bash
# Franka V1 — wrist cameras scene smoke test. STATUS: ready (no Isaac, no GPU).
# Validates the two-camera config/adapter contract, that the registry registers
# both cameras, prints the camera axes (confirm VLA not straight down), and
# byte-compiles the Isaac-side scene cfg + entry hooks. Does NOT launch Isaac
# Sim (the rendered cameras need a GPU + --enable_cameras).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_d435] 1/4 wrist cameras contract (FP rgbd + VLA rgb + logical keys + axes)"
"$PY" projects/franka_v1_skill_lab/sensors/d435/test_d435_observation.py | sed 's/^/  /'

echo "[smoke_d435] 2/4 registry registers all three cameras (enabled)"
"$PY" projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py --expect_sensor vla_front_static | sed 's/^/  /'
"$PY" projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py --expect_sensor vla_libero_eye_in_hand >/dev/null
"$PY" projects/franka_v1_skill_lab/scene/tools/check_scene_v1.py --expect_sensor foundationpose_d435_rgbd >/dev/null
echo "  vla_front_static (image) + vla_libero_eye_in_hand (wrist_image) + foundationpose_d435_rgbd (depth) present"

echo "[smoke_d435] 3/4 byte-compile Isaac-side D435 scene cfg + entry hooks"
"$PY" -m py_compile \
  projects/franka_v1_skill_lab/sensors/d435/d435_scene_cfg.py \
  projects/franka_v1_skill_lab/sensors/d435/d435_visual_debug.py \
  projects/franka_v1_skill_lab/sensors/d435/check_three_cameras.py \
  projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  projects/franka_v1_skill_lab/skill_runtime/entries/skill_test_ui_joint_v1.py
echo "  compile OK"

echo "[smoke_d435] 4/4 mount summary"
"$PY" - <<'PYEOF' | sed 's/^/  /'
import sys
sys.path.insert(0, "projects")
from franka_v1_skill_lab.sensors.d435.d435_visual_debug import static_summary_lines
for line in static_summary_lines():
    print(line)
PYEOF

cat <<'CMD'

[smoke_d435] To SEE the wrist D435 on the Franka in the layout editor (GPU):

  ./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
    --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0 --enable_wrist_d435

  (add --enable_cameras to also render RGB-D; without it you still see the body.)

CMD
echo "[smoke_d435] OK"
