#!/usr/bin/env bash
# Franka V1 — layout editor smoke test. STATUS: ready (compile check); GUI run needs GPU.
# Byte-compiles the V1 layout editor and prints the real launch command.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_layout] byte-compile layout editor + scene registry"
"$PY" -m py_compile \
  projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
  projects/franka_v1_skill_lab/scene/scene_registry.py
echo "  compile OK"

cat <<'CMD'

[smoke_layout] To run the V1 layout editor (needs GPU + display):

  ./isaaclab.sh -p projects/franka_v1_skill_lab/layout_editor/layout_v1_ui.py \
    --num_envs 1 --task Isaac-Stack-Cube-Franka-JointPolicy-v0

  Move/add objects in the viewport, click "Save V1 (USD+JSON+registry)".
  That writes scene_v1_latest.usd/.json into scene/saved_scenes/v1_active/
  and updates scene_v1_registry.json. Re-open it with --load_latest_v1.

CMD
echo "[smoke_layout] OK"
