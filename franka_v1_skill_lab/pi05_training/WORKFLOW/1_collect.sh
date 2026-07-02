#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — collect joint teleop demos (V1 wrapper).
# STATUS: ready (mock; --dry_run no-Isaac). LIVE collection needs ./isaaclab.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
COLLECT="projects/franka_v1_skill_lab/teleop_collection/entries/collect_teleop_demos_joint_v1.py"
echo "[1_collect] V1 task: $V1_TASK_ID (joint action)"
echo "[1_collect] V1 collection runs through teleop_collection (joint), NOT the legacy IK-Rel keyboard collector."

# --dry_run is fully offline; the live loop must run under ./isaaclab.sh.
if [[ " $* " == *" --dry_run "* ]]; then
  exec python "$COLLECT" --task "$V1_TASK_ID" --out_dir "$RAW_HDF5_DIR" --task_config "$TASK_CONFIG" "$@"
fi
echo "[1_collect] LIVE collection -> launching Isaac (./isaaclab.sh). Default: mock GELLO, 1 episode."
exec ./isaaclab.sh -p "$COLLECT" \
  --num_envs 1 --task "$V1_TASK_ID" \
  --scene_registry "projects/franka_v1_skill_lab/scene/saved_scenes/v1_active/scene_v1_registry.json" \
  --teleop_device mock --record_objects --task_config "$TASK_CONFIG" \
  --out_dir "$RAW_HDF5_DIR" --max_steps_per_episode 300 --seed 1 "$@"
