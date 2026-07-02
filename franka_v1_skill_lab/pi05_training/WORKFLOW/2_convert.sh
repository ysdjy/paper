#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — convert demos -> normalized + LeRobot (joint) (V1 wrapper).
# STATUS: ready (normalized always; LeRobot best-effort).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
echo "[2_convert] V1 joint-action HDF5 -> normalized + LeRobot ($LEROBOT_DIR/$V1_REPO_ID)"
# No args -> --dry_run (validate mapping only). Otherwise pass through, defaulting
# --input to latest.hdf5 and --output to the V1 LeRobot dir if not provided.
if [ "$#" -eq 0 ]; then
  exec python projects/franka_v1_skill_lab/pi05_training/adapters/skill_demo_to_lerobot.py --dry_run
fi
exec python projects/franka_v1_skill_lab/pi05_training/adapters/skill_demo_to_lerobot.py \
  --input "$RAW_HDF5_DIR/latest.hdf5" \
  --output "$LEROBOT_DIR/$V1_REPO_ID" \
  --task_config "$TASK_CONFIG" "$@"
