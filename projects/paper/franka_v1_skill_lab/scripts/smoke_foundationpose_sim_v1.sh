#!/usr/bin/env bash
# Franka V1 — FoundationPose sim-GT smoke test. STATUS: ready (no Isaac, no GPU).
# Exercises the unified pose-entry schema + sim-GT adapter offline.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_fp_sim] 1/2 pose-adapter unit test"
"$PY" projects/franka_v1_skill_lab/perception_foundationpose/sim/test_sim_pose_adapter.py

echo "[smoke_fp_sim] 2/2 sim-GT --demo PoseResult"
"$PY" projects/franka_v1_skill_lab/perception_foundationpose/sim/sim_gt_pose_as_foundationpose.py --demo \
  | head -n 20

echo "[smoke_fp_sim] OK"
