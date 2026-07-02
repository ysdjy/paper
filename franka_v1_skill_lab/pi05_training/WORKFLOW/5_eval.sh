#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — closed-loop eval in IsaacLab (joint) (V1 wrapper).
# STATUS: wrapper -> legacy eval, V1 task
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
echo "[5_eval] eval V1 task $V1_TASK_ID (joint action) against policy server :$POLICY_PORT"
echo "[5_eval] Real command (needs ./isaaclab.sh + running policy server):"
echo "  ./isaaclab.sh -p $LEGACY_PI05_DIR/scripts/isaaclab/run_policy_in_isaaclab.py \\"
echo "    --task $V1_TASK_ID --num_rollouts 5 --policy_port $POLICY_PORT"
echo "[5_eval] Ensure the legacy run script env-kind detection picks JOINT for this task."
