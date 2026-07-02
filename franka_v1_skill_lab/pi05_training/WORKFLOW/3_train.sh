#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — pi0.5 LoRA fine-tune (V1 wrapper).
# STATUS: wrapper -> legacy (OpenPI, GPU; not auto-run)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
echo "[3_train] NOTE: training runs OpenPI in its OWN venv (.venv_openpi); never in env_isaaclab."
echo "[3_train] This wrapper does NOT auto-start a long training run."
echo "[3_train] Real command (run manually when ready):"
echo "  bash $LEGACY_WORKFLOW_DIR/3_train.sh --repo-id $V1_REPO_ID --steps $TRAIN_STEPS"
echo "[3_train] V1 action space is JOINT — ensure the OpenPI config matches (see docs/pi05_v1_migration.md)."
