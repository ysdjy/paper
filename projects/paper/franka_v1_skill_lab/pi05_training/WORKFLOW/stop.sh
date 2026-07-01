#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — stop policy server / pipeline procs (V1 wrapper).
# STATUS: wrapper -> legacy stop
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
STOP="$LEGACY_WORKFLOW_DIR/stop.sh"
if [ -f "$STOP" ]; then exec bash "$STOP" "${@:-}"; fi
echo "[stop] legacy stop script not found ($STOP). Kill the policy server on :$POLICY_PORT manually."
