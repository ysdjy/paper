#!/usr/bin/env bash
# Franka V1 pi0.5 pipeline — serve policy over HTTP (V1 wrapper).
# STATUS: wrapper -> legacy mock/openpi server
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/pipeline.env"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"
echo "[4_serve] policy server backend=$POLICY_BACKEND host=$POLICY_HOST port=$POLICY_PORT"
LEGACY_SERVE="$LEGACY_PI05_DIR/scripts/serve_pi05_checkpoint.sh"
MOCK_SERVE="$LEGACY_PI05_DIR/scripts/start_mock_server.sh"
if [ "$POLICY_BACKEND" = "mock" ] && [ -f "$MOCK_SERVE" ]; then
  exec bash "$MOCK_SERVE" --port "$POLICY_PORT" "${@:-}"
fi
echo "[4_serve] For openpi backend run: bash $LEGACY_SERVE --port $POLICY_PORT --backend openpi ..."
echo "[4_serve] (mock server script not found — see $LEGACY_PI05_DIR/scripts/)"
