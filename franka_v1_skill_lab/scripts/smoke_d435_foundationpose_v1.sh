#!/usr/bin/env bash
# Franka V1 — D435 -> FoundationPose input smoke test. STATUS: ready (no Isaac, no GPU).
# Builds a FoundationPose input bundle from an offline wrist-D435 frame + sim-GT
# objects and checks the bundle is well-formed (camera + intrinsics + pose + objects).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

OUT="${TMPDIR:-/tmp}/franka_v1_fp_input.json"

echo "[smoke_d435_fp] 1/2 build FoundationPose input bundle (rgb+depth)"
"$PY" projects/franka_v1_skill_lab/perception_foundationpose/sim/test_d435_foundationpose_input.py \
  --out "$OUT" | sed 's/^/  /'
echo "  wrote $OUT"

echo "[smoke_d435_fp] 2/2 no-depth path emits an explicit WARNING (not silent)"
if "$PY" projects/franka_v1_skill_lab/perception_foundationpose/sim/test_d435_foundationpose_input.py \
     --no_depth 2>&1 | grep -q "WARNING: depth missing"; then
  echo "  depth-missing WARNING present — OK"
else
  echo "  ERROR: expected a depth-missing WARNING in --no_depth mode" >&2
  exit 1
fi

echo "[smoke_d435_fp] OK"
