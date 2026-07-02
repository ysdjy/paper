#!/usr/bin/env bash
# Franka V1 — GELLO teleop smoke test. STATUS: ready (no hardware, no Isaac).
# Exercises the GELLO-read -> joint-target safety path with a synthetic source.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
PY="${PYTHON:-python3}"

echo "[smoke_gello] mock GELLO -> Franka joint target (safety pipeline)"
"$PY" projects/franka_v1_skill_lab/teleop_collection/entries/gello_to_isaac_joint_test.py \
  --mock_gello --steps 30

cat <<'NOTE'

[smoke_gello] NOTE: this is the hardware-free path.
To read a REAL GELLO (stage 1, read-only), use the legacy reader in its
isolated venv (it does NOT touch Isaac or this env):

    cd projects/gello_franka_teleop
    bash scripts/detect_gello_port.sh
    source .venv-gello/bin/activate
    python scripts/read_gello_joints.py --config configs/gello_franka.yaml --hz 30

NOTE
echo "[smoke_gello] OK"
