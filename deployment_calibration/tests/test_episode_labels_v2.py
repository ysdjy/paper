"""Stage-0: episode labels are consistent with the physical outcome.

Asserts, for the latest stage0 run: success == (final_joint_position >= target - tolerance);
failure_reason is a known token; task_outcome_error == |final - target|; overshoot >= 0.
Run: python projects/paper/deployment_calibration/tests/test_episode_labels_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
DATA = _PAPER / "deployment_calibration" / "data"
sys.path.insert(0, str(_PAPER / "deployment_calibration"))
from contracts.episode_schema_v2 import FAILURE_REASONS  # noqa: E402


def main() -> int:
    runs = sorted(DATA.glob("stage0_drawer_validation_v2_*"))
    if not runs:
        raise SystemExit("no stage0 run found")
    run = runs[-1]
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    fails = []
    for e in eps:
        y, g = e["y"], e["g"]
        fp = y["final_joint_position"]
        if fp != fp:  # nan
            fails.append(f"{e['episode_id']}: final_joint_position is nan"); continue
        expect = fp >= g["target_open_position"] - g["target_tolerance"]
        if bool(y["success"]) != bool(expect):
            fails.append(f"{e['episode_id']}: success={y['success']} but final={fp:.4f} vs "
                         f"target-tol={g['target_open_position']-g['target_tolerance']:.4f}")
        if y["failure_reason"] not in FAILURE_REASONS:
            fails.append(f"{e['episode_id']}: unknown failure_reason {y['failure_reason']}")
        if abs(y["task_outcome_error"] - abs(fp - g["target_open_position"])) > 1e-4:
            fails.append(f"{e['episode_id']}: task_outcome_error mismatch")
        if y["overshoot"] < -1e-6:
            fails.append(f"{e['episode_id']}: negative overshoot")
        if not y["success"] and y["failure_reason"] == "NONE":
            fails.append(f"{e['episode_id']}: failed but failure_reason=NONE")
    for f in fails[:20]:
        print("FAIL:", f)
    print(f"[test_episode_labels_v2] {run.name}: {len(eps)} eps, {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
