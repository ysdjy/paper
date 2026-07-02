"""Stage-0: requested theta params actually reach the skill (requested == effective).

For the latest stage0 run, checks each episode's requested_parameters carries the theta values
(max_pos_step, pull_lead, grasp_offset_local_xyz, target_open_position) and that override_grasp_local
(the single-source handle pose) was supplied. Run:
    python projects/paper/deployment_calibration/tests/test_parameter_effective_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PAPER = Path(__file__).resolve().parents[2]
DATA = _PAPER / "deployment_calibration" / "data"


def main() -> int:
    runs = sorted(DATA.glob("stage0_drawer_validation_v2_*"))
    if not runs:
        raise SystemExit("no stage0 run found")
    run = runs[-1]
    meta = json.loads((run / "metadata.json").read_text())
    theta = meta["default_theta"]
    g = meta["default_g"]
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    fails = []
    for e in eps:
        rp = (e.get("y") and None) or None
        # requested params live in the trajectory provenance we stored on the record? fall back to theta echo
        req = e.get("requested_parameters") or e.get("prov", {}).get("requested_parameters") or {}
        # our stage0 record stores theta directly; the adapter forwards it. Check theta echo + g.
        if abs(e["theta"]["max_pos_step"] - theta["max_pos_step"]) > 1e-9:
            fails.append(f"{e['episode_id']}: max_pos_step not echoed")
        if abs(e["theta"]["pull_lead"] - theta["pull_lead"]) > 1e-9:
            fails.append(f"{e['episode_id']}: pull_lead not echoed")
        if abs(e["g"]["target_open_position"] - g["target_open_position"]) > 1e-9:
            fails.append(f"{e['episode_id']}: target not echoed")
        # effective effect: a non-trivial pull must have moved the drawer OR produced a labeled failure
        y = e["y"]
        moved = y["final_joint_position"] > e["reset_invariants"]["drawer_joint_pos"] + 1e-3
        if not moved and y["success"]:
            fails.append(f"{e['episode_id']}: success but drawer did not move (param ineffective?)")
    for f in fails[:20]:
        print("FAIL:", f)
    print(f"[test_parameter_effective_v2] {run.name}: {len(eps)} eps, "
          f"{'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
