"""Stage-0: reset invariants across repeats of the same drawer are within thresholds.

Loads the latest stage0_drawer_validation_v2_* run and asserts, per drawer, that the 5 full resets
produced identical initial robot joints / TCP / drawer joint (within thresholds) and ~zero velocity.
Pure-python; run: python projects/paper/deployment_calibration/tests/test_reset_invariants_v2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PAPER = Path(__file__).resolve().parents[2]
DATA = _PAPER / "deployment_calibration" / "data"
THRESH = {"robot_joint": 1e-3, "tcp": 2e-3, "drawer": 1e-4, "vel": 5e-2}


def _latest_run():
    runs = sorted(DATA.glob("stage0_drawer_validation_v2_*"))
    if not runs:
        raise SystemExit("no stage0 run found")
    return runs[-1]


def _maxdiff(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), list)]
    if len(vals) < 2:
        return 0.0
    a = np.array(vals, dtype=float)
    return float(np.max(np.max(a, axis=0) - np.min(a, axis=0)))


def main() -> int:
    run = _latest_run()
    eps = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    by = {}
    for e in eps:
        by.setdefault(e["drawer_name"], []).append(e["reset_invariants"])
    fails = []
    for dn, invs in by.items():
        rjd = _maxdiff(invs, "robot_joint_pos")
        tcd = _maxdiff(invs, "tcp_pos")
        drd = float(np.ptp([i["drawer_joint_pos"] for i in invs])) if len(invs) > 1 else 0.0
        rv = max(i["robot_joint_vel_absmax"] for i in invs)
        dv = max(abs(i["drawer_joint_vel"]) for i in invs)
        print(f"{dn:22s} robot_jd={rjd:.2e} tcp_d={tcd:.2e} drawer_d={drd:.2e} robot_v={rv:.2e} drawer_v={dv:.2e}")
        if rjd > THRESH["robot_joint"]: fails.append(f"{dn}: robot_joint reset diff {rjd:.2e} > {THRESH['robot_joint']}")
        if tcd > THRESH["tcp"]: fails.append(f"{dn}: tcp reset diff {tcd:.2e} > {THRESH['tcp']}")
        if drd > THRESH["drawer"]: fails.append(f"{dn}: drawer reset diff {drd:.2e} > {THRESH['drawer']}")
        if rv > THRESH["vel"]: fails.append(f"{dn}: robot init vel {rv:.2e} > {THRESH['vel']}")
        if dv > THRESH["vel"]: fails.append(f"{dn}: drawer init vel {dv:.2e} > {THRESH['vel']}")
    for f in fails:
        print("FAIL:", f)
    print(f"[test_reset_invariants_v2] {run.name}: {'PASS' if not fails else str(len(fails))+' FAIL'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
