#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline smoke test for the sim-GT pose adapter (no Isaac, no torch).

Verifies:
  * PoseEntry / PoseResult round-trip through dict;
  * the unified entry schema validates;
  * poses_from_provider() works against a fake provider (w-first -> xyzw);
  * the --demo PoseResult is contract-valid.

Run:
    python projects/franka_v1_skill_lab/perception_foundationpose/sim/test_sim_pose_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECTS_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECTS_DIR))

from franka_v1_skill_lab.perception_foundationpose.sim.pose_entry import (  # noqa: E402
    PoseEntry,
    PoseResult,
    validate_entry,
)
from franka_v1_skill_lab.perception_foundationpose.sim.sim_gt_pose_as_foundationpose import (  # noqa: E402
    _demo_result,
    poses_from_provider,
)


class _FakeObj:
    def __init__(self, pose):
        self.pose = pose  # [x,y,z, qw,qx,qy,qz]


class _FakeState:
    def __init__(self):
        self.objects = {
            "cube_1": _FakeObj([0.45, 0.05, 0.02, 1.0, 0.0, 0.0, 0.0]),
            "knife": _FakeObj([0.40, -0.10, 0.02, 0.9239, 0.0, 0.0, 0.3827]),
        }


class _FakeProvider:
    def get_state(self):
        return _FakeState()


def main() -> int:
    failures = []

    # 1) round-trip
    e = PoseEntry(name="cube_1", position=[1, 2, 3], quat=[0, 0, 0, 1], confidence=1.0)
    if PoseEntry.from_dict(e.to_dict()).to_dict() != e.to_dict():
        failures.append("PoseEntry dict round-trip mismatch")

    # 2) schema validation
    if validate_entry(e.to_dict()):
        failures.append(f"valid entry flagged: {validate_entry(e.to_dict())}")
    if not validate_entry({"name": "x"}):
        failures.append("invalid entry not flagged")

    # 3) provider conversion (w-first -> xyzw)
    res = poses_from_provider(_FakeProvider(), tracked=("cube_1", "knife"))
    knife = next(o for o in res.objects if o.name == "knife")
    # input wxyz = [0.9239, 0, 0, 0.3827] -> xyzw = [0, 0, 0.3827, 0.9239]
    expected = [0.0, 0.0, 0.3827, 0.9239]
    if any(abs(a - b) > 1e-6 for a, b in zip(knife.quat, expected)):
        failures.append(f"quat conversion wrong: {knife.quat} != {expected}")
    if res.source != "sim_gt":
        failures.append("provider result source should be sim_gt")

    # 4) demo result is contract-valid
    demo = _demo_result()
    for o in demo.objects:
        probs = validate_entry(o.to_dict())
        if probs:
            failures.append(f"demo entry {o.name} invalid: {probs}")
    if len(demo.objects) < 6:
        failures.append("demo should emit >= 6 objects")

    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK — sim-GT pose adapter smoke test passed ({len(demo.objects)} demo objects).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
