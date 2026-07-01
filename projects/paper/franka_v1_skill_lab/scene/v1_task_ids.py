# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Canonical task ids for the Franka V1 Skill Lab.

This is the ONE place that names the gym task id every V1 consumer must use.
If the base task id ever changes, change it here and nowhere else.

The V1 contract is JOINT ACTION. Do NOT point V1 at an IK-Rel / IK-Abs task.
"""

from __future__ import annotations

# --- V1 base task -----------------------------------------------------------
# Joint-position policy env: the action sent to env.step() is a joint target
# (7 arm joints + gripper), NOT a 7-D EE delta. This is the contract the whole
# franka_v1_skill_lab depends on.
V1_BASE_TASK_ID = "Isaac-Stack-Cube-Franka-JointPolicy-v0"

# The joint-position teleop/record sibling (same robot, same control mode).
# Kept here for the teleop / pi0.5 data-collection pipeline.
V1_JOINT_POS_TASK_ID = "Isaac-Stack-Cube-Franka-JointPos-v0"

# Control mode string stamped into every registry / manifest / dataset.
V1_CONTROL_MODE = "joint"

# Tasks that are explicitly NOT part of the V1 mainline (IK-Rel / IK-Abs).
# Listed only so tooling can warn loudly if someone wires them into V1.
NON_V1_TASK_IDS = (
    "Isaac-Stack-Cube-Franka-IK-Rel-v0",
    "Isaac-Stack-Cube-Franka-IK-Abs-v0",
)


def is_v1_task(task_id: str) -> bool:
    """True if ``task_id`` is an accepted V1 joint-action task id."""
    return task_id in (V1_BASE_TASK_ID, V1_JOINT_POS_TASK_ID)
