# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO read -> safe Franka joint target + gripper command. STATUS: ready.

Wraps :class:`gello.joint_teleop_safety.JointTeleopSafety` and adds the gripper
convention mapping. One :meth:`map` call turns a raw GELLO read into:
  * ``q_des`` [7]  — safe Franka arm joint target (filtered + clamped), and
  * ``gripper_cmd`` — binary {-1.0 open, +1.0 close} for the joint-position env
    action manager, derived from the GELLO gripper scalar in [0, 1] (0=open).
  * ``gripper_norm`` — the continuous [0, 1] (open..closed) value, recorded as
    the pi0.5 gripper action.

This is the single place the GELLO frame is converted; the test entry and the
collection entry both call it so behaviour is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

from .joint_teleop_safety import JointTeleopSafety


@dataclass
class GelloToFrankaMapper:
    safety: JointTeleopSafety = None  # type: ignore[assignment]
    gripper_open_below: float = 0.5   # gello gripper <= this -> open

    def __post_init__(self):
        if self.safety is None:
            self.safety = JointTeleopSafety()

    def reset(self, q0: list[float] | None = None) -> None:
        self.safety.reset(q0)

    def estop(self) -> None:
        self.safety.estop()

    def release_estop(self) -> None:
        self.safety.release_estop()

    def map(self, q_raw: list[float], gripper: float) -> dict:
        """Return {"q_des":[7], "gripper_cmd": ±1.0, "gripper_norm": [0,1]}."""
        q_des = self.safety.step(list(q_raw))
        g = max(0.0, min(1.0, float(gripper)))
        gripper_cmd = 1.0 if g <= self.gripper_open_below else -1.0  # +1 open, -1 close
        return {"q_des": q_des, "gripper_cmd": gripper_cmd, "gripper_norm": g}
