# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 action adapter: policy joint action -> IsaacLab joint-position action. STATUS: ready (logic), wrapper (Isaac apply).

The V1 mainline action space is JOINT (7 arm joint targets + 1 gripper command),
NOT the legacy IK-Rel 7-D EE delta. This adapter takes a raw policy action and
turns it into a safe Franka joint target, then (on the Isaac side) hands it to
``SceneStateProvider.make_joint_action_from_q_des(q_des, gripper)``.

Pure-python clipping logic (unit-testable). The Isaac handoff is a one-liner the
eval script calls.
"""

from __future__ import annotations

from dataclasses import dataclass

# Reuse the Franka soft limits from the teleop safety module to stay consistent.
FRANKA_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


@dataclass
class JointActionAdapter:
    num_joints: int = 7
    limits_margin_rad: float = 0.02
    max_step_rad: float | None = 0.10   # None disables per-step clamp

    _last_q: list[float] | None = None

    ACTION_SPEC = {"joint_target": 7, "gripper_command": 1}

    def reset(self, q0: list[float] | None = None) -> None:
        self._last_q = list(q0) if q0 is not None else None

    def adapt(self, action: dict | list, current_q: list[float] | None = None) -> dict:
        """Return {"joint_target": [7], "gripper_command": float} after safety clips.

        ``action`` may be a dict {joint_target, gripper_command} or a flat list
        [q0..q6, gripper].
        """
        if isinstance(action, dict):
            q = list(action.get("joint_target", []))[: self.num_joints]
            gripper = float(action.get("gripper_command", 0.0))
        else:
            flat = list(action)
            q = flat[: self.num_joints]
            gripper = float(flat[self.num_joints]) if len(flat) > self.num_joints else 0.0

        if len(q) < self.num_joints:
            raise ValueError(f"joint_target needs {self.num_joints} elements, got {len(q)}")

        prev = current_q if current_q is not None else self._last_q
        out = []
        for i, target in enumerate(q):
            if self.max_step_rad is not None and prev is not None:
                target = prev[i] + _clamp(target - prev[i], -self.max_step_rad, self.max_step_rad)
            lo, hi = FRANKA_JOINT_LIMITS[i]
            out.append(_clamp(target, lo + self.limits_margin_rad, hi - self.limits_margin_rad))

        self._last_q = out
        gripper = _clamp(gripper, 0.0, 1.0)
        return {"joint_target": out, "gripper_command": gripper}

    def to_isaac_action(self, provider, action: dict | list, state=None):
        """Adapt then build the Isaac joint action via the provider. Needs Isaac."""
        current_q = None
        if state is not None:
            current_q = getattr(state.robot, "joint_pos", None)
        safe = self.adapt(action, current_q=current_q)
        return provider.make_joint_action_from_q_des(safe["joint_target"], safe["gripper_command"])
