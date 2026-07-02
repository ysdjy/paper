# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""GELLO q -> Franka joint-target safety pipeline (stage 2). STATUS: ready (logic), wrapper (Isaac wiring TODO).

Pure python (no numpy required; works on lists). Turns a raw GELLO joint read
into a safe Franka joint target via, in order:
  1. EMA low-pass filter (smooth sensor noise / jitter);
  2. per-step delta clamp (limit how far the target can jump per control step);
  3. joint-limit clamp (keep within Franka soft limits w/ margin);
  4. e-stop latch (freeze the target on stop).

These four are exactly the safety requirements in
``projects/gello_franka_teleop/docs/next_stage_isaac_control_plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Franka Panda joint soft limits (rad), 7 arm joints.
FRANKA_JOINT_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class JointTeleopSafety:
    num_joints: int = 7
    lowpass_alpha: float = 0.3       # 0..1, higher = snappier (less smoothing)
    max_step_rad: float = 0.05       # max per-step joint delta
    limits_margin_rad: float = 0.05  # shrink each soft limit by this
    joint_limits: list[tuple[float, float]] = field(default_factory=lambda: list(FRANKA_JOINT_LIMITS))

    _ema: list[float] | None = None
    _last_target: list[float] | None = None
    _estopped: bool = False

    def reset(self, q0: list[float] | None = None) -> None:
        self._ema = list(q0) if q0 is not None else None
        self._last_target = list(q0) if q0 is not None else None
        self._estopped = False

    def estop(self) -> None:
        self._estopped = True

    def release_estop(self) -> None:
        self._estopped = False

    def step(self, q_raw: list[float]) -> list[float]:
        """Map one raw GELLO joint vector to a safe Franka joint target."""
        q_raw = list(q_raw)[: self.num_joints]
        if self._estopped and self._last_target is not None:
            return list(self._last_target)

        # 1) EMA low-pass
        if self._ema is None:
            self._ema = list(q_raw)
        else:
            a = self.lowpass_alpha
            self._ema = [a * r + (1.0 - a) * e for r, e in zip(q_raw, self._ema)]

        # 2) per-step delta clamp (relative to last issued target)
        prev = self._last_target if self._last_target is not None else self._ema
        stepped = []
        for i, (target, p) in enumerate(zip(self._ema, prev)):
            delta = _clamp(target - p, -self.max_step_rad, self.max_step_rad)
            stepped.append(p + delta)

        # 3) joint-limit clamp (with margin)
        clamped = []
        for i, v in enumerate(stepped):
            lo, hi = self.joint_limits[i]
            clamped.append(_clamp(v, lo + self.limits_margin_rad, hi - self.limits_margin_rad))

        self._last_target = clamped
        return list(clamped)
