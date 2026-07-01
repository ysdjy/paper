# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Re-export of the joint-teleop safety pipeline under the spec name. STATUS: ready.

The real implementation is :class:`gello.joint_teleop_safety.JointTeleopSafety`
(EMA low-pass -> per-step clamp -> joint-limit clamp -> e-stop latch). This
module exists so the recording layer can ``from ..recording.teleop_safety_filter
import TeleopSafetyFilter`` without a second copy of the logic.
"""

from __future__ import annotations

from ..gello.joint_teleop_safety import FRANKA_JOINT_LIMITS, JointTeleopSafety  # noqa: F401

# Spec alias.
TeleopSafetyFilter = JointTeleopSafety
