# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 re-export shim — STATUS: wrapper.

DLS IK -> joint-target adapter used by grasp/place.

Real implementation: projects/franka_skill_state_machine/runtime/ik_joint_adapter
"""

from __future__ import annotations

from .._legacy import ensure_legacy_on_path as _ensure

_ensure()

from runtime.ik_joint_adapter import *  # noqa: E402,F401,F403
