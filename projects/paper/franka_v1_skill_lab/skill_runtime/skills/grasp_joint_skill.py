# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 re-export shim — STATUS: wrapper.

Joint-IK grasp skill.

Real implementation: projects/franka_skill_state_machine/skills/grasp_skill
"""

from __future__ import annotations

from .._legacy import ensure_legacy_on_path as _ensure

_ensure()

from skills.grasp_skill import *  # noqa: E402,F401,F403
