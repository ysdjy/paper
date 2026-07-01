# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""V1 re-export shim — STATUS: wrapper.

SkillRequest dataclass.

Real implementation: projects/franka_skill_state_machine/runtime/skill_request
"""

from __future__ import annotations

from .._legacy import ensure_legacy_on_path as _ensure

_ensure()

from runtime.skill_request import *  # noqa: E402,F401,F403
