# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Bootstrap access to the legacy skill-state-machine implementation.

STATUS: wrapper.

The real, battle-tested joint-action skill code still lives in
``projects/franka_skill_state_machine``. Rather than copy ~5k lines of working
runtime into V1, the V1 ``skill_runtime`` package re-exports from it. This
helper puts the legacy package root on ``sys.path`` so its top-level imports
(``runtime.*``, ``state_machine.*``, ``skills.*``, ``learned_drawer.*``)
resolve.

When the legacy code is eventually folded into V1, only this file and the thin
re-export shims need to change.
"""

from __future__ import annotations

import sys
from pathlib import Path

# …/projects/franka_v1_skill_lab/skill_runtime/_legacy.py
#   parents[0] = skill_runtime
#   parents[1] = franka_v1_skill_lab
#   parents[2] = projects
LEGACY_ROOT = (Path(__file__).resolve().parents[2] / "franka_skill_state_machine").resolve()


def ensure_legacy_on_path() -> Path:
    """Insert the legacy skill-state-machine root onto ``sys.path`` (idempotent)."""
    root = str(LEGACY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return LEGACY_ROOT
