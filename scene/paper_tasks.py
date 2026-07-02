"""Register the paper-owned, self-contained open_drawer scene task.

This is a PRIVATE copy of the shared scene (stackpkg = frozen copy of manipulation/stack + franka
cfg), so the paper experiment is decoupled from edits the other project makes to the shared
`isaaclab_tasks` franka config. Import this module (after putting `projects/paper/scene` on sys.path)
to register:

    Isaac-Paper-OpenDrawer-Franka-v0   (== the JointPolicy env, Sektion disabled, paper assets)
"""

from __future__ import annotations

import gymnasium as gym

# ensure the frozen scene package resolves even if only this file's dir is on the path
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

PAPER_OPEN_DRAWER_TASK = "Isaac-Paper-OpenDrawer-Franka-v0"

if PAPER_OPEN_DRAWER_TASK not in gym.registry:
    gym.register(
        id=PAPER_OPEN_DRAWER_TASK,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        kwargs={
            "env_cfg_entry_point":
                "stackpkg.franka.stack_joint_policy_env_cfg:FrankaCubeStackJointPolicyEnvCfg",
        },
        disable_env_checker=True,
    )
