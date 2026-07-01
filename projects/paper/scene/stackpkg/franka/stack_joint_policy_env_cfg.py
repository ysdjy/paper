# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position variant of the modified Franka stack scene for the joint-action state machine.

This config is intentionally close to :class:`stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg`
but changes the arm :class:`JointPositionActionCfg` ``scale`` to ``1.0`` so that the action
contract matches Isaac Lab's official Franka open-drawer PPO policy (which trains with
``scale=1.0, use_default_offset=True``).

The custom cabinet's ``BottomHandleProxy`` is NOT a separate rigid body, so it cannot be the
target of a FrameTransformer. The drawer observation adapter therefore derives the handle world
pose from the cabinet ``link_1`` body pose plus the (scaled) proxy local offset. See
``skill_runtime/drawer_obs_adapter.py``.

All skills (grasp / place via internal IK, open-drawer via the learned policy) drive this single
joint-position environment.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from stackpkg import mdp

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaCubeStackJointPolicyEnvCfg(stack_joint_pos_env_cfg.FrankaCubeStackEnvCfg):
    """Joint-position stack scene for the method-B joint-action state machine.

    Priority: faithfully reproduce the IK-Abs grasp/place behaviour through joint-position control.
    We therefore use the SAME stiff PD robot as the IK-Abs env (``FRANKA_PANDA_HIGH_PD_CFG``: stiffer
    PD + gravity disabled on the arm) so the arm tracks the IK ``q_des`` tightly and does not sag.

    NOTE: this stiff-PD / gravity-off robot differs from the soft-PD robot used to train the official
    open-drawer PPO policy. The learned-drawer backend is currently deprioritized; revisit the robot
    config (e.g. a dedicated env or re-enabling gravity) when re-integrating that policy.

    Arm action keeps ``scale=1.0, use_default_offset=True`` so q_des -> raw is a clean offset.
    """

    def __post_init__(self):
        # post init of parent (builds the full stack + cabinet + knife scene)
        super().__post_init__()

        # Stiff PD robot (matches stack_ik_abs_env_cfg) for tight joint tracking, no sag.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.usd_path = stack_joint_pos_env_cfg._repo_path(
            "Connection/assets/Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Match the official open-drawer PPO action scale (1.0); also gives a clean q_des->raw map.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
