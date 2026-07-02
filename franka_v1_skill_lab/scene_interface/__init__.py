# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""scene_interface —— V1 场景的统一进程内接口（场景管理 AI 拥有/维护）。

其他模块从这里启动底层 Isaac 场景、读观测、发机器人 joint+夹爪指令；不用 ZMQ/ROS。

    from franka_v1_skill_lab.scene_interface import SceneSession, SceneConfig, ResetMode
    with SceneSession.launch(SceneConfig(enable_fp=True)) as session:
        obs = session.reset()
        obs = session.step(joint_target=q_des7, gripper=0)   # 绝对关节 + 0=open/1=close

顶层只 import 纯 python（config/observation）；SceneSession 的 isaac 依赖延迟到 launch()，
所以本包在 plain venv 也能 import。
"""

from __future__ import annotations

from .config import ResetMode, SceneConfig, SceneMode
from .observation import (
    CameraView,
    FoundationPoseBlock,
    Observation,
    Pi05Block,
    RawBlock,
)
from .session import SceneSession, pi05_gripper_to_env_cmd

__all__ = [
    "SceneSession",
    "SceneConfig",
    "ResetMode",
    "SceneMode",
    "Observation",
    "Pi05Block",
    "FoundationPoseBlock",
    "RawBlock",
    "CameraView",
    "pi05_gripper_to_env_cmd",
]
