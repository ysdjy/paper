"""Drawer target config for the state-machine runtime.

Single source of truth lives in the task package so both the RL env (source/) and this runtime
(scripts/) read the SAME mapping:
    isaaclab_tasks.manager_based.manipulation.stack.config.franka.custom_drawer_config

Confirmed by debug_drawer_joint_scan.py: top=joint_0/link_0, middle=joint_2/link_2,
bottom=joint_1/link_1 (LOCKED). All prismatic, closed=0, open_direction=+1. Gripper +1 open / -1 close.
"""

from __future__ import annotations

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.custom_drawer_config import (  # noqa: F401
    CABINET_USD_SCALE,
    DEFAULT_TARGET,
    FUNCTIONAL_DRAWERS as _BASE_FUNCTIONAL,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.custom_drawer_config import (
    DRAWER_TARGETS as _BASE_DRAWER_TARGETS,
)

# 右下角地面的【白色 Sektion 橱柜】(env cfg 里作为 articulation 'sektion_cabinet' 接入)的两个抽屉。
# member 字段指明它属于哪个场景成员(默认 'cabinet'=旧的 Cabinet_44853)。link_name 是抽屉刚体；真正的
# 抓取把手位姿由用户在场景编辑里标定 handle_sektion_*_drawer(grasp_poses.json)覆盖，故 handle_offset 只是
# 兜底(0=抽屉刚体原点)。joint 名取自官方 sektion 资产：drawer_top_joint / drawer_bottom_joint。
SEKTION_DRAWER_TARGETS = {
    "sektion_top_drawer": {
        "display_name_zh": "白柜上抽屉",
        "member": "sektion_cabinet",
        "joint_name": "drawer_top_joint",
        "link_name": "drawer_handle_top",   # 把手刚体(随抽屉滑动)；兜底抓取目标≈把手杆
        "handle_frame": "sektion_top_drawer_handle",
        "handle_offset": (0.0, 0.0, 0.0),
        "success_threshold": 0.20,
        "closed_pos": 0.0,
        "open_direction": 1,
        "functional": True,
    },
    "sektion_bottom_drawer": {
        "display_name_zh": "白柜下抽屉",
        "member": "sektion_cabinet",
        "joint_name": "drawer_bottom_joint",
        "link_name": "drawer_handle_bottom",
        "handle_frame": "sektion_bottom_drawer_handle",
        "handle_offset": (0.0, 0.0, 0.0),
        "success_threshold": 0.20,
        "closed_pos": 0.0,
        "open_direction": 1,
        "functional": True,
    },
}

# 运行时(技能)用的扩展表 = 旧 Cabinet_44853 抽屉 + Sektion 白柜抽屉。RL 端仍 import 原始 custom_drawer_config，
# 不受影响。每个 entry 的 'member' 默认 'cabinet'(旧表没有该字段)。
DRAWER_TARGETS = {**_BASE_DRAWER_TARGETS, **SEKTION_DRAWER_TARGETS}
SEKTION_DRAWERS = list(SEKTION_DRAWER_TARGETS.keys())

# bottom-drawer handle proxy authored offset (link-local, pre-scale); see stack_joint_pos_env_cfg.py
HANDLE_PROXY_LOCAL_OFFSET = {
    "bottom_drawer": (0.11946, 0.01491, 1.06183),
}


def get_drawer_config(target_drawer: str) -> dict:
    if target_drawer not in DRAWER_TARGETS:
        raise KeyError(f"unknown target_drawer '{target_drawer}'; valid={list(DRAWER_TARGETS)}")
    return DRAWER_TARGETS[target_drawer]


def member_for(target_drawer: str) -> str:
    return get_drawer_config(target_drawer).get("member", "cabinet")


def joint_name_for(target_drawer: str) -> str:
    return get_drawer_config(target_drawer)["joint_name"]


def functional_drawers() -> list[str]:
    return list(_BASE_FUNCTIONAL)
