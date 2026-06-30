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
    DRAWER_TARGETS,
    FUNCTIONAL_DRAWERS,
    get_drawer_config,
)

# bottom-drawer handle proxy authored offset (link-local, pre-scale); see stack_joint_pos_env_cfg.py
HANDLE_PROXY_LOCAL_OFFSET = {
    "bottom_drawer": (0.11946, 0.01491, 1.06183),
}


def joint_name_for(target_drawer: str) -> str:
    return get_drawer_config(target_drawer)["joint_name"]


def functional_drawers() -> list[str]:
    return list(FUNCTIONAL_DRAWERS)
