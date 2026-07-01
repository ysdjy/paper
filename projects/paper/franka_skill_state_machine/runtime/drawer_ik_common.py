"""Shared geometry helpers for the IK-based open/close drawer skills.

These let the open/close drawer skills work for ANY drawer / cabinet pose: the state machine only
provides the target drawer (its handle pose + drawer joint are read live from the scene), and the
open/pull direction and grasp orientation are derived from the live handle vs cabinet geometry.
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

_UP = (0.0, 0.0, 1.0)

# 抽屉技能 turn-to-face 用的 home 臂姿（7 关节）：末端【垂直朝下】且比原 home 抬高 ~15cm。
# 由 IK 离线解出(目标=原 home TCP +Z15cm、朝向竖直)。抬高 + 竖直是为了机器人转身去拉抽屉时
# 不会用手肘/腕把桌面上的物品扫掉。drawer 技能 seed 用它，只把 joint1 转到朝向把手方位。
HOME_Q_VERTICAL_RAISED = (0.028, -0.330, -0.092, -2.223, -0.030, 1.912, 0.734)

# Cabinet-local axis the drawers slide out along (this asset opens along local -X). Transformed by
# the cabinet's world orientation it gives the world opening direction — the SAME for every drawer,
# so the grasp orientation is consistent across top/middle/bottom regardless of per-handle geometry.
LOCAL_OPEN_AXIS = (-1.0, 0.0, 0.0)


def open_direction_world(cabinet_quat: torch.Tensor) -> torch.Tensor:
    """Unit world vector the drawers open along, derived from the cabinet orientation (consistent
    for all drawers)."""
    axis = torch.tensor(LOCAL_OPEN_AXIS, device=cabinet_quat.device)
    d = math_utils.quat_apply(cabinet_quat.unsqueeze(0), axis.unsqueeze(0))[0].clone()
    d[2] = 0.0  # horizontal only
    n = torch.linalg.norm(d)
    if float(n) < 1e-6:
        return torch.tensor([0.0, -1.0, 0.0], device=cabinet_quat.device)
    return d / n


def grasp_quat_from_open_dir(open_dir: torch.Tensor, device) -> torch.Tensor:
    """TCP quaternion (w,x,y,z) for a front grasp of a horizontal handle bar.

    approach axis (TCP +Z) = -open_dir (gripper moves inward onto the handle);
    finger-open axis (TCP +Y) = world up (fingers straddle the bar top/bottom);
    the remaining axis runs along the (horizontal) handle bar.
    """
    up = torch.tensor(_UP, device=device)
    z = -open_dir / torch.linalg.norm(open_dir)
    x = torch.linalg.cross(up, z)
    nx = torch.linalg.norm(x)
    if float(nx) < 1e-6:  # approach nearly vertical: fall back to world X for the bar axis
        x = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        x = x / nx
    y = torch.linalg.cross(z, x)
    R = torch.stack((x, y, z), dim=1)  # columns = hand axes in world
    return math_utils.quat_from_matrix(R.unsqueeze(0))[0]
