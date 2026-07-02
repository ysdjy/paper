# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""目标位姿箭头可视化 —— USE/TEST 通用的预留接口。

其他模块通过 SceneSession.show_target_poses(poses) 画一组目标位姿（坐标轴/箭头 marker），
平时不调用就不画。薄封装 franka_skill_state_machine 的 DebugVisualizer（懒构造 marker、
按 name 复用，避免每帧泄漏 prim）。headless / 未启用时全 no-op。

顶层只 import 纯 python；torch / isaac / DebugVisualizer 延迟到方法体内（构造在 launch 之后）。
"""

from __future__ import annotations

import re


def _normalize_poses(poses) -> list[list[float]]:
    """把多种输入统一成 [[x,y,z, qw,qx,qy,qz], ...]。

    支持：单个/列表的 PoseState(pos_w,quat_w) / (pos3,quat4) / 7元序列 / Nx7 tensor。
    """
    if poses is None:
        return []

    # Nx7 / 7 tensor 或 numpy
    tolist = getattr(poses, "tolist", None)
    if tolist is not None and not isinstance(poses, (list, tuple)):
        poses = tolist()

    # 单个 PoseState
    if hasattr(poses, "pos_w") and hasattr(poses, "quat_w"):
        poses = [poses]
    # 单个 7 元序列
    elif isinstance(poses, (list, tuple)) and len(poses) == 7 and all(_is_num(v) for v in poses):
        poses = [poses]
    # 单个 (pos, quat)
    elif (isinstance(poses, (list, tuple)) and len(poses) == 2
          and hasattr(poses[0], "__len__") and len(poses[0]) == 3):
        poses = [poses]

    out: list[list[float]] = []
    for p in poses:
        if hasattr(p, "pos_w") and hasattr(p, "quat_w"):
            pos = _seq(p.pos_w)
            quat = _seq(p.quat_w)
            out.append([*pos[:3], *quat[:4]])
        elif isinstance(p, (list, tuple)) and len(p) == 2:
            out.append([*_seq(p[0])[:3], *_seq(p[1])[:4]])
        elif hasattr(p, "__len__") and len(p) >= 7:
            row = _seq(p)
            out.append([float(row[i]) for i in range(7)])
    return out


def _is_num(v) -> bool:
    return isinstance(v, (int, float))


def _seq(v) -> list[float]:
    tolist = getattr(v, "tolist", None)
    if tolist is not None:
        v = tolist()
    return [float(x) for x in v]


class TargetPoseMarkers:
    """画一组目标位姿箭头；按 name 复用 marker，可 clear/hide。"""

    def __init__(self, enabled: bool = True, prefix: str = "target"):
        self.enabled = bool(enabled)
        self.prefix = prefix
        self._viz = None
        self._shown: set[str] = set()
        if self.enabled:
            try:
                from runtime.debug_visualizer import DebugVisualizer  # legacy path 已就绪

                self._viz = DebugVisualizer(enabled=True)
            except Exception:
                self.enabled = False

    def show(self, poses, names=None, use_arrows: bool = True, axis_length: float = 0.08) -> None:
        if not self.enabled or self._viz is None:
            return
        import torch

        items = _normalize_poses(poses)
        for i, p7 in enumerate(items):
            name = names[i] if (names is not None and i < len(names)) else f"{self.prefix}_{i}"
            self._viz.update_pose(
                name, torch.as_tensor(p7, dtype=torch.float32),
                axis_length=axis_length, use_coordinate_arrows=use_arrows,
            )
            self._shown.add(name)
            self._set_visible(name, True)

    def clear(self) -> None:
        for name in list(self._shown):
            self._set_visible(name, False)
        self._shown.clear()

    hide = clear

    def _set_visible(self, name: str, vis: bool) -> None:
        if self._viz is None:
            return
        path = "/Visuals/SkillRuntime/" + re.sub(r"[^A-Za-z0-9_]", "_", name)
        marker = getattr(self._viz, "_arrow_items", {}).get(path)
        if marker is not None and hasattr(marker, "set_visibility"):
            try:
                marker.set_visibility(bool(vis))
            except Exception:
                pass
