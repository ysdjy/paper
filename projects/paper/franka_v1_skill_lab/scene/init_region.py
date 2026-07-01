# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Cube / knife 随机初始化区域 —— 由场景里的四个黄色 InitCorner 方块定义。

契约（HARD RULE）
-----------------
布局编辑器 (`layout_editor/layout_v1_ui.py`) 在场景里放了四个黄色方块
``InitCorner_0..3``。用户在编辑器里拖动它们，**Save V1** 时它们的世界坐标被写进
manifest (`scene_v1_latest.json`) 的 objects 里。这四个点构成一个四边形，**就是**
三个 cube (`cube_1/2/3`) 和小刀 (`knife`) 每次场景初始化时允许出现的区域。

任何做场景 reset / 随机化的模块，都必须把 ``cube_1/2/3`` 和 ``knife`` 采样在
**这四个点构成的区域内**（带最小间距，避免互相穿插）。区域随用户编辑而变，所以
**必须每次从 manifest 动态读取**，不要把坐标写死。

本模块是纯 python（仅 json/math/pathlib），任何 venv 都能 import，不依赖 Isaac/torch。

用法
----
    from franka_v1_skill_lab.scene.init_region import (
        load_init_region, sample_in_region, INIT_REGION_OBJECTS,
    )
    region = load_init_region()                 # -> [(x,y), (x,y), (x,y), (x,y)]  (CCW 凸多边形)
    xy = sample_in_region(region, count=4, seed=0)   # cube_1/2/3 + knife 的 (x,y)
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

# 四个黄色标定块的 prim 名（layout_v1_ui._spawn_init_region_markers 创建）。
INIT_CORNER_NAMES = ["InitCorner_0", "InitCorner_1", "InitCorner_2", "InitCorner_3"]

# 必须落在区域内初始化的物体（与场景契约的 movable 物体一致）。
INIT_REGION_OBJECTS = ["cube_1", "cube_2", "cube_3", "knife"]

# 采样默认值（与 runtime/simple_scene_layout.py 对齐）。
DEFAULT_MIN_SEPARATION = 0.15
DEFAULT_MAX_TRIES = 5000


def _active_manifest_path() -> Path:
    """定位当前最新场景的 manifest（scene_v1_latest.json）。"""
    from .scene_registry import load_registry

    return load_registry().manifest_path()


def _order_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """把四个角按质心极角排序成逆时针凸多边形（point_in_region 要求 CCW）。"""
    if not points:
        return []
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def load_init_region(manifest_path: str | Path | None = None) -> list[tuple[float, float]]:
    """从 manifest 读 InitCorner_0..3 的世界 (x,y)，返回 CCW 凸多边形顶点。

    缺少四个角时抛 ValueError —— 这是“场景没标定初始化区域”的明确错误，不要静默兜底。
    """
    path = Path(manifest_path) if manifest_path else _active_manifest_path()
    if not Path(path).is_file():
        raise FileNotFoundError(f"找不到场景 manifest：{path}（先用 layout UI 保存一个场景）。")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    found: dict[str, tuple[float, float]] = {}
    for obj in data.get("objects", []):
        name = obj.get("name", "")
        if name in INIT_CORNER_NAMES:
            t = (obj.get("world_transform") or {}).get("translation")
            if t and len(t) >= 2:
                found[name] = (float(t[0]), float(t[1]))
    missing = [n for n in INIT_CORNER_NAMES if n not in found]
    if missing:
        raise ValueError(
            f"manifest {path} 缺少初始化区域标定块 {missing}；"
            "请在 layout UI 里确认四个黄色 InitCorner 都在场景中并重新 Save。"
        )
    return _order_ccw([found[n] for n in INIT_CORNER_NAMES])


def load_init_corner_poses(manifest_path: str | Path | None = None) -> dict:
    """从 manifest 读 InitCorner_0..3 的世界 (x,y,z)，按【名字】返回 {name:(x,y,z)}。

    给 spawn_init_markers 用：保留每个角的身份 + 高度(z)，这样用户把方块抬到桌面上、Save 后，
    重载时各方块回到保存的 (x,y,z) 而不是默认落地。缺角不报错（返回已找到的部分；区域采样另由
    load_init_region 负责，它仍要求四角齐全）。
    """
    path = Path(manifest_path) if manifest_path else _active_manifest_path()
    if not Path(path).is_file():
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, tuple[float, float, float]] = {}
    for obj in data.get("objects", []):
        name = obj.get("name", "")
        if name in INIT_CORNER_NAMES:
            t = (obj.get("world_transform") or {}).get("translation")
            if t and len(t) >= 3:
                out[name] = (float(t[0]), float(t[1]), float(t[2]))
    return out


def point_in_region(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    """点是否在 CCW 凸多边形内（落在边上算在内）。"""
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < -1e-9:
            return False
    return True


def sample_in_region(
    polygon: list[tuple[float, float]],
    count: int,
    seed: int | None = None,
    rng: random.Random | None = None,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    max_tries: int = DEFAULT_MAX_TRIES,
) -> list[tuple[float, float]]:
    """在多边形内拒绝采样 ``count`` 个 (x,y)，两两间距 > min_separation。

    与 runtime/simple_scene_layout.py 的采样逻辑一致：在包围盒里采样，落在多边形外或
    离已接受点太近就拒绝；最后一次尝试无条件接受以保证有结果。
    """
    if rng is None:
        rng = random.Random(seed)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    points: list[tuple[float, float]] = []
    for _ in range(count):
        for j in range(max_tries):
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            if not point_in_region(x, y, polygon):
                continue
            far_enough = all(math.dist((x, y), p) > min_separation for p in points)
            if not points or far_enough or j == max_tries - 1:
                points.append((x, y))
                break
    return points
