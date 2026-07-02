# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""机器人 home 关节位姿的持久化(7 个手臂关节弧度)。

Franka 控制面板的「Set Home = current」把当前关节配置存到 saved_scenes/v1_active/robot_home.json，
面板的 Home 按钮 + 抽屉/门技能(runtime.drawer_ik_common)都从这里读，做到"设一次，处处一致"。
纯 stdlib，任何 venv 可 import。
"""

from __future__ import annotations

import json

HOME_FILE_NAME = "robot_home.json"


def _home_path():
    from pathlib import Path

    from franka_v1_skill_lab.scene.scene_registry import V1_ACTIVE_DIR
    return Path(V1_ACTIVE_DIR) / HOME_FILE_NAME


def load_home_q(default):
    """读保存的 home 关节配置(7 浮点)。缺失/损坏则返回 default(原样元组)。"""
    try:
        p = _home_path()
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            q = d.get("home_q")
            if isinstance(q, (list, tuple)) and len(q) == 7:
                return tuple(float(v) for v in q)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[robot_home] load failed: {exc}", flush=True)
    return tuple(float(v) for v in default)


def saved_home_q():
    """返回保存的 home 关节配置(7 浮点)，没有保存过则返回 None(不带默认)。
    cfg_hook 用它判断是否要把机器人 reset/起始姿态设成 home。"""
    try:
        p = _home_path()
        if p.is_file():
            q = json.loads(p.read_text(encoding="utf-8")).get("home_q")
            if isinstance(q, (list, tuple)) and len(q) == 7:
                return tuple(float(v) for v in q)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[robot_home] saved_home_q failed: {exc}", flush=True)
    return None


def save_home_q(q) -> bool:
    """把 home 关节配置(7 浮点)存盘。返回是否成功。"""
    try:
        p = _home_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"home_q": [float(v) for v in q]}, indent=2) + "\n", encoding="utf-8")
        print(f"[robot_home] saved home_q -> {p}", flush=True)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[robot_home] save failed: {exc}", flush=True)
        return False
