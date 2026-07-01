# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""reset 随机化策略（STATIC / REGION / LEGACY）。

torch / 旧 runtime 全部延迟 import（这些类只在 SceneSession.launch 之后、env 存在时构造）。
统一接口：``apply(env, provider, *, seed, reset_index) -> dict``（返回 object_poses，可空）。

- STATIC：不动物体（USD 原样，调试 / FoundationPose 标定）。
- LEGACY：复用 SimpleSceneLayoutManager（硬编码 PLACEMENT_REGION），向后兼容旧 eval。
- REGION：子类化 SimpleSceneLayoutManager，只把采样源换成四黄块 InitCorner 区域
  （init_region.load_init_region / sample_in_region），写位姿 + 物理重置全继承。
"""

from __future__ import annotations

from .config import ResetMode

# REGION 采样避让半径（env-local，米）：物体不能初始化在机器人基座 / 家电脚印圆内，
# 否则一进场景就和它们穿模、被物理弹飞。机器人在 env-local (0,0)。
ROBOT_KEEPOUT_R = 0.32
APPLIANCE_KEEPOUT_R = 0.28


class StaticReset:
    """不随机化：物体停在 USD 保存的位置。"""

    def apply(self, env, provider, *, seed: int, reset_index: int) -> dict:
        return {}


class LegacyReset:
    """复用 SimpleSceneLayoutManager 的硬编码 PLACEMENT_REGION 采样。"""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self._mgr = None

    def _manager(self, env):
        if self._mgr is None:
            from runtime.simple_scene_layout import SimpleSceneLayoutManager

            self._mgr = SimpleSceneLayoutManager(env=env, base_seed=self.base_seed)
        return self._mgr

    def apply(self, env, provider, *, seed: int, reset_index: int) -> dict:
        result = self._manager(env).reset_layout(reset_index=reset_index)
        return getattr(result, "object_poses", {}) or {}


class RegionReset:
    """在四黄块 InitCorner 区域内随机（动态读 manifest，符合初始化区域契约）。"""

    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self._mgr = None

    def _manager(self, env):
        if self._mgr is None:
            self._mgr = _build_region_manager(env, self.base_seed)
        return self._mgr

    def apply(self, env, provider, *, seed: int, reset_index: int) -> dict:
        result = self._manager(env).reset_layout(reset_index=reset_index)
        return getattr(result, "object_poses", {}) or {}


def build_reset_strategy(mode: ResetMode, base_seed: int):
    if mode == ResetMode.STATIC:
        return StaticReset()
    if mode == ResetMode.LEGACY:
        return LegacyReset(base_seed)
    return RegionReset(base_seed)


def _build_region_manager(env, base_seed: int):
    """延迟构造 RegionLayoutManager：覆写采样源 + 区域校验，其余继承。"""
    from runtime.simple_scene_layout import (
        MIN_MOVABLE_DISTANCE,
        PLACE_MIN_SEPARATION,
        SimpleSceneLayoutManager,
    )

    from franka_v1_skill_lab.scene.init_region import (
        load_init_region,
        point_in_region,
        sample_in_region,
    )

    import torch

    class RegionLayoutManager(SimpleSceneLayoutManager):
        def __init__(self, env, base_seed):
            super().__init__(env=env, base_seed=base_seed)
            self._region = None

        def _place_cabinet(self):
            # 不移动家电：柜子保持 env.reset 的位姿（= 保存场景/scene_sync 的位置）。
            # 只读当前 root 局部位姿返回给 result（供显示/校验），不写 sim。
            cab = self.scene["cabinet"]
            env_origin = self.scene.env_origins[self.env_id]
            local = cab.data.root_pos_w[self.env_id] - env_origin
            quat = cab.data.root_quat_w[self.env_id]
            return [float(v) for v in torch.cat((local, quat)).detach().cpu().tolist()]

        def _sample_region_xy(self, rng, count):
            # 每次 reset 动态读 manifest 的四黄块区域（随用户编辑而变，不缓存）。
            self._region = load_init_region()
            keepouts = self._keepout_circles()   # 机器人基座 + 家电脚印（env-local xy, r）
            # 区域可能被用户拖到压住机器人/家电；这里拒绝落在 keep-out 圆内的采样，重采直到全清。
            last = None
            for _ in range(200):
                pts = sample_in_region(
                    self._region, count, rng=rng, min_separation=PLACE_MIN_SEPARATION
                )
                last = pts
                if all(not self._hits_keepout(x, y, keepouts) for (x, y) in pts):
                    return pts
            print("[reset] WARNING: region overlaps robot/appliance keep-out; could not find a fully "
                  "clear layout in 200 tries -- using last sample (move the InitCorner markers off the "
                  "robot/appliances).", flush=True)
            return last

        def _keepout_circles(self):
            # 物体不能初始化在这些圆内（env-local xy, 半径 m）：机器人基座 + 各家电脚印。
            circles = [(0.0, 0.0, ROBOT_KEEPOUT_R)]
            env_origin = self.scene.env_origins[self.env_id]
            for name in ("cabinet", "microwave", "coffee_machine"):
                try:
                    if name in self.scene.keys():
                        p = (self.scene[name].data.root_pos_w[self.env_id] - env_origin).detach().cpu().tolist()
                        circles.append((float(p[0]), float(p[1]), APPLIANCE_KEEPOUT_R))
                except Exception:
                    pass
            return circles

        @staticmethod
        def _hits_keepout(x, y, circles):
            for cx, cy, r in circles:
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    return True
            return False

        def _validate_layout(self, object_poses):
            # 用 InitCorner 区域校验（基类校验的是硬编码 PLACEMENT_REGION，对不上）。
            import math

            region = self._region or load_init_region()
            for name in ("cube_1", "cube_2", "cube_3", "knife"):
                pose = object_poses.get(name)
                if pose is None:
                    continue
                x, y = pose[0], pose[1]
                if not all(math.isfinite(v) for v in pose):
                    raise ValueError(f"{name} has non-finite pose: {pose}")
                if not point_in_region(x, y, region):
                    raise ValueError(f"{name} sampled outside InitCorner region: ({x:.3f}, {y:.3f})")
            # 最小间距（与基类一致）
            movable = [object_poses[n] for n in ("cube_1", "cube_2", "cube_3", "knife") if n in object_poses]
            for i in range(len(movable)):
                for j in range(i + 1, len(movable)):
                    d = math.dist(movable[i][:2], movable[j][:2])
                    if d < MIN_MOVABLE_DISTANCE:
                        raise ValueError(f"movable objects too close: {d:.3f} < {MIN_MOVABLE_DISTANCE}")

    return RegionLayoutManager(env=env, base_seed=base_seed)
