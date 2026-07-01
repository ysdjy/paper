# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""SceneSession 的配置 + reset 模式枚举（纯 python，任何 venv 可 import）。

这里不 import isaac/torch/gym —— 只是普通 dataclass，描述「怎么启动场景」。
真正的启动在 session.py（延迟 import）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# scene.v1_task_ids 是纯 python（无 isaac），作为任务 id 的唯一来源，不写死字面量。
from franka_v1_skill_lab.scene.v1_task_ids import V1_BASE_TASK_ID


class ResetMode(str, Enum):
    """reset() 时 cube/knife 如何初始化。"""

    STATIC = "static"   # USD 原样，不动物体（调试 / FoundationPose 标定）
    REGION = "region"   # 在四黄块 InitCorner 区域内随机（推荐默认，符合初始化区域契约）
    LEGACY = "legacy"   # SimpleSceneLayoutManager 硬编码 PLACEMENT_REGION（向后兼容旧 eval）


class SceneMode(str, Enum):
    """场景模式。"""

    USE = "use"     # 使用模式：只可见资产；target 箭头接口预留，平时不画
    TEST = "test"   # 测试模式：碰撞viz + target viz + 双 UI（由 test_mode_ui entry 驱动）


@dataclass
class SceneConfig:
    """启动场景所需的全部可调参数。"""

    # --- 模式 ---
    mode: SceneMode = SceneMode.USE   # USE=只可见资产+预留箭头接口；TEST=碰撞/目标viz+双UI

    # --- 任务 / 设备 ---
    task_id: str = V1_BASE_TASK_ID
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    use_fabric: bool = True
    seed: int = 1

    # --- 控制 ---
    control_hz: float = 50.0              # -> decimation = round(100/control_hz)
    arm_stiffness: float | None = 400.0   # None = 用任务默认 (软 80/4)；400/80 跟踪更紧
    arm_damping: float | None = 80.0
    free_microwave_door: bool = True

    # --- 相机（两相机模型：front + wrist，各出 RGB+RGBD，同一相机同视角）---
    # 同一物理相机同时渲染 RGB(给 VLA) 和 depth(给 FoundationPose)，所以 VLA 的 RGB 与
    # FP 的 RGBD 视角完全一致。两相机都用 FP 原生 640x480 渲染；VLA 端再缩到 vla_rgb_size。
    enable_cameras: bool = True               # 挂 front+wrist 两相机（需 --enable_cameras）
    enable_fp: bool = True                    # 渲染 depth（FoundationPose 用）；False=仅 RGB，FP 块为 None
    camera_resolution: tuple[int, int] = (640, 480)   # 统一渲染分辨率（FP 原生）
    vla_rgb_size: tuple[int, int] = (256, 256)        # pi05 image/wrist_image 缩放目标（匹配训练形状）

    # --- 最新场景同步 ---
    # True = 建 env 前用最新保存场景 (scene_v1_latest) 的物体位姿/缩放覆盖 base cfg。
    # 默认 False：用 base task cfg 里校准好的家电位姿（稳定）。仅当 scene_v1_latest 是干净的
    # 保存时再开 True，否则脏 manifest（家电悬空/非均匀缩放）会污染场景。
    load_latest_scene: bool = False
    scene_registry_path: str | None = None   # None = 用默认 active registry
    add_microwave_stand: bool = True          # 微波炉悬空时在底下加带碰撞的台子（自动判断）
    spawn_init_markers: bool = False          # spawn 黄色 InitCorner 方框（仅 TEST 用，保存时保留区域）
    refine_handle_collisions: bool = False    # spawn 时把把手 link 设 convexDecomposition（物理 init 前，安全）
    apply_saved_camera_offsets: bool = False  # 重载时用 registry 里保存的相机 offset 覆盖默认（TEST 用）

    # --- 资产替换 ---
    replace_microwave_with_fridge: bool = False                # 把 microwave 成员换成冰箱（保留位姿）
    fridge_usd: str = "SapienAssetPipeline/usd_assets/Fridge_10797/fridge.usd"
    fridge_scale: tuple[float, float, float] = (0.5, 0.5, 0.5)  # 冰箱缩放（默认；test 后再调）
    lock_knife: bool = False                                    # 锁死小刀关节（刀片不动）
    enable_collision_monitor: bool = False                      # 给机器人各连杆加 ContactSensor，运行时读碰撞力

    # --- 额外资产（往最新场景加任意 USD：本地路径 或 云端直链 http/https/omniverse://）---
    # 每项 dict: {name, usd_path, pos=(x,y,z), rot=(w,x,y,z), scale=(x,y,z), rigid=bool, mass=float}
    extra_assets: tuple = ()

    # --- 删除场景成员（把 base cfg 里的成员置 None 不 spawn，如不需要的冰箱/洗碗机）---
    exclude_members: tuple = ()

    # --- 静态场景：关掉自动重置 + 物体随机化（交互/teleop/采集场景不希望场景自己 reset）---
    # True = 关掉 randomize_cube_positions 事件 + episode 超时 + cube 掉落/堆叠成功等会触发
    # 自动 reset 的 termination。这样场景保持静止，物体不再每次 reset 随机跳位。
    disable_auto_reset: bool = False

    # --- 隐藏 contract 成员（cube_1/2/3 等）：它们是 base cfg 成员、不能直接置 None（obs/term 会引用），
    # 但可以把 init_state 挪到很远(场景外)使其不可见。apply_latest 之后再覆盖，确保生效。---
    # 每项 = 成员名；统一挪到 (HIDDEN_XY, HIDDEN_XY+i*dy, 0.05) 落地停在场景外，相机看不到。
    hidden_members: tuple = ()

    # --- 机器人支架（机器人坐在 Isaac stand 上，支架顶面对齐基座，机器人不移动）---
    add_robot_stand: bool = False
    robot_stand_usd: str = "SapienAssetPipeline/usd_assets/IsaacProps/Props/Mounts/Stand/stand_instanceable.usd"

    # --- 图像流（外部实时查看器）---
    enable_image_stream: bool = False
    image_stream_addr: str = "tcp://*:5557"

    # --- reset ---
    reset_mode: ResetMode = ResetMode.REGION
    tracked_objects: tuple[str, ...] = ("cube_1", "cube_2", "cube_3", "knife")

    # reset 后让渲染/物理稳定的步数（相机首帧需要它，否则黑图/旧位姿）
    settle_steps: int = 8

    def wants_cameras(self) -> bool:
        return bool(self.enable_cameras)
