# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""观测的结构化容器（纯 python dataclass，numpy 仅类型注解，不顶层 import）。

两相机模型（front + wrist）：每个相机是 ONE 物理相机，同时渲染 RGB 和 depth，
所以 VLA 用的 RGB 和 FoundationPose 用的 RGBD **视角完全一致**。`Observation.cameras`
是这两路 RGBD 的单一数据源；pi05 / foundationpose 两块只是它的不同视图：

  * CameraView          —— 一个相机的一帧：rgb(640) / depth(640) / intrinsics / 世界位姿
  * Observation.cameras —— {"front": CameraView, "wrist": CameraView}
  * Pi05Block           —— image/wrist_image（由 front/wrist 的 rgb 缩放而来）+ state8
  * FoundationPoseBlock —— 每个相机的 rgb+depth+内参+位姿 + 物体 GT 位姿；fp_input(view)
  * RawBlock            —— 关节/速度/TCP/夹爪宽度

`to_dict()` 给跨进程脚本序列化用：只放数值与图像 shape（不内联大图数组，图像请由
调用方自行 base64，参考 eval 的 _b64_png）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 接口里 front/wrist 两相机的逻辑名（与 sensors 包的实际相机名解耦）
FRONT_VIEW = "front"
WRIST_VIEW = "wrist"


def _shape_of(arr) -> list | None:
    if arr is None:
        return None
    shp = getattr(arr, "shape", None)
    return list(shp) if shp is not None else None


@dataclass
class CameraView:
    """一个相机的一帧 RGBD（同一视角同时供 VLA-RGB 与 FP-RGBD）。"""

    name: str                       # 逻辑名 front / wrist
    sensor_name: str                # 底层 Isaac 相机名（vla_front_static / vla_libero_eye_in_hand）
    rgb: Any = None                 # HxWx3 uint8（原生 640x480）
    depth: Any = None               # HxW float32 米（enable_fp 时有）
    intrinsics: dict = field(default_factory=dict)
    camera_pose_world: dict = field(default_factory=dict)
    _frame: dict | None = field(default=None, repr=False)   # 完整 frame dict，供 fp_input 复用

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sensor_name": self.sensor_name,
            "rgb_shape": _shape_of(self.rgb),
            "depth_shape": _shape_of(self.depth),
            "intrinsics": self.intrinsics,
            "camera_pose_world": self.camera_pose_world,
        }


@dataclass
class RawBlock:
    """底层机器人状态（绝对量）。"""

    joint_pos: list[float]        # 全部关节（arm7 + finger2 = 9）
    arm_joint_pos: list[float]    # 7 维手臂绝对关节弧度
    joint_vel: list[float]
    tcp_pose: list[float]         # [x,y,z, qx,qy,qz, qw]，position 为 env-local
    gripper_width: float          # 两指关节位置之和（米），不是 0/1
    base_pose_w: list[float] = field(default_factory=list)  # 机器人基座世界位姿 [x,y,z, qw,qx,qy,qz]

    def to_dict(self) -> dict:
        return {
            "joint_pos": self.joint_pos,
            "arm_joint_pos": self.arm_joint_pos,
            "joint_vel": self.joint_vel,
            "tcp_pose": self.tcp_pose,
            "gripper_width": self.gripper_width,
            "base_pose_w": self.base_pose_w,
        }


@dataclass
class Pi05Block:
    """pi0.5 / LeRobot 观测。state8 与训练数据布局一致：[arm_joint_pos(7), gripper_width(1)]。

    image / wrist_image 是 front / wrist 相机 RGB 缩放到 vla_rgb_size 的结果（与 FP 的 RGBD 同源同视角）。
    """

    image: Any = None          # front RGB -> LeRobot images.image
    wrist_image: Any = None    # wrist RGB -> LeRobot images.wrist_image
    state8: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state8": self.state8,
            "image_shape": _shape_of(self.image),
            "wrist_image_shape": _shape_of(self.wrist_image),
        }


@dataclass
class FoundationPoseBlock:
    """FoundationPose 输入：front + wrist 两个相机的 RGBD（与 VLA-RGB 同视角）+ 物体 GT 位姿（世界系）。"""

    views: dict = field(default_factory=dict)          # {"front": CameraView, "wrist": CameraView}
    object_gt_poses: list[dict] = field(default_factory=list)
    _tracked: tuple = field(default=(), repr=False)
    _pose_source: str = field(default="sim_gt", repr=False)

    def view(self, which: str = WRIST_VIEW) -> CameraView | None:
        return self.views.get(which)

    def fp_input(self, which: str = WRIST_VIEW, dump_dir: str | None = None) -> dict:
        """组装某个相机视角的 FoundationPose 输入包（复用现有 builder）。默认用腕部视角。"""
        from franka_v1_skill_lab.perception_foundationpose.foundationpose.foundationpose_input_builder import (
            build_foundationpose_input,
        )

        cam = self.views.get(which)
        if cam is None:
            raise KeyError(f"FoundationPose view {which!r} not available; have {list(self.views)}")
        frame = cam._frame or {
            "name": cam.sensor_name, "frame_id": cam.sensor_name,
            "rgb": cam.rgb, "depth": cam.depth,
            "intrinsics": cam.intrinsics, "camera_pose_world": cam.camera_pose_world,
        }
        return build_foundationpose_input(
            frame, objects=list(self._tracked), pose_source=self._pose_source, dump_dir=dump_dir
        )

    def to_dict(self) -> dict:
        return {
            "views": {k: v.to_dict() for k, v in self.views.items()},
            "object_gt_poses": self.object_gt_poses,
        }


@dataclass
class Observation:
    """一次 observe() 的完整结果。"""

    pi05: Pi05Block
    raw: RawBlock
    cameras: dict = field(default_factory=dict)        # {"front": CameraView, "wrist": CameraView}（RGBD 单一数据源）
    foundationpose: FoundationPoseBlock | None = None
    # 门把手/抽屉把手/拉手在**机器人基坐标系**下的 pose（给状态机模块读）。
    # {name: {asset, link, calibrated, position[3], quat_wxyz[4], position_world[3], quat_wxyz_world[4]}}
    handles: dict = field(default_factory=dict)
    sim_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sim_time": self.sim_time,
            "pi05": self.pi05.to_dict(),
            "raw": self.raw.to_dict(),
            "cameras": {k: v.to_dict() for k, v in self.cameras.items()},
            "foundationpose": self.foundationpose.to_dict() if self.foundationpose else None,
            "handles": self.handles,
        }
