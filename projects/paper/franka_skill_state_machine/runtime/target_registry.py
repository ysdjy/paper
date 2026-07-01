"""Target registration and grasp affordance computation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils

from isaaclab_tasks.manager_based.manipulation.stack.config.franka import stack_joint_pos_env_cfg

from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_types import FailureReason


CUBE_SIZE = 0.0406
CUBE_PRE_GRASP_HEIGHT = 0.100
CUBE_GRASP_Z_OFFSET = 0.000
CUBE_PROBE_LIFT_HEIGHT = 0.030
CUBE_FULL_LIFT_HEIGHT = 0.120
CUBE_MIN_GRASP_WIDTH = 0.020
CUBE_MAX_GRASP_WIDTH = 0.060

TOP_DOWN_GRASP_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)

KNIFE_ASSET_SCALE = 0.12
KNIFE_HANDLE_CONFIG_CENTER_LOCAL = stack_joint_pos_env_cfg.KNIFE_HANDLE_PROXY_OFFSET
KNIFE_HANDLE_CENTER_LOCAL = tuple(value * KNIFE_ASSET_SCALE for value in KNIFE_HANDLE_CONFIG_CENTER_LOCAL)
KNIFE_HANDLE_LOCAL_QUAT = (1.0, 0.0, 0.0, 0.0)
KNIFE_HANDLE_CONFIG_SIZE = stack_joint_pos_env_cfg.KNIFE_HANDLE_PROXY_SIZE
KNIFE_HANDLE_SIZE = tuple(value * KNIFE_ASSET_SCALE for value in KNIFE_HANDLE_CONFIG_SIZE)
KNIFE_PRE_GRASP_HEIGHT = 0.100
KNIFE_GRASP_Z_OFFSET = 0.020
KNIFE_EXTRA_DESCENT = 0.5 * KNIFE_HANDLE_SIZE[2]
KNIFE_PROBE_LIFT_HEIGHT = 0.030
KNIFE_FULL_LIFT_HEIGHT = 0.120
KNIFE_GRIP_YAW_OFFSET = math.pi

WORKSPACE_X_MIN = 0.13  # cover the placement region (cubes randomized down to x~0.15)
WORKSPACE_X_MAX = 0.85
WORKSPACE_Y_ABS_MAX = 0.65
WORKSPACE_Z_MIN = 0.010
WORKSPACE_Z_MAX = 0.80


@dataclass(frozen=True)
class TargetConfig:
    name: str
    scene_key: str
    display_name: str
    geometry_type: str
    size: tuple[float, float, float]
    local_grasp_pos: tuple[float, float, float]
    pre_grasp_clearance: float
    probe_lift_height: float
    full_lift_height: float
    gripper_open_command: float
    gripper_close_command: float
    min_gripper_width: float
    max_gripper_width: float


@dataclass
class GraspPlan:
    target_name: str
    target_pose: PoseState
    grasp_pose: PoseState
    pre_grasp_pose: PoseState
    probe_lift_pose: PoseState
    full_lift_pose: PoseState
    approach_dir_w: torch.Tensor
    lift_distance: float
    gripper_open_command: float
    gripper_close_command: float
    min_gripper_width: float
    max_gripper_width: float
    valid: bool = True
    failure_reason: FailureReason = FailureReason.NONE
    message: str = ""
    object_quat_w: torch.Tensor | None = None
    object_yaw: float | None = None
    grasp_yaw: float | None = None


class TargetRegistry:
    """Keeps static target affordances separate from real-time object state."""

    def __init__(self, device: torch.device | str, cube_grasp_z_offset: float = CUBE_GRASP_Z_OFFSET):
        if cube_grasp_z_offset < -0.010 or cube_grasp_z_offset > 0.015:
            raise ValueError("--cube_grasp_z_offset must be in [-0.010, 0.015] m")
        self.device = torch.device(device)
        self.cube_grasp_z_offset = float(cube_grasp_z_offset)
        cube_size = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        self.targets: dict[str, TargetConfig] = {
            "cube_1": self._cube_cfg("cube_1", "Blue Cube (cube_1)", cube_size),
            "cube_2": self._cube_cfg("cube_2", "Red Cube (cube_2)", cube_size),
            "cube_3": self._cube_cfg("cube_3", "Green Cube (cube_3)", cube_size),
            "knife": TargetConfig(
                name="knife",
                scene_key="knife",
                display_name="Knife (knife)",
                geometry_type="knife_handle",
                size=KNIFE_HANDLE_SIZE,
                local_grasp_pos=KNIFE_HANDLE_CENTER_LOCAL,
                pre_grasp_clearance=KNIFE_PRE_GRASP_HEIGHT,
                probe_lift_height=KNIFE_PROBE_LIFT_HEIGHT,
                full_lift_height=KNIFE_FULL_LIFT_HEIGHT,
                gripper_open_command=1.0,
                gripper_close_command=-1.0,
                min_gripper_width=0.0,
                max_gripper_width=0.080,
            ),
        }

    def display_targets(self) -> list[tuple[str, str]]:
        return [(key, cfg.display_name) for key, cfg in self.targets.items()]

    def compute_grasp_plan(self, target_name: str, state: SceneState) -> GraspPlan:
        cfg = self.targets.get(target_name)
        if cfg is None:
            return self._invalid(target_name, FailureReason.REQUEST_INVALID, f"unknown target: {target_name}", state)
        if cfg.scene_key not in state.objects:
            return self._invalid(target_name, FailureReason.TARGET_LOST, f"target not found: {cfg.scene_key}", state)
        if cfg.geometry_type == "cube":
            return self._cube_grasp_plan(cfg, state)
        if cfg.geometry_type == "knife_handle":
            return self._knife_grasp_plan(cfg, state)
        return self._invalid(target_name, FailureReason.REQUEST_INVALID, "unsupported target type", state)

    def current_target_pose(self, target_name: str, state: SceneState) -> PoseState | None:
        plan = self.compute_grasp_plan(target_name, state)
        if not plan.valid:
            return None
        return plan.target_pose

    def _cube_cfg(self, scene_key: str, display_name: str, size: tuple[float, float, float]) -> TargetConfig:
        return TargetConfig(
            name=scene_key,
            scene_key=scene_key,
            display_name=display_name,
            geometry_type="cube",
            size=size,
            local_grasp_pos=(0.0, 0.0, self.cube_grasp_z_offset),
            pre_grasp_clearance=CUBE_PRE_GRASP_HEIGHT,
            probe_lift_height=CUBE_PROBE_LIFT_HEIGHT,
            full_lift_height=CUBE_FULL_LIFT_HEIGHT,
            gripper_open_command=1.0,
            gripper_close_command=-1.0,
            min_gripper_width=CUBE_MIN_GRASP_WIDTH,
            max_gripper_width=CUBE_MAX_GRASP_WIDTH,
        )

    def _snapped_gripper_yaw(self, state: SceneState) -> torch.Tensor:
        """Yaw of the current gripper snapped to the nearest 90 deg.

        A cube is symmetric, so the gripper can grab either opposite side-pair (yaw multiples of
        90 deg). Snapping to the nearest 90 deg of the CURRENT gripper yaw makes the arm rotate the
        least, and decouples the grasp from the cube's own orientation (robust to the cube flipping).
        """
        x_w = math_utils.quat_apply(state.robot.tcp_pose.quat_w.unsqueeze(0), self._vec3(1.0, 0.0, 0.0).unsqueeze(0))[0]
        yaw = float(torch.atan2(x_w[1], x_w[0]).detach().cpu())
        half_pi = math.pi / 2.0
        snapped = round(yaw / half_pi) * half_pi
        return torch.tensor(snapped, dtype=torch.float32, device=self.device)

    def _cube_grasp_plan(self, cfg: TargetConfig, state: SceneState) -> GraspPlan:
        obj = state.objects[cfg.scene_key]
        cube_center_w = obj.pose.pos_w
        cube_quat_w = math_utils.normalize(obj.pose.quat_w.unsqueeze(0))[0]
        # grasp orientation is NOT bound to the cube frame: always top-down, yaw chosen to minimize
        # arm rotation (nearest 90 deg of the current gripper yaw).
        grasp_yaw = self._snapped_gripper_yaw(state)
        grasp_quat = self._top_down_quat(grasp_yaw)
        grasp_pos = cube_center_w + self._vec3(0.0, 0.0, cfg.local_grasp_pos[2])
        pre_grasp_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.pre_grasp_clearance)
        probe_lift_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.probe_lift_height)
        full_lift_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.full_lift_height)
        grasp_pose = PoseState(grasp_pos, grasp_quat)
        pre_grasp_pose = PoseState(pre_grasp_pos, grasp_quat)
        safety_error = self._pose_safety_error(grasp_pose, pre_grasp_pose)
        if safety_error is not None:
            return self._invalid(cfg.name, FailureReason.TARGET_UNSAFE, safety_error, state)
        return self._valid_plan(
            cfg,
            obj.pose,
            grasp_pose,
            pre_grasp_pose,
            probe_lift_pos,
            full_lift_pos,
            object_quat_w=cube_quat_w,
            object_yaw=None,
            grasp_yaw=grasp_yaw,
        )

    def _knife_grasp_plan(self, cfg: TargetConfig, state: SceneState) -> GraspPlan:
        obj = state.objects[cfg.scene_key]
        link_pose = obj.links.get(stack_joint_pos_env_cfg.KNIFE_BODY_LINK, obj.links.get("base", obj.pose))
        handle_offset = torch.tensor(KNIFE_HANDLE_CENTER_LOCAL, dtype=torch.float32, device=self.device).unsqueeze(0)
        handle_quat_local = torch.tensor(KNIFE_HANDLE_LOCAL_QUAT, dtype=torch.float32, device=self.device).unsqueeze(0)
        handle_pos, handle_quat = math_utils.combine_frame_transforms(
            link_pose.pos_w.unsqueeze(0),
            link_pose.quat_w.unsqueeze(0),
            handle_offset,
            handle_quat_local,
        )
        handle_pos = handle_pos[0]
        handle_quat = math_utils.normalize(handle_quat)[0]
        knife_yaw = self._yaw_from_local_x(handle_quat)
        knife_grasp_yaw = knife_yaw + KNIFE_GRIP_YAW_OFFSET
        grasp_quat = self._top_down_quat(knife_grasp_yaw)
        grasp_pos = handle_pos + self._vec3(0.0, 0.0, KNIFE_GRASP_Z_OFFSET - KNIFE_EXTRA_DESCENT)
        pre_grasp_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.pre_grasp_clearance)
        probe_lift_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.probe_lift_height)
        full_lift_pos = grasp_pos + self._vec3(0.0, 0.0, cfg.full_lift_height)
        grasp_pose = PoseState(grasp_pos, grasp_quat)
        pre_grasp_pose = PoseState(pre_grasp_pos, grasp_quat)
        safety_error = self._pose_safety_error(grasp_pose, pre_grasp_pose)
        if safety_error is not None:
            self._print_knife_unsafe_debug(obj, link_pose, handle_pos, handle_quat, grasp_pose, pre_grasp_pose, safety_error)
            return self._invalid(cfg.name, FailureReason.TARGET_UNSAFE, safety_error, state)
        target_pose = PoseState(handle_pos, handle_quat)
        return self._valid_plan(
            cfg,
            target_pose,
            grasp_pose,
            pre_grasp_pose,
            probe_lift_pos,
            full_lift_pos,
            object_quat_w=handle_quat,
            object_yaw=knife_yaw,
            grasp_yaw=knife_grasp_yaw,
        )

    def _valid_plan(
        self,
        cfg: TargetConfig,
        target_pose: PoseState,
        grasp_pose: PoseState,
        pre_grasp_pose: PoseState,
        probe_lift_pos: torch.Tensor,
        full_lift_pos: torch.Tensor,
        object_quat_w: torch.Tensor | None = None,
        object_yaw: torch.Tensor | None = None,
        grasp_yaw: torch.Tensor | None = None,
    ) -> GraspPlan:
        return GraspPlan(
            target_name=cfg.name,
            target_pose=target_pose,
            grasp_pose=grasp_pose,
            pre_grasp_pose=pre_grasp_pose,
            probe_lift_pose=PoseState(probe_lift_pos, grasp_pose.quat_w),
            full_lift_pose=PoseState(full_lift_pos, grasp_pose.quat_w),
            approach_dir_w=self._vec3(0.0, 0.0, 1.0),
            lift_distance=cfg.full_lift_height,
            gripper_open_command=cfg.gripper_open_command,
            gripper_close_command=cfg.gripper_close_command,
            min_gripper_width=cfg.min_gripper_width,
            max_gripper_width=cfg.max_gripper_width,
            object_quat_w=object_quat_w,
            object_yaw=None if object_yaw is None else float(object_yaw.detach().cpu()),
            grasp_yaw=None if grasp_yaw is None else float(grasp_yaw.detach().cpu()),
        )

    def _top_down_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(1, dtype=torch.float32, device=self.device)
        yaw_tensor = yaw.reshape(1).to(device=self.device, dtype=torch.float32)
        yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw_tensor)[0]
        down_quat = torch.tensor(TOP_DOWN_GRASP_QUAT_WXYZ, dtype=torch.float32, device=self.device)
        quat = math_utils.quat_mul(yaw_quat.unsqueeze(0), down_quat.unsqueeze(0))[0]
        return math_utils.normalize(quat.unsqueeze(0))[0]

    def _yaw_from_local_x(self, object_quat: torch.Tensor) -> torch.Tensor:
        local_x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=object_quat.device)
        world_x = math_utils.quat_apply(object_quat.unsqueeze(0), local_x)[0]
        return torch.atan2(world_x[1], world_x[0])

    def _pose_safety_error(self, grasp_pose: PoseState, pre_grasp_pose: PoseState) -> str | None:
        for label, pose in (("grasp", grasp_pose), ("pre_grasp", pre_grasp_pose)):
            if not torch.isfinite(pose.pos_w).all() or not torch.isfinite(pose.quat_w).all():
                return f"{label} pose is not finite: pose={self._pose_list(pose)}"
            x, y, z = [float(v) for v in pose.pos_w.detach().cpu()]
            if x < WORKSPACE_X_MIN or x > WORKSPACE_X_MAX:
                return f"{label} x outside workspace: x={x:.4f}"
            if abs(y) > WORKSPACE_Y_ABS_MAX:
                return f"{label} y outside workspace: y={y:.4f}"
            if z < WORKSPACE_Z_MIN or z > WORKSPACE_Z_MAX:
                return f"{label} z outside workspace: z={z:.4f}"
        return None

    def _print_knife_unsafe_debug(
        self,
        obj,
        link_pose: PoseState,
        handle_pos: torch.Tensor,
        handle_quat: torch.Tensor,
        grasp_pose: PoseState,
        pre_grasp_pose: PoseState,
        reason: str,
    ) -> None:
        debug = {
            "knife_root_pose": self._pose_list(obj.pose),
            "knife_base_pose": self._pose_list(link_pose),
            "handle_config_local_pose": [*KNIFE_HANDLE_CONFIG_CENTER_LOCAL, *KNIFE_HANDLE_LOCAL_QUAT],
            "handle_scaled_local_pose": [*KNIFE_HANDLE_CENTER_LOCAL, *KNIFE_HANDLE_LOCAL_QUAT],
            "handle_size": list(KNIFE_HANDLE_SIZE),
            "computed_handle_world_pose": [*self._tensor_list(handle_pos), *self._tensor_list(handle_quat)],
            "pre_grasp_pose": self._pose_list(pre_grasp_pose),
            "grasp_pose": self._pose_list(grasp_pose),
            "unsafe_reason": reason,
        }
        print(f"[TargetRegistry] knife TARGET_UNSAFE {debug}", flush=True)

    def _invalid(self, target_name: str, reason: FailureReason, message: str, state: SceneState) -> GraspPlan:
        tcp = state.robot.tcp_pose
        return GraspPlan(
            target_name=target_name,
            target_pose=tcp,
            grasp_pose=tcp,
            pre_grasp_pose=tcp,
            probe_lift_pose=tcp,
            full_lift_pose=tcp,
            approach_dir_w=self._vec3(0.0, 0.0, 1.0),
            lift_distance=0.0,
            gripper_open_command=1.0,
            gripper_close_command=-1.0,
            min_gripper_width=0.0,
            max_gripper_width=1.0,
            valid=False,
            failure_reason=reason,
            message=message,
        )

    def _vec3(self, x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([x, y, z], dtype=torch.float32, device=self.device)

    def _pose_list(self, pose: PoseState) -> list[float]:
        return [round(v, 5) for v in self._tensor_list(pose.as_pose_tensor())]

    def _tensor_list(self, tensor: torch.Tensor) -> list[float]:
        return [float(v) for v in tensor.detach().cpu().tolist()]
