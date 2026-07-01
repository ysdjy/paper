"""Advanced experimental layout manager; not used by the default skill test.

Scene layout management for deterministic, collision-free initialization.

The layout logic is separated from skill execution. It samples and writes the
cabinet plus movable objects, then validates the reset layout before control
starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
import random
from typing import Any

import torch


@dataclass(frozen=True)
class WorkspaceRegion:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


@dataclass(frozen=True)
class SceneLayoutConfig:
    cabinet_xy: tuple[float, float] = (0.78, 0.52)
    movable_region: WorkspaceRegion = WorkspaceRegion(
        x_min=0.38,
        x_max=0.72,
        y_min=-0.48,
        y_max=-0.10,
    )
    ground_top_z: float = 0.0
    ground_clearance: float = 0.002
    cube_size: float = 0.0406
    cube_safety_margin: float = 0.040
    knife_safety_margin: float = 0.045
    cabinet_keepout_margin: float = 0.120
    robot_exclusion_radius: float = 0.300
    drawer_sweep_length: float = 0.350
    drawer_sweep_side_margin: float = 0.080
    max_candidate_attempts: int = 2000
    max_layout_attempts: int = 10
    settling_steps: int = 20


@dataclass
class AssetGeometry:
    name: str
    bottom_offset_z: float
    footprint_radius: float
    aabb_size: tuple[float, float, float]


@dataclass
class SampledPose:
    position_base: tuple[float, float, float]
    yaw: float
    attempts: int


@dataclass
class LayoutValidationResult:
    valid: bool
    violations: list[str]
    min_pair_clearance: float
    cabinet_min_z: float
    object_min_z: dict[str, float]


@dataclass
class LayoutResult:
    seed: int
    reset_index: int
    object_poses: dict[str, SampledPose]
    validation: LayoutValidationResult


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _normalize_xy(vec: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec)
    if float(norm) < 1.0e-9:
        raise ValueError("Cannot normalize a zero-length 2D vector.")
    return vec / norm


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if v.shape[-1] != 3:
        raise ValueError("Expected a 3D vector.")
    zeros = torch.zeros_like(v[..., :1])
    v_quat = torch.cat((zeros, v), dim=-1)
    rotated = _quat_mul(_quat_mul(q, v_quat), _quat_conjugate(q))
    return rotated[..., 1:]


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _combine_frame_transforms(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_pos = parent_pos + _quat_apply(parent_quat, child_pos)
    world_quat = _quat_mul(parent_quat, child_quat)
    return world_pos, world_quat


def _subtract_frame_transforms(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent_quat_conj = _quat_conjugate(parent_quat)
    rel_pos = _quat_apply(parent_quat_conj, child_pos - parent_pos)
    rel_quat = _quat_mul(parent_quat_conj, child_quat)
    return rel_pos, rel_quat


def _circle_intersects_aabb(cx: float, cy: float, radius: float, aabb: tuple[float, float, float, float]) -> bool:
    min_x, min_y, max_x, max_y = aabb
    closest_x = _clamp(cx, min_x, max_x)
    closest_y = _clamp(cy, min_y, max_y)
    dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
    return dist_sq < radius**2


def _circle_intersects_obb(
    cx: float,
    cy: float,
    radius: float,
    center: tuple[float, float],
    front: torch.Tensor,
    side: torch.Tensor,
    half_front: float,
    half_side: float,
) -> bool:
    relative = torch.tensor([cx - center[0], cy - center[1]], dtype=torch.float32)
    local_front = float(torch.dot(relative, front))
    local_side = float(torch.dot(relative, side))
    closest_front = _clamp(local_front, -half_front, half_front)
    closest_side = _clamp(local_side, -half_side, half_side)
    distance_sq = (local_front - closest_front) ** 2 + (local_side - closest_side) ** 2
    return distance_sq < radius**2


def _pairwise_clearance(pose_a: SampledPose, radius_a: float, pose_b: SampledPose, radius_b: float) -> float:
    dx = pose_a.position_base[0] - pose_b.position_base[0]
    dy = pose_a.position_base[1] - pose_b.position_base[1]
    return math.hypot(dx, dy) - radius_a - radius_b


def sample_movable_layout_static(
    rng: random.Random,
    config: SceneLayoutConfig,
    radii: dict[str, float],
    cabinet_keepout: tuple[float, float, float, float],
    drawer_sweep_center: tuple[float, float],
    drawer_front: torch.Tensor,
    drawer_side: torch.Tensor,
    drawer_half_front: float,
    drawer_half_side: float,
    max_candidate_attempts: int | None = None,
) -> tuple[dict[str, SampledPose], dict[str, int]]:
    """Sample a collision-free layout in robot-base coordinates."""

    attempt_limit = max_candidate_attempts or config.max_candidate_attempts
    order = ["knife", "cube_1", "cube_2", "cube_3"]
    for layout_attempt in range(1, config.max_layout_attempts + 1):
        placements: dict[str, SampledPose] = {}
        attempts: dict[str, int] = {}
        layout_failed = False

        for name in order:
            radius = radii[name]
            placed = False
            for attempt in range(1, attempt_limit + 1):
                x = rng.uniform(config.movable_region.x_min + radius, config.movable_region.x_max - radius)
                y = rng.uniform(config.movable_region.y_min + radius, config.movable_region.y_max - radius)
                yaw = rng.uniform(-math.pi, math.pi)
                if math.hypot(x, y) < config.robot_exclusion_radius + radius:
                    continue
                if _circle_intersects_aabb(x, y, radius, cabinet_keepout):
                    continue
                if _circle_intersects_obb(
                    x,
                    y,
                    radius,
                    drawer_sweep_center,
                    drawer_front,
                    drawer_side,
                    drawer_half_front,
                    drawer_half_side,
                ):
                    continue
                collision = False
                for other_name, other_pose in placements.items():
                    if _pairwise_clearance(
                        SampledPose((x, y, 0.0), yaw, attempt),
                        radius,
                        other_pose,
                        radii[other_name],
                    ) < 0.0:
                        collision = True
                        break
                if collision:
                    continue
                placements[name] = SampledPose((x, y, 0.0), _wrap_to_pi(yaw), attempt)
                attempts[name] = attempt
                placed = True
                break

            if placed:
                continue

            grid_attempts = []
            grid_x_count = 32
            grid_y_count = 32
            x_low = config.movable_region.x_min + radius
            x_high = config.movable_region.x_max - radius
            y_low = config.movable_region.y_min + radius
            y_high = config.movable_region.y_max - radius
            for ix in range(grid_x_count):
                x = x_low if grid_x_count == 1 else x_low + (x_high - x_low) * ix / (grid_x_count - 1)
                for iy in range(grid_y_count):
                    y = y_low if grid_y_count == 1 else y_low + (y_high - y_low) * iy / (grid_y_count - 1)
                    grid_attempts.append((x, y))
            rng.shuffle(grid_attempts)
            for extra_idx, (x, y) in enumerate(grid_attempts, start=1):
                yaw = rng.uniform(-math.pi, math.pi)
                if math.hypot(x, y) < config.robot_exclusion_radius + radius:
                    continue
                if _circle_intersects_aabb(x, y, radius, cabinet_keepout):
                    continue
                if _circle_intersects_obb(
                    x,
                    y,
                    radius,
                    drawer_sweep_center,
                    drawer_front,
                    drawer_side,
                    drawer_half_front,
                    drawer_half_side,
                ):
                    continue
                collision = False
                for other_name, other_pose in placements.items():
                    if _pairwise_clearance(
                        SampledPose((x, y, 0.0), yaw, attempt_limit + extra_idx),
                        radius,
                        other_pose,
                        radii[other_name],
                    ) < 0.0:
                        collision = True
                        break
                if collision:
                    continue
                placements[name] = SampledPose((x, y, 0.0), _wrap_to_pi(yaw), attempt_limit + extra_idx)
                attempts[name] = attempt_limit + extra_idx
                placed = True
                break

            if not placed:
                layout_failed = True
                break

        if not layout_failed:
            return placements, attempts

    raise RuntimeError(
        f"Failed to sample collision-free layout after {config.max_layout_attempts} layout attempts"
    )


def validate_layout_static(
    config: SceneLayoutConfig,
    cabinet_min_z: float,
    object_min_z: dict[str, float],
    sampled_poses: dict[str, SampledPose],
    actual_poses_base: dict[str, SampledPose],
    radii: dict[str, float],
    cabinet_keepout: tuple[float, float, float, float],
    drawer_sweep_center: tuple[float, float],
    drawer_front: torch.Tensor,
    drawer_side: torch.Tensor,
    drawer_half_front: float,
    drawer_half_side: float,
    settling_before: dict[str, tuple[float, float, float]],
    settling_after: dict[str, tuple[float, float, float]],
) -> LayoutValidationResult:
    violations: list[str] = []
    region_eps = 1.0e-6

    if not (-0.002 <= cabinet_min_z - config.ground_top_z <= 0.006):
        violations.append("CABINET_NOT_ON_GROUND")

    for name, min_z in object_min_z.items():
        if name == "knife":
            if not (-0.003 <= min_z - config.ground_top_z <= 0.008):
                violations.append("KNIFE_NOT_ON_GROUND")
        else:
            if not (-0.003 <= min_z - config.ground_top_z <= 0.008):
                violations.append(f"{name.upper()}_NOT_ON_GROUND")

    for name, pose in actual_poses_base.items():
        radius = radii[name]
        x, y, _ = pose.position_base
        if not (
            config.movable_region.x_min + radius - region_eps <= x <= config.movable_region.x_max - radius + region_eps
            and config.movable_region.y_min + radius - region_eps <= y <= config.movable_region.y_max - radius + region_eps
        ):
            violations.append(f"{name.upper()}_OUTSIDE_MOVABLE_REGION")
        if math.hypot(x, y) < config.robot_exclusion_radius + radius:
            violations.append(f"{name.upper()}_IN_ROBOT_EXCLUSION")
        if _circle_intersects_aabb(x, y, radius, cabinet_keepout):
            violations.append(f"{name.upper()}_INTERSECTS_CABINET")
        if _circle_intersects_obb(
            x,
            y,
            radius,
            drawer_sweep_center,
            drawer_front,
            drawer_side,
            drawer_half_front,
            drawer_half_side,
        ):
            violations.append(f"{name.upper()}_INTERSECTS_DRAWER_SWEEP")
        dx = settling_after[name][0] - settling_before[name][0]
        dy = settling_after[name][1] - settling_before[name][1]
        if math.hypot(dx, dy) > 0.015:
            violations.append(f"{name.upper()}_SETTLING_DRIFT")

    names = list(actual_poses_base.keys())
    min_pair_clearance = float("inf")
    for i, name_i in enumerate(names):
        for name_j in names[i + 1 :]:
            clearance = _pairwise_clearance(actual_poses_base[name_i], radii[name_i], actual_poses_base[name_j], radii[name_j])
            min_pair_clearance = min(min_pair_clearance, clearance)
            if clearance < -0.002:
                violations.append(f"{name_i.upper()}_{name_j.upper()}_OVERLAP")

    valid = len(violations) == 0
    if not math.isfinite(min_pair_clearance):
        min_pair_clearance = 0.0
    return LayoutValidationResult(
        valid=valid,
        violations=violations,
        min_pair_clearance=min_pair_clearance,
        cabinet_min_z=cabinet_min_z,
        object_min_z=object_min_z,
    )


class SceneLayoutManager:
    def __init__(self, env, config: SceneLayoutConfig, base_seed: int):
        self.env = env
        self.scene = env.unwrapped.scene
        self.config = config
        self.base_seed = base_seed
        self.env_id = 0
        self.device = self.scene.device
        self._bbox_cache = None
        self._geometry: dict[str, AssetGeometry] = {}
        self._effective_config = config
        self._cabinet_front_xy_zero_yaw: torch.Tensor | None = None
        self._cabinet_keepout: tuple[float, float, float, float] | None = None
        self._drawer_sweep_center: tuple[float, float] | None = None
        self._drawer_front: torch.Tensor | None = None
        self._drawer_side: torch.Tensor | None = None
        self._drawer_half_front: float | None = None
        self._drawer_half_side: float | None = None
        self._latest_result: LayoutResult | None = None
        self._last_sampled_poses: dict[str, SampledPose] = {}
        self._last_written_world_poses: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._last_settled_base_poses: dict[str, SampledPose] = {}
        self._cabinet_pose_world: tuple[torch.Tensor, torch.Tensor] | None = None

    def calibrate_asset_geometry(self) -> None:
        self._geometry["cabinet"] = self._calibrate_articulation_geometry("cabinet")
        self._geometry["knife"] = self._calibrate_articulation_geometry("knife")
        cube_radius = 0.5 * math.sqrt(2.0 * self.config.cube_size**2)
        self._geometry["cube_1"] = AssetGeometry(
            name="cube_1",
            bottom_offset_z=self.config.cube_size * 0.5,
            footprint_radius=cube_radius,
            aabb_size=(self.config.cube_size, self.config.cube_size, self.config.cube_size),
        )
        self._geometry["cube_2"] = self._geometry["cube_1"]
        self._geometry["cube_3"] = self._geometry["cube_1"]

        cabinet_pose = self._read_asset_root_pose(self.scene["cabinet"])
        handle_world = self._read_handle_proxy_world_xy("cabinet", "link_1/BottomHandleProxy")
        self._cabinet_front_xy_zero_yaw = _normalize_xy(handle_world - cabinet_pose[0][:2])

    def reset_layout(self, reset_index: int, hold_action: torch.Tensor | None = None) -> LayoutResult:
        if not self._geometry:
            raise RuntimeError("call calibrate_asset_geometry() before reset_layout()")

        layout_seed = self.base_seed + reset_index * 1009
        rng = random.Random(layout_seed)
        hold_action = hold_action if hold_action is not None else self._make_hold_action()

        last_validation: LayoutValidationResult | None = None
        last_sampled: dict[str, SampledPose] | None = None
        for _layout_attempt in range(self.config.max_layout_attempts):
            self.place_cabinet()
            sampled = self._sample_movable_layout_from_rng(rng)
            self.write_movable_poses(sampled)
            self.settle(hold_action)
            validation = self.validate_layout()
            if validation.valid:
                self._latest_result = LayoutResult(
                    seed=layout_seed,
                    reset_index=reset_index,
                    object_poses=sampled,
                    validation=validation,
                )
                self._print_layout_log(self._latest_result)
                return self._latest_result
            last_validation = validation
            last_sampled = sampled

        if last_validation is None or last_sampled is None:
            raise RuntimeError("Failed to produce any candidate layout.")
        raise RuntimeError(f"Failed to produce valid layout after {self.config.max_layout_attempts} attempts: {last_validation.violations}")

    def place_cabinet(self) -> None:
        cabinet = self.scene["cabinet"]
        robot_pos_w, robot_quat_w = self._base_pose_world()
        base_xy = torch.tensor(self.config.cabinet_xy, dtype=torch.float32, device=self.device)
        cabinet_xy_world = self._base_to_world_xy(base_xy, robot_pos_w, robot_quat_w)
        cabinet_yaw = self._compute_cabinet_yaw(cabinet_xy_world)
        cabinet_bottom_offset_z = self._geometry["cabinet"].bottom_offset_z
        cabinet_root_z = self.config.ground_top_z + self.config.ground_clearance - cabinet_bottom_offset_z
        cabinet_pose_local = torch.tensor([base_xy[0].item(), base_xy[1].item(), cabinet_root_z], device=self.device)
        cabinet_quat_local = _quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([0.0], device=self.device),
            torch.tensor([cabinet_yaw], device=self.device),
        )[0]
        cabinet_pos_w, cabinet_quat_w = _combine_frame_transforms(
            robot_pos_w, robot_quat_w, cabinet_pose_local, cabinet_quat_local
        )
        cabinet_root_pose = torch.cat((cabinet_pos_w.unsqueeze(0), cabinet_quat_w.unsqueeze(0)), dim=-1)
        zero_vel = torch.zeros((1, 6), device=self.device)
        cabinet.write_root_pose_to_sim(cabinet_root_pose, env_ids=torch.tensor([self.env_id], device=self.device))
        cabinet.write_root_velocity_to_sim(zero_vel, env_ids=torch.tensor([self.env_id], device=self.device))
        cabinet_root_pose, cabinet_min_z = self._ground_correct_root_pose(
            cabinet,
            cabinet_root_pose,
            target_min_z=self.config.ground_top_z + self.config.ground_clearance,
        )
        joint_pos = torch.zeros_like(cabinet.data.joint_pos[:1])
        joint_vel = torch.zeros_like(cabinet.data.joint_vel[:1])
        cabinet.set_joint_position_target(joint_pos, env_ids=torch.tensor([self.env_id], device=self.device))
        cabinet.set_joint_velocity_target(joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
        cabinet.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
        self._cabinet_pose_world = (cabinet_root_pose[0, :3].clone(), cabinet_root_pose[0, 3:7].clone())

    def sample_movable_layout(self, reset_index: int) -> dict[str, SampledPose]:
        layout_seed = self.base_seed + reset_index * 1009
        rng = random.Random(layout_seed)
        return self._sample_movable_layout_from_rng(rng)

    def write_movable_poses(self, poses: dict[str, SampledPose]) -> None:
        robot_pos_w, robot_quat_w = self._base_pose_world()
        self._last_sampled_poses = poses
        zero_vel = torch.zeros((1, 6), device=self.device)

        for name, sampled in poses.items():
            asset = self.scene[name]
            local_pos = torch.tensor(sampled.position_base, dtype=torch.float32, device=self.device)
            local_quat = _quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([sampled.yaw], device=self.device),
            )[0]
            pos_w, quat_w = _combine_frame_transforms(robot_pos_w, robot_quat_w, local_pos, local_quat)
            root_pose = torch.cat((pos_w.unsqueeze(0), quat_w.unsqueeze(0)), dim=-1)
            asset.write_root_pose_to_sim(root_pose, env_ids=torch.tensor([self.env_id], device=self.device))
            asset.write_root_velocity_to_sim(zero_vel, env_ids=torch.tensor([self.env_id], device=self.device))
            root_pose, _ = self._ground_correct_root_pose(
                asset,
                root_pose,
                target_min_z=self.config.ground_top_z + self.config.ground_clearance,
            )
            if name == "knife":
                joint_pos = torch.zeros_like(asset.data.joint_pos[:1])
                joint_vel = torch.zeros_like(asset.data.joint_vel[:1])
                asset.set_joint_position_target(joint_pos, env_ids=torch.tensor([self.env_id], device=self.device))
                asset.set_joint_velocity_target(joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
                asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
            self._last_written_world_poses[name] = (root_pose[0, :3].clone(), root_pose[0, 3:7].clone())

    def settle(self, hold_action: torch.Tensor) -> None:
        for _ in range(self.config.settling_steps):
            self.env.step(hold_action)

    def validate_layout(self) -> LayoutValidationResult:
        if self._cabinet_pose_world is None:
            raise RuntimeError("Cabinet has not been placed yet.")

        robot_pos_w, robot_quat_w = self._base_pose_world()
        desired_base_poses = self._last_sampled_poses
        actual_base_poses: dict[str, SampledPose] = {}
        object_min_z: dict[str, float] = {}
        settling_before: dict[str, tuple[float, float, float]] = {}
        settling_after: dict[str, tuple[float, float, float]] = {}

        for name in ("knife", "cube_1", "cube_2", "cube_3"):
            pose_w = self._read_asset_root_pose(self.scene[name])
            base_pos, _ = _subtract_frame_transforms(robot_pos_w, robot_quat_w, pose_w[0], pose_w[1])
            actual_base_poses[name] = SampledPose(
                (float(base_pos[0]), float(base_pos[1]), float(base_pos[2])),
                0.0 if name == "knife" else desired_base_poses[name].yaw,
                desired_base_poses[name].attempts,
            )
            settling_after[name] = actual_base_poses[name].position_base
            object_min_z[name] = self._compute_aabb_min_z(self.scene[name])
            settling_before[name] = desired_base_poses[name].position_base

        cabinet_min_z = self._compute_aabb_min_z(self.scene["cabinet"])
        cabinet_keepout = self._cabinet_keepout_bounds_base()
        drawer_center, drawer_front, drawer_side, drawer_half_front, drawer_half_side = self._drawer_sweep_obb_base()
        radii = self._radii()

        validation = validate_layout_static(
            self._effective_config,
            cabinet_min_z=cabinet_min_z,
            object_min_z=object_min_z,
            sampled_poses=desired_base_poses,
            actual_poses_base=actual_base_poses,
            radii=radii,
            cabinet_keepout=cabinet_keepout,
            drawer_sweep_center=drawer_center,
            drawer_front=drawer_front,
            drawer_side=drawer_side,
            drawer_half_front=drawer_half_front,
            drawer_half_side=drawer_half_side,
            settling_before=settling_before,
            settling_after=settling_after,
        )
        self._last_settled_base_poses = actual_base_poses
        return validation

    def _sample_movable_layout_from_rng(self, rng: random.Random) -> dict[str, SampledPose]:
        radii = self._radii()
        cabinet_keepout = self._cabinet_keepout_bounds_base()
        drawer_center, drawer_front, drawer_side, drawer_half_front, drawer_half_side = self._drawer_sweep_obb_base()
        for expansion in (0.0, 0.20, 0.40, 0.60, 0.80, 1.00):
            effective_config = replace(
                self.config,
                movable_region=WorkspaceRegion(
                    x_min=self.config.movable_region.x_min,
                    x_max=self.config.movable_region.x_max + expansion,
                    y_min=self.config.movable_region.y_min - expansion,
                    y_max=self.config.movable_region.y_max,
                ),
            )
            try:
                poses, _attempts = sample_movable_layout_static(
                    rng=rng,
                    config=effective_config,
                    radii=radii,
                    cabinet_keepout=cabinet_keepout,
                    drawer_sweep_center=drawer_center,
                    drawer_front=drawer_front,
                    drawer_side=drawer_side,
                    drawer_half_front=drawer_half_front,
                    drawer_half_side=drawer_half_side,
                    max_candidate_attempts=self.config.max_candidate_attempts,
                )
                self._effective_config = effective_config
                return poses
            except RuntimeError:
                continue
        self._effective_config = self.config
        raise RuntimeError("Failed to sample collision-free movable layout after region expansion retries")

    def _make_hold_action(self) -> torch.Tensor:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]
        tcp_pos_w = ee_frame.data.target_pos_w[self.env_id, 0].clone()
        tcp_quat_w = ee_frame.data.target_quat_w[self.env_id, 0].clone()
        pos_env = tcp_pos_w - self.scene.env_origins[self.env_id]
        action = torch.zeros((self.env.unwrapped.num_envs, 8), device=self.device)
        action[:, :3] = pos_env
        action[:, 3:7] = tcp_quat_w
        action[:, 7] = 1.0
        return action

    def _radii(self) -> dict[str, float]:
        cube_radius = 0.5 * math.sqrt(2.0 * self.config.cube_size**2)
        return {
            "knife": self._geometry["knife"].footprint_radius + self.config.knife_safety_margin,
            "cube_1": cube_radius + self.config.cube_safety_margin,
            "cube_2": cube_radius + self.config.cube_safety_margin,
            "cube_3": cube_radius + self.config.cube_safety_margin,
        }

    def _cabinet_keepout_bounds_base(self) -> tuple[float, float, float, float]:
        cabinet_aabb = self._compute_aabb_world(self.scene["cabinet"])
        robot_pos_w, robot_quat_w = self._base_pose_world()
        corners = self._aabb_corners(cabinet_aabb)
        corners_base = []
        for corner in corners:
            pos_base, _ = _subtract_frame_transforms(
                robot_pos_w,
                robot_quat_w,
                corner,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device),
            )
            corners_base.append(pos_base[:2])
        xy = torch.stack(corners_base)
        min_x = float(torch.min(xy[:, 0])) - self.config.cabinet_keepout_margin
        min_y = float(torch.min(xy[:, 1])) - self.config.cabinet_keepout_margin
        max_x = float(torch.max(xy[:, 0])) + self.config.cabinet_keepout_margin
        max_y = float(torch.max(xy[:, 1])) + self.config.cabinet_keepout_margin
        self._cabinet_keepout = (min_x, min_y, max_x, max_y)
        return self._cabinet_keepout

    def _drawer_sweep_obb_base(self) -> tuple[tuple[float, float], torch.Tensor, torch.Tensor, float, float]:
        if self._cabinet_front_xy_zero_yaw is None:
            raise RuntimeError("call calibrate_asset_geometry() before layout sampling")
        cabinet_xy = torch.tensor(self.config.cabinet_xy, dtype=torch.float32, device=self.device)
        front = self._cabinet_front_xy_zero_yaw
        side = torch.tensor([-front[1], front[0]], dtype=torch.float32, device=self.device)
        cabinet_size = self._geometry["cabinet"].aabb_size
        drawer_half_front = 0.5 * self.config.drawer_sweep_length
        drawer_half_side = 0.5 * min(cabinet_size[0], cabinet_size[1]) + self.config.drawer_sweep_side_margin
        center = cabinet_xy + front * (self.config.drawer_sweep_length * 0.5)
        self._drawer_sweep_center = (float(center[0]), float(center[1]))
        self._drawer_front = front
        self._drawer_side = side
        self._drawer_half_front = drawer_half_front
        self._drawer_half_side = drawer_half_side
        return self._drawer_sweep_center, front, side, drawer_half_front, drawer_half_side

    def _compute_cabinet_yaw(self, target_cabinet_xy_world: torch.Tensor) -> float:
        if self._cabinet_front_xy_zero_yaw is None:
            raise RuntimeError("Cabinet geometry has not been calibrated.")
        robot_pos_w, _ = self._base_pose_world()
        front_xy = self._cabinet_front_xy_zero_yaw
        target_front_xy = _normalize_xy(robot_pos_w[:2] - target_cabinet_xy_world)
        source_angle = math.atan2(float(front_xy[1]), float(front_xy[0]))
        target_angle = math.atan2(float(target_front_xy[1]), float(target_front_xy[0]))
        return _wrap_to_pi(target_angle - source_angle)

    def _base_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.scene["robot"]
        return robot.data.root_pos_w[self.env_id].clone(), robot.data.root_quat_w[self.env_id].clone()

    def _base_to_world_xy(
        self, xy_base: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor
    ) -> torch.Tensor:
        local = torch.tensor([xy_base[0], xy_base[1], 0.0], dtype=torch.float32, device=self.device)
        world_pos, _ = _combine_frame_transforms(
            base_pos_w, base_quat_w, local, torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        )
        return world_pos[:2]

    def _read_asset_root_pose(self, asset: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return asset.data.root_pos_w[self.env_id].clone(), asset.data.root_quat_w[self.env_id].clone()

    def _resolve_asset_prim_path(self, asset: Any) -> str:
        if hasattr(asset, "prim_paths") and getattr(asset, "prim_paths"):
            prim_path = asset.prim_paths[0]
            if not re.search(r"[\*\[\]\(\)]", prim_path):
                return prim_path
        cfg_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
        if cfg_path is None:
            raise RuntimeError(f"Cannot resolve prim path for asset {asset}.")
        env_path = self.scene.env_prim_paths[self.env_id]
        if "{ENV_REGEX_NS}" in cfg_path:
            return cfg_path.replace("{ENV_REGEX_NS}", env_path)
        env_leaf = env_path.rsplit("/", 1)[-1]
        if "env_" in cfg_path:
            return re.sub(r"env_[^/]+", env_leaf, cfg_path)
        return cfg_path

    def _get_stage(self):
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _compute_aabb_world(self, asset: Any) -> tuple[float, float, float, float, float, float]:
        prim_path = self._resolve_asset_prim_path(asset)
        try:
            import isaacsim.core.experimental.utils.bounds as bounds_utils  # type: ignore

            if self._bbox_cache is None:
                self._bbox_cache = bounds_utils.create_bbox_cache()
            return tuple(bounds_utils.compute_aabb(prim_path, bbox_cache=self._bbox_cache, include_children=True))  # type: ignore
        except Exception:
            from pxr import Gf, Usd, UsdGeom

            stage = self._get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            bound = cache.ComputeWorldBound(prim)
            aligned_range = bound.ComputeAlignedRange()
            min_corner = aligned_range.GetMin()
            max_corner = aligned_range.GetMax()
            return (
                float(min_corner[0]),
                float(min_corner[1]),
                float(min_corner[2]),
                float(max_corner[0]),
                float(max_corner[1]),
                float(max_corner[2]),
            )

    def _aabb_corners(self, aabb: tuple[float, float, float, float, float, float]) -> list[torch.Tensor]:
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        return [
            torch.tensor([x, y, z], dtype=torch.float32, device=self.device)
            for x in (min_x, max_x)
            for y in (min_y, max_y)
            for z in (min_z, max_z)
        ]

    def _compute_aabb_min_z(self, asset: Any) -> float:
        return self._compute_aabb_world(asset)[2]

    def _ground_correct_root_pose(
        self, asset: Any, root_pose: torch.Tensor, target_min_z: float
    ) -> tuple[torch.Tensor, float]:
        actual_min_z = self._compute_aabb_min_z(asset)
        correction = target_min_z - actual_min_z
        if abs(correction) > 1.0e-5:
            corrected_root_pose = root_pose.clone()
            corrected_root_pose[:, 2] += correction
            asset.write_root_pose_to_sim(corrected_root_pose, env_ids=torch.tensor([self.env_id], device=self.device))
            asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device), env_ids=torch.tensor([self.env_id], device=self.device))
            actual_min_z = self._compute_aabb_min_z(asset)
            return corrected_root_pose, actual_min_z
        return root_pose, actual_min_z

    def _calibrate_articulation_geometry(self, name: str) -> AssetGeometry:
        asset = self.scene[name]
        aabb = self._compute_aabb_world(asset)
        size_x = aabb[3] - aabb[0]
        size_y = aabb[4] - aabb[1]
        size_z = aabb[5] - aabb[2]
        root_z = float(asset.data.root_pos_w[self.env_id, 2].detach().cpu())
        bottom_offset_z = aabb[2] - root_z
        footprint_radius = 0.5 * math.sqrt(size_x**2 + size_y**2)
        return AssetGeometry(
            name=name,
            bottom_offset_z=bottom_offset_z,
            footprint_radius=footprint_radius,
            aabb_size=(size_x, size_y, size_z),
        )

    def _read_handle_proxy_world_xy(self, asset_name: str, handle_suffix: str) -> torch.Tensor:
        asset = self.scene[asset_name]
        prim_path = self._resolve_asset_prim_path(asset)
        stage = self._get_stage()
        from pxr import Usd, UsdGeom

        handle_prim = stage.GetPrimAtPath(f"{prim_path}/{handle_suffix}")
        xform = UsdGeom.Xformable(handle_prim)
        matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        handle_world = matrix.ExtractTranslation()
        return torch.tensor([float(handle_world[0]), float(handle_world[1])], dtype=torch.float32, device=self.device)

    def _print_layout_log(self, layout_result: LayoutResult) -> None:
        cabinet_pose = self._cabinet_pose_world
        message = {
            "layout_seed": layout_result.seed,
            "reset_index": layout_result.reset_index,
            "cabinet_pose": None if cabinet_pose is None else [float(x) for x in torch.cat((cabinet_pose[0], cabinet_pose[1])).detach().cpu().tolist()],
            "cube_1_pose": layout_result.object_poses.get("cube_1").position_base if "cube_1" in layout_result.object_poses else None,
            "cube_2_pose": layout_result.object_poses.get("cube_2").position_base if "cube_2" in layout_result.object_poses else None,
            "cube_3_pose": layout_result.object_poses.get("cube_3").position_base if "cube_3" in layout_result.object_poses else None,
            "knife_pose": layout_result.object_poses.get("knife").position_base if "knife" in layout_result.object_poses else None,
            "sampling_attempts": {name: pose.attempts for name, pose in layout_result.object_poses.items()},
            "minimum_pair_clearance": layout_result.validation.min_pair_clearance,
            "cabinet_min_z": layout_result.validation.cabinet_min_z,
            "knife_min_z": layout_result.validation.object_min_z.get("knife"),
            "valid": layout_result.validation.valid,
            "violations": layout_result.validation.violations,
        }
        print(message)
