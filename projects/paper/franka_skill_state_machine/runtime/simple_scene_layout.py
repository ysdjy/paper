"""Simple slot-based scene layout for skill testing.

This intentionally avoids USD bounds queries and geometric rejection sampling.
The default skill-test entry uses this manager so scene reset is stable and
cheap while grasp/place/drawer skills are still under development.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

import torch

from isaaclab.utils import math as math_utils


# Unified with the RL training env cfg (stack_joint_pos_env_cfg.py cabinet init pos) so the learned
# drawer policy sees the SAME cabinet pose in training and deployment. Matches the user's fixed scene
# (saved_scenes/v0_layout): cabinet within the arm workspace, yaw +90deg (carried by the cfg
# default quaternion, so CABINET_YAW_PI stays False).
CABINET_LOCAL_X = 0.27402
CABINET_LOCAL_Y = 0.91583
CABINET_Z_CORRECTION = 0.0
CABINET_YAW_PI = False

OBJECT_SLOTS = {
    "slot_a": (0.42, -0.14),
    "slot_b": (0.42, -0.38),
    "slot_c": (0.64, -0.14),
    "slot_d": (0.66, -0.40),
}

# User-defined placement region (world XY quadrilateral). cube_1/2/3 and knife are randomly
# initialized inside this convex polygon, with rejection sampling + min separation to avoid
# interference (mirrors isaaclab franka_stack_events.sample_object_poses). Corners are ordered CCW.
PLACEMENT_REGION = [(0.15, -0.30), (0.58, -0.30), (0.70, 0.25), (0.22, 0.37)]
PLACE_MIN_SEPARATION = 0.15
PLACE_MAX_TRIES = 5000

MIN_MOVABLE_DISTANCE = 0.14


def _point_in_region(x: float, y: float, polygon=PLACEMENT_REGION) -> bool:
    """True if (x, y) is inside the convex CCW polygon (on-edge counts as inside)."""
    n = len(polygon)
    for i in range(n):
        ax, ay = polygon[i]
        bx, by = polygon[(i + 1) % n]
        # cross product of edge (A->B) and (A->P); >= 0 for all edges => inside (CCW)
        if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < -1e-9:
            return False
    return True


@dataclass
class SimpleLayoutResult:
    seed: int
    reset_index: int
    object_poses: dict[str, list[float]]


@dataclass
class _PlannedPose:
    local_position: torch.Tensor
    quaternion: torch.Tensor


class SimpleSceneLayoutManager:
    """Place the cabinet and tabletop objects into deterministic safe slots."""

    def __init__(self, env, base_seed: int):
        self.env = env
        self.scene = env.unwrapped.scene
        self.device = self.scene.device
        self.base_seed = base_seed
        self.env_id = 0
        self._env_ids = torch.tensor([self.env_id], dtype=torch.long, device=self.device)

    def reset_layout(self, reset_index: int) -> SimpleLayoutResult:
        layout_seed = self.base_seed + reset_index * 1009
        rng = random.Random(layout_seed)

        object_poses: dict[str, list[float]] = {}
        object_poses["cabinet"] = self._place_cabinet()
        object_poses.update(self._place_cubes_and_knife(rng))

        for name in ("cabinet", "cube_1", "cube_2", "cube_3", "knife"):
            self._zero_root_velocity(name)
        self._reset_articulation_joints("cabinet")
        self._reset_articulation_joints("knife")

        self._validate_layout(object_poses)
        result = SimpleLayoutResult(seed=layout_seed, reset_index=reset_index, object_poses=object_poses)
        return result

    def _place_cabinet(self) -> list[float]:
        cabinet = self.scene["cabinet"]
        default_root_state = cabinet.data.default_root_state[self.env_id].clone()
        env_origin = self.scene.env_origins[self.env_id]
        local_z = default_root_state[2] - env_origin[2] + CABINET_Z_CORRECTION
        local_position = torch.tensor(
            [CABINET_LOCAL_X, CABINET_LOCAL_Y, float(local_z)],
            dtype=torch.float32,
            device=self.device,
        )
        quaternion = default_root_state[3:7].clone()
        if CABINET_YAW_PI:
            yaw_quat = self._yaw_quat(math.pi)
            quaternion = math_utils.quat_mul(yaw_quat.unsqueeze(0), quaternion.unsqueeze(0))[0]
        return self._write_root_pose("cabinet", local_position, quaternion)

    def _sample_region_xy(self, rng: random.Random, count: int) -> list[tuple[float, float]]:
        """Rejection-sample ``count`` (x, y) points inside PLACEMENT_REGION with min separation.

        Mirrors isaaclab franka_stack_events.sample_object_poses: sample in the polygon bounding box,
        reject points outside the polygon or closer than PLACE_MIN_SEPARATION to already-accepted
        points; accept the last try unconditionally to guarantee progress.
        """
        xs = [p[0] for p in PLACEMENT_REGION]
        ys = [p[1] for p in PLACEMENT_REGION]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        points: list[tuple[float, float]] = []
        for _ in range(count):
            for j in range(PLACE_MAX_TRIES):
                x = rng.uniform(x_min, x_max)
                y = rng.uniform(y_min, y_max)
                if not _point_in_region(x, y):
                    continue
                far_enough = all(math.dist((x, y), p) > PLACE_MIN_SEPARATION for p in points)
                if not points or far_enough or j == PLACE_MAX_TRIES - 1:
                    points.append((x, y))
                    break
        return points

    def _place_cubes_and_knife(self, rng: random.Random) -> dict[str, list[float]]:
        names = ["cube_1", "cube_2", "cube_3", "knife"]
        samples = self._sample_region_xy(rng, len(names))

        planned: dict[str, _PlannedPose] = {}
        knife_default_quat = self.scene["knife"].data.default_root_state[self.env_id, 3:7].clone()
        for name, (x, y) in zip(names, samples):
            yaw = rng.uniform(-math.pi, math.pi)
            if name == "knife":
                quat = math_utils.quat_mul(self._yaw_quat(yaw).unsqueeze(0), knife_default_quat.unsqueeze(0))[0]
            else:
                quat = self._yaw_quat(yaw)
            planned[name] = _PlannedPose(
                local_position=self._default_local_position(name, x, y),
                quaternion=quat,
            )

        object_poses = {name: self._write_root_pose(name, pose.local_position, pose.quaternion) for name, pose in planned.items()}
        return object_poses

    def _write_root_pose(self, name: str, local_position: torch.Tensor, quaternion: torch.Tensor) -> list[float]:
        if not torch.isfinite(local_position).all() or not torch.isfinite(quaternion).all():
            raise ValueError(f"Non-finite pose for {name}: position={local_position}, quat={quaternion}")
        asset = self.scene[name]
        env_origin = self.scene.env_origins[self.env_id]
        world_position = env_origin + local_position
        root_pose = torch.cat((world_position.unsqueeze(0), quaternion.unsqueeze(0)), dim=-1)
        asset.write_root_pose_to_sim(root_pose, env_ids=self._env_ids)
        self._zero_root_velocity(name)
        return [float(v) for v in torch.cat((local_position, quaternion)).detach().cpu().tolist()]

    def _reset_articulation_joints(self, name: str) -> None:
        asset = self.scene[name]
        if not hasattr(asset.data, "default_joint_pos"):
            return
        joint_pos = asset.data.default_joint_pos[:1].clone()
        joint_vel = torch.zeros_like(asset.data.default_joint_vel[:1])
        asset.set_joint_position_target(joint_pos, env_ids=self._env_ids)
        asset.set_joint_velocity_target(joint_vel, env_ids=self._env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=self._env_ids)

    def _default_local_position(self, name: str, x: float, y: float) -> torch.Tensor:
        asset = self.scene[name]
        env_origin = self.scene.env_origins[self.env_id]
        default_root_state = asset.data.default_root_state[self.env_id].clone()
        local_z = default_root_state[2] - env_origin[2]
        return torch.tensor([x, y, float(local_z)], dtype=torch.float32, device=self.device)

    def _yaw_quat(self, yaw: float) -> torch.Tensor:
        yaw_tensor = torch.tensor([yaw], dtype=torch.float32, device=self.device)
        zero = torch.zeros_like(yaw_tensor)
        return math_utils.quat_from_euler_xyz(zero, zero, yaw_tensor)[0]

    def _zero_root_velocity(self, name: str) -> None:
        asset = self.scene[name]
        if hasattr(asset, "write_root_velocity_to_sim"):
            zero_vel = torch.zeros((1, 6), dtype=torch.float32, device=self.device)
            asset.write_root_velocity_to_sim(zero_vel, env_ids=self._env_ids)

    def _validate_layout(self, object_poses: dict[str, list[float]]) -> None:
        cabinet_pose = object_poses["cabinet"]
        # cabinet is intentionally placed on the -y side at (1.0, -0.8) to match the RL training env
        # and to clear the arm during drawer opening; just sanity-check it is finite.
        if not all(math.isfinite(value) for value in cabinet_pose):
            raise ValueError(f"Cabinet has non-finite pose: {cabinet_pose}")

        movable = {name: object_poses[name] for name in ("cube_1", "cube_2", "cube_3", "knife")}
        for name, pose in movable.items():
            x, y = pose[0], pose[1]
            if not all(math.isfinite(value) for value in pose):
                raise ValueError(f"{name} has non-finite pose: {pose}")
            if not _point_in_region(x, y):
                raise ValueError(f"{name} sampled outside placement region: ({x:.3f}, {y:.3f})")

        min_distance = self._minimum_movable_distance(
            {
                name: _PlannedPose(
                    local_position=torch.tensor(pose[:3], dtype=torch.float32, device=self.device),
                    quaternion=torch.tensor(pose[3:7], dtype=torch.float32, device=self.device),
                )
                for name, pose in movable.items()
            }
        )
        if min_distance < MIN_MOVABLE_DISTANCE:
            raise ValueError(f"Movable objects too close after fallback: min_distance={min_distance:.3f}")

    def _minimum_movable_distance(self, poses: dict[str, _PlannedPose]) -> float:
        names = list(poses.keys())
        min_distance = float("inf")
        for i, name_i in enumerate(names):
            for name_j in names[i + 1 :]:
                pi = poses[name_i].local_position
                pj = poses[name_j].local_position
                distance = float(torch.linalg.norm(pi[:2] - pj[:2]).detach().cpu())
                min_distance = min(min_distance, distance)
        return min_distance if math.isfinite(min_distance) else 0.0
