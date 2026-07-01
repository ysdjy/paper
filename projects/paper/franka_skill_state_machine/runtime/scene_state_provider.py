"""Centralized scene state reads for skill execution.

Frame contract used by this package:
    * Control frame: Franka TCP.
    * TCP transform: panda_hand local +Z offset of 0.1034 m.
    * Quaternion order: (w, x, y, z), matching Isaac Lab math utilities.
    * IK action pose order: [x, y, z, qw, qx, qy, qz, gripper].

The IK action is configured with the same body_offset as the ee_frame sensor, so
the observed TCP, commanded TCP, debug visuals, and error metrics describe the
same physical frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PoseState:
    pos_w: torch.Tensor
    quat_w: torch.Tensor

    def as_pose_tensor(self) -> torch.Tensor:
        return torch.cat((self.pos_w, self.quat_w), dim=-1)


@dataclass
class ObjectState:
    name: str
    pose: PoseState
    lin_vel_w: torch.Tensor | None = None
    ang_vel_w: torch.Tensor | None = None
    links: dict[str, PoseState] = field(default_factory=dict)
    joint_pos: dict[str, float] = field(default_factory=dict)
    joint_vel: dict[str, float] = field(default_factory=dict)


@dataclass
class RobotState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    tcp_pose: PoseState
    gripper_width: float


@dataclass
class SceneState:
    env_id: int
    env_origin_w: torch.Tensor
    robot: RobotState
    objects: dict[str, ObjectState]
    sim_time: float


class SceneStateProvider:
    """Reads all runtime state needed by skills from the Isaac Lab scene."""

    def __init__(self, env, env_id: int = 0):
        self.env = env
        self.scene = env.unwrapped.scene
        self.env_id = env_id
        self.device = self.scene.device
        self._finger_joint_ids = self._find_joints("panda_finger_.*")
        self._arm_joint_ids = self._find_joints("panda_joint.*")
        self._joint_action_cache: dict | None = None
        self._sim_time = 0.0
        self._cabinet_joint_id_cache: dict[str, int] = {}
        self._cabinet_control_method: dict[str, str] = {}
        self._logged_cabinet_joints: set[str] = set()

    def set_sim_time(self, sim_time: float):
        self._sim_time = sim_time

    def get_state(self) -> SceneState:
        robot = self.scene["robot"]
        ee_frame = self.scene["ee_frame"]
        env_origin_w = self.scene.env_origins[self.env_id]
        tcp_pos_w = ee_frame.data.target_pos_w[self.env_id, 0].clone()
        tcp_quat_w = ee_frame.data.target_quat_w[self.env_id, 0].clone()
        gripper_width = self._read_gripper_width(robot)

        objects: dict[str, ObjectState] = {}
        for name in ("cube_1", "cube_2", "cube_3"):
            if name in self.scene.keys():
                objects[name] = self._rigid_object_state(name)
        if "knife" in self.scene.keys():
            objects["knife"] = self._articulation_state("knife", link_names=("base", "joint_0", "HandleProxy"))
        if "cabinet" in self.scene.keys():
            objects["cabinet"] = self._articulation_state(
                "cabinet", link_names=("link_1", "drawer", "handle", "HandleProxy")
            )

        return SceneState(
            env_id=self.env_id,
            env_origin_w=env_origin_w.clone(),
            robot=RobotState(
                joint_pos=robot.data.joint_pos[self.env_id].clone(),
                joint_vel=robot.data.joint_vel[self.env_id].clone(),
                tcp_pose=PoseState(tcp_pos_w, tcp_quat_w),
                gripper_width=gripper_width,
            ),
            objects=objects,
            sim_time=self._sim_time,
        )

    def make_action(self, tcp_pose_w: PoseState | torch.Tensor, gripper_command: float) -> torch.Tensor:
        """Build an absolute IK action for the configured TCP frame.

        Isaac Lab manager actions use environment-local position with world-frame
        quaternion. The returned tensor has shape (num_envs, 8).
        """

        if isinstance(tcp_pose_w, PoseState):
            pos_w = tcp_pose_w.pos_w
            quat_w = tcp_pose_w.quat_w
        else:
            pos_w = tcp_pose_w[:3]
            quat_w = tcp_pose_w[3:7]
        pos_env = pos_w - self.scene.env_origins[self.env_id]
        action = torch.zeros((self.env.unwrapped.num_envs, 8), device=self.device)
        action[:, :3] = pos_env
        action[:, 3:7] = quat_w
        action[:, 7] = gripper_command
        return action

    def hold_action(self, state: SceneState, gripper_command: float) -> torch.Tensor:
        if gripper_command not in (-1.0, 1.0):
            raise ValueError("gripper_command must be -1.0 or 1.0")
        return self.make_action(state.robot.tcp_pose, gripper_command)

    # ------------------------------------------------------------------
    # Joint-position action helpers (method-B joint state machine)
    # ------------------------------------------------------------------
    def _resolve_joint_action(self) -> dict:
        """Resolve and cache the arm joint-position action term layout (scale/offset/indices).

        The processed joint target is ``q = raw * scale + offset`` with ``offset`` cloned at action
        term construction (``use_default_offset`` -> ``default_joint_pos``). We therefore read the
        action term's own scale/offset rather than ``robot.data.default_joint_pos`` (which a reset
        event may overwrite without touching the cached offset).
        """
        if self._joint_action_cache is not None:
            return self._joint_action_cache
        action_manager = self.env.unwrapped.action_manager
        arm_term = action_manager.get_term("arm_action")
        # location of the arm block inside the concatenated action vector
        arm_start = 0
        for name in action_manager.active_terms:
            term = action_manager.get_term(name)
            if term is arm_term:
                break
            arm_start += term.action_dim
        arm_dim = arm_term.action_dim
        scale = arm_term._scale
        offset = arm_term._offset
        if not isinstance(scale, torch.Tensor):
            scale = torch.full((arm_dim,), float(scale), device=self.device)
        else:
            scale = scale[self.env_id].to(self.device)
        if not isinstance(offset, torch.Tensor):
            offset = torch.full((arm_dim,), float(offset), device=self.device)
        else:
            offset = offset[self.env_id].to(self.device)
        self._joint_action_cache = {
            "total_dim": action_manager.total_action_dim,
            "arm_start": arm_start,
            "arm_dim": arm_dim,
            "gripper_index": arm_start + arm_dim,
            "scale": scale,
            "offset": offset,
        }
        print(
            "[SceneStateProvider] joint_action_layout "
            f"total_dim={self._joint_action_cache['total_dim']} arm_start={arm_start} arm_dim={arm_dim} "
            f"gripper_index={self._joint_action_cache['gripper_index']} "
            f"scale={[round(float(v), 4) for v in scale.tolist()]} "
            f"offset={[round(float(v), 4) for v in offset.tolist()]}",
            flush=True,
        )
        return self._joint_action_cache

    def make_joint_action_from_q_des(self, q_des: torch.Tensor, gripper_command: float) -> torch.Tensor:
        """Build a raw joint-position env action from absolute arm joint targets ``q_des`` [7].

        Returns a ``(num_envs, total_action_dim)`` tensor. The gripper slot keeps the binary
        convention used elsewhere (1.0 = open, -1.0 = close).
        """
        layout = self._resolve_joint_action()
        if gripper_command not in (-1.0, 1.0):
            raise ValueError("gripper_command must be -1.0 (close) or 1.0 (open)")
        q_des = torch.as_tensor(q_des, dtype=torch.float32, device=self.device).reshape(-1)
        if q_des.numel() != layout["arm_dim"]:
            raise ValueError(f"q_des must have {layout['arm_dim']} entries, got {q_des.numel()}")
        raw_arm = (q_des - layout["offset"]) / layout["scale"]
        num_envs = self.env.unwrapped.num_envs
        action = torch.zeros((num_envs, layout["total_dim"]), device=self.device)
        action[:, layout["arm_start"] : layout["arm_start"] + layout["arm_dim"]] = raw_arm
        action[:, layout["gripper_index"]] = gripper_command
        return action

    def make_joint_action_from_raw(self, raw_joint_action: torch.Tensor) -> torch.Tensor:
        """Broadcast a learned policy's raw joint action to ``(num_envs, total_action_dim)``.

        The policy output is already a raw action for the joint-position action manager; it must NOT
        be re-interpreted as ``q_des``.
        """
        layout = self._resolve_joint_action()
        num_envs = self.env.unwrapped.num_envs
        raw = torch.as_tensor(raw_joint_action, dtype=torch.float32, device=self.device)
        if raw.dim() == 1:
            raw = raw.unsqueeze(0).expand(num_envs, -1).contiguous()
        if raw.shape[-1] != layout["total_dim"]:
            raise ValueError(
                f"raw_joint_action last dim {raw.shape[-1]} != action_dim {layout['total_dim']}"
            )
        return raw

    def make_hold_joint_action(self, state: SceneState, gripper_command: float | None = None) -> torch.Tensor:
        """Hold the current arm joint position (used for pause / idle / skill switch)."""
        layout = self._resolve_joint_action()
        q_hold = state.robot.joint_pos[self._arm_joint_ids].to(self.device)
        if q_hold.numel() != layout["arm_dim"]:
            q_hold = q_hold.reshape(-1)[: layout["arm_dim"]]
        if gripper_command is None:
            gripper_command = 1.0 if state.robot.gripper_width > 0.05 else -1.0
        return self.make_joint_action_from_q_des(q_hold, gripper_command)

    def arm_joint_pos(self, state: SceneState) -> torch.Tensor:
        """Current absolute arm joint positions [arm_dim] for the configured env."""
        return state.robot.joint_pos[self._arm_joint_ids].clone()

    def get_cabinet_joint_pos(self, joint_name: str) -> float:
        cabinet = self.scene["cabinet"]
        joint_id = self._cabinet_joint_id(joint_name)
        return float(cabinet.data.joint_pos[self.env_id, joint_id].detach().cpu())

    def set_cabinet_joint_target(self, joint_name: str, target: float):
        cabinet = self.scene["cabinet"]
        joint_id = self._cabinet_joint_id(joint_name)
        env_ids = torch.tensor([self.env_id], dtype=torch.long, device=self.device)
        joint_ids = torch.tensor([joint_id], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor([[float(target)]], dtype=cabinet.data.joint_pos.dtype, device=self.device)
        method = self._cabinet_control_method.get(joint_name)
        if method != "write_joint_state_to_sim fallback":
            try:
                cabinet.set_joint_position_target(target_tensor, joint_ids=joint_ids, env_ids=env_ids)
                self._cabinet_control_method[joint_name] = "set_joint_position_target"
                self._log_cabinet_joint_control(joint_name, joint_id)
                return
            except Exception as exc:
                self._cabinet_control_method[joint_name] = "write_joint_state_to_sim fallback"
                print(
                    "[SceneStateProvider] set_joint_position_target failed; "
                    f"using write_joint_state_to_sim fallback for {joint_name}: {exc}",
                    flush=True,
                )
        self._write_single_cabinet_joint(joint_name, joint_id, target_tensor, env_ids, joint_ids)
        self._log_cabinet_joint_control(joint_name, joint_id)

    def reset_cabinet_joint(self, joint_name: str, target: float = 0.0):
        self.set_cabinet_joint_target(joint_name, target)
        cabinet = self.scene["cabinet"]
        joint_id = self._cabinet_joint_id(joint_name)
        env_ids = torch.tensor([self.env_id], dtype=torch.long, device=self.device)
        joint_ids = torch.tensor([joint_id], dtype=torch.long, device=self.device)
        target_tensor = torch.tensor([[float(target)]], dtype=cabinet.data.joint_pos.dtype, device=self.device)
        self._write_single_cabinet_joint(joint_name, joint_id, target_tensor, env_ids, joint_ids)

    def reset_scene_deterministic(self):
        raise RuntimeError("Scene reset moved to SceneLayoutManager.reset_layout()")

    def _find_joints(self, pattern: str) -> torch.Tensor:
        robot = self.scene["robot"]
        try:
            joint_ids, _ = robot.find_joints(pattern)
            return torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        except Exception:
            names = getattr(robot.data, "joint_names", [])
            ids = [idx for idx, name in enumerate(names) if "panda_finger" in name]
            return torch.as_tensor(ids, dtype=torch.long, device=self.device)

    def _read_gripper_width(self, robot) -> float:
        if self._finger_joint_ids.numel() == 0:
            return 0.0
        values = robot.data.joint_pos[self.env_id, self._finger_joint_ids]
        return float(values.sum().detach().cpu())

    def _cabinet_joint_id(self, joint_name: str) -> int:
        if joint_name in self._cabinet_joint_id_cache:
            return self._cabinet_joint_id_cache[joint_name]
        cabinet = self.scene["cabinet"]
        names = list(getattr(cabinet.data, "joint_names", []))
        if joint_name in names:
            joint_id = names.index(joint_name)
        else:
            joint_ids, _ = cabinet.find_joints(joint_name)
            if not joint_ids:
                raise RuntimeError(f"Cabinet joint not found: {joint_name}; available={names}")
            joint_id = int(joint_ids[0])
        self._cabinet_joint_id_cache[joint_name] = joint_id
        return joint_id

    def _write_single_cabinet_joint(
        self,
        joint_name: str,
        joint_id: int,
        target_tensor: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
    ):
        cabinet = self.scene["cabinet"]
        if not hasattr(cabinet, "write_joint_state_to_sim"):
            raise RuntimeError("cabinet.write_joint_state_to_sim is unavailable")
        joint_vel = torch.zeros_like(target_tensor)
        try:
            cabinet.write_joint_state_to_sim(target_tensor, joint_vel, joint_ids=joint_ids, env_ids=env_ids)
        except TypeError:
            full_pos = cabinet.data.joint_pos[self.env_id : self.env_id + 1].clone()
            full_vel = cabinet.data.joint_vel[self.env_id : self.env_id + 1].clone()
            full_pos[0, joint_id] = target_tensor[0, 0]
            full_vel[0, joint_id] = 0.0
            cabinet.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)
        self._cabinet_control_method[joint_name] = "write_joint_state_to_sim fallback"

    def _log_cabinet_joint_control(self, joint_name: str, joint_id: int):
        if joint_name in self._logged_cabinet_joints:
            return
        cabinet = self.scene["cabinet"]
        print(
            "[SceneStateProvider] cabinet_joint_control "
            f"cabinet_joint_names={list(getattr(cabinet.data, 'joint_names', []))} "
            f"selected_drawer_joint_name={joint_name} "
            f"selected_drawer_joint_id={joint_id} "
            f"control_method={self._cabinet_control_method.get(joint_name)}",
            flush=True,
        )
        self._logged_cabinet_joints.add(joint_name)

    def _rigid_object_state(self, name: str) -> ObjectState:
        asset = self.scene[name]
        return ObjectState(
            name=name,
            pose=PoseState(asset.data.root_pos_w[self.env_id].clone(), asset.data.root_quat_w[self.env_id].clone()),
            lin_vel_w=getattr(asset.data, "root_lin_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            ang_vel_w=getattr(asset.data, "root_ang_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
        )

    def _articulation_state(self, name: str, link_names: tuple[str, ...]) -> ObjectState:
        asset = self.scene[name]
        links = {}
        for link_name in link_names:
            pose = self._link_pose(asset, link_name)
            if pose is not None:
                links[link_name] = pose
        joint_pos: dict[str, float] = {}
        joint_vel: dict[str, float] = {}
        for joint_name, joint_id in self._joint_id_map(asset).items():
            joint_pos[joint_name] = float(asset.data.joint_pos[self.env_id, joint_id].detach().cpu())
            joint_vel[joint_name] = float(asset.data.joint_vel[self.env_id, joint_id].detach().cpu())
        return ObjectState(
            name=name,
            pose=PoseState(asset.data.root_pos_w[self.env_id].clone(), asset.data.root_quat_w[self.env_id].clone()),
            lin_vel_w=getattr(asset.data, "root_lin_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            ang_vel_w=getattr(asset.data, "root_ang_vel_w", torch.zeros_like(asset.data.root_pos_w))[self.env_id].clone(),
            links=links,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )

    def _link_pose(self, asset: Any, link_name: str) -> PoseState | None:
        names = getattr(asset.data, "body_names", [])
        candidates = [idx for idx, name in enumerate(names) if link_name in name]
        if not candidates:
            try:
                ids, _ = asset.find_bodies(link_name)
                candidates = list(ids)
            except Exception:
                return None
        idx = int(candidates[0])
        if not hasattr(asset.data, "body_pos_w") or idx >= asset.data.body_pos_w.shape[1]:
            return None
        return PoseState(asset.data.body_pos_w[self.env_id, idx].clone(), asset.data.body_quat_w[self.env_id, idx].clone())

    def _joint_id_map(self, asset: Any) -> dict[str, int]:
        names = getattr(asset.data, "joint_names", [])
        return {name: idx for idx, name in enumerate(names)}

    def _zero_articulation_joints(self, asset: Any):
        if not hasattr(asset, "write_joint_state_to_sim") or not hasattr(asset.data, "joint_pos"):
            return
        joint_pos = torch.zeros_like(asset.data.joint_pos[:1])
        joint_vel = torch.zeros_like(asset.data.joint_vel[:1])
        try:
            asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=torch.tensor([self.env_id], device=self.device))
        except Exception:
            return
