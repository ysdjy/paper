"""Base helpers for skill state machines."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils import math as math_utils

from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


@dataclass
class SkillCommand:
    """Command emitted by a skill for one control step.

    Two control modes are supported:

    * ``"ik_pose"`` (default, legacy): ``tcp_pose_w`` + ``gripper_command`` are consumed by the
      IK-Abs action path (``SceneStateProvider.make_action``). Used by the original
      ``skill_test_ui.py`` and the IK-Abs skills.
    * ``"joint"``: the command targets a joint-position env. Exactly one of ``joint_target``
      (absolute Franka arm joint angles ``q_des`` of shape [7], from internal IK) or
      ``raw_joint_action`` (a full raw action ready for the joint-position action manager, from a
      learned policy) is set. ``tcp_pose_w`` is kept only for debug / visualization.

    ``drawer_joint_target`` defaults to ``None`` and must stay ``None`` for the learned
    open-drawer policy (it may only be set by the explicit scripted-joint baseline).
    """

    tcp_pose_w: PoseState | None
    gripper_command: float
    status: ExecutionStatus
    drawer_joint_name: str | None = None
    drawer_joint_target: float | None = None
    control_mode: str = "ik_pose"
    joint_target: torch.Tensor | None = None
    raw_joint_action: torch.Tensor | None = None


@dataclass
class PoseError:
    position: float
    orientation: float


class BaseSkill:
    status = ExecutionStatus.IDLE
    failure_reason = FailureReason.NONE

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        raise NotImplementedError

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        return SkillCommand(state.robot.tcp_pose, 1.0, self.status)

    def result(self, state: SceneState) -> SkillResult:
        raise NotImplementedError


def pose_error(current: PoseState, desired: PoseState) -> PoseError:
    pos_error = float(torch.linalg.norm(current.pos_w - desired.pos_w).detach().cpu())
    current_quat = math_utils.normalize(current.quat_w.unsqueeze(0))[0]
    desired_quat = math_utils.normalize(desired.quat_w.unsqueeze(0))[0]
    dot = torch.abs(torch.dot(current_quat, desired_quat)).clamp(max=1.0)
    ori_error = float((2.0 * torch.acos(dot)).detach().cpu())
    return PoseError(pos_error, ori_error)


# --- Global skill speed scale -------------------------------------------------
# A single multiplier applied to every per-step position/orientation increment that any skill
# feeds through ``step_pose`` (grasp / place / open-drawer / close-drawer / open-door / close-door).
# 1.0 = original tuned speed. The interactive UI exposes this so the user can speed up / slow down
# all skill motion live without re-tuning per-skill configs. Headless sequence runs default to 1.0.
_SPEED_SCALE = [1.0]
SPEED_SCALE_MIN = 0.25
SPEED_SCALE_MAX = 8.0


def set_speed_scale(scale: float) -> float:
    """Set the global skill motion speed multiplier (clamped to [SPEED_SCALE_MIN, SPEED_SCALE_MAX])."""
    _SPEED_SCALE[0] = float(min(max(scale, SPEED_SCALE_MIN), SPEED_SCALE_MAX))
    return _SPEED_SCALE[0]


def get_speed_scale() -> float:
    return _SPEED_SCALE[0]


def step_pose(current: PoseState, desired: PoseState, max_pos_step: float, max_ori_step: float) -> PoseState:
    scale = _SPEED_SCALE[0]
    max_pos_step = max_pos_step * scale
    max_ori_step = max_ori_step * scale
    delta = desired.pos_w - current.pos_w
    dist = torch.linalg.norm(delta)
    if float(dist) > max_pos_step:
        pos = current.pos_w + delta / dist * max_pos_step
    else:
        pos = desired.pos_w
    current_quat = math_utils.normalize(current.quat_w.unsqueeze(0))[0]
    desired_quat = math_utils.normalize(desired.quat_w.unsqueeze(0))[0]
    if float(torch.dot(current_quat, desired_quat).detach().cpu()) < 0.0:
        desired_quat = -desired_quat
    angle = math_utils.quat_error_magnitude(current_quat.unsqueeze(0), desired_quat.unsqueeze(0))[0]
    if float(angle) > max_ori_step and float(angle) > 1.0e-6:
        tau = max_ori_step / max(float(angle), 1.0e-6)
        quat = math_utils.quat_slerp(current_quat, desired_quat, tau)
    else:
        quat = desired_quat
    return PoseState(pos, math_utils.normalize(quat.unsqueeze(0))[0])


def pose_tensor(pose: PoseState | None) -> torch.Tensor | None:
    if pose is None:
        return None
    return torch.cat((pose.pos_w, pose.quat_w), dim=-1)


def finite_pose(pose: PoseState) -> bool:
    return bool(torch.isfinite(pose.pos_w).all() and torch.isfinite(pose.quat_w).all())


def is_pose_reached(error: PoseError, pos_threshold: float, ori_threshold: float) -> bool:
    return error.position <= pos_threshold and error.orientation <= ori_threshold


def clamp_workspace(pose: PoseState, lower: torch.Tensor, upper: torch.Tensor) -> PoseState:
    pos = torch.minimum(torch.maximum(pose.pos_w, lower), upper)
    return PoseState(pos, pose.quat_w)


def format_pose(pose: PoseState) -> list[float]:
    return [round(float(x), 5) for x in pose.as_pose_tensor().detach().cpu().tolist()]


def safe_acos_dot(q1: torch.Tensor, q2: torch.Tensor) -> float:
    dot = float(torch.abs(torch.dot(q1, q2)).clamp(max=1.0).detach().cpu())
    return 2.0 * math.acos(dot)
