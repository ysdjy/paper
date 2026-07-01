"""Open-drawer skill driven by Isaac Lab's official Franka open-drawer PPO policy.

Each step: state -> DrawerObsAdapter.build -> policy(obs) -> raw joint action. The action is
returned as a SkillCommand with ``control_mode="joint"`` and ``raw_joint_action`` set, so it is sent
unchanged to the joint-position action manager (NOT re-interpreted as q_des).

This skill must NEVER set the cabinet joint target directly: the drawer is opened only by the Franka
physically interacting with the handle. Success is read from the drawer joint position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.base_skill import SkillCommand, pose_tensor
from runtime.drawer_obs_adapter import DrawerObsAdapter
from learned_drawer.official_drawer_policy import OfficialDrawerPolicyWrapper
from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    drawer_control_mode: str = "official_joint_policy"
    drawer_joint_name: str = "joint_0"
    drawer_joint_target: float | None = None  # always None: learned policy never sets joint target
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    max_joint_displacement: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    obs_shape: tuple | None = None
    action_shape: tuple | None = None
    history: list[dict] = field(default_factory=list)


class OfficialDrawerJointSkill:
    backend = "official_joint_policy"

    def __init__(
        self,
        request: SkillRequest,
        policy: OfficialDrawerPolicyWrapper,
        obs_adapter: DrawerObsAdapter,
        drawer_joint_name: str = "joint_0",
        success_threshold: float = 0.20,
        timeout: float = 8.0,
    ):
        self.request = request
        self.policy = policy
        self.obs_adapter = obs_adapter
        self.drawer_joint_name = drawer_joint_name
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime(drawer_joint_name=drawer_joint_name)

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime(
            state="RUN_POLICY",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            drawer_joint_name=self.drawer_joint_name,
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self._drawer_pos(state)
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self._record_transition(state, "IDLE", "RUN_POLICY")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold_command(state, None)

        obs = self.obs_adapter.build(state)
        action = self.policy.act(obs)
        self.runtime.obs_shape = tuple(obs.shape)
        self.runtime.action_shape = tuple(action.shape)

        self.runtime.current_joint_pos = self._drawer_pos(state)
        self.runtime.max_joint_displacement = max(
            self.runtime.max_joint_displacement,
            abs(self.runtime.current_joint_pos - self.runtime.initial_joint_pos),
        )
        if self.runtime.current_joint_pos >= self.success_threshold:
            self._succeed(state)
        elif state.sim_time - self.runtime.start_time > self.timeout:
            self._fail(
                state,
                FailureReason.DRAWER_OPEN_TIMEOUT,
                f"learned policy did not open drawer: pos={self.runtime.current_joint_pos:.4f} "
                f"threshold={self.success_threshold:.2f}",
            )

        self.runtime.last_command_pose = state.robot.tcp_pose
        return SkillCommand(
            tcp_pose_w=state.robot.tcp_pose,
            gripper_command=1.0,
            status=self.status,
            control_mode="joint",
            raw_joint_action=action,
            drawer_joint_target=None,
        )

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return self._hold_command(state, None)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.request.destination_object,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
        )

    def _drawer_pos(self, state: SceneState) -> float:
        cabinet = state.objects.get("cabinet")
        if cabinet is None or self.drawer_joint_name not in cabinet.joint_pos:
            return 0.0
        return float(cabinet.joint_pos[self.drawer_joint_name])

    def _hold_command(self, state: SceneState, _unused) -> SkillCommand:
        return SkillCommand(
            tcp_pose_w=state.robot.tcp_pose,
            gripper_command=1.0,
            status=self.status,
            control_mode="joint",
            joint_target=None,
            raw_joint_action=None,
            drawer_joint_target=None,
        )

    def _transition(self, state: SceneState, new_state: str):
        old = self.runtime.state
        if old == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self._record_transition(state, old, new_state)

    def _record_transition(self, state: SceneState, old_state: str, new_state: str):
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": self.request.skill_type.value,
            "backend": self.backend,
            "from": old_state,
            "to": new_state,
            "drawer_control_mode": self.runtime.drawer_control_mode,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "drawer_joint_pos": round(self.runtime.current_joint_pos, 5),
            "drawer_joint_target": None,
            "obs_shape": list(self.runtime.obs_shape) if self.runtime.obs_shape else None,
            "action_shape": list(self.runtime.action_shape) if self.runtime.action_shape else None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[OfficialDrawerJointSkill] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")
        print(f"[OfficialDrawerJointSkill] failure reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(
            "[OfficialDrawerJointSkill] success "
            f"joint_pos={self.runtime.current_joint_pos:.5f} "
            f"max_displacement={self.runtime.max_joint_displacement:.5f}",
            flush=True,
        )
