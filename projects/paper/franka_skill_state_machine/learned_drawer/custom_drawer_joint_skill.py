"""Open-drawer skill driven by the learned custom selected-drawer policy.

The state machine passes only ``target_drawer`` (top_drawer / middle_drawer / bottom_drawer). This
skill resolves the joint / handle for that drawer from the central config, builds the 31-d selected
obs, runs the learned policy, and returns the raw joint action. It NEVER sets the cabinet joint
target (``drawer_joint_target`` is always None) — the drawer must be opened by physical interaction.

bottom_drawer is currently locked / non-functional and is rejected here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.base_skill import SkillCommand, pose_tensor
from runtime.drawer_obs_adapter import SelectedDrawerObsAdapter
from runtime.drawer_target_config import DRAWER_TARGETS, get_drawer_config
from learned_drawer.official_drawer_policy import OfficialDrawerPolicyWrapper
from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    drawer_control_mode: str = "custom_selected_policy"
    target_drawer: str = ""
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None  # always None for the learned policy
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


class CustomDrawerJointSkill:
    backend = "custom_selected_policy"

    def __init__(
        self,
        request: SkillRequest,
        policy: OfficialDrawerPolicyWrapper,
        env,
        timeout: float = 8.0,
    ):
        self.request = request
        self.policy = policy
        self.env = env
        self.timeout = timeout
        self.target_drawer = request.destination_object or "top_drawer"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.obs_adapter: SelectedDrawerObsAdapter | None = None
        cfg = DRAWER_TARGETS.get(self.target_drawer, {})
        self.success_threshold = cfg.get("success_threshold", 0.20)
        self.open_direction = cfg.get("open_direction", 1)
        self.runtime = _Runtime(
            target_drawer=self.target_drawer, drawer_joint_name=cfg.get("joint_name", "")
        )

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.failure_reason = FailureReason.NONE
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        if cfg is None:
            self.status = ExecutionStatus.RUNNING
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        if not cfg.get("functional", False):
            self.status = ExecutionStatus.RUNNING
            self._fail(
                state,
                FailureReason.REQUEST_INVALID,
                f"{self.target_drawer} is currently locked / non-functional; "
                "train/test top and middle first.",
            )
            print(
                f"[CustomDrawerJointSkill][WARNING] {self.target_drawer} is currently locked / "
                "non-functional; train/test top and middle first.",
                flush=True,
            )
            return

        self.status = ExecutionStatus.RUNNING
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer)
        self.runtime = _Runtime(
            state="RUN_POLICY",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            target_drawer=self.target_drawer,
            drawer_joint_name=cfg["joint_name"],
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self.obs_adapter.selected_drawer_joint_pos()
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self._record_transition(state, "IDLE", "RUN_POLICY")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold_command(state)

        obs = self.obs_adapter.build(state)
        action = self.policy.act(obs)
        self.runtime.obs_shape = tuple(obs.shape)
        self.runtime.action_shape = tuple(action.shape)
        self.runtime.current_joint_pos = self.obs_adapter.selected_drawer_joint_pos()
        self.runtime.max_joint_displacement = max(
            self.runtime.max_joint_displacement,
            abs(self.runtime.current_joint_pos - self.runtime.initial_joint_pos),
        )

        if self.runtime.current_joint_pos * self.open_direction >= self.success_threshold:
            self._succeed(state)
        elif state.sim_time - self.runtime.start_time > self.timeout:
            self._fail(
                state,
                FailureReason.DRAWER_OPEN_TIMEOUT,
                f"learned policy did not open {self.target_drawer}: pos="
                f"{self.runtime.current_joint_pos:.4f} threshold={self.success_threshold:.2f}",
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
        return self._hold_command(state)

    def result(self, state: SceneState) -> SkillResult:
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
            target_name=self.target_drawer,
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

    def _hold_command(self, state: SceneState) -> SkillCommand:
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
            "target_drawer": self.runtime.target_drawer,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "drawer_joint_pos": round(self.runtime.current_joint_pos, 5),
            "drawer_joint_target": None,
            "obs_shape": list(self.runtime.obs_shape) if self.runtime.obs_shape else None,
            "action_shape": list(self.runtime.action_shape) if self.runtime.action_shape else None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[CustomDrawerJointSkill] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")
        print(f"[CustomDrawerJointSkill] failure target={self.target_drawer} reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(
            f"[CustomDrawerJointSkill] success target={self.target_drawer} "
            f"joint_pos={self.runtime.current_joint_pos:.5f} max_disp={self.runtime.max_joint_displacement:.5f}",
            flush=True,
        )
