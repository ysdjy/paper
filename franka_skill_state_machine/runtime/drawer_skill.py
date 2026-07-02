"""Bottom-drawer open/close scripted joint baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.base_skill import SkillCommand, pose_tensor
from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


from runtime.drawer_target_config import DRAWER_TARGETS

DRAWER_CONTROL_MODE = "scripted_joint"
BOTTOM_DRAWER_LINK = "link_1"
BOTTOM_DRAWER_JOINT = "joint_1"  # confirmed by debug_drawer_joint_scan: bottom drawer = joint_1

# target_drawer -> cabinet joint, from the confirmed central config (top=joint_0, middle=joint_2, bottom=joint_1)
DRAWER_JOINT_BY_TARGET = {name: cfg["joint_name"] for name, cfg in DRAWER_TARGETS.items()}


@dataclass
class DrawerSkillConfig:
    use_scripted_joint_control: bool = True
    open_target_joint_pos: float = 0.25
    open_success_threshold: float = 0.20
    close_target_joint_pos: float = 0.0
    close_success_threshold: float = 0.02
    stable_cycles: int = 3
    state_timeout: float = 8.0


@dataclass
class DrawerRuntime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = DRAWER_CONTROL_MODE
    drawer_joint_name: str = BOTTOM_DRAWER_JOINT
    drawer_joint_target: float | None = None
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    max_joint_displacement: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class DrawerSkill:
    def __init__(self, request: SkillRequest, config: DrawerSkillConfig | None = None):
        self.request = request
        self.cfg = config or DrawerSkillConfig()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = DrawerRuntime()

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        self.runtime = DrawerRuntime(
            state="ACQUIRE_DRAWER",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            last_command_pose=state.robot.tcp_pose,
        )
        self._record_transition(state, "IDLE", "ACQUIRE_DRAWER")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._command(state, None)

        if self.runtime.state == "ACQUIRE_DRAWER":
            if not self._acquire_drawer(state):
                return self._command(state, None)
            self._transition(state, "COMMAND_JOINT_TARGET")

        if self.runtime.state == "COMMAND_JOINT_TARGET":
            self.runtime.drawer_joint_target = self._target_joint_pos()
            self._transition(state, "WAIT_JOINT_REACHED")

        if self.runtime.state == "WAIT_JOINT_REACHED":
            self._update_joint_progress(state)
            if self._success_reached():
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= self.cfg.stable_cycles:
                    self._succeed(state)
                    return self._command(state, self.runtime.drawer_joint_target)
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.state_timeout:
                self._fail(
                    state,
                    FailureReason.DRAWER_OPEN_TIMEOUT
                    if self.request.skill_type == SkillType.OPEN_DRAWER
                    else FailureReason.DRAWER_CLOSE_TIMEOUT,
                    (
                        f"drawer joint did not reach target: pos={self.runtime.current_joint_pos:.5f}, "
                        f"target={self.runtime.drawer_joint_target:.5f}"
                    ),
                )

        return self._command(state, self.runtime.drawer_joint_target)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self.runtime.drawer_joint_target = None
        self._transition(state, "CANCELLED")
        return self._command(state, None)

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

    def _acquire_drawer(self, state: SceneState) -> bool:
        if not self.cfg.use_scripted_joint_control:
            self._fail(state, FailureReason.REQUEST_INVALID, "scripted joint control is disabled")
            return False
        dest = self.request.destination_object or "bottom_drawer"
        if dest not in DRAWER_JOINT_BY_TARGET:
            self._fail(
                state,
                FailureReason.REQUEST_INVALID,
                f"unsupported drawer target: {dest}; supported={list(DRAWER_JOINT_BY_TARGET)}",
            )
            return False
        # explicit drawer_joint param overrides the target->joint mapping
        joint_name = self.request.parameters.get("drawer_joint") or DRAWER_JOINT_BY_TARGET[dest]
        cabinet = state.objects.get("cabinet")
        if cabinet is None:
            self._fail(state, FailureReason.TARGET_LOST, "cabinet state missing")
            return False
        if joint_name not in cabinet.joint_pos:
            self._fail(
                state,
                FailureReason.DRAWER_JOINT_MISSING,
                f"{joint_name} missing; available={list(cabinet.joint_pos.keys())}",
            )
            return False
        self.runtime.drawer_joint_name = joint_name
        self.runtime.initial_joint_pos = float(cabinet.joint_pos[joint_name])
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self.runtime.max_joint_displacement = 0.0
        print(
            "[DrawerSkill] scripted_joint_start "
            f"skill_type={self.request.skill_type.value} "
            f"drawer_joint_names={list(cabinet.joint_pos.keys())} "
            f"selected_drawer_joint={joint_name} "
            f"initial_joint_pos={self.runtime.initial_joint_pos:.5f}",
            flush=True,
        )
        return True

    def _target_joint_pos(self) -> float:
        if self.request.skill_type == SkillType.OPEN_DRAWER:
            return self.cfg.open_target_joint_pos
        return self.cfg.close_target_joint_pos

    def _success_reached(self) -> bool:
        if self.request.skill_type == SkillType.OPEN_DRAWER:
            return self.runtime.current_joint_pos >= self.cfg.open_success_threshold
        return abs(self.runtime.current_joint_pos) <= self.cfg.close_success_threshold

    def _update_joint_progress(self, state: SceneState) -> None:
        cabinet = state.objects.get("cabinet")
        if cabinet is None:
            self._fail(state, FailureReason.TARGET_LOST, "cabinet state missing")
            return
        joint_pos = cabinet.joint_pos.get(self.runtime.drawer_joint_name)
        if joint_pos is None:
            self._fail(state, FailureReason.DRAWER_JOINT_MISSING, f"{self.runtime.drawer_joint_name} missing")
            return
        self.runtime.current_joint_pos = float(joint_pos)
        displacement = abs(self.runtime.current_joint_pos - self.runtime.initial_joint_pos)
        self.runtime.max_joint_displacement = max(self.runtime.max_joint_displacement, displacement)

    def _command(self, state: SceneState, drawer_joint_target: float | None) -> SkillCommand:
        self.runtime.last_command_pose = state.robot.tcp_pose
        return SkillCommand(
            tcp_pose_w=state.robot.tcp_pose,
            gripper_command=1.0,
            status=self.status,
            drawer_joint_name=self.runtime.drawer_joint_name,
            drawer_joint_target=drawer_joint_target,
        )

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old_state = self.runtime.state
        if old_state == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record_transition(state, old_state, new_state)

    def _record_transition(self, state: SceneState, old_state: str, new_state: str):
        cabinet = state.objects.get("cabinet")
        joint_pos = None if cabinet is None else cabinet.joint_pos.get(self.runtime.drawer_joint_name)
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": self.request.skill_type.value,
            "from": old_state,
            "to": new_state,
            "drawer_control_mode": self.runtime.drawer_control_mode,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "drawer_joint_pos": None if joint_pos is None else round(float(joint_pos), 5),
            "drawer_joint_target": None
            if self.runtime.drawer_joint_target is None
            else round(self.runtime.drawer_joint_target, 5),
            "gripper_width": round(state.robot.gripper_width, 5),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[DrawerSkill] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self.runtime.drawer_joint_target = None
        self._transition(state, "FAILED")
        print(f"[DrawerSkill] failure request={self.request.request_id} reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(
            "[DrawerSkill] success "
            f"request={self.request.request_id} "
            f"joint_pos={self.runtime.current_joint_pos:.5f} "
            f"target={self.runtime.drawer_joint_target:.5f} "
            f"max_joint_displacement={self.runtime.max_joint_displacement:.5f}",
            flush=True,
        )
