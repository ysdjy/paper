"""Rule-based grasp skill for cube and knife skill tests."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from runtime.base_skill import SkillCommand, finite_pose, format_pose, pose_error, pose_tensor, step_pose
from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, GraspState, SkillType
from runtime.target_registry import GraspPlan, TargetRegistry


@dataclass
class GraspSkillConfig:
    move_position_threshold: float = 0.015
    move_stable_cycles: int = 5
    align_position_threshold: float = 0.012
    align_orientation_threshold: float = math.radians(8.0)
    align_stable_cycles: int = 5
    descend_position_threshold: float = 0.006
    descend_orientation_threshold: float = math.radians(5.0)
    descend_stable_cycles: int = 5
    full_lift_stable_cycles: int = 10
    max_position_step: float = 0.020
    max_orientation_step: float = math.radians(12.0)
    align_max_position_step: float = 0.005
    align_max_orientation_step: float = math.radians(5.0)
    descend_max_position_step: float = 0.003
    descend_max_orientation_step: float = math.radians(3.0)
    lift_max_position_step: float = 0.006
    lift_max_orientation_step: float = math.radians(5.0)
    max_target_jump: float = 0.120
    settle_duration: float = 0.25
    close_duration: float = 0.60
    verify_duration: float = 0.30
    state_timeout: float = 10.0
    align_timeout: float = 8.0
    descend_timeout: float = 10.0
    lift_timeout: float = 6.0
    probe_min_lift_height: float = 0.015
    probe_relative_motion_tolerance: float = 0.015
    full_lift_height_tolerance: float = 0.025
    full_lift_relative_motion_tolerance: float = 0.018
    workspace_lower: tuple[float, float, float] = (0.13, -0.65, 0.010)
    workspace_upper: tuple[float, float, float] = (0.85, 0.65, 0.80)


@dataclass
class _Runtime:
    state: GraspState = GraspState.IDLE
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    state_cycles: int = 0
    filtered_plan: GraspPlan | None = None
    locked_grasp_pose: PoseState | None = None
    locked_probe_lift_pose: PoseState | None = None
    locked_full_lift_pose: PoseState | None = None
    locked_object_pose: PoseState | None = None
    locked_tcp_object_offset: torch.Tensor | None = None
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    logged_plan_debug: bool = False
    history: list[dict] = field(default_factory=list)


class GraspSkill:
    """Grasps a registered object using live target pose until the close phase."""

    def __init__(self, request: SkillRequest, registry: TargetRegistry, config: GraspSkillConfig | None = None):
        self.request = request
        self.registry = registry
        self.cfg = config or GraspSkillConfig()
        self.runtime = _Runtime()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE

    @property
    def current_state(self) -> str:
        return self.runtime.state.value

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        now = state.sim_time
        self.runtime = _Runtime(
            state=GraspState.ACQUIRE_TARGET,
            start_time=now,
            state_start_time=now,
            last_command_pose=state.robot.tcp_pose,
        )
        self._record_transition(state, GraspState.IDLE, GraspState.ACQUIRE_TARGET)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return SkillCommand(state.robot.tcp_pose, self._close_command(), self.status)

        self.runtime.state_cycles += 1
        desired = state.robot.tcp_pose
        command_pose = state.robot.tcp_pose
        gripper = self._open_command()

        plan = self._plan_for_state(state)
        if plan is None or not plan.valid:
            self._fail(state, plan.failure_reason if plan else FailureReason.TARGET_LOST, plan.message if plan else "")
            return SkillCommand(state.robot.tcp_pose, self._open_command(), self.status)

        if self.runtime.state == GraspState.ACQUIRE_TARGET:
            if not self._poses_are_finite(plan):
                self._fail(state, FailureReason.TARGET_UNSAFE, "computed grasp plan contains non-finite pose")
            else:
                self._transition(state, GraspState.MOVE_TO_PRE_GRASP)

        if self.runtime.state == GraspState.MOVE_TO_PRE_GRASP:
            desired = plan.pre_grasp_pose
            command_pose = self._bounded_command(state, desired, 0.012, math.radians(4.0))
            self._advance_when_reached(
                state,
                desired,
                GraspState.DESCEND_TO_GRASP,
                0.012,
                math.radians(6.0),
                self.cfg.move_stable_cycles,
                self.cfg.state_timeout,
            )
        elif self.runtime.state == GraspState.ALIGN_GRASP:
            desired = plan.pre_grasp_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.align_max_position_step,
                self.cfg.align_max_orientation_step,
            )
            self._advance_when_reached(
                state,
                desired,
                GraspState.DESCEND_TO_GRASP,
                self.cfg.align_position_threshold,
                self.cfg.align_orientation_threshold,
                self.cfg.align_stable_cycles,
                self.cfg.align_timeout,
            )
        elif self.runtime.state == GraspState.DESCEND_TO_GRASP:
            desired = plan.grasp_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.descend_max_position_step,
                self.cfg.descend_max_orientation_step,
            )
            self._log_descend_progress(state, desired)
            if self._advance_when_reached(
                state,
                desired,
                GraspState.SETTLE_AT_GRASP,
                self.cfg.descend_position_threshold,
                self.cfg.descend_orientation_threshold,
                self.cfg.descend_stable_cycles,
                self.cfg.descend_timeout,
                timeout_reason=FailureReason.GRASP_POSE_TIMEOUT,
            ):
                self._lock_grasp_pose(plan)
        elif self.runtime.state == GraspState.SETTLE_AT_GRASP:
            desired = self.runtime.locked_grasp_pose or plan.grasp_pose
            command_pose = desired
            if self._state_elapsed(state) >= self.cfg.settle_duration:
                if self._lock_close_reference(state, plan):
                    self._transition(state, GraspState.CLOSE_GRIPPER)
        elif self.runtime.state == GraspState.CLOSE_GRIPPER:
            desired = self.runtime.locked_grasp_pose or plan.grasp_pose
            command_pose = desired
            gripper = plan.gripper_close_command
            if self._state_elapsed(state) >= self.cfg.close_duration:
                self._transition(state, GraspState.VERIFY_GRASP)
        elif self.runtime.state == GraspState.VERIFY_GRASP:
            desired = self.runtime.locked_grasp_pose or plan.grasp_pose
            command_pose = desired
            gripper = plan.gripper_close_command
            if self._state_elapsed(state) >= self.cfg.verify_duration:
                ok, message = self._verify_closed_width(state, plan)
                if ok:
                    self._transition(state, GraspState.PROBE_LIFT)
                else:
                    self._fail(state, FailureReason.GRASP_VERIFICATION_FAILED, message)
        elif self.runtime.state == GraspState.PROBE_LIFT:
            desired = self.runtime.locked_probe_lift_pose or plan.probe_lift_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.lift_max_position_step,
                self.cfg.lift_max_orientation_step,
            )
            self._advance_when_reached(
                state,
                desired,
                GraspState.VERIFY_PROBE_LIFT,
                self.cfg.align_position_threshold,
                self.cfg.align_orientation_threshold,
                self.cfg.align_stable_cycles,
                self.cfg.lift_timeout,
            )
            gripper = plan.gripper_close_command
        elif self.runtime.state == GraspState.VERIFY_PROBE_LIFT:
            desired = self.runtime.locked_probe_lift_pose or plan.probe_lift_pose
            command_pose = desired
            gripper = plan.gripper_close_command
            if self._verify_probe_lift(state, plan):
                self._transition(state, GraspState.FULL_LIFT)
            elif self._state_elapsed(state) > self.cfg.verify_duration + 0.50:
                self._fail(state, FailureReason.GRASP_VERIFICATION_FAILED, "object did not follow probe lift")
        elif self.runtime.state == GraspState.FULL_LIFT:
            desired = self.runtime.locked_full_lift_pose or plan.full_lift_pose
            command_pose = self._bounded_command(
                state,
                desired,
                self.cfg.lift_max_position_step,
                self.cfg.lift_max_orientation_step,
            )
            self._advance_when_reached(
                state,
                desired,
                GraspState.VERIFY_FULL_LIFT,
                self.cfg.align_position_threshold,
                self.cfg.align_orientation_threshold,
                self.cfg.align_stable_cycles,
                self.cfg.lift_timeout,
            )
            gripper = plan.gripper_close_command
        elif self.runtime.state == GraspState.VERIFY_FULL_LIFT:
            desired = self.runtime.locked_full_lift_pose or plan.full_lift_pose
            command_pose = desired
            gripper = plan.gripper_close_command
            if self._verify_full_lift(state, plan):
                self._transition(state, GraspState.HOLD)
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.verify_duration + 1.50:
                self._fail(state, FailureReason.OBJECT_DROPPED, "object did not remain stable after full lift")
        elif self.runtime.state == GraspState.HOLD:
            desired = self.runtime.locked_full_lift_pose or plan.full_lift_pose
            command_pose = desired
            gripper = plan.gripper_close_command

        error = pose_error(state.robot.tcp_pose, desired)
        self.runtime.final_error_pos = error.position
        self.runtime.final_error_ori = error.orientation
        self.runtime.last_command_pose = command_pose
        return SkillCommand(command_pose, gripper, self.status)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, GraspState.CANCELLED)
        return SkillCommand(state.robot.tcp_pose, self._open_command(), self.status)

    def result(self, state: SceneState) -> SkillResult:
        obj_pose = self._current_target_pose(state)
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=SkillType.GRASP,
            target_name=self.request.source_object,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=self.failure_reason.value or None,
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
            final_tcp_pose=pose_tensor(state.robot.tcp_pose),
            final_object_pose=pose_tensor(obj_pose),
            position_error=self.runtime.final_error_pos,
            orientation_error=self.runtime.final_error_ori,
            gripper_width=state.robot.gripper_width,
            state_history=self.runtime.history,
        )

    def _plan_for_state(self, state: SceneState) -> GraspPlan | None:
        if self.runtime.state in {
            GraspState.ACQUIRE_TARGET,
            GraspState.MOVE_TO_PRE_GRASP,
            GraspState.ALIGN_GRASP,
            GraspState.DESCEND_TO_GRASP,
        }:
            return self._update_plan(state)
        return self.runtime.filtered_plan

    def _update_plan(self, state: SceneState) -> GraspPlan:
        plan = self.registry.compute_grasp_plan(self.request.source_object or "", state)
        if not plan.valid:
            return plan
        previous = self.runtime.filtered_plan
        if previous is not None:
            jump = torch.linalg.norm(plan.grasp_pose.pos_w - previous.grasp_pose.pos_w)
            if float(jump) > self.cfg.max_target_jump:
                plan.valid = False
                plan.failure_reason = FailureReason.TARGET_LOST
                plan.message = f"target jump exceeded limit: {float(jump):.3f} m"
                return plan
        self.runtime.filtered_plan = plan
        if not self.runtime.logged_plan_debug:
            self._log_initial_plan_debug(state, plan)
            self.runtime.logged_plan_debug = True
        return plan

    def _bounded_command(
        self,
        state: SceneState,
        desired: PoseState,
        max_position_step: float | None = None,
        max_rotation_step: float | None = None,
    ) -> PoseState:
        command_from = self.runtime.last_command_pose or state.robot.tcp_pose
        lower = torch.tensor(self.cfg.workspace_lower, dtype=torch.float32, device=desired.pos_w.device)
        upper = torch.tensor(self.cfg.workspace_upper, dtype=torch.float32, device=desired.pos_w.device)
        bounded = PoseState(torch.minimum(torch.maximum(desired.pos_w, lower), upper), desired.quat_w)
        if not finite_pose(bounded):
            self._fail(state, FailureReason.IK_UNREACHABLE, "non-finite target pose")
            return state.robot.tcp_pose
        return step_pose(
            command_from,
            bounded,
            self.cfg.max_position_step if max_position_step is None else max_position_step,
            self.cfg.max_orientation_step if max_rotation_step is None else max_rotation_step,
        )

    def _advance_when_reached(
        self,
        state: SceneState,
        desired: PoseState,
        next_state: GraspState,
        position_threshold: float,
        orientation_threshold: float,
        stable_cycles: int,
        timeout: float,
        timeout_reason: FailureReason | None = None,
    ) -> bool:
        error = pose_error(state.robot.tcp_pose, desired)
        if error.position <= position_threshold and error.orientation <= orientation_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > timeout:
            reason = timeout_reason or (
                FailureReason.POSITION_TIMEOUT if error.position > position_threshold else FailureReason.ORIENTATION_TIMEOUT
            )
            self._fail(
                state,
                reason,
                f"pose error p={error.position:.4f}, orientation_error_deg={math.degrees(error.orientation):.2f}",
            )
        return False

    def _lock_grasp_pose(self, plan: GraspPlan) -> None:
        self.runtime.locked_grasp_pose = plan.grasp_pose
        self.runtime.locked_probe_lift_pose = plan.probe_lift_pose
        self.runtime.locked_full_lift_pose = plan.full_lift_pose

    def _lock_close_reference(self, state: SceneState, plan: GraspPlan) -> bool:
        target_pose = self._current_target_pose(state)
        if target_pose is None:
            self._fail(state, FailureReason.TARGET_LOST, "target disappeared before close")
            return False
        tcp_pose = state.robot.tcp_pose
        self.runtime.locked_object_pose = target_pose
        self.runtime.locked_tcp_object_offset = target_pose.pos_w - tcp_pose.pos_w
        self.runtime.locked_grasp_pose = self.runtime.locked_grasp_pose or plan.grasp_pose
        self.runtime.locked_probe_lift_pose = self.runtime.locked_probe_lift_pose or plan.probe_lift_pose
        self.runtime.locked_full_lift_pose = self.runtime.locked_full_lift_pose or plan.full_lift_pose
        return True

    def _verify_closed_width(self, state: SceneState, plan: GraspPlan) -> tuple[bool, str]:
        width = state.robot.gripper_width
        if width < plan.min_gripper_width:
            return False, f"gripper closed empty width={width:.5f} min={plan.min_gripper_width:.5f}"
        if width > plan.max_gripper_width:
            return False, f"gripper too open width={width:.5f} max={plan.max_gripper_width:.5f}"
        return True, ""

    def _verify_probe_lift(self, state: SceneState, plan: GraspPlan) -> bool:
        target_pose = self._current_target_pose(state)
        locked_target = self.runtime.locked_object_pose
        locked_offset = self.runtime.locked_tcp_object_offset
        if target_pose is None or locked_target is None or locked_offset is None:
            return False
        width_ok, _ = self._verify_closed_width(state, plan)
        height_gain = float((target_pose.pos_w[2] - locked_target.pos_w[2]).detach().cpu())
        relative_offset = target_pose.pos_w - state.robot.tcp_pose.pos_w
        relative_error = float(torch.linalg.norm(relative_offset - locked_offset).detach().cpu())
        return bool(
            width_ok
            and height_gain >= self.cfg.probe_min_lift_height
            and relative_error <= self.cfg.probe_relative_motion_tolerance
        )

    def _verify_full_lift(self, state: SceneState, plan: GraspPlan) -> bool:
        target_pose = self._current_target_pose(state)
        locked_target = self.runtime.locked_object_pose
        locked_offset = self.runtime.locked_tcp_object_offset
        if target_pose is None or locked_target is None or locked_offset is None:
            self.runtime.stable_count = 0
            return False
        width_ok, _ = self._verify_closed_width(state, plan)
        height_gain = float((target_pose.pos_w[2] - locked_target.pos_w[2]).detach().cpu())
        relative_offset = target_pose.pos_w - state.robot.tcp_pose.pos_w
        relative_error = float(torch.linalg.norm(relative_offset - locked_offset).detach().cpu())
        expected_height = max(plan.lift_distance - self.cfg.full_lift_height_tolerance, self.cfg.probe_min_lift_height)
        stable = (
            width_ok
            and height_gain >= expected_height
            and relative_error <= self.cfg.full_lift_relative_motion_tolerance
        )
        if stable:
            self.runtime.stable_count += 1
        else:
            self.runtime.stable_count = 0
        return self.runtime.stable_count >= self.cfg.full_lift_stable_cycles

    def _current_target_pose(self, state: SceneState) -> PoseState | None:
        if self.request.source_object not in state.objects:
            return None
        return self.registry.current_target_pose(self.request.source_object or "", state)

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _open_command(self) -> float:
        plan = self.runtime.filtered_plan
        return 1.0 if plan is None else plan.gripper_open_command

    def _close_command(self) -> float:
        plan = self.runtime.filtered_plan
        return -1.0 if plan is None else plan.gripper_close_command

    def _transition(self, state: SceneState, new_state: GraspState):
        old_state = self.runtime.state
        if old_state == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self.runtime.state_cycles = 0
        if new_state == GraspState.DESCEND_TO_GRASP:
            self._log_pre_grasp_error(state)
        self._record_transition(state, old_state, new_state)

    def _record_transition(self, state: SceneState, old_state: GraspState, new_state: GraspState):
        plan = self.runtime.filtered_plan
        command_pose = self.runtime.last_command_pose or state.robot.tcp_pose
        error = pose_error(state.robot.tcp_pose, command_pose)
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": SkillType.GRASP.value,
            "target": self.request.source_object,
            "from": old_state.value,
            "to": new_state.value,
            "object_pose": None if plan is None else format_pose(plan.target_pose),
            "tcp_pose": format_pose(state.robot.tcp_pose),
            "pre_grasp_pose": None if plan is None else format_pose(plan.pre_grasp_pose),
            "grasp_pose": None if plan is None else format_pose(plan.grasp_pose),
            "current_tcp_quat": format_pose(state.robot.tcp_pose)[3:7],
            "target_tcp_quat": None if plan is None else format_pose(plan.grasp_pose)[3:7],
            "position_error": round(error.position, 5),
            "orientation_error_deg": round(math.degrees(error.orientation), 3),
            "gripper_width": round(state.robot.gripper_width, 5),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[SkillRuntime] transition {record}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, GraspState.FAILED)
        print(f"[SkillRuntime] failure request={self.request.request_id} reason={reason.value} {message}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, GraspState.SUCCEEDED)
        print(f"[SkillRuntime] success request={self.request.request_id} target={self.request.source_object}", flush=True)

    def _poses_are_finite(self, plan: GraspPlan) -> bool:
        return all(
            finite_pose(pose)
            for pose in (
                plan.target_pose,
                plan.grasp_pose,
                plan.pre_grasp_pose,
                plan.probe_lift_pose,
                plan.full_lift_pose,
            )
        )

    def _log_initial_plan_debug(self, state: SceneState, plan: GraspPlan) -> None:
        orientation_error = pose_error(state.robot.tcp_pose, plan.grasp_pose).orientation
        object_quat = plan.target_pose.quat_w if plan.object_quat_w is None else plan.object_quat_w
        record = {
            "target_name": plan.target_name,
            "object_quat_wxyz": self._quat_list(object_quat),
            "object_yaw_deg": None if plan.object_yaw is None else round(math.degrees(plan.object_yaw), 3),
            "planned_grasp_quat_wxyz": self._quat_list(plan.grasp_pose.quat_w),
            "current_tcp_quat_wxyz": self._quat_list(state.robot.tcp_pose.quat_w),
            "initial_orientation_error_deg": round(math.degrees(orientation_error), 3),
        }
        print(f"[GraspPlanDebug] {record}", flush=True)
        if plan.target_name == "knife" and plan.object_yaw is not None and plan.grasp_yaw is not None:
            knife_record = {
                "knife_object_yaw_deg": round(math.degrees(plan.object_yaw), 3),
                "knife_grasp_yaw_deg": round(math.degrees(plan.grasp_yaw), 3),
                "yaw_difference_deg": round(math.degrees(plan.grasp_yaw - plan.object_yaw), 3),
            }
            print(f"[GraspPlanDebug] {knife_record}", flush=True)

    def _log_pre_grasp_error(self, state: SceneState) -> None:
        plan = self.runtime.filtered_plan
        if plan is None:
            return
        error = pose_error(state.robot.tcp_pose, plan.pre_grasp_pose)
        record = {
            "pre_grasp_position_error": round(error.position, 5),
            "pre_grasp_orientation_error_deg": round(math.degrees(error.orientation), 3),
        }
        print(f"[GraspPlanDebug] {record}", flush=True)

    def _quat_list(self, quat: torch.Tensor) -> list[float]:
        return [round(float(x), 5) for x in quat.detach().cpu().tolist()]

    def _log_descend_progress(self, state: SceneState, desired: PoseState) -> None:
        if self.runtime.state_cycles % 10 != 0:
            return
        z_error = float((desired.pos_w[2] - state.robot.tcp_pose.pos_w[2]).detach().cpu())
        print(
            "[SkillRuntime] descend "
            f"target={self.request.source_object} "
            f"current_tcp_z={float(state.robot.tcp_pose.pos_w[2].detach().cpu()):.5f} "
            f"target_grasp_z={float(desired.pos_w[2].detach().cpu()):.5f} "
            f"z_error={z_error:.5f}",
            flush=True,
        )
