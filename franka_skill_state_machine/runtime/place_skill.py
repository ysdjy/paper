"""Place skill (stage-1 parameterised + object-verified).

State machine (task spec section 4.4)::

    MOVE_TO_PRE_PLACE -> DESCEND_TO_RELEASE -> OPEN_GRIPPER -> WAIT_FOR_SETTLE
                      -> RETREAT -> VERIFY_PLACE -> SUCCEEDED / FAILED

Key stage-1 properties:

* Execution parameters come from ``request.parameters`` via :class:`ParamResolver`
  (priority request > config default > legacy constant); every value is traced.
* ``release_clearance`` changes ONLY the execution (the height at which the object is released);
  the evaluation target is always the original task surface pose (section 4.3 / 4.6).
* ``success`` is decided in VERIFY_PLACE from the real OBJECT state (position/orientation error,
  linear/angular speed, settled cycles, not dropped, not out of bounds, not timed out) -- never from
  the state machine merely reaching the end (the old behaviour is kept only as the debug flag
  ``legacy_reached``).
* The object pose<->TCP relationship (``object_to_tcp`` captured at grasp) and the DLS-IK reverse
  mapping are preserved unchanged from the original skill (section 4.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.utils import math as math_utils

from runtime.base_skill import SkillCommand, finite_pose, format_pose, pose_error, pose_tensor, step_pose
from runtime.experiment_params import InvalidExperimentParameter, ParamResolver
from runtime.scene_state_provider import PoseState, SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType


OBJECT_SUPPORT_OFFSET_Z = {
    "cube_1": 0.0203,
    "cube_2": 0.0203,
    "cube_3": 0.0203,
    "knife": 0.095,
}
# nominal contact gap baked into the TASK target (object底面 just above the surface). NOT a research
# parameter; ``release_clearance`` is the experiment knob layered on top of this for execution only.
BASE_CLEARANCE = 0.002


@dataclass
class PlaceSkillConfig:
    # --- geometric / motion / timing execution params (research knobs, overridable via request) ---
    pre_place_height: float = 0.100
    release_clearance: float = 0.0
    move_max_position_step: float = 0.012
    move_max_orientation_step: float = math.radians(5.0)
    descend_max_position_step: float = 0.006
    descend_max_orientation_step: float = math.radians(4.0)
    open_duration: float = 0.45
    settle_after_release_duration: float = 0.5
    retreat_height: float = 0.10
    retreat_max_position_step: float = 0.012
    # --- judgment params (recorded + fixed; NOT first-batch research knobs) ---
    # phase-advance reach gate. 0.020 m (=2 cm) is the steady-state residual of the single-step DLS
    # IK at the workspace edge (the old 0.012 m gate stalled MOVE_TO_PRE_PLACE -> POSITION_TIMEOUT);
    # a 2 cm "close enough to advance" gate is physically fine (final accuracy is judged on the OBJECT
    # in VERIFY_PLACE, tol=3 cm), and keeps default placements succeeding.
    position_threshold: float = 0.020
    orientation_threshold: float = math.radians(10.0)
    stable_cycles: int = 5
    state_timeout: float = 12.0
    # --- success-verification thresholds (sized to the scene: cube edge ~0.04 m) ---
    object_position_tolerance: float = 0.030
    object_orientation_tolerance: float = math.radians(20.0)
    settle_linear_speed_threshold: float = 0.02
    settle_angular_speed_threshold: float = 0.5
    settle_required_cycles: int = 3
    out_of_bounds_radius: float = 0.15
    dropped_z_margin: float = 0.05


@dataclass
class PlacePlan:
    point_name: str
    held_object_name: str
    target_surface_xyz: list[float]
    support_offset_z: float
    # evaluation target (object should rest here -- no release_clearance)
    task_target_pose: PoseState
    # execution target (object released from here -- includes release_clearance)
    release_object_pose: PoseState
    pre_object_pose: PoseState
    release_tcp_pose: PoseState
    pre_place_tcp_pose: PoseState


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    settle_count: int = 0
    settling_time: float | None = None
    plan: PlacePlan | None = None
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    legacy_reached: bool = False
    # running TCP tracking-error stats over the motion phases
    track_err_max: float = 0.0
    track_err_sum: float = 0.0
    track_err_n: int = 0
    history: list[dict] = field(default_factory=list)


class PlaceSkill:
    def __init__(self, request: SkillRequest, held_object: Any | None = None, config: PlaceSkillConfig | None = None):
        self.request = request
        self.held_object = held_object
        self.cfg = config or PlaceSkillConfig()
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime()
        self.resolver = ParamResolver(request.parameters)
        self.last_telemetry: dict[str, Any] = {}
        self._eff: dict[str, Any] = {}
        self._param_error: str | None = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- parameter resolution -------------------------------------------------
    def _resolve_params(self) -> bool:
        """Resolve all execution params from the request. Returns False (and sets failure) on a
        rejected experiment parameter."""
        c = self.cfg
        at = "place_skill"
        try:
            r = self.resolver
            e = {}
            e["pre_place_height"], _ = r.float("pre_place_height", c.pre_place_height,
                                               minimum=0.02, maximum=0.30, hard_minimum=0.0, hard_maximum=0.6, applied_at=at)
            e["release_clearance"], _ = r.float("release_clearance", c.release_clearance,
                                                minimum=0.0, maximum=0.10, hard_minimum=0.0, hard_maximum=0.3, applied_at=at)
            e["move_max_position_step"], _ = r.float("move_max_position_step", c.move_max_position_step,
                                                     minimum=0.002, maximum=0.05, hard_minimum=0.0005, hard_maximum=0.2, applied_at=at)
            e["move_max_orientation_step"], _ = r.float("move_max_orientation_step", c.move_max_orientation_step,
                                                        minimum=math.radians(1.0), maximum=math.radians(20.0), hard_maximum=math.radians(45.0), applied_at=at)
            e["descend_max_position_step"], _ = r.float("descend_max_position_step", c.descend_max_position_step,
                                                        minimum=0.001, maximum=0.03, hard_minimum=0.0005, hard_maximum=0.2, applied_at=at)
            e["descend_max_orientation_step"], _ = r.float("descend_max_orientation_step", c.descend_max_orientation_step,
                                                           minimum=math.radians(1.0), maximum=math.radians(20.0), hard_maximum=math.radians(45.0), applied_at=at)
            e["open_duration"], _ = r.float("open_duration", c.open_duration,
                                            minimum=0.1, maximum=2.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["settle_after_release_duration"], _ = r.float("settle_after_release_duration", c.settle_after_release_duration,
                                                            minimum=0.1, maximum=2.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["retreat_height"], _ = r.float("retreat_height", c.retreat_height,
                                             minimum=0.02, maximum=0.30, hard_minimum=0.0, hard_maximum=0.6, applied_at=at)
            e["retreat_max_position_step"], _ = r.float("retreat_max_position_step", c.retreat_max_position_step,
                                                        minimum=0.002, maximum=0.05, hard_minimum=0.0005, hard_maximum=0.2, applied_at=at)
            # judgment params (recorded, normally left at default)
            e["position_threshold"], _ = r.float("position_threshold", c.position_threshold,
                                                 minimum=0.004, maximum=0.05, applied_at=at)
            e["orientation_threshold"], _ = r.float("orientation_threshold", c.orientation_threshold,
                                                    minimum=math.radians(2.0), maximum=math.radians(30.0), applied_at=at)
            e["stable_cycles"], _ = r.int("stable_cycles", c.stable_cycles, minimum=1, maximum=30, applied_at=at)
            e["state_timeout"], _ = r.float("state_timeout", c.state_timeout, minimum=2.0, maximum=60.0, applied_at=at)
            self._eff = e
            return True
        except InvalidExperimentParameter as exc:
            self._param_error = str(exc)
            return False

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        self.runtime = _Runtime(
            state="MOVE_TO_PRE_PLACE",
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            last_command_pose=state.robot.tcp_pose,
        )
        if not self._resolve_params():
            self._fail(state, FailureReason.INVALID_EXPERIMENT_PARAMETER, self._param_error or "bad parameter")
            return
        plan = self._make_plan(state)
        if plan is None:
            return
        self.runtime.plan = plan
        self._log_plan(plan)
        self._record_transition(state, "IDLE", "MOVE_TO_PRE_PLACE")

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            self._update_telemetry(state, state.robot.tcp_pose, self._terminal_gripper_command())
            return SkillCommand(state.robot.tcp_pose, self._terminal_gripper_command(), self.status)

        plan = self.runtime.plan
        if plan is None:
            self._fail(state, FailureReason.REQUEST_INVALID, "place plan is missing")
            return SkillCommand(state.robot.tcp_pose, -1.0, self.status)

        e = self._eff
        desired = state.robot.tcp_pose
        command_pose = state.robot.tcp_pose
        gripper = -1.0

        if self.runtime.state == "MOVE_TO_PRE_PLACE":
            desired = plan.pre_place_tcp_pose
            command_pose = self._bounded_command(state, desired, e["move_max_position_step"], e["move_max_orientation_step"])
            self._advance_when_reached(state, desired, "DESCEND_TO_RELEASE")
        elif self.runtime.state == "DESCEND_TO_RELEASE":
            desired = plan.release_tcp_pose
            command_pose = self._bounded_command(state, desired, e["descend_max_position_step"], e["descend_max_orientation_step"])
            self._advance_when_reached(state, desired, "OPEN_GRIPPER")
        elif self.runtime.state == "OPEN_GRIPPER":
            desired = plan.release_tcp_pose
            command_pose = desired
            gripper = 1.0
            if self._state_elapsed(state) >= e["open_duration"]:
                self.runtime.legacy_reached = True   # old state machine's "done" point (debug only)
                self._transition(state, "WAIT_FOR_SETTLE")
        elif self.runtime.state == "WAIT_FOR_SETTLE":
            desired = plan.release_tcp_pose
            command_pose = desired
            gripper = 1.0
            self._track_settle(state)
            if self._state_elapsed(state) >= e["settle_after_release_duration"]:
                self._transition(state, "RETREAT")
        elif self.runtime.state == "RETREAT":
            desired = self._retreat_pose(plan)
            command_pose = self._bounded_command(state, desired, e["retreat_max_position_step"], e["move_max_orientation_step"])
            gripper = 1.0
            if self._reached(state, desired) or self._state_elapsed(state) > e["state_timeout"]:
                self._transition(state, "VERIFY_PLACE")
        elif self.runtime.state == "VERIFY_PLACE":
            desired = self._retreat_pose(plan)
            command_pose = desired
            gripper = 1.0
            self._verify(state)

        error = pose_error(state.robot.tcp_pose, desired)
        self.runtime.final_error_pos = error.position
        self.runtime.final_error_ori = error.orientation
        # accumulate TCP tracking-error stats over the active motion phases
        if self.runtime.state in ("MOVE_TO_PRE_PLACE", "DESCEND_TO_RELEASE", "RETREAT"):
            self.runtime.track_err_max = max(self.runtime.track_err_max, error.position)
            self.runtime.track_err_sum += error.position
            self.runtime.track_err_n += 1
        self.runtime.last_command_pose = command_pose
        self._update_telemetry(state, desired, gripper)
        return SkillCommand(command_pose, gripper, self.status)

    # ---- settle / retreat / verify -------------------------------------------
    def _track_settle(self, state: SceneState):
        obj = state.objects.get(self.request.source_object or "")
        if obj is None:
            return
        lin = float(torch.linalg.norm(obj.lin_vel_w)) if obj.lin_vel_w is not None else float("nan")
        ang = float(torch.linalg.norm(obj.ang_vel_w)) if obj.ang_vel_w is not None else float("nan")
        c = self.cfg
        if math.isfinite(lin) and math.isfinite(ang) and lin <= c.settle_linear_speed_threshold and ang <= c.settle_angular_speed_threshold:
            self.runtime.settle_count += 1
            if self.runtime.settle_count >= c.settle_required_cycles and self.runtime.settling_time is None:
                self.runtime.settling_time = max(0.0, state.sim_time - self.runtime.state_start_time)
        else:
            self.runtime.settle_count = 0

    def _retreat_pose(self, plan: PlacePlan) -> PoseState:
        pos = plan.release_tcp_pose.pos_w.clone()
        pos[2] = pos[2] + self._eff["retreat_height"]
        return PoseState(pos, plan.release_tcp_pose.quat_w)

    def _object_errors(self, state: SceneState):
        obj = state.objects.get(self.request.source_object or "")
        plan = self.runtime.plan
        if obj is None or plan is None:
            return None
        pos_err = float(torch.linalg.norm(obj.pose.pos_w - plan.task_target_pose.pos_w))
        q_cur = math_utils.normalize(obj.pose.quat_w.unsqueeze(0))[0]
        q_tgt = math_utils.normalize(plan.task_target_pose.quat_w.unsqueeze(0))[0]
        ori_err = float(math_utils.quat_error_magnitude(q_cur.unsqueeze(0), q_tgt.unsqueeze(0))[0])
        lin = float(torch.linalg.norm(obj.lin_vel_w)) if obj.lin_vel_w is not None else float("nan")
        ang = float(torch.linalg.norm(obj.ang_vel_w)) if obj.ang_vel_w is not None else float("nan")
        # dropped: object fell well below its intended rest height
        rest_z = float(plan.task_target_pose.pos_w[2])
        dropped = float(obj.pose.pos_w[2]) < (rest_z - self.cfg.dropped_z_margin)
        xy_err = float(torch.linalg.norm(obj.pose.pos_w[:2] - plan.task_target_pose.pos_w[:2]))
        oob = xy_err > self.cfg.out_of_bounds_radius
        return {
            "object_position_error": pos_err,
            "object_orientation_error": ori_err,
            "object_final_linear_speed": lin,
            "object_final_angular_speed": ang,
            "object_dropped": bool(dropped),
            "object_out_of_bounds": bool(oob),
        }

    def _verify(self, state: SceneState):
        m = self._object_errors(state)
        if m is None:
            self._fail(state, FailureReason.TARGET_LOST, "object not found at verify")
            return
        c = self.cfg
        speeds_ok = (math.isfinite(m["object_final_linear_speed"])
                     and m["object_final_linear_speed"] <= c.settle_linear_speed_threshold
                     and m["object_final_angular_speed"] <= c.settle_angular_speed_threshold)
        pos_ok = m["object_position_error"] <= c.object_position_tolerance
        ori_ok = m["object_orientation_error"] <= c.object_orientation_tolerance
        settled = self.runtime.settling_time is not None
        if m["object_dropped"]:
            self._fail(state, FailureReason.OBJECT_DROPPED, f"object dropped (z below rest by >{c.dropped_z_margin})")
        elif m["object_out_of_bounds"]:
            self._fail(state, FailureReason.OBJECT_OUT_OF_BOUNDS, "object outside placement bounds")
        elif not (pos_ok and ori_ok):
            self._fail(state, FailureReason.PLACE_VERIFICATION_FAILED,
                       f"pos_err={m['object_position_error']:.4f} ori_err_deg={math.degrees(m['object_orientation_error']):.2f}")
        elif not (speeds_ok and settled):
            self._fail(state, FailureReason.PLACE_NOT_SETTLED,
                       f"lin={m['object_final_linear_speed']:.3f} ang={m['object_final_angular_speed']:.3f} settled={settled}")
        else:
            self._succeed(state)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return SkillCommand(state.robot.tcp_pose, -1.0, self.status)

    # ---- result / outcomes ---------------------------------------------------
    def outcomes(self, state: SceneState) -> dict[str, Any]:
        m = self._object_errors(state) or {}
        mean_track = (self.runtime.track_err_sum / self.runtime.track_err_n) if self.runtime.track_err_n else None
        return {
            "elapsed_time": max(0.0, state.sim_time - self.runtime.start_time),
            "object_position_error": m.get("object_position_error"),
            "object_orientation_error": m.get("object_orientation_error"),
            "settling_time": self.runtime.settling_time,
            "maximum_tcp_tracking_error": self.runtime.track_err_max if self.runtime.track_err_n else None,
            "mean_tcp_tracking_error": mean_track,
            "object_final_linear_speed": m.get("object_final_linear_speed"),
            "object_final_angular_speed": m.get("object_final_angular_speed"),
            "object_dropped": m.get("object_dropped"),
            "object_out_of_bounds": m.get("object_out_of_bounds"),
            "tcp_final_position_error": self.runtime.final_error_pos,
            "tcp_final_orientation_error": self.runtime.final_error_ori,
            "legacy_reached": self.runtime.legacy_reached,
            "success": self.status == ExecutionStatus.SUCCEEDED,
            "failure_reason": self.failure_reason.value or None,
            "contact_available": False,
            "maximum_contact_force": None,
            "collision_available": False,
            "collision_count": None,
        }

    def result(self, state: SceneState) -> SkillResult:
        obj_pose = None
        if self.request.source_object in state.objects:
            obj_pose = state.objects[self.request.source_object or ""].pose
        plan = self.runtime.plan
        task_target = {}
        if plan is not None:
            task_target = {
                "target_surface_xyz": plan.target_surface_xyz,
                "task_target_object_pose": [round(float(v), 6) for v in plan.task_target_pose.as_pose_tensor().tolist()],
            }
        return SkillResult(
            request_id=self.request.request_id,
            skill_type=self.request.skill_type,
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
            outcomes=self.outcomes(state),
            legacy_reached=self.runtime.legacy_reached,
            task_target=task_target,
            requested_parameters=self.resolver.requested_subset(),
            effective_parameters=self.resolver.trace_dicts(),
        )

    # ---- plan ----------------------------------------------------------------
    def _make_plan(self, state: SceneState) -> PlacePlan | None:
        held_name = self.request.source_object
        if not held_name:
            self._fail(state, FailureReason.REQUEST_INVALID, "place request missing source_object")
            return None
        if self.held_object is None or getattr(self.held_object, "object_name", None) != held_name:
            self._fail(state, FailureReason.REQUEST_INVALID, "place request missing held object context")
            return None
        held = state.objects.get(held_name)
        if held is None:
            self._fail(state, FailureReason.TARGET_LOST, f"held object not found: {held_name}")
            return None
        if held_name not in OBJECT_SUPPORT_OFFSET_Z:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unsupported place object: {held_name}")
            return None

        surface_xyz = self._parse_surface_xyz(state)
        if surface_xyz is None:
            return None

        device = state.env_origin_w.device
        support_offset_z = OBJECT_SUPPORT_OFFSET_Z[held_name]
        # task target (evaluation): object底面 just above surface, NO release_clearance
        task_local = torch.tensor(
            [surface_xyz[0], surface_xyz[1], surface_xyz[2] + support_offset_z + BASE_CLEARANCE],
            dtype=torch.float32, device=device,
        )
        task_target_pos_w = state.env_origin_w + task_local
        # release target (execution): add release_clearance in +Z
        release_pos_w = task_target_pos_w + torch.tensor([0.0, 0.0, self._eff["release_clearance"]], dtype=torch.float32, device=device)
        target_quat_w = math_utils.normalize(held.pose.quat_w.unsqueeze(0))[0]
        pre_object_pos_w = release_pos_w + torch.tensor([0.0, 0.0, self._eff["pre_place_height"]], dtype=torch.float32, device=device)

        o2tcp_pos = self.held_object.object_to_tcp_pos.to(device=device, dtype=torch.float32).unsqueeze(0)
        o2tcp_quat = self.held_object.object_to_tcp_quat.to(device=device, dtype=torch.float32).unsqueeze(0)
        release_tcp_pos, release_tcp_quat = math_utils.combine_frame_transforms(
            release_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0), o2tcp_pos, o2tcp_quat)
        pre_tcp_pos, pre_tcp_quat = math_utils.combine_frame_transforms(
            pre_object_pos_w.unsqueeze(0), target_quat_w.unsqueeze(0), o2tcp_pos, o2tcp_quat)

        task_target_pose = PoseState(task_target_pos_w, target_quat_w)
        release_object_pose = PoseState(release_pos_w, target_quat_w)
        pre_object_pose = PoseState(pre_object_pos_w, target_quat_w)
        release_tcp_pose = PoseState(release_tcp_pos[0], math_utils.normalize(release_tcp_quat)[0])
        pre_place_tcp_pose = PoseState(pre_tcp_pos[0], math_utils.normalize(pre_tcp_quat)[0])
        if not all(finite_pose(p) for p in (task_target_pose, release_object_pose, pre_object_pose, release_tcp_pose, pre_place_tcp_pose)):
            self._fail(state, FailureReason.REQUEST_INVALID, "computed place target contains non-finite values")
            return None
        return PlacePlan(
            point_name=self.request.destination_object or "point",
            held_object_name=held_name,
            target_surface_xyz=surface_xyz,
            support_offset_z=support_offset_z,
            task_target_pose=task_target_pose,
            release_object_pose=release_object_pose,
            pre_object_pose=pre_object_pose,
            release_tcp_pose=release_tcp_pose,
            pre_place_tcp_pose=pre_place_tcp_pose,
        )

    def _parse_surface_xyz(self, state: SceneState) -> list[float] | None:
        value = self.request.parameters.get("target_surface_xyz")
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz must contain exactly three values")
            return None
        try:
            xyz = [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz values must be numeric")
            return None
        if not all(math.isfinite(v) for v in xyz):
            self._fail(state, FailureReason.REQUEST_INVALID, "target_surface_xyz values must be finite")
            return None
        return xyz

    # ---- motion helpers ------------------------------------------------------
    def _bounded_command(self, state, desired, max_position_step, max_rotation_step) -> PoseState:
        command_from = self.runtime.last_command_pose or state.robot.tcp_pose
        return step_pose(command_from, desired, max_position_step, max_rotation_step)

    def _reached(self, state, desired) -> bool:
        error = pose_error(state.robot.tcp_pose, desired)
        return error.position <= self._eff["position_threshold"] and error.orientation <= self._eff["orientation_threshold"]

    def _advance_when_reached(self, state, desired, next_state) -> bool:
        if self._reached(state, desired):
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self._eff["stable_cycles"]:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self._eff["state_timeout"]:
            error = pose_error(state.robot.tcp_pose, desired)
            self._fail(state, FailureReason.POSITION_TIMEOUT,
                       f"place pose error p={error.position:.4f}, ori_deg={math.degrees(error.orientation):.2f}")
        return False

    def _state_elapsed(self, state) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state, new_state):
        old_state = self.runtime.state
        if old_state == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record_transition(state, old_state, new_state)

    def _update_telemetry(self, state, desired: PoseState, gripper: float):
        err = pose_error(state.robot.tcp_pose, desired)
        self.last_telemetry = {
            "skill_state": self.runtime.state,
            "target_tcp_position": [float(v) for v in desired.pos_w.tolist()],
            "target_tcp_orientation": [float(v) for v in desired.quat_w.tolist()],
            "tcp_position_error": err.position,
            "tcp_orientation_error": err.orientation,
            "gripper_command": float(gripper),
        }

    def _record_transition(self, state, old_state, new_state):
        plan = self.runtime.plan
        command_pose = self.runtime.last_command_pose or state.robot.tcp_pose
        error = pose_error(state.robot.tcp_pose, command_pose)
        record = {
            "time": round(state.sim_time, 4),
            "request_id": self.request.request_id,
            "skill": SkillType.PLACE.value,
            "target": self.request.source_object,
            "from": old_state,
            "to": new_state,
            "task_target_object_pose": None if plan is None else format_pose(plan.task_target_pose),
            "tcp_pose": format_pose(state.robot.tcp_pose),
            "position_error": round(error.position, 5),
            "orientation_error_deg": round(math.degrees(error.orientation), 3),
            "gripper_width": round(state.robot.gripper_width, 5),
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(record)
        print(f"[PlaceSkill] transition {record}", flush=True)

    def _fail(self, state, reason, message):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")
        print(f"[PlaceSkill] failure request={self.request.request_id} reason={reason.value} {message}", flush=True)

    def _succeed(self, state):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(f"[PlaceSkill] success request={self.request.request_id} target={self.request.source_object}", flush=True)

    def _log_plan(self, plan: PlacePlan):
        record = {
            "point_name": plan.point_name,
            "target_surface_xyz": plan.target_surface_xyz,
            "support_offset_z": plan.support_offset_z,
            "base_clearance": BASE_CLEARANCE,
            "release_clearance": self._eff["release_clearance"],
            "task_target_object_pos_w": self._tensor_list(plan.task_target_pose.pos_w),
            "release_tcp_pos_w": self._tensor_list(plan.release_tcp_pose.pos_w),
        }
        print(f"[PlaceSkill] target {record}", flush=True)

    def _tensor_list(self, tensor: torch.Tensor) -> list[float]:
        return [round(float(v), 5) for v in tensor.detach().cpu().tolist()]

    def _terminal_gripper_command(self) -> float:
        return 1.0 if self.status == ExecutionStatus.SUCCEEDED else 1.0
