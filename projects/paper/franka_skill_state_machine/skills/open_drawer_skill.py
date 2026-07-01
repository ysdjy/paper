"""General IK-based OPEN-DRAWER skill (stage-1 parameterised; no learned policy, no joint cheating).

Preserved from the original design (task spec section 5.1):
* the handle pose is read LIVE every step (link body pose o link-local handle offset);
* the open direction is derived from the cabinet orientation;
* the drawer is opened ONLY by physically gripping + pulling the handle;
* ``drawer_joint_target`` is always None and ``set_cabinet_joint_target`` is never called;
* top / middle drawers supported (bottom is locked, excluded from experiments).

Stage-1 additions:
* execution parameters come from ``request.parameters`` via :class:`ParamResolver`
  (pre_grasp_clearance / pull_lead / max_pos_step / max_ori_step / approach_line_lead /
  grasp_offset_local_xyz / close_duration / settle_duration / release_duration / reach_timeout /
  pull_timeout), all traced and really entering the geometry;
* ``target_open_position`` is a TASK TARGET (request > DRAWER_TARGETS default 0.20, clamped to the
  joint limit); success = ``actual >= target - tolerance``; ``drawer_overshoot`` is recorded because
  the free-sliding drawer can coast past the target;
* ``grasp_offset_local_xyz`` is added in the drawer LINK local frame (NOT a world offset) before the
  link transform, so it produces accurate / edge / slipped / missed grasps;
* a handle-detachment detector (no contact sensor): sustained large TCP-vs-handle error together with
  no drawer progress while in PULL;
* per-step telemetry + running TCP / handle tracking stats + a full ``outcomes`` dict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

import isaaclab.utils.math as math_utils

from runtime.base_skill import SkillCommand, PoseState, pose_error, pose_tensor, step_pose
from runtime.drawer_ik_common import HOME_Q_VERTICAL_RAISED, grasp_quat_from_open_dir, open_direction_world
from runtime.drawer_obs_adapter import SelectedDrawerObsAdapter
from runtime.drawer_target_config import DRAWER_TARGETS
from runtime.experiment_params import InvalidExperimentParameter, ParamResolver
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason, SkillType

# prismatic drawer joint physical range (custom_drawer_config.py): limits [0, 0.8]
DRAWER_JOINT_MIN = 0.0
DRAWER_JOINT_MAX = 0.8


@dataclass
class OpenDrawerIKConfig:
    start_from_current: bool = True
    use_turn_to_face: bool = True
    face_joint_threshold: float = 0.12
    face_timeout: float = 5.0
    pre_grasp_clearance: float = 0.12
    pull_lead: float = 0.08
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.03
    reach_ori_threshold: float = math.radians(18.0)
    reach_stable_cycles: int = 6
    soft_reach_advance: float = 0.10
    soft_reach_ori: float = math.radians(40.0)
    approach_line_lead: float = 0.03
    close_duration: float = 1.0
    reach_timeout: float = 16.0
    pull_timeout: float = 16.0
    settle_duration: float = 0.5
    release_duration: float = 0.4
    home_joint_threshold: float = 0.12
    home_timeout: float = 6.0
    # --- task target + verification (stage-1) ---
    target_open_position: float = 0.20        # default == old success_threshold
    target_tolerance: float = 0.02
    # --- handle-detachment detector thresholds (calibrate from default-success runs) ---
    handle_detach_error: float = 0.06         # TCP-vs-live-handle distance (m) considered "lost grip"
    handle_detach_no_progress_window: float = 1.5   # s of no drawer progress
    handle_detach_progress_eps: float = 0.003       # m drawer advance counted as progress


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = "ik_pull"
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None  # always None (physical pull only)
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    handle_detached: bool = False
    target_reached_time: float | None = None
    timed_out: bool = False
    # running stats
    track_err_max: float = 0.0
    track_err_sum: float = 0.0
    track_err_n: int = 0
    handle_err_max: float = 0.0
    handle_err_sum: float = 0.0
    handle_err_n: int = 0
    history: list[dict] = field(default_factory=list)


class OpenDrawerIKSkill:
    backend = "ik_pull"

    def __init__(self, request: SkillRequest, env, ik_adapter, config: OpenDrawerIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or OpenDrawerIKConfig()
        self.target_drawer = request.destination_object or "top_drawer"
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        cfg = DRAWER_TARGETS.get(self.target_drawer, {})
        self.runtime = _Runtime(drawer_joint_name=cfg.get("joint_name", ""))
        self.obs_adapter: SelectedDrawerObsAdapter | None = None
        self.last_q = None
        self._approach_start = None
        self._approach_end = None
        self._approach_quat = None
        self.resolver = ParamResolver(request.parameters)
        self.last_telemetry: dict = {}
        self._eff: dict = {}
        self._param_error: str | None = None
        self._grasp_offset_local = torch.zeros(3)
        self._last_progress_time = 0.0
        self._progress_ref = 0.0
        self.target_open_position = float(cfg.get("success_threshold", 0.20))
        self.initial_handle_pos = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    # ---- param resolution -------------------------------------------------
    def _resolve_params(self) -> bool:
        c = self.cfg
        at = "open_drawer_skill"
        try:
            r = self.resolver
            e = {}
            e["pre_grasp_clearance"], _ = r.float("pre_grasp_clearance", c.pre_grasp_clearance,
                                                  minimum=0.02, maximum=0.30, hard_minimum=0.0, hard_maximum=0.6, applied_at=at)
            e["pull_lead"], _ = r.float("pull_lead", c.pull_lead,
                                        minimum=0.02, maximum=0.20, hard_minimum=0.0, hard_maximum=0.5, applied_at=at)
            e["max_pos_step"], _ = r.float("max_pos_step", c.max_pos_step,
                                           minimum=0.005, maximum=0.05, hard_minimum=0.001, hard_maximum=0.2, applied_at=at)
            e["max_ori_step"], _ = r.float("max_ori_step", c.max_ori_step,
                                           minimum=math.radians(1.0), maximum=math.radians(15.0), hard_maximum=math.radians(45.0), applied_at=at)
            e["approach_line_lead"], _ = r.float("approach_line_lead", c.approach_line_lead,
                                                 minimum=0.01, maximum=0.10, hard_maximum=0.3, applied_at=at)
            e["close_duration"], _ = r.float("close_duration", c.close_duration,
                                             minimum=0.3, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["settle_duration"], _ = r.float("settle_duration", c.settle_duration,
                                              minimum=0.1, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["release_duration"], _ = r.float("release_duration", c.release_duration,
                                               minimum=0.1, maximum=3.0, hard_minimum=0.0, hard_maximum=10.0, applied_at=at)
            e["reach_timeout"], _ = r.float("reach_timeout", c.reach_timeout, minimum=4.0, maximum=40.0, applied_at=at)
            e["pull_timeout"], _ = r.float("pull_timeout", c.pull_timeout, minimum=4.0, maximum=40.0, applied_at=at)
            off, _ = r.vec3("grasp_offset_local_xyz", [0.0, 0.0, 0.0],
                            minimum=-0.06, maximum=0.06, hard_minimum=-0.2, hard_maximum=0.2, applied_at=at)
            self._grasp_offset_local = torch.tensor(off, dtype=torch.float32, device=self.adapter.device)
            # task target: target_open_position (request > config default), clamped to joint limit
            tgt, _ = r.float("target_open_position", c.target_open_position,
                             minimum=DRAWER_JOINT_MIN + 0.02, maximum=DRAWER_JOINT_MAX,
                             hard_minimum=DRAWER_JOINT_MIN, hard_maximum=DRAWER_JOINT_MAX, applied_at=at + ".task_target")
            self.target_open_position = float(tgt)
            e["target_open_position"] = self.target_open_position
            self._eff = e
            return True
        except InvalidExperimentParameter as exc:
            self._param_error = str(exc)
            return False

    # ---- live scene reads -------------------------------------------------
    def _handle_pos(self) -> torch.Tensor:
        """Live world position of the actual handle bar (link o handle_offset, NO grasp_offset)."""
        return self.obs_adapter.selected_handle_pos_w()[self.adapter.env_id]

    def _cabinet_quat(self) -> torch.Tensor:
        return self.env.unwrapped.scene["cabinet"].data.root_quat_w[self.adapter.env_id]

    def _drawer_pos(self) -> float:
        return self.obs_adapter.selected_drawer_joint_pos()

    def _faced_seed_q(self) -> torch.Tensor:
        robot = self.env.unwrapped.scene["robot"]
        eid = self.adapter.env_id
        base_pos = robot.data.root_pos_w[eid]
        base_quat = robot.data.root_quat_w[eid]
        d = (self._handle_pos() - base_pos)
        d_base = math_utils.quat_apply(math_utils.quat_inv(base_quat).unsqueeze(0), d.unsqueeze(0))[0]
        azimuth = math.atan2(float(d_base[1]), float(d_base[0]))
        seed = self.home_q.clone()
        lo, hi = float(self.adapter._joint_lower[0]), float(self.adapter._joint_upper[0])
        seed[0] = max(lo, min(hi, azimuth))
        print(f"[OpenDrawerIKSkill] turn-to-face: handle azimuth(base) = {math.degrees(azimuth):.1f}deg "
              f"-> joint1 = {math.degrees(float(seed[0])):.1f}deg", flush=True)
        return seed

    def _grasp_pose(self, lead: float) -> PoseState:
        """Live grasp/pull target: (handle link-local point + grasp_offset_local) -> world, + lead*open_dir.

        ``grasp_offset_local_xyz`` is added in the drawer LINK local frame (combined with the base
        handle offset BEFORE the link transform), so it tracks the sliding drawer and yields a real
        edge/slip grasp -- never a fixed world offset (task spec section 5.4).
        """
        open_dir = open_direction_world(self._cabinet_quat())
        eid = self.adapter.env_id
        cab = self.obs_adapter.cabinet
        li = self.obs_adapter._link_idx
        link_pos = cab.data.body_pos_w[eid, li]
        link_quat = cab.data.body_quat_w[eid, li]
        ov = (self.request.parameters or {}).get("override_grasp_local")
        if ov is not None:
            base_local = torch.tensor(ov["pos"], dtype=torch.float32, device=link_pos.device)
            lq = torch.tensor(ov["quat"], dtype=torch.float32, device=link_pos.device)
            total_local = base_local + self._grasp_offset_local
            gpos, gquat = math_utils.combine_frame_transforms(
                link_pos.unsqueeze(0), link_quat.unsqueeze(0), total_local.unsqueeze(0), lq.unsqueeze(0))
            return PoseState(gpos[0] + open_dir * lead, gquat[0])
        total_local = self.obs_adapter.handle_offset + self._grasp_offset_local
        gpos, _ = math_utils.combine_frame_transforms(
            link_pos.unsqueeze(0), link_quat.unsqueeze(0), total_local.unsqueeze(0))
        quat = grasp_quat_from_open_dir(open_dir, link_pos.device)
        return PoseState(gpos[0] + open_dir * lead, quat)

    # ---- skill API --------------------------------------------------------
    def start(self, state: SceneState):
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if cfg is None:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer, env_id=self.adapter.env_id)
        if not self._resolve_params():
            self._fail(state, FailureReason.INVALID_EXPERIMENT_PARAMETER, self._param_error or "bad parameter")
            return
        robot = self.env.unwrapped.scene["robot"]
        self.home_q = torch.tensor(HOME_Q_VERTICAL_RAISED, dtype=torch.float32,
                                   device=robot.data.default_joint_pos.device)
        self.faced_q = self._faced_seed_q()
        if self.cfg.use_turn_to_face:
            initial_state = "TURN_TO_FACE"
        elif self.cfg.start_from_current:
            initial_state = "MOVE_TO_PRE_GRASP"
        else:
            initial_state = "MOVE_TO_HOME"
        self.runtime = _Runtime(
            state=initial_state,
            start_time=state.sim_time,
            state_start_time=state.sim_time,
            drawer_joint_name=cfg["joint_name"],
            last_command_pose=state.robot.tcp_pose,
        )
        self.runtime.initial_joint_pos = self._drawer_pos()
        self.runtime.current_joint_pos = self.runtime.initial_joint_pos
        self.initial_handle_pos = self._handle_pos().clone()
        self._last_progress_time = state.sim_time
        self._progress_ref = self.runtime.initial_joint_pos
        print(f"[OpenDrawerIKSkill] target_open_position={self.target_open_position:.3f} "
              f"initial_drawer={self.runtime.initial_joint_pos:.4f} "
              f"grasp_offset_local={[round(float(v),4) for v in self._grasp_offset_local.tolist()]}", flush=True)
        self._record(state, "IDLE", initial_state)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)

        self.runtime.current_joint_pos = self._drawer_pos()
        gripper = 1.0
        target = None

        if self.runtime.state == "MOVE_TO_HOME":
            return self._home_step(state)
        if self.runtime.state == "TURN_TO_FACE":
            arm_q = state.robot.joint_pos[self.adapter._joint_ids]
            dq1 = abs(float(arm_q[0] - self.faced_q[0]))
            if dq1 <= self.cfg.face_joint_threshold:
                self.runtime.stable_count += 1
                if self.runtime.stable_count >= 3:
                    self._transition(state, "MOVE_TO_PRE_GRASP")
            else:
                self.runtime.stable_count = 0
            if self._state_elapsed(state) > self.cfg.face_timeout:
                self._transition(state, "MOVE_TO_PRE_GRASP")
            self.last_q = self.faced_q.clone()
            self._update_telemetry(state, state.robot.tcp_pose, 1.0)
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=self.faced_q.clone(), drawer_joint_target=None,
            )
        if self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self._eff["pre_grasp_clearance"])
            gripper = 1.0
            self._advance_when_reached(state, target, "APPROACH", self._eff["reach_timeout"])
        elif self.runtime.state == "APPROACH":
            gp = self._grasp_pose(0.0)
            if self._approach_end is None:
                self._approach_start = state.robot.tcp_pose.pos_w.clone()
                self._approach_end = gp.pos_w.clone()
                self._approach_quat = gp.quat_w.clone()
            carrot = self._line_setpoint(state.robot.tcp_pose.pos_w, self._approach_start, self._approach_end)
            target = PoseState(carrot, self._approach_quat)
            gripper = 1.0
            self._advance_when_reached(
                state, PoseState(self._approach_end, self._approach_quat), "CLOSE_GRIPPER", self._eff["reach_timeout"])
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self._eff["close_duration"]:
                self._last_progress_time = state.sim_time
                self._progress_ref = self.runtime.current_joint_pos
                self._transition(state, "PULL")
        elif self.runtime.state == "PULL":
            target = self._grasp_pose(self._eff["pull_lead"])
            gripper = -1.0
            self._track_pull(state)
            if self.runtime.current_joint_pos >= self.target_open_position - self.cfg.target_tolerance:
                if self.runtime.target_reached_time is None:
                    self.runtime.target_reached_time = max(0.0, state.sim_time - self.runtime.start_time)
                self._transition(state, "SETTLE")
            elif self.runtime.handle_detached:
                self._fail(state, FailureReason.HANDLE_DETACHED,
                           f"handle lost during pull at drawer={self.runtime.current_joint_pos:.4f}")
            elif self._state_elapsed(state) > self._eff["pull_timeout"]:
                self.runtime.timed_out = True
                self._fail(state, FailureReason.DRAWER_OPEN_TIMEOUT,
                           f"pull did not reach target {self.target_open_position:.3f}: "
                           f"pos={self.runtime.current_joint_pos:.4f}")
        elif self.runtime.state == "SETTLE":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self._eff["settle_duration"]:
                self._transition(state, "RELEASE")
        elif self.runtime.state == "RELEASE":
            target = self._grasp_pose(0.0)
            gripper = 1.0
            if self._state_elapsed(state) >= self._eff["release_duration"]:
                self._succeed(state)

        self._last_phase_target = target
        cmd = self._command(state, target, gripper)
        self._update_telemetry(state, target if target is not None else state.robot.tcp_pose, gripper)
        return cmd

    # ---- pull tracking / detachment --------------------------------------
    def _track_pull(self, state: SceneState):
        handle_w = self._handle_pos()
        tcp = state.robot.tcp_pose.pos_w
        handle_err = float(torch.linalg.norm(tcp - handle_w))
        self.runtime.handle_err_max = max(self.runtime.handle_err_max, handle_err)
        self.runtime.handle_err_sum += handle_err
        self.runtime.handle_err_n += 1
        # drawer progress watchdog
        if self.runtime.current_joint_pos >= self._progress_ref + self.cfg.handle_detach_progress_eps:
            self._progress_ref = self.runtime.current_joint_pos
            self._last_progress_time = state.sim_time
        no_progress = (state.sim_time - self._last_progress_time) > self.cfg.handle_detach_no_progress_window
        not_reached = self.runtime.current_joint_pos < (self.target_open_position - self.cfg.target_tolerance)
        if handle_err > self.cfg.handle_detach_error and no_progress and not_reached:
            self.runtime.handle_detached = True

    def viz_poses(self) -> list:
        out = []
        tgt = getattr(self, "_last_phase_target", None)
        if tgt is not None:
            out.append(("phase_target", tgt))
        try:
            handle = self._handle_pos()
            quat = grasp_quat_from_open_dir(open_direction_world(self._cabinet_quat()), handle.device)
            out.append(("drawer_handle", PoseState(handle, quat)))
        except Exception:
            pass
        return out

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.failure_reason = FailureReason.CANCELLED_BY_USER
        self._transition(state, "CANCELLED")
        return self._hold(state)

    # ---- outcomes / result -----------------------------------------------
    def outcomes(self, state: SceneState) -> dict:
        final = self._drawer_pos() if self.obs_adapter is not None else self.runtime.current_joint_pos
        mean_track = (self.runtime.track_err_sum / self.runtime.track_err_n) if self.runtime.track_err_n else None
        mean_handle = (self.runtime.handle_err_sum / self.runtime.handle_err_n) if self.runtime.handle_err_n else None
        return {
            "elapsed_time": max(0.0, state.sim_time - self.runtime.start_time),
            "initial_drawer_position": self.runtime.initial_joint_pos,
            "target_drawer_position": self.target_open_position,
            "final_drawer_position": final,
            "drawer_position_error": abs(final - self.target_open_position),
            "drawer_overshoot": final - self.target_open_position,
            "maximum_tcp_tracking_error": self.runtime.track_err_max if self.runtime.track_err_n else None,
            "mean_tcp_tracking_error": mean_track,
            "maximum_handle_relative_error": self.runtime.handle_err_max if self.runtime.handle_err_n else None,
            "mean_handle_relative_error": mean_handle,
            "handle_detached": self.runtime.handle_detached,
            "target_reached_time": self.runtime.target_reached_time,
            "timeout": self.runtime.timed_out,
            "success": self.status == ExecutionStatus.SUCCEEDED,
            "failure_reason": self.failure_reason.value or None,
            "contact_available": False,
            "maximum_contact_force": None,
            "collision_available": False,
            "collision_count": None,
        }

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
            outcomes=self.outcomes(state),
            task_target={"target_open_position": self.target_open_position, "drawer_name": self.target_drawer},
            requested_parameters=self.resolver.requested_subset(),
            effective_parameters=self.resolver.trace_dicts(),
        )

    # ---- helpers ----------------------------------------------------------
    def _line_setpoint(self, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d = b - a
        L = float(torch.linalg.norm(d))
        if L < 1e-6:
            return b.clone()
        u = d / L
        t = float(torch.clamp(torch.dot(c - a, u), 0.0, L))
        s = min(t + self._eff["approach_line_lead"], L)
        return a + s * u

    def _home_step(self, state: SceneState) -> SkillCommand:
        arm_q = state.robot.joint_pos[self.adapter._joint_ids]
        dist = float(torch.linalg.norm(arm_q - self.home_q))
        if dist <= self.cfg.home_joint_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= 3:
                self._transition(state, "MOVE_TO_PRE_GRASP")
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > self.cfg.home_timeout:
            self._transition(state, "MOVE_TO_PRE_GRASP")
        self.last_q = self.home_q.clone()
        return SkillCommand(
            state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
            joint_target=self.home_q.clone(), drawer_joint_target=None,
        )

    def _command(self, state: SceneState, target: PoseState | None, gripper: float) -> SkillCommand:
        if target is None:
            return self._hold(state, gripper)
        cmd = step_pose(state.robot.tcp_pose, target, self._eff["max_pos_step"], self._eff["max_ori_step"])
        ik = self.adapter.solve(cmd)
        err = pose_error(state.robot.tcp_pose, target)
        self.runtime.final_error_pos = err.position
        self.runtime.final_error_ori = err.orientation
        self.runtime.last_command_pose = cmd
        # TCP tracking-error stats over the reaching/pulling phases
        if self.runtime.state in ("MOVE_TO_PRE_GRASP", "APPROACH", "PULL"):
            self.runtime.track_err_max = max(self.runtime.track_err_max, err.position)
            self.runtime.track_err_sum += err.position
            self.runtime.track_err_n += 1
        if not ik.success:
            q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                                drawer_joint_target=None)
        self.last_q = ik.q_des
        return SkillCommand(cmd, gripper, self.status, control_mode="joint", joint_target=ik.q_des, drawer_joint_target=None)

    def _hold(self, state: SceneState, gripper: float = -1.0) -> SkillCommand:
        q = self.last_q if self.last_q is not None else state.robot.joint_pos[self.adapter._joint_ids].clone()
        return SkillCommand(state.robot.tcp_pose, gripper, self.status, control_mode="joint", joint_target=q,
                            drawer_joint_target=None)

    def _advance_when_reached(self, state: SceneState, target: PoseState, next_state: str, timeout: float) -> bool:
        err = pose_error(state.robot.tcp_pose, target)
        if err.position <= self.cfg.reach_pos_threshold and err.orientation <= self.cfg.reach_ori_threshold:
            self.runtime.stable_count += 1
            if self.runtime.stable_count >= self.cfg.reach_stable_cycles:
                self._transition(state, next_state)
                return True
        else:
            self.runtime.stable_count = 0
        if self._state_elapsed(state) > timeout:
            if err.position <= self.cfg.soft_reach_advance and err.orientation <= self.cfg.soft_reach_ori:
                print(f"[OpenDrawerIKSkill] soft-advance {self.runtime.state}->{next_state} "
                      f"(pos_err={err.position:.3f} ori_err={math.degrees(err.orientation):.1f}deg)", flush=True)
                self._transition(state, next_state)
                return True
            self._fail(state, FailureReason.POSITION_TIMEOUT,
                       f"{self.runtime.state} reach timeout (pos={err.position:.3f} "
                       f"ori={math.degrees(err.orientation):.1f}deg)")
        return False

    def _state_elapsed(self, state: SceneState) -> float:
        return max(0.0, state.sim_time - self.runtime.state_start_time)

    def _transition(self, state: SceneState, new_state: str):
        old = self.runtime.state
        if old == new_state:
            return
        self.runtime.state = new_state
        self.runtime.state_start_time = state.sim_time
        self.runtime.stable_count = 0
        self._record(state, old, new_state)

    def _update_telemetry(self, state: SceneState, target: PoseState, gripper: float):
        err = pose_error(state.robot.tcp_pose, target)
        handle_w = None
        handle_rel = float("nan")
        try:
            handle_w = self._handle_pos()
            handle_rel = float(torch.linalg.norm(state.robot.tcp_pose.pos_w - handle_w))
        except Exception:
            pass
        self.last_telemetry = {
            "skill_state": self.runtime.state,
            "target_tcp_position": [float(v) for v in target.pos_w.tolist()],
            "target_tcp_orientation": [float(v) for v in target.quat_w.tolist()],
            "tcp_position_error": err.position,
            "tcp_orientation_error": err.orientation,
            "gripper_command": float(gripper),
            "drawer_joint_position": float(self.runtime.current_joint_pos),
            "handle_position": None if handle_w is None else [float(v) for v in handle_w.tolist()],
            "target_handle_grasp_position": [float(v) for v in target.pos_w.tolist()],
            "handle_relative_position_error": handle_rel,
            "drawer_progress": float(self.runtime.current_joint_pos - self.runtime.initial_joint_pos),
        }

    def _record(self, state: SceneState, old: str, new: str):
        rec = {
            "time": round(state.sim_time, 4),
            "skill": self.request.skill_type.value,
            "backend": self.backend,
            "target_drawer": self.target_drawer,
            "drawer_joint_name": self.runtime.drawer_joint_name,
            "from": old,
            "to": new,
            "drawer_joint_pos": round(self.runtime.current_joint_pos, 5),
            "target_open_position": round(self.target_open_position, 5),
            "drawer_joint_target": None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[OpenDrawerIKSkill] {rec}", flush=True)

    def _fail(self, state: SceneState, reason: FailureReason, message: str):
        if self.status == ExecutionStatus.FAILED:
            return
        self.status = ExecutionStatus.FAILED
        self.failure_reason = reason
        self.runtime.last_failure_message = message
        self._transition(state, "FAILED")

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.failure_reason = FailureReason.NONE
        self._transition(state, "SUCCEEDED")
        print(f"[OpenDrawerIKSkill] success target={self.target_drawer} drawer_pos={self.runtime.current_joint_pos:.4f}", flush=True)
