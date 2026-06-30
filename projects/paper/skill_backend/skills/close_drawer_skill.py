"""General IK-based CLOSE-DRAWER skill (no learned policy, no joint-target cheating).

Mirror of OpenDrawerIKSkill: reach the (currently open) drawer's handle, grip it, and push along the
drawer's CLOSING direction (toward the cabinet body) until the drawer joint returns near closed.
Handle pose + orientation are read live every step; geometry is derived from the live handle-vs-cabinet
vector, so it works for any drawer / cabinet. ``drawer_joint_target`` stays None;
``set_cabinet_joint_target`` is never called.
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
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


@dataclass
class CloseDrawerIKConfig:
    start_from_current: bool = True   # True: begin at MOVE_TO_PRE_GRASP from the CURRENT arm pose (no
    #                                   detour home). Important here: after open_drawer the gripper is
    #                                   already AT the handle, so going home first then re-reaching wastes
    #                                   motion and can miss; start where we are.
    use_turn_to_face: bool = True     # rotate joint 1 to FACE the drawer first (the local DLS won't turn
    #                                   the base -> leans back for a side/rear drawer); fixes posture+reach
    face_joint_threshold: float = 0.12
    face_timeout: float = 5.0
    pre_grasp_clearance: float = 0.0  # no OUTWARD back-off: would drag a gripped/free drawer further open
    push_lead: float = 0.12  # how far toward the cabinet to aim while pushing (seat it fully closed)
    max_pos_step: float = 0.020
    max_ori_step: float = math.radians(6.0)
    reach_pos_threshold: float = 0.03
    reach_ori_threshold: float = math.radians(18.0)  # also require wrist alignment before grasping
    reach_stable_cycles: int = 6
    soft_reach_advance: float = 0.10   # advance on reach timeout if within this; let physical push engage
    soft_reach_ori: float = math.radians(40.0)  # orientation tolerance for the timeout soft-advance
    approach_line_lead: float = 0.03   # APPROACH glides the TCP along a STRAIGHT line via a carrot
    close_duration: float = 1.0
    reach_timeout: float = 12.0
    push_timeout: float = 8.0
    close_success_threshold: float = 0.01  # drawer considered closed when joint <= this (less gap)
    home_joint_threshold: float = 0.12
    home_timeout: float = 6.0


@dataclass
class _Runtime:
    state: str = "IDLE"
    start_time: float = 0.0
    state_start_time: float = 0.0
    stable_count: int = 0
    drawer_control_mode: str = "ik_push"
    drawer_joint_name: str = ""
    drawer_joint_target: float | None = None
    initial_joint_pos: float = 0.0
    current_joint_pos: float = 0.0
    last_command_pose: PoseState | None = None
    final_error_pos: float | None = None
    final_error_ori: float | None = None
    last_failure_message: str | None = None
    history: list[dict] = field(default_factory=list)


class CloseDrawerIKSkill:
    backend = "ik_pull"

    def __init__(self, request: SkillRequest, env, ik_adapter, config: CloseDrawerIKConfig | None = None):
        self.request = request
        self.env = env
        self.adapter = ik_adapter
        self.cfg = config or CloseDrawerIKConfig()
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

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def _line_setpoint(self, c: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Carrot on the straight line a->b (project TCP, aim approach_line_lead further) -> straight glide."""
        d = b - a
        L = float(torch.linalg.norm(d))
        if L < 1e-6:
            return b.clone()
        u = d / L
        t = float(torch.clamp(torch.dot(c - a, u), 0.0, L))
        s = min(t + self.cfg.approach_line_lead, L)
        return a + s * u

    def _handle_pos(self) -> torch.Tensor:
        return self.obs_adapter.selected_handle_pos_w()[self.adapter.env_id]

    def _cabinet_quat(self) -> torch.Tensor:
        return self.env.unwrapped.scene["cabinet"].data.root_quat_w[self.adapter.env_id]

    def _drawer_pos(self) -> float:
        return self.obs_adapter.selected_drawer_joint_pos()

    def _grasp_pose(self, lead: float) -> PoseState:
        """Live target: handle + lead*open_dir. lead>0 = outward (pre-grasp), lead<0 = toward cabinet (push).

        open_dir comes from the cabinet orientation (same for every drawer). If the request carries
        ``parameters['override_grasp_local']`` (user's edited grasp pose in the drawer LINK frame),
        use it: world grasp = link_pose o local (tracks the drawer, matches the panel marker).
        """
        open_dir = open_direction_world(self._cabinet_quat())
        ov = (self.request.parameters or {}).get("override_grasp_local")
        if ov is not None:
            eid = self.adapter.env_id
            cab = self.obs_adapter.cabinet
            li = self.obs_adapter._link_idx
            link_pos = cab.data.body_pos_w[eid, li]
            link_quat = cab.data.body_quat_w[eid, li]
            lp = torch.tensor(ov["pos"], dtype=torch.float32, device=link_pos.device)
            lq = torch.tensor(ov["quat"], dtype=torch.float32, device=link_pos.device)
            gpos, gquat = math_utils.combine_frame_transforms(
                link_pos.unsqueeze(0), link_quat.unsqueeze(0), lp.unsqueeze(0), lq.unsqueeze(0))
            return PoseState(gpos[0] + open_dir * lead, gquat[0])
        handle = self._handle_pos()
        quat = grasp_quat_from_open_dir(open_dir, handle.device)
        return PoseState(handle + open_dir * lead, quat)

    def _faced_seed_q(self) -> torch.Tensor:
        """Home arm posture with joint 1 turned to FACE the handle azimuth in the robot base frame."""
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
        print(f"[CloseDrawerIKSkill] turn-to-face: joint1 -> {math.degrees(float(seed[0])):.1f}deg", flush=True)
        return seed

    def start(self, state: SceneState):
        cfg = DRAWER_TARGETS.get(self.target_drawer)
        self.status = ExecutionStatus.RUNNING
        self.failure_reason = FailureReason.NONE
        if cfg is None:
            self._fail(state, FailureReason.REQUEST_INVALID, f"unknown target_drawer '{self.target_drawer}'")
            return
        self.obs_adapter = SelectedDrawerObsAdapter(self.env, self.target_drawer, env_id=self.adapter.env_id)
        robot = self.env.unwrapped.scene["robot"]
        # turn-to-face 用【竖直 + 抬高 15cm】的 home 臂姿(防止转身拉抽屉时碰掉桌面物品)，不是默认 home。
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
        self._record(state, "IDLE", initial_state)

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
            return SkillCommand(
                state.robot.tcp_pose, 1.0, self.status, control_mode="joint",
                joint_target=self.faced_q.clone(), drawer_joint_target=None,
            )
        if self.runtime.state == "MOVE_TO_PRE_GRASP":
            target = self._grasp_pose(self.cfg.pre_grasp_clearance)
            gripper = 1.0
            self._advance_when_reached(state, target, "APPROACH", self.cfg.reach_timeout)
        elif self.runtime.state == "APPROACH":
            gp = self._grasp_pose(0.0)
            if self._approach_end is None:
                self._approach_start = state.robot.tcp_pose.pos_w.clone()
                self._approach_end = gp.pos_w.clone()
                self._approach_quat = gp.quat_w.clone()
            carrot = self._line_setpoint(state.robot.tcp_pose.pos_w, self._approach_start, self._approach_end)
            target = PoseState(carrot, self._approach_quat)   # straight-line glide onto the handle
            gripper = 1.0
            self._advance_when_reached(
                state, PoseState(self._approach_end, self._approach_quat), "CLOSE_GRIPPER", self.cfg.reach_timeout)
        elif self.runtime.state == "CLOSE_GRIPPER":
            target = self._grasp_pose(0.0)
            gripper = -1.0
            if self._state_elapsed(state) >= self.cfg.close_duration:
                self._transition(state, "PUSH")
        elif self.runtime.state == "PUSH":
            target = self._grasp_pose(-self.cfg.push_lead)  # aim toward the cabinet (closing)
            gripper = -1.0
            if self.runtime.current_joint_pos <= self.cfg.close_success_threshold:
                self._succeed(state)
            elif self._state_elapsed(state) > self.cfg.push_timeout:
                self._fail(
                    state,
                    FailureReason.DRAWER_CLOSE_TIMEOUT,
                    f"push did not close {self.target_drawer}: pos={self.runtime.current_joint_pos:.4f}",
                )

        self._last_phase_target = target
        return self._command(state, target, gripper)

    def viz_poses(self) -> list:
        """(name, PoseState) pairs for live target-pose arrows: current phase target + drawer handle."""
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

    def _command(self, state: SceneState, target: PoseState | None, gripper: float) -> SkillCommand:
        if target is None:
            return self._hold(state, gripper)
        cmd = step_pose(state.robot.tcp_pose, target, self.cfg.max_pos_step, self.cfg.max_ori_step)
        ik = self.adapter.solve(cmd)
        err = pose_error(state.robot.tcp_pose, target)
        self.runtime.final_error_pos = err.position
        self.runtime.final_error_ori = err.orientation
        self.runtime.last_command_pose = cmd
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
                print(f"[CloseDrawerIKSkill] soft-advance {self.runtime.state}->{next_state} "
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
            "drawer_joint_target": None,
            "failure_reason": self.failure_reason.value or None,
            "failure_message": self.runtime.last_failure_message,
        }
        self.runtime.history.append(rec)
        print(f"[CloseDrawerIKSkill] {rec}", flush=True)

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
        print(f"[CloseDrawerIKSkill] success target={self.target_drawer} drawer_pos={self.runtime.current_joint_pos:.4f}", flush=True)
