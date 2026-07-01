# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""MoveToPoseSkill: collision-aware SAFE AIR-TRANSFER between contact skills, via cuRobo.

The pure-physical contact skills (grasp/place/open-door/open-drawer) are unchanged. This skill only
moves the arm THROUGH FREE SPACE to a target TCP pose without hitting the cabinet / microwave / open
drawer / other objects, using a cuRobo-planned collision-free joint trajectory executed through the
joint-position action (PD tracks each waypoint) -- the same env action the other joint skills use.

Usage (state-machine style):
    transit = CuroboTransit(env, robot)         # build ONCE (warmup ~12s)
    transit.build_appliance_cuboids(); transit.refresh_world()
    skill = MoveToPoseSkill(transit, target_pos, target_quat, gripper=1.0)
    skill.start(state)
    while not done: cmd = skill.step(state, dt)  # SkillCommand(control_mode="joint", joint_target=q7)

Phases: PLAN -> EXECUTE -> SUCCEEDED/FAILED.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from runtime.base_skill import SkillCommand
from runtime.scene_state_provider import SceneState
from runtime.skill_types import ExecutionStatus, FailureReason


@dataclass
class MoveToPoseConfig:
    waypoint_pos_tol: float = 0.05      # advance to next waypoint when arm joints are this close (rad L2)
    max_dwell_steps: int = 8            # ...or after this many steps on one waypoint (don't stall)
    settle_steps: int = 6              # extra steps holding the final waypoint to settle
    plan_timeout_s: float = 8.0
    refresh_world_on_start: bool = True  # re-extract obstacles + cuboids before planning


@dataclass
class _RT:
    state: str = "IDLE"
    idx: int = 0
    dwell: int = 0
    settle: int = 0
    n_waypoints: int = 0
    start_time: float = 0.0
    fail_msg: str | None = None
    history: list = field(default_factory=list)


class MoveToPoseSkill:
    backend = "curobo"

    def __init__(self, transit, target_pos, target_quat=(0.0, 1.0, 0.0, 0.0), gripper: float = 1.0,
                 config: MoveToPoseConfig | None = None, label: str = "move"):
        self.transit = transit
        self.robot = transit.robot
        self.target_pos = list(target_pos)
        self.target_quat = list(target_quat)
        self.gripper = float(gripper)
        self.cfg = config or MoveToPoseConfig()
        self.label = label
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.runtime = _RT()
        self._traj = None
        self._arm_ids, _ = self.robot.find_joints("panda_joint.*")
        self._last_q = None

    @property
    def current_state(self) -> str:
        return self.runtime.state

    def start(self, state: SceneState):
        self.status = ExecutionStatus.RUNNING
        self.runtime = _RT(state="PLAN", start_time=state.sim_time)
        if self.cfg.refresh_world_on_start:
            try:
                self.transit.refresh_world()
            except Exception as exc:
                print(f"[MoveToPose:{self.label}] WARN refresh_world failed: {exc}", flush=True)
        traj = self.transit.plan(self.target_pos, self.target_quat)
        if traj is None or traj.shape[0] == 0:
            self._fail(state, "cuRobo found no collision-free plan to target")
            return
        self._traj = traj
        self.runtime.n_waypoints = int(traj.shape[0])
        self.runtime.state = "EXECUTE"
        print(f"[MoveToPose:{self.label}] planned {self.runtime.n_waypoints} waypoints -> {self.target_pos}", flush=True)

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        if self.status == ExecutionStatus.IDLE:
            self.start(state)
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            return self._hold(state)
        if self.runtime.state != "EXECUTE" or self._traj is None:
            return self._hold(state)

        idx = min(self.runtime.idx, self.runtime.n_waypoints - 1)
        q_des = self._traj[idx][:7].to(self.robot.device)
        self._last_q = q_des
        cur = state.robot.joint_pos[self._arm_ids].to(q_des.device)
        err = float(torch.linalg.norm(cur - q_des))

        self.runtime.dwell += 1
        if err <= self.cfg.waypoint_pos_tol or self.runtime.dwell >= self.cfg.max_dwell_steps:
            self.runtime.dwell = 0
            if self.runtime.idx >= self.runtime.n_waypoints - 1:
                # at final waypoint: settle then succeed
                self.runtime.settle += 1
                if self.runtime.settle >= self.cfg.settle_steps:
                    self._succeed(state)
            else:
                self.runtime.idx += 1
        return SkillCommand(state.robot.tcp_pose, self.gripper, self.status, control_mode="joint",
                            joint_target=q_des, drawer_joint_target=None)

    def _hold(self, state: SceneState) -> SkillCommand:
        q = self._last_q if self._last_q is not None else state.robot.joint_pos[self._arm_ids].clone()
        return SkillCommand(state.robot.tcp_pose, self.gripper, self.status, control_mode="joint",
                            joint_target=q, drawer_joint_target=None)

    def _fail(self, state: SceneState, msg: str):
        self.status = ExecutionStatus.FAILED
        self.failure_reason = FailureReason.IK_UNREACHABLE
        self.runtime.state = "FAILED"
        self.runtime.fail_msg = msg
        print(f"[MoveToPose:{self.label}] FAILED: {msg}", flush=True)

    def _succeed(self, state: SceneState):
        self.status = ExecutionStatus.SUCCEEDED
        self.runtime.state = "SUCCEEDED"
        print(f"[MoveToPose:{self.label}] SUCCEEDED (reached target)", flush=True)

    def cancel(self, state: SceneState) -> SkillCommand:
        self.status = ExecutionStatus.STOPPED
        self.runtime.state = "CANCELLED"
        return self._hold(state)

    def result(self, state: SceneState):
        from runtime.skill_result import SkillResult

        return SkillResult(
            request_id=getattr(self, "request_id", f"move_{self.label}"),
            skill_type=None,
            target_name=self.label,
            success=self.status == ExecutionStatus.SUCCEEDED,
            final_status=self.status,
            failure_reason=(self.runtime.fail_msg if self.status == ExecutionStatus.FAILED else None),
            elapsed_time=max(0.0, state.sim_time - self.runtime.start_time),
        )
