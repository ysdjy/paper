"""Joint-action place skill: reuse the IK-pose PlaceSkill state machine, output q_des via IK.

Same pattern as :class:`GraspJointSkill`. The pre_place / descend / open phases live in
:class:`PlaceSkill`; this wrapper solves the encapsulated DLS IK for the commanded TCP pose and
emits a joint command for the joint-position env.
"""

from __future__ import annotations

import math
from typing import Any

from runtime.base_skill import SkillCommand, pose_error
from runtime.ik_joint_adapter import IKJointAdapter
from runtime.place_skill import PlaceSkill
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


class PlaceJointSkill:
    backend = "joint_ik"

    def __init__(self, request: SkillRequest, adapter: IKJointAdapter, held_object: Any | None = None):
        self.request = request
        self.adapter = adapter
        self.inner = PlaceSkill(request, held_object=held_object)
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE
        self.last_q_des = None
        self.last_ik_success = None

    @property
    def runtime(self):
        return self.inner.runtime

    @property
    def current_state(self) -> str:
        return self.inner.current_state

    @property
    def resolver(self):
        return self.inner.resolver

    @property
    def last_telemetry(self) -> dict:
        return self.inner.last_telemetry

    def outcomes(self, state: SceneState) -> dict:
        return self.inner.outcomes(state)

    def start(self, state: SceneState):
        self.inner.start(state)
        self.status = self.inner.status

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        pose_command = self.inner.step(state, dt)
        self.status = self.inner.status
        self.failure_reason = self.inner.failure_reason
        return self._to_joint_command(state, pose_command)

    def cancel(self, state: SceneState) -> SkillCommand:
        pose_command = self.inner.cancel(state)
        self.status = self.inner.status
        return self._to_joint_command(state, pose_command)

    def result(self, state: SceneState) -> SkillResult:
        return self.inner.result(state)

    def viz_poses(self) -> list:
        """(name, PoseState) pairs for live target-pose arrows: the current desired TCP pose."""
        tgt = getattr(self, "_last_desired", None)
        return [("phase_target", tgt)] if tgt is not None else []

    def _to_joint_command(self, state: SceneState, pose_command: SkillCommand) -> SkillCommand:
        desired = pose_command.tcp_pose_w
        self._last_desired = desired
        gripper = pose_command.gripper_command
        if self.status in (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.STOPPED):
            q_des = self.last_q_des
            if q_des is None:
                q_des = state.robot.joint_pos[self.adapter._joint_ids].clone()
            return SkillCommand(
                tcp_pose_w=desired,
                gripper_command=gripper,
                status=self.status,
                control_mode="joint",
                joint_target=q_des,
            )

        ik = self.adapter.solve(desired)
        self.last_ik_success = ik.success
        err = pose_error(state.robot.tcp_pose, desired)
        if not ik.success:
            self.inner._fail(state, FailureReason.IK_UNREACHABLE, f"IK failed: {ik.message}")
            self.status = self.inner.status
            self.failure_reason = self.inner.failure_reason
            return SkillCommand(
                tcp_pose_w=desired,
                gripper_command=gripper,
                status=self.status,
                control_mode="joint",
                joint_target=self.last_q_des,
            )
        self.last_q_des = ik.q_des
        print(
            "[PlaceJointSkill] step "
            f"target_point={self.request.destination_object} state={self.current_state} "
            f"target_tcp={[round(float(v),4) for v in desired.pos_w.tolist()]} "
            f"ik_success={ik.success} pos_err={err.position:.4f} ori_err_deg={math.degrees(err.orientation):.2f} "
            f"q_des={[round(float(v),4) for v in ik.q_des.tolist()]}",
            flush=True,
        )
        return SkillCommand(
            tcp_pose_w=desired,
            gripper_command=gripper,
            status=self.status,
            control_mode="joint",
            joint_target=ik.q_des,
        )
