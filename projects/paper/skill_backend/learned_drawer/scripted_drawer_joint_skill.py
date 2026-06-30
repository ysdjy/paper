"""Scripted-joint open/close drawer baseline for the joint-action state machine.

This wraps the existing :class:`DrawerSkill` (which directly commands the cabinet joint target) and
emits a joint command that (a) holds the Franka arm at its current joints and (b) carries the
``drawer_joint_target`` for the runtime to apply via ``set_cabinet_joint_target``.

It is a debug / baseline backend only (``--drawer_backend scripted_joint``). The learned policy
backend (:class:`OfficialDrawerJointSkill`) must be used to satisfy "drawer opened by physical
interaction"; this baseline intentionally sets the joint target directly.
"""

from __future__ import annotations

from runtime.base_skill import SkillCommand
from runtime.drawer_skill import DrawerSkill
from runtime.scene_state_provider import SceneState
from runtime.skill_request import SkillRequest
from runtime.skill_result import SkillResult
from runtime.skill_types import ExecutionStatus, FailureReason


class ScriptedDrawerJointSkill:
    backend = "scripted_joint"

    def __init__(self, request: SkillRequest, arm_joint_ids):
        self.request = request
        self.inner = DrawerSkill(request)
        self._arm_joint_ids = arm_joint_ids
        self.status = ExecutionStatus.IDLE
        self.failure_reason = FailureReason.NONE

    @property
    def runtime(self):
        return self.inner.runtime

    @property
    def current_state(self) -> str:
        return self.inner.current_state

    def start(self, state: SceneState):
        self.inner.start(state)
        self.status = self.inner.status

    def step(self, state: SceneState, dt: float) -> SkillCommand:
        command = self.inner.step(state, dt)
        self.status = self.inner.status
        self.failure_reason = self.inner.failure_reason
        return self._to_joint_command(state, command)

    def cancel(self, state: SceneState) -> SkillCommand:
        command = self.inner.cancel(state)
        self.status = self.inner.status
        return self._to_joint_command(state, command)

    def result(self, state: SceneState) -> SkillResult:
        return self.inner.result(state)

    def _to_joint_command(self, state: SceneState, command: SkillCommand) -> SkillCommand:
        q_hold = state.robot.joint_pos[self._arm_joint_ids].clone()
        return SkillCommand(
            tcp_pose_w=state.robot.tcp_pose,
            gripper_command=command.gripper_command,
            status=self.status,
            control_mode="joint",
            joint_target=q_hold,
            drawer_joint_name=command.drawer_joint_name,
            drawer_joint_target=command.drawer_joint_target,
        )
