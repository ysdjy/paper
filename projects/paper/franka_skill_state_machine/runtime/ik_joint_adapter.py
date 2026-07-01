"""Damped-least-squares IK adapter that turns a world TCP pose into Franka arm joint targets.

This reuses Isaac Lab's :class:`DifferentialIKController` with the same configuration the IK-Abs
stack env uses (``command_type="pose"``, absolute, ``ik_method="dls"``) and the same TCP frame
(``panda_hand`` + 0.1034 m local Z). It performs a single DLS Newton step per call, matching the
per-frame behaviour of the IK-Abs action term, so grasp/place reproduce the IK-Abs trajectory when
fed the same bounded TCP targets.

The adapter outputs ``q_des`` (absolute arm joint angles). It does NOT train anything and protects
against NaNs and joint-limit violations; on failure it reports ``success=False`` so the calling
skill can fail safely instead of diverging.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from runtime.scene_state_provider import PoseState


# TCP offset on panda_hand local +Z, matching ee_frame sensor and IK-Abs body_offset.
TCP_LOCAL_OFFSET_POS = (0.0, 0.0, 0.1034)
TCP_LOCAL_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)


@dataclass
class IKResult:
    success: bool
    q_des: torch.Tensor | None
    position_error: float
    orientation_error: float
    message: str = ""


class IKJointAdapter:
    """Wraps a DLS DifferentialIKController for a single env id.

    Args:
        env: the manager-based env (joint-position variant).
        env_id: which environment to control.
        max_joint_step: clamp on |q_des - q_current| per call to avoid jumps (rad).
    """

    def __init__(self, env, env_id: int = 0, max_joint_step: float = 0.20):
        self.env = env
        self.env_id = env_id
        self.robot = env.unwrapped.scene["robot"]
        self.device = self.robot.device
        self.max_joint_step = float(max_joint_step)

        self._joint_ids, self._joint_names = self.robot.find_joints("panda_joint.*")
        body_ids, body_names = self.robot.find_bodies("panda_hand")
        if len(body_ids) != 1:
            raise RuntimeError(f"expected one panda_hand body, found {body_names}")
        self._body_idx = body_ids[0]
        if self.robot.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        self._offset_pos = torch.tensor(TCP_LOCAL_OFFSET_POS, device=self.device).repeat(self.robot.num_instances, 1)
        self._offset_rot = torch.tensor(TCP_LOCAL_OFFSET_ROT, device=self.device).repeat(self.robot.num_instances, 1)

        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self._controller = DifferentialIKController(cfg=cfg, num_envs=self.robot.num_instances, device=self.device)

        # joint limits for safety clamping
        limits = getattr(self.robot.data, "soft_joint_pos_limits", None)
        if limits is None:
            limits = self.robot.data.joint_pos_limits
        self._joint_lower = limits[self.env_id, self._joint_ids, 0].clone()
        self._joint_upper = limits[self.env_id, self._joint_ids, 1].clone()

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self.robot.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ee_pos_b, ee_quat_b = math_utils.combine_frame_transforms(
            ee_pos_b, ee_quat_b, self._offset_pos, self._offset_rot
        )
        return ee_pos_b, ee_quat_b

    def _compute_frame_jacobian(self) -> torch.Tensor:
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = self.robot.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        # account for the TCP offset
        jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
        jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian

    def solve(self, target_tcp_pose_w: PoseState, null_gain: float = 0.0) -> IKResult:
        """One DLS IK step toward an absolute world TCP pose. Returns q_des [arm_dim].

        ``null_gain`` > 0 enables redundancy resolution: a null-space term that biases the arm toward
        its joint CENTERS without moving the TCP, so a redundant (7-DOF) reach keeps the wrist (j5/j6)
        off its limits instead of saturating it. Used by the door pull, where the gripper must rotate
        ~90deg with the door and the plain DLS solution pins j6 at its limit. Default 0 = unchanged
        (grasp/place/drawer keep their verified behaviour)."""
        if not (torch.isfinite(target_tcp_pose_w.pos_w).all() and torch.isfinite(target_tcp_pose_w.quat_w).all()):
            return IKResult(False, None, float("inf"), float("inf"), "target TCP pose is not finite")

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        des_pos_b, des_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            target_tcp_pose_w.pos_w.to(self.device).unsqueeze(0).expand(self.robot.num_instances, -1),
            math_utils.normalize(target_tcp_pose_w.quat_w.to(self.device).unsqueeze(0)).expand(
                self.robot.num_instances, -1
            ),
        )
        command = torch.cat((des_pos_b, des_quat_b), dim=-1)

        ee_pos_b, ee_quat_b = self._compute_frame_pose()
        jacobian = self._compute_frame_jacobian()
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]

        self._controller.set_command(command, ee_pos_b, ee_quat_b)
        if float(ee_quat_b[self.env_id].norm()) == 0.0:
            return IKResult(False, None, float("inf"), float("inf"), "degenerate current ee quaternion")
        q_des_all = self._controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        q_des = q_des_all[self.env_id]

        if not torch.isfinite(q_des).all():
            return IKResult(False, None, float("inf"), float("inf"), "IK produced non-finite joint targets")

        q_curr = joint_pos[self.env_id]
        # redundancy resolution: add a null-space step toward joint centers (TCP pose unchanged) so the
        # wrist does not saturate during a large reach (e.g. the door pull rotating the gripper ~90deg).
        if null_gain > 0.0:
            try:
                J = jacobian[self.env_id]                                  # [6, 7]
                JT = J.transpose(0, 1)
                JJt = J @ JT + 1.0e-4 * torch.eye(J.shape[0], device=self.device)
                Jpinv = JT @ torch.linalg.inv(JJt)                        # [7, 6]
                N = torch.eye(J.shape[1], device=self.device) - Jpinv @ J  # [7, 7] null-space projector
                q_pref = 0.5 * (self._joint_lower + self._joint_upper)     # joint centers
                q_des = q_des + null_gain * (N @ (q_pref - q_curr))
            except Exception:  # pragma: no cover - linalg guard
                pass

        # step-size limit relative to current joints, then clamp to joint limits
        delta = torch.clamp(q_des - q_curr, -self.max_joint_step, self.max_joint_step)
        q_des = q_curr + delta
        q_des = torch.maximum(torch.minimum(q_des, self._joint_upper), self._joint_lower)

        # report residual error of the (clamped) one-step solution vs current ee pose
        pos_err = float(torch.linalg.norm(des_pos_b[self.env_id] - ee_pos_b[self.env_id]).detach().cpu())
        ori_err = float(
            math_utils.quat_error_magnitude(
                ee_quat_b[self.env_id].unsqueeze(0), des_quat_b[self.env_id].unsqueeze(0)
            )[0].detach().cpu()
        )
        return IKResult(True, q_des.detach(), pos_err, ori_err, "")
